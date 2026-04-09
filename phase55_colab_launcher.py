"""DELTA Phase 55 Launcher — Local or Colab Execution

Runs phase55_brain_port.py with optimized parameters.
Autodetects environment (local vs Colab) and adjusts paths accordingly.

Local usage:
  cd c:\\dev\\DELTA
  python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30

Colab usage:
  !git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
  %cd /content/DELTA
  !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  !python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30

This will:
- Run brain_hybrid and delta_full baseline
- Use 1 seed (42) for speed validation
- Reduce epochs to 150 (still sufficient for convergence validation)
- Save results to phase55_output.json
- Stream all output to the terminal in real time
"""

import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def run_phase55_colab(
    seeds='42',
    epochs=150,
    eval_every=30,
    batch_size=512,
    lr=0.001,
    target_density=0.02,  # 2% to avoid memory issues
    models='brain_hybrid,delta_full',
):
    """
    Run Phase 55 on Colab with optimized parameters.
    """
    print("="*80)
    print("🧠 DELTA PHASE 55: Brain Architecture Port — COLAB EXECUTION")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Epochs: {epochs} | Eval every: {eval_every}")
    print(f"Batch size: {batch_size} | LR: {lr}")
    print(f"Target density: {target_density} | Seeds: {seeds}")
    print(f"Models: {models}")
    print()
    
    # Verify GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ WARNING: No GPU detected. Will run on CPU (slow).")
    except Exception as e:
        print(f"⚠️ Error checking GPU: {e}")
    
    print()
    
    # Run the actual phase55_brain_port.py
    cmd = [
        sys.executable,
        'experiments/phase55_brain_port.py',
        '--seeds', seeds,
        '--epochs', str(epochs),
        '--eval_every', str(eval_every),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--target_density', str(target_density),
        '--models', models,
    ]
    
    # Determine working directory: use script's parent (works locally and on Colab)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = script_dir  # DELTA root, regardless of where we're invoked from

    print(f"Working directory: {work_dir}")
    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        # Stream output in real time (no buffering)
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0:
            print(f"\n❌ Phase 55 failed with exit code {process.returncode}")
            return False
        print("\n✅ Phase 55 completed successfully!")
    except FileNotFoundError:
        print(f"\n❌ Error: experiments/phase55_brain_port.py not found")
        print("Make sure you're running this from the DELTA directory")
        return False
    
    # Check for output (search in work_dir first, then common locations)
    output_path = None
    for candidate in [
        Path(work_dir) / 'phase55_output.json',
        Path(work_dir) / 'DELTA' / 'phase55_output.json',
        Path('/content/DELTA/phase55_output.json'),
    ]:
        if candidate.exists():
            output_path = candidate
            break

    if output_path and output_path.exists():
        print(f"\n💾 Results saved to {output_path}")

        # Display summary
        try:
            with open(output_path) as f:
                results = json.load(f)

            print("\n" + "="*80)
            print("PHASE 55 SUMMARY")
            print("="*80)

            # Handle both 'summary' and 'results' formats
            summary_data = results.get('summary', {})
            if not summary_data and 'results' in results:
                for r in results['results']:
                    if isinstance(r, dict) and 'model' in r:
                        name = r['model']
                        print(f"\n{name}:")
                        for k, v in r.items():
                            if 'MRR' in k or 'Hits' in k:
                                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                for model_name, data in summary_data.items():
                    if isinstance(data, dict) and 'best_val_mrr' in data:
                        print(f"\n{model_name}:")
                        print(f"  Best Val MRR:  {data['best_val_mrr']:.4f}")
                        if 'test_mrr' in data:
                            print(f"  Test MRR:      {data['test_mrr']:.4f}")

            print("\n" + "="*80)

            # Copy to Drive if mounted (Colab only)
            drive_path = Path('/content/drive/MyDrive/DELTA')
            if drive_path.exists():
                drive_output = drive_path / 'phase55_output.json'
                shutil.copy(output_path, drive_output)
                print(f"\n📤 Results also saved to {drive_output}")

        except Exception as e:
            print(f"⚠️ Could not parse results: {e}")
    else:
        print(f"\n⚠️ Output file not found")
        print("Check the logs above for errors.")
    
    print(f"\nEnd time: {datetime.now().isoformat()}")
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Phase 55 (local or Colab)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (1-2 hours)
  python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30
  
  # Full validation (3-4 hours)
  python phase55_colab_launcher.py --seeds 42,123,456 --epochs 300 --eval_every 50
  
  # Brain models only (fast)
  python phase55_colab_launcher.py --seeds 42 --epochs 150 --models brain_hybrid,brain_pure
        """
    )
    parser.add_argument('--seeds', default='42', help='Seeds to run (comma-separated, default: 42)')
    parser.add_argument('--epochs', type=int, default=150, help='Epochs to train (default: 150)')
    parser.add_argument('--eval_every', type=int, default=30, help='Evaluate every N epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--target_density', type=float, default=0.02, help='Constructor density (default: 0.02)')
    parser.add_argument('--models', default='brain_hybrid,delta_full', help='Models to train (comma-separated)')
    
    args = parser.parse_args()
    
    success = run_phase55_colab(
        seeds=args.seeds,
        epochs=args.epochs,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        lr=args.lr,
        target_density=args.target_density,
        models=args.models,
    )
    
    sys.exit(0 if success else 1)
