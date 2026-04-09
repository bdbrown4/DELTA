"""
DELTA Phase 55 - Colab Launcher

Runs phase55_brain_port.py on Google Colab with optimized parameters.
Mount your Google Drive at /content/drive to save results.

Installation & Setup:
```
!git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
%cd /content/DELTA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q tqdm numpy scipy scikit-learn
```

Then run this script:
```
!python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30
```

This will:
- Run brain_hybrid and delta_full baseline
- Use 1 seed (42) for speed validation
- Reduce epochs to 150 (still sufficient for convergence validation)
- Save results to phase55_output.json
- If Drive mounted, also copy to /content/drive/MyDrive/DELTA/
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
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd='/content/DELTA', check=True)
        print("\n✅ Phase 55 completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Phase 55 failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Error: experiments/phase55_brain_port.py not found")
        print("Make sure you're running this from the DELTA directory")
        return False
    
    # Check for output
    output_path = Path('/content/DELTA/phase55_output.json')
    if output_path.exists():
        print(f"\n💾 Results saved to {output_path}")
        
        # Display summary
        try:
            with open(output_path) as f:
                results = json.load(f)
            
            print("\n" + "="*80)
            print("PHASE 55 SUMMARY")
            print("="*80)
            
            if 'summary' in results:
                summary = results['summary']
                for model_name, data in summary.items():
                    if isinstance(data, dict) and 'best_val_mrr' in data:
                        print(f"\n{model_name}:")
                        print(f"  Best Val MRR:  {data['best_val_mrr']:.4f}")
                        if 'test_mrr' in data:
                            print(f"  Test MRR:      {data['test_mrr']:.4f}")
            
            print("\n" + "="*80)
            
            # Copy to Drive if mounted
            drive_path = Path('/content/drive/MyDrive/DELTA')
            if drive_path.exists():
                drive_output = drive_path / 'phase55_output.json'
                shutil.copy(output_path, drive_output)
                print(f"\n📤 Results also saved to {drive_output}")
            
        except Exception as e:
            print(f"⚠️ Could not parse results: {e}")
    else:
        print(f"\n⚠️ Output file not found at {output_path}")
        print("Check the logs above for errors.")
    
    print(f"\nEnd time: {datetime.now().isoformat()}")
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Phase 55 on Colab',
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
