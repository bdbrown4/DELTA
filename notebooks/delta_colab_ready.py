"""
DELTA — Google Colab-Ready Infrastructure Script

This script automates environment setup and experiment execution for
running DELTA on Google Colab Pro+ (A100 GPU). It can also run locally
on CPU for validation.

Usage in Google Colab:
    !git clone https://github.com/bdbrown4/DELTA.git
    %cd DELTA
    !pip install torch>=2.0.0 numpy>=1.24.0
    !python notebooks/delta_colab_ready.py

Usage locally (CPU, for testing):
    python notebooks/delta_colab_ready.py --cpu-only

Sections:
    1. Environment detection and GPU setup
    2. Repository validation (run all tests)
    3. Phase 34: DELTA vs GraphGPS vs GRIT comparison
    4. Phase 31 preparation: mini-batching scaffolding for full FB15k-237
    5. Results export
"""

import sys
import os
import time
import argparse
import json
from datetime import datetime

# Ensure we can import delta from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from delta.baselines import GraphGPSModel, GRITModel
from delta.utils import (
    create_analogy_task,
    create_noisy_kg_benchmark,
    create_multi_relational_reasoning_task,
    create_knowledge_graph,
    calculate_graph_statistics,
)


# ===================================================================
# Section 1: Environment Detection & GPU Setup
# ===================================================================

def detect_environment():
    """Detect runtime environment and available hardware."""
    env = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_memory_gb': None,
        'is_colab': False,
        'device': 'cpu',
    }

    # Check if running in Google Colab
    try:
        import google.colab  # noqa: F401
        env['is_colab'] = True
    except ImportError:
        pass

    # GPU detection
    if env['cuda_available']:
        env['device'] = 'cuda'
        env['gpu_name'] = torch.cuda.get_device_name(0)
        env['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return env


def print_environment(env):
    """Display environment info."""
    print("=" * 70)
    print("DELTA COLAB INFRASTRUCTURE — ENVIRONMENT")
    print("=" * 70)
    print(f"  Timestamp:  {env['timestamp']}")
    print(f"  Python:     {env['python_version'].split()[0]}")
    print(f"  PyTorch:    {env['torch_version']}")
    print(f"  CUDA:       {'Yes' if env['cuda_available'] else 'No'}")
    if env['gpu_name']:
        print(f"  GPU:        {env['gpu_name']}")
        print(f"  VRAM:       {env['gpu_memory_gb']:.1f} GB")
    print(f"  Colab:      {'Yes' if env['is_colab'] else 'No'}")
    print(f"  Device:     {env['device']}")
    print()


# ===================================================================
# Section 2: Repository Validation
# ===================================================================

def run_validation_tests():
    """Run all unit tests to verify the environment is set up correctly."""
    print("=" * 70)
    print("VALIDATION: Running all unit tests")
    print("=" * 70)

    test_files = [
        'tests/test_graph.py',
        'tests/test_attention.py',
        'tests/test_router.py',
        'tests/test_memory.py',
        'tests/test_utils.py',
        'tests/test_baselines.py',
    ]

    all_passed = True
    for test_file in test_files:
        full_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), test_file)
        if os.path.exists(full_path):
            ret = os.system(f'{sys.executable} {full_path}')
            if ret != 0:
                print(f"  FAIL: {test_file}")
                all_passed = False
        else:
            print(f"  SKIP: {test_file} (not found)")

    if all_passed:
        print("\n  All tests passed ✓\n")
    else:
        print("\n  Some tests failed — check output above.\n")

    return all_passed


# ===================================================================
# Section 3: Phase 34 — DELTA vs GraphGPS vs GRIT
# ===================================================================

def train_model(model, graph, labels, epochs=200, lr=1e-3, seed=42, device='cpu'):
    """Train a model on edge classification and return results."""
    torch.manual_seed(seed)

    # Move model and data to device
    model = model.to(device)
    graph = graph.to(device)
    labels = labels.to(device)

    E = labels.shape[0]
    perm = torch.randperm(E, device=device)
    train_end = int(E * 0.7)
    train_idx = perm[:train_end]
    test_idx = perm[train_end:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        out = model(graph)
        logits = model.classify_edges(out)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(graph)
                logits = model.classify_edges(out)
                test_acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean().item()
                best_test_acc = max(best_test_acc, test_acc)

    return {
        'best_test_acc': best_test_acc,
        'training_time_s': time.time() - start,
    }


def run_phase34_comparison(device='cpu', num_seeds=3, epochs=200):
    """Run Phase 34: DELTA vs GraphGPS vs GRIT on synthetic tasks."""
    print("=" * 70)
    print("PHASE 34: DELTA vs GraphGPS vs GRIT Comparison")
    print("=" * 70)
    print(f"  Device: {device}, Seeds: {num_seeds}, Epochs: {epochs}")
    print()

    d_node, d_edge = 64, 32
    num_patterns = 6

    all_results = {}

    # --- Task 1: Edge Classification ---
    print("--- Task 1: Edge Classification (Synthetic KG) ---")
    task1_results = {name: [] for name in ['DELTA', 'GraphGPS', 'GRIT']}

    for seed in range(num_seeds):
        graph, labels = create_analogy_task(
            num_patterns=num_patterns, instances_per_pattern=8,
            d_node=d_node, d_edge=d_edge, seed=seed,
        )
        print(f"  Seed {seed}: {graph.num_nodes} nodes, {graph.num_edges} edges")

        models = {
            'DELTA': DELTAModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                num_heads=4, num_classes=num_patterns),
            'GraphGPS': GraphGPSModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                      num_heads=4, num_classes=num_patterns),
            'GRIT': GRITModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                              num_heads=4, walk_length=8, d_pe=16,
                              num_classes=num_patterns),
        }

        for name, model in models.items():
            r = train_model(model, graph, labels, epochs, seed=seed, device=device)
            task1_results[name].append(r['best_test_acc'])
            print(f"    {name:<10s}  Test Acc: {r['best_test_acc']:.3f}  "
                  f"({r['training_time_s']:.1f}s)")

    all_results['edge_classification'] = task1_results

    # --- Task 2: Noise Robustness ---
    print("\n--- Task 2: Noise Robustness ---")
    for noise in [0.0, 0.3, 0.6]:
        task_key = f'noise_{noise:.1f}'
        noise_results = {name: [] for name in ['DELTA', 'GraphGPS', 'GRIT']}

        for seed in range(num_seeds):
            graph, labels, metadata = create_noisy_kg_benchmark(
                num_entities=50, num_relations=6,
                num_triples=200,
                d_node=d_node, d_edge=d_edge,
                noise_ratio=noise, seed=seed,
            )
            num_classes = labels.max().item() + 1
            models = {
                'DELTA': DELTAModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                    num_heads=4, num_classes=num_classes),
                'GraphGPS': GraphGPSModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                          num_heads=4, num_classes=num_classes),
                'GRIT': GRITModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                  num_heads=4, walk_length=8, d_pe=16,
                                  num_classes=num_classes),
            }

            for name, model in models.items():
                r = train_model(model, graph, labels, epochs, seed=seed, device=device)
                noise_results[name].append(r['best_test_acc'])

        all_results[task_key] = noise_results
        print(f"  Noise={noise:.1f}: " + "  ".join(
            f"{n}: {sum(v)/len(v):.3f}" for n, v in noise_results.items()))

    # --- Task 3: Path Composition ---
    print("\n--- Task 3: Compositional Relational Reasoning ---")
    path_results = {name: [] for name in ['DELTA', 'GraphGPS', 'GRIT']}

    for seed in range(num_seeds):
        graph, labels, metadata = create_multi_relational_reasoning_task(
            num_entities=60, num_base_relations=4, num_derived_rules=3,
            d_node=d_node, d_edge=d_edge, seed=seed,
        )
        num_classes = metadata['num_total_relations']
        models = {
            'DELTA': DELTAModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                num_heads=4, num_classes=num_classes),
            'GraphGPS': GraphGPSModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                                      num_heads=4, num_classes=num_classes),
            'GRIT': GRITModel(d_node=d_node, d_edge=d_edge, num_layers=3,
                              num_heads=4, walk_length=8, d_pe=16,
                              num_classes=num_classes),
        }

        for name, model in models.items():
            r = train_model(model, graph, labels, epochs, seed=seed, device=device)
            path_results[name].append(r['best_test_acc'])
            print(f"    Seed {seed} {name:<10s}  Test Acc: {r['best_test_acc']:.3f}")

    all_results['path_composition'] = path_results

    return all_results


def print_summary(results):
    """Print formatted summary of Phase 34 results."""
    print()
    print("=" * 70)
    print("PHASE 34 RESULTS SUMMARY")
    print("=" * 70)

    for task_name, task_results in results.items():
        print(f"\n  {task_name}:")
        for name in ['DELTA', 'GraphGPS', 'GRIT']:
            accs = task_results[name]
            if accs:
                mean = sum(accs) / len(accs)
                if len(accs) > 1:
                    variance = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                    std = variance ** 0.5
                    print(f"    {name:<10s}  {mean:.3f} ± {std:.3f}  (n={len(accs)})")
                else:
                    print(f"    {name:<10s}  {mean:.3f}  (n=1)")


# ===================================================================
# Section 4: Phase 31 Preparation — Mini-Batching Scaffolding
# ===================================================================

def check_phase31_readiness(env):
    """Check if environment supports Phase 31 (full FB15k-237)."""
    print()
    print("=" * 70)
    print("PHASE 31 READINESS CHECK")
    print("=" * 70)

    requirements = {
        'CUDA available': env['cuda_available'],
        'VRAM >= 20GB': (env.get('gpu_memory_gb') or 0) >= 20,
        'A100 or better': env['gpu_name'] is not None and 'A100' in (env['gpu_name'] or ''),
    }

    for req, met in requirements.items():
        status = '✓' if met else '✗'
        print(f"  [{status}] {req}")

    if all(requirements.values()):
        print("\n  Environment ready for Phase 31!")
        print("  Full FB15k-237 (14,505 entities) can be loaded without subsampling.")
    elif requirements['CUDA available']:
        print(f"\n  GPU detected ({env['gpu_name']}) but may need larger VRAM for full FB15k-237.")
        print("  Consider: gradient accumulation, or subsample to 5000 entities first.")
    else:
        print("\n  Running on CPU — Phase 31 requires GPU for full-scale training.")
        print("  Use this script on Google Colab Pro+ (see docs/COLAB_SETUP.md).")
        print("  Phase 34 synthetic comparison can still run on CPU.")

    print()
    return all(requirements.values())


# ===================================================================
# Section 5: Results Export
# ===================================================================

def export_results(results, env, output_dir='results'):
    """Save results to JSON for later analysis."""
    os.makedirs(output_dir, exist_ok=True)

    output = {
        'environment': env,
        'results': results,
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'phase34_results_{timestamp}.json')

    # Convert any non-serializable values
    def serialize(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        return obj

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=serialize)

    print(f"  Results saved to: {filepath}")
    return filepath


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="DELTA Colab Infrastructure")
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU execution (for local testing)')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip validation tests')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds for Phase 34')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs for Phase 34')
    args = parser.parse_args()

    # Section 1: Environment
    env = detect_environment()
    if args.cpu_only:
        env['device'] = 'cpu'
    print_environment(env)

    # Section 2: Validation
    if not args.skip_tests:
        tests_ok = run_validation_tests()
        if not tests_ok:
            print("Tests failed — fix issues before running experiments.")
            return

    # Section 3: Phase 34
    results = run_phase34_comparison(env['device'], args.seeds, args.epochs)
    print_summary(results)

    # Section 4: Phase 31 readiness
    phase31_ready = check_phase31_readiness(env)

    # Section 5: Export
    print("=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    filepath = export_results(results, env)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Next steps:")
    if phase31_ready:
        print("  1. Run Phase 31 (full FB15k-237 with mini-batching)")
        print("  2. Run Phase 34b (this comparison on real data)")
    else:
        print("  1. Set up Google Colab Pro+ (see docs/COLAB_SETUP.md)")
        print("  2. Rerun this script on A100 GPU")
    print("  3. Update docs/RESEARCH_AGENDA.md with results")
    print()


if __name__ == '__main__':
    main()
