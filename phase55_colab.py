"""
Phase 55: Brain Architecture Port — Colab Execution Script

Runs a minimal validation (1 seed, 200 epochs) of BrainEncoder on FB15k-237 LP.
Designed for Colab's 15-16GB GPU with fresh CUDA state.

Hypothesis: brain_hybrid achieves LP MRR ≥ 0.475 (baseline A = 0.4744)
Target: Prove learned graph augmentation improves LP link prediction.

Usage in Colab:
    !git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
    %cd /content/DELTA
    !python phase55_colab.py --seed 42 --epochs 200 --target_density 0.02 --eval_every 50
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path

# Add DELTA to path
sys.path.insert(0, '/content/DELTA' if os.path.exists('/content/DELTA') else '.')

from delta.model import DELTAModel
from delta.brain import BrainEncoder
from data.fb15k_237 import FB15k237
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

def main():
    parser = argparse.ArgumentParser(description='Phase 55: Brain Port on Colab')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--target_density', type=float, default=0.02, help='Constructor edge density (2%)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"🧠 Phase 55: Brain Architecture Port (Colab)")
    print(f"Device: {args.device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Load FB15k-237
    print("\n📊 Loading FB15k-237...")
    data = FB15k237(subset='fb15k237_debug' if not torch.cuda.is_available() else 'fb15k237')
    num_nodes = data.num_nodes
    num_relations = data.num_relations
    
    print(f"Nodes: {num_nodes} | Relations: {num_relations}")
    print(f"Train: {len(data.train)} | Val: {len(data.val)} | Test: {len(data.test)}")
    
    # Prepare training data (link prediction)
    train_pos = torch.tensor(data.train, dtype=torch.long)
    train_neg = torch.randint(0, num_nodes, (len(data.train), 1), dtype=torch.long)
    train_pos_full = torch.cat([train_pos[:, :2], train_pos[:, 2:3]], dim=1)  # (h, r, t)
    
    train_h = train_pos[:, 0]
    train_r = train_pos[:, 1]
    train_t_pos = train_pos[:, 2]
    train_t_neg = train_neg.squeeze(1)
    
    val_pos = torch.tensor(data.val, dtype=torch.long)
    val_h = val_pos[:, 0]
    val_r = val_pos[:, 1]
    val_t = val_pos[:, 2]
    
    test_pos = torch.tensor(data.test, dtype=torch.long)
    test_h = test_pos[:, 0]
    test_r = test_pos[:, 1]
    test_t = test_pos[:, 2]
    
    # Create models: brain_hybrid and delta_full baseline
    print("\n🔧 Creating models...")
    
    brain_hybrid = BrainEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        d_node=64,
        d_edge=32,
        num_layers=3,
        num_heads=4,
        target_density=args.target_density,
        constructor_mode='hybrid'
    ).to(device)
    
    delta_full = DELTAModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        d_node=64,
        d_edge=32,
        num_layers=3,
        num_heads=4
    ).to(device)
    
    print(f"BrainEncoder (hybrid): {sum(p.numel() for p in brain_hybrid.parameters())} params")
    print(f"DELTAModel (baseline): {sum(p.numel() for p in delta_full.parameters())} params")
    
    # Optimizers
    opt_brain = Adam(brain_hybrid.parameters(), lr=args.lr)
    opt_delta = Adam(delta_full.parameters(), lr=args.lr)
    
    results = {
        'phase': 55,
        'seed': args.seed,
        'epochs': args.epochs,
        'target_density': args.target_density,
        'device': str(device),
        'gpu_memory_gb': float(torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else None,
        'models': {
            'brain_hybrid': {'params': sum(p.numel() for p in brain_hybrid.parameters()), 'results': []},
            'delta_full': {'params': sum(p.numel() for p in delta_full.parameters()), 'results': []}
        }
    }
    
    # Training loop
    best_val_mrr_brain = 0.0
    best_val_mrr_delta = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train brain_hybrid
        brain_hybrid.train()
        train_loss_brain = 0.0
        
        for i in range(0, len(train_h), args.batch_size):
            end = min(i + args.batch_size, len(train_h))
            batch_h = train_h[i:end].to(device)
            batch_r = train_r[i:end].to(device)
            batch_t_pos = train_t_pos[i:end].to(device)
            batch_t_neg = train_t_neg[i:end].to(device)
            
            opt_brain.zero_grad()
            
            # Forward pass on positive
            logits_pos = brain_hybrid(batch_h, batch_r, batch_t_pos)
            loss_pos = F.cross_entropy(logits_pos, torch.ones_like(logits_pos[:, 0]))
            
            # Forward pass on negative
            logits_neg = brain_hybrid(batch_h, batch_r, batch_t_neg)
            loss_neg = F.cross_entropy(logits_neg, torch.zeros_like(logits_neg[:, 0]))
            
            loss = loss_pos + loss_neg
            loss.backward()
            opt_brain.step()
            
            train_loss_brain += loss.item()
        
        # Train delta_full baseline
        delta_full.train()
        train_loss_delta = 0.0
        
        for i in range(0, len(train_h), args.batch_size):
            end = min(i + args.batch_size, len(train_h))
            batch_h = train_h[i:end].to(device)
            batch_r = train_r[i:end].to(device)
            batch_t_pos = train_t_pos[i:end].to(device)
            batch_t_neg = train_t_neg[i:end].to(device)
            
            opt_delta.zero_grad()
            
            logits_pos = delta_full(batch_h, batch_r, batch_t_pos)
            loss_pos = F.cross_entropy(logits_pos, torch.ones_like(logits_pos[:, 0]))
            
            logits_neg = delta_full(batch_h, batch_r, batch_t_neg)
            loss_neg = F.cross_entropy(logits_neg, torch.zeros_like(logits_neg[:, 0]))
            
            loss = loss_pos + loss_neg
            loss.backward()
            opt_delta.step()
            
            train_loss_delta += loss.item()
        
        # Evaluate
        if epoch % args.eval_every == 0 or epoch == 1:
            brain_hybrid.eval()
            delta_full.eval()
            
            with torch.no_grad():
                # Validation
                val_logits_brain = []
                for i in range(0, len(val_h), args.batch_size):
                    end = min(i + args.batch_size, len(val_h))
                    batch_h = val_h[i:end].to(device)
                    batch_r = val_r[i:end].to(device)
                    batch_t = val_t[i:end].to(device)
                    val_logits_brain.append(brain_hybrid(batch_h, batch_r, batch_t).cpu())
                
                val_logits_brain = torch.cat(val_logits_brain, dim=0)
                val_mrr_brain = (1.0 / (val_logits_brain.argsort(dim=1, descending=True).argsort(dim=1)[:, 0].float() + 1)).mean().item()
                
                val_logits_delta = []
                for i in range(0, len(val_h), args.batch_size):
                    end = min(i + args.batch_size, len(val_h))
                    batch_h = val_h[i:end].to(device)
                    batch_r = val_r[i:end].to(device)
                    batch_t = val_t[i:end].to(device)
                    val_logits_delta.append(delta_full(batch_h, batch_r, batch_t).cpu())
                
                val_logits_delta = torch.cat(val_logits_delta, dim=0)
                val_mrr_delta = (1.0 / (val_logits_delta.argsort(dim=1, descending=True).argsort(dim=1)[:, 0].float() + 1)).mean().item()
                
                best_val_mrr_brain = max(best_val_mrr_brain, val_mrr_brain)
                best_val_mrr_delta = max(best_val_mrr_delta, val_mrr_delta)
                
                results['models']['brain_hybrid']['results'].append({
                    'epoch': epoch,
                    'train_loss': train_loss_brain / (len(train_h) // args.batch_size),
                    'val_mrr': val_mrr_brain,
                    'best_val_mrr': best_val_mrr_brain
                })
                results['models']['delta_full']['results'].append({
                    'epoch': epoch,
                    'train_loss': train_loss_delta / (len(train_h) // args.batch_size),
                    'val_mrr': val_mrr_delta,
                    'best_val_mrr': best_val_mrr_delta
                })
                
                print(f"Ep {epoch:3d} | Brain val_MRR={val_mrr_brain:.4f} (best={best_val_mrr_brain:.4f}) | "
                      f"DELTA val_MRR={val_mrr_delta:.4f} (best={best_val_mrr_delta:.4f})")
    
    # Final test evaluation
    print("\n📈 Final Test Evaluation...")
    brain_hybrid.eval()
    delta_full.eval()
    
    with torch.no_grad():
        test_logits_brain = []
        for i in range(0, len(test_h), args.batch_size):
            end = min(i + args.batch_size, len(test_h))
            batch_h = test_h[i:end].to(device)
            batch_r = test_r[i:end].to(device)
            batch_t = test_t[i:end].to(device)
            test_logits_brain.append(brain_hybrid(batch_h, batch_r, batch_t).cpu())
        
        test_logits_brain = torch.cat(test_logits_brain, dim=0)
        test_mrr_brain = (1.0 / (test_logits_brain.argsort(dim=1, descending=True).argsort(dim=1)[:, 0].float() + 1)).mean().item()
        
        test_logits_delta = []
        for i in range(0, len(test_h), args.batch_size):
            end = min(i + args.batch_size, len(test_h))
            batch_h = test_h[i:end].to(device)
            batch_r = test_r[i:end].to(device)
            batch_t = test_t[i:end].to(device)
            test_logits_delta.append(delta_full(batch_h, batch_r, batch_t).cpu())
        
        test_logits_delta = torch.cat(test_logits_delta, dim=0)
        test_mrr_delta = (1.0 / (test_logits_delta.argsort(dim=1, descending=True).argsort(dim=1)[:, 0].float() + 1)).mean().item()
    
    results['final_test'] = {
        'brain_hybrid_mrr': test_mrr_brain,
        'delta_full_mrr': test_mrr_delta,
        'delta_improvement': test_mrr_brain - test_mrr_delta,
        'baseline_a': 0.4744  # Baseline A from Phase 54
    }
    
    # Summary
    print("\n" + "="*70)
    print(f"PHASE 55 RESULTS (Seed {args.seed})")
    print("="*70)
    print(f"BrainEncoder (hybrid)  | Best Val MRR: {best_val_mrr_brain:.4f} | Test MRR: {test_mrr_brain:.4f}")
    print(f"DELTAModel (baseline)  | Best Val MRR: {best_val_mrr_delta:.4f} | Test MRR: {test_mrr_delta:.4f}")
    print(f"Baseline A (Phase 54)  | LP MRR: 0.4744")
    print("-"*70)
    
    if test_mrr_brain >= 0.475:
        print(f"✅ HYPOTHESIS CONFIRMED: brain_hybrid ({test_mrr_brain:.4f}) ≥ 0.475")
    else:
        print(f"❌ HYPOTHESIS REJECTED: brain_hybrid ({test_mrr_brain:.4f}) < 0.475")
    
    if test_mrr_brain > test_mrr_delta:
        print(f"✅ Brain augmentation improves LP: +{(test_mrr_brain - test_mrr_delta):.4f}")
    else:
        print(f"❌ Brain augmentation hurts LP: {(test_mrr_brain - test_mrr_delta):.4f}")
    
    print("="*70)
    
    # Save results
    output_path = Path('/content/DELTA' if os.path.exists('/content/DELTA') else '.') / 'phase55_output.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")
    
    return results

if __name__ == '__main__':
    main()
