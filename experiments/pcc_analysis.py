"""
PCC Analysis Script - Refined Version

Generates a leaderboard of functions ranked by their PCC (Persistent Cycle Complexity) scores.
Outputs "Topological Winners and Losers" summary.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from boolean_gen import generate_truth_table
from topology_calc import compute_persistence, calculate_pcc

def run_pcc_analysis(n=8):
    print("=" * 60)
    print(f"PCC ANALYSIS - N={n}")
    print("=" * 60)
    
    results = []  # List of (name, pcc_score)
    
    # =============================================
    # Section 1: Core Named Functions
    # =============================================
    print("\n[1] Analyzing Core Named Functions...")
    
    named_functions = [
        ('Majority', 'majority', {}),
        ('Parity', 'parity', {}),
        ('Threshold_k3', 'threshold', {'k': 3}),
        ('Threshold_k4', 'threshold', {'k': 4}),
        ('Threshold_k5', 'threshold', {'k': 5}),
    ]
    
    for name, f_type, kwargs in named_functions:
        tt, coords = generate_truth_table(n, f_type, **kwargs)
        if len(coords) == 0:
            print(f"  {name}: Empty On-Set, skipping.")
            continue
        dgms = compute_persistence(coords, max_dim=1)
        pcc = calculate_pcc(dgms)
        results.append((name, pcc))
        print(f"  {name}: PCC = {pcc:.4f}")
    
    # =============================================
    # Section 2: Random Functions (50)
    # =============================================
    print("\n[2] Analyzing 50 Random Functions...")
    
    for i in range(50):
        tt, coords = generate_truth_table(n, 'random', seed=i*42)
        if len(coords) == 0:
            continue
        dgms = compute_persistence(coords, max_dim=1)
        pcc = calculate_pcc(dgms)
        results.append((f"Random_{i+1}", pcc))
    
    print(f"  Completed 50 Random function analyses.")
    
    # =============================================
    # Section 3: Structured (Noisy) Functions (50)
    # =============================================
    print("\n[3] Analyzing 50 Structured (Noisy) Functions...")
    
    np.random.seed(12345)
    structured_bases = ['majority', 'parity', 'threshold']
    
    for i in range(50):
        base = np.random.choice(structured_bases)
        tt_base, _ = generate_truth_table(n, base)
        
        # Add small noise (2% bit flips)
        noise_mask = np.random.random(tt_base.shape) < 0.02
        tt_noisy = tt_base.copy()
        tt_noisy[noise_mask] = 1 - tt_noisy[noise_mask]
        
        # Get coords for noisy truth table
        masks = 1 << np.arange(n)[::-1]
        all_indices = np.arange(2**n, dtype=int)
        all_coords = ((all_indices[:, None] & masks) > 0).astype(int)
        
        indices = np.where(tt_noisy == 1)[0]
        coords = all_coords[indices]
        
        if len(coords) == 0:
            continue
            
        dgms = compute_persistence(coords, max_dim=1)
        pcc = calculate_pcc(dgms)
        results.append((f"Noisy_{base.capitalize()}_{i+1}", pcc))
    
    print(f"  Completed 50 Structured (Noisy) function analyses.")
    
    # =============================================
    # Section 4: Leaderboard
    # =============================================
    print("\n" + "=" * 60)
    print("TOPOLOGICAL WINNERS AND LOSERS")
    print("=" * 60)
    
    # Sort by PCC
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("\n🏆 TOP 5 HIGHEST PCC (Most Topologically Complex):")
    print("-" * 40)
    for i, (name, pcc) in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {name}: PCC = {pcc:.4f}")
    
    print("\n📉 BOTTOM 5 LOWEST PCC (Least Topologically Complex):")
    print("-" * 40)
    for i, (name, pcc) in enumerate(sorted_results[-5:], 1):
        print(f"  {i}. {name}: PCC = {pcc:.4f}")
    
    # =============================================
    # Section 5: Hypothesis Check
    # =============================================
    print("\n" + "=" * 60)
    print("HYPOTHESIS CHECK: Named Functions")
    print("=" * 60)
    
    named_pcc = {name: pcc for name, pcc in results if not name.startswith(('Random_', 'Noisy_'))}
    
    print("\n| Function       | PCC Score |")
    print("|----------------|-----------|")
    for name, pcc in sorted(named_pcc.items(), key=lambda x: x[1], reverse=True):
        print(f"| {name:<14} | {pcc:>9.4f} |")
    
    # Summary stats
    random_pccs = [pcc for name, pcc in results if name.startswith('Random_')]
    structured_pccs = [pcc for name, pcc in results if name.startswith('Noisy_')]
    
    print("\n" + "-" * 40)
    print(f"Random Functions (n=50):     Mean PCC = {np.mean(random_pccs):.4f} ± {np.std(random_pccs):.4f}")
    print(f"Structured Functions (n=50): Mean PCC = {np.mean(structured_pccs):.4f} ± {np.std(structured_pccs):.4f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_pcc_analysis(n=8)
