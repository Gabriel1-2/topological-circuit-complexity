"""
Scale Experiment: N=16 with 2,000-point samples

Tests whether PCC analysis scales to larger N using lazy sampling.
Compares Threshold_k8 vs Random over 5 trials each.
Uses Hamming distance for proper hypercube topology.
"""

import sys
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ripser

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from lazy_sampler import sample_active_points
from topology_calc import calculate_pcc_normalized

def compute_persistence_hamming(points: np.ndarray, max_dim: int = 1):
    """Compute persistence using Hamming distance matrix."""
    # Compute Hamming distance matrix
    dist_matrix = squareform(pdist(points, metric='hamming')) * points.shape[1]
    # Run Ripser
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
    return results['dgms']

def run_scale_experiment():
    print("=" * 70)
    print("SCALE EXPERIMENT: N=16, M=2000 (Hamming Distance)")
    print("=" * 70)
    
    N = 16
    M = 2000
    TRIALS = 5
    
    print(f"\nConfiguration:")
    print(f"  • N = {N} (Total space = {2**N:,} points)")
    print(f"  • M = {M} samples ({100*M/2**N:.1f}% of data)")
    print(f"  • Trials = {TRIALS}")
    print(f"  • Distance = Hamming (proper hypercube metric)")
    
    results = []
    
    # Test Threshold_k8
    print(f"\n[1] Testing Threshold_k8...")
    for trial in range(1, TRIALS + 1):
        pts = sample_active_points(N, M, 'threshold', k=8, seed=trial * 100)
        dgms = compute_persistence_hamming(pts, max_dim=1)
        metrics = calculate_pcc_normalized(dgms)
        results.append(('Threshold_k8', trial, metrics))
        print(f"    Trial {trial}: H0_entropy={metrics['h0_entropy']:.4f}, H0_max={metrics['h0_max_lifetime']:.2f}, H1={metrics['h1_total']:.4f}")
    
    # Test Random
    print(f"\n[2] Testing Random...")
    for trial in range(1, TRIALS + 1):
        pts = sample_active_points(N, M, 'random', seed=trial * 200)
        dgms = compute_persistence_hamming(pts, max_dim=1)
        metrics = calculate_pcc_normalized(dgms)
        results.append(('Random', trial, metrics))
        print(f"    Trial {trial}: H0_entropy={metrics['h0_entropy']:.4f}, H0_max={metrics['h0_max_lifetime']:.2f}, H1={metrics['h1_total']:.4f}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Function':<15} | {'Trial':<6} | {'H0 Entropy':<12} | {'H0 Mean':<10} | {'H0 Max':<10}")
    print("-" * 65)
    
    for func, trial, m in results:
        print(f"{func:<15} | {trial:<6} | {m['h0_entropy']:<12.4f} | {m['h0_mean_lifetime']:<10.4f} | {m['h0_max_lifetime']:<10.4f}")
    
    # Statistics
    thresh_entropy = [m['h0_entropy'] for f, t, m in results if f == 'Threshold_k8']
    random_entropy = [m['h0_entropy'] for f, t, m in results if f == 'Random']
    thresh_mean = [m['h0_mean_lifetime'] for f, t, m in results if f == 'Threshold_k8']
    random_mean = [m['h0_mean_lifetime'] for f, t, m in results if f == 'Random']
    
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"\nThreshold_k8:")
    print(f"  H0 Entropy:      {np.mean(thresh_entropy):.4f} ± {np.std(thresh_entropy):.4f}")
    print(f"  H0 Mean Lifetime: {np.mean(thresh_mean):.4f} ± {np.std(thresh_mean):.4f}")
    
    print(f"\nRandom:")
    print(f"  H0 Entropy:      {np.mean(random_entropy):.4f} ± {np.std(random_entropy):.4f}")
    print(f"  H0 Mean Lifetime: {np.mean(random_mean):.4f} ± {np.std(random_mean):.4f}")
    
    # Hypothesis check
    print("\n" + "=" * 70)
    print("HYPOTHESIS CHECK")
    print("=" * 70)
    
    entropy_sep = np.mean(thresh_entropy) - np.mean(random_entropy)
    mean_sep = np.mean(thresh_mean) - np.mean(random_mean)
    
    print(f"\n  Entropy Separation: {entropy_sep:.4f}")
    print(f"  Mean Lifetime Separation: {mean_sep:.4f}")
    
    if abs(entropy_sep) > 0.1 or abs(mean_sep) > 0.01:
        print("\n  ✓ SCALING SUCCESS: Metrics show distinguishable differences.")
    else:
        print("\n  ✗ SCALING CONCERN: Metrics too similar to distinguish.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_scale_experiment()
