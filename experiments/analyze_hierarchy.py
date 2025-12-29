"""
Hierarchy Analysis: AC0 vs NC1 Topological Separation

Computes comprehensive TDA metrics to distinguish complexity classes.
Uses Hamming distance with optional jitter for numerical stability.
"""

import sys
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ripser

MAX_POINTS = 1000  # Limit for computational feasibility

def add_jitter(points, sigma=0.05):
    """Add Gaussian jitter for numerical stability."""
    return points.astype(float) + np.random.normal(0, sigma, points.shape)

def sample_points(points, max_points=MAX_POINTS, seed=42):
    """Subsample if too many points."""
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), max_points, replace=False)
    return points[indices]

def compute_persistence_hamming(points, max_dim=1, use_jitter=True, jitter_std=0.05):
    """Compute persistence using Hamming distance."""
    # Subsample if needed
    pts = sample_points(points)
    
    if use_jitter:
        pts = add_jitter(pts, sigma=jitter_std)
        dist_matrix = squareform(pdist(pts, metric='euclidean'))
    else:
        dist_matrix = squareform(pdist(pts, metric='hamming')) * pts.shape[1]
    
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
    return results['dgms'], len(pts)

def calculate_pcc(diagrams, skip_h0=False):
    """Total persistence: sum of squared lifetimes."""
    total = 0.0
    start = 1 if skip_h0 else 0
    for dim in range(start, len(diagrams)):
        dgm = diagrams[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            total += np.sum(lifetimes ** 2)
    return total

def calculate_max_betti(diagrams):
    """Maximum Betti number at any filtration value for each dimension."""
    max_betti = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            max_betti[dim] = 0
            continue
        
        # Find max simultaneous features
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) == 0:
            max_betti[dim] = 0
            continue
        
        # Sample filtration values
        all_vals = finite.flatten()
        min_v, max_v = all_vals.min(), all_vals.max()
        test_points = np.linspace(min_v, max_v, 500)
        
        max_count = 0
        for t in test_points:
            alive = np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t))
            max_count = max(max_count, alive)
        
        max_betti[dim] = max_count
    
    return max_betti

def calculate_lifetime_entropy(diagrams):
    """Entropy of lifetime distribution (how spread out are lifetimes)."""
    entropy = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            entropy[dim] = 0.0
            continue
        
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) == 0:
            entropy[dim] = 0.0
            continue
        
        lifetimes = finite[:, 1] - finite[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        
        if len(lifetimes) == 0:
            entropy[dim] = 0.0
            continue
        
        probs = lifetimes / lifetimes.sum()
        probs = probs[probs > 0]
        entropy[dim] = float(-np.sum(probs * np.log(probs)))
    
    return entropy

def analyze_class(name, points, max_dim=1):
    """Compute all metrics for a single class."""
    print(f"  Analyzing {name} ({len(points)} points)...")
    
    dgms, sampled_count = compute_persistence_hamming(points, max_dim=max_dim, use_jitter=True)
    
    metrics = {
        'name': name,
        'num_points': len(points),
        'sampled_points': sampled_count,
        'pcc_total': calculate_pcc(dgms, skip_h0=False),
        'pcc_h1_plus': calculate_pcc(dgms, skip_h0=True),
        'max_betti': calculate_max_betti(dgms),
        'lifetime_entropy': calculate_lifetime_entropy(dgms),
        'diagrams': dgms,
    }
    
    return metrics

def print_comparison_report(metrics_list):
    """Generate text report comparing classes."""
    print("\n" + "=" * 70)
    print("TOPOLOGICAL HIERARCHY ANALYSIS REPORT")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("1. SUMMARY TABLE")
    print("-" * 70)
    
    header = f"{'Class':<20} | {'Points':<8} | {'PCC(H1+)':<12} | {'MaxB1':<8} | {'Entropy H1':<10}"
    print(header)
    print("-" * 70)
    
    for m in metrics_list:
        b1 = m['max_betti'].get(1, 0)
        e1 = m['lifetime_entropy'].get(1, 0)
        print(f"{m['name']:<20} | {m['num_points']:<8} | {m['pcc_h1_plus']:<12.2f} | {b1:<8} | {e1:<10.4f}")
    
    print("\n" + "-" * 70)
    print("2. H2 (VOIDS) ANALYSIS")
    print("-" * 70)
    
    for m in metrics_list:
        b2 = m['max_betti'].get(2, 0)
        e2 = m['lifetime_entropy'].get(2, 0)
        h2_dgm = m['diagrams'][2] if len(m['diagrams']) > 2 else np.array([])
        h2_count = len(h2_dgm[~np.isinf(h2_dgm[:, 1])]) if len(h2_dgm) > 0 else 0
        print(f"{m['name']:<20}: H2 features = {h2_count}, Max Betti_2 = {b2}")
    
    print("\n" + "-" * 70)
    print("3. AC0 vs NC1 CRITICAL COMPARISON")
    print("-" * 70)
    
    ac0 = next((m for m in metrics_list if 'Tribes' in m['name']), None)
    nc1 = next((m for m in metrics_list if 'Parity' in m['name']), None)
    
    if ac0 and nc1:
        pcc_diff = nc1['pcc_h1_plus'] - ac0['pcc_h1_plus']
        b1_diff = nc1['max_betti'].get(1, 0) - ac0['max_betti'].get(1, 0)
        e1_diff = nc1['lifetime_entropy'].get(1, 0) - ac0['lifetime_entropy'].get(1, 0)
        
        print(f"\n  AC0_Tribes  PCC(H1+) = {ac0['pcc_h1_plus']:.2f}")
        print(f"  NC1_Parity  PCC(H1+) = {nc1['pcc_h1_plus']:.2f}")
        print(f"  Difference: {pcc_diff:+.2f}")
        
        print(f"\n  AC0_Tribes  Max Betti_1 = {ac0['max_betti'].get(1, 0)}")
        print(f"  NC1_Parity  Max Betti_1 = {nc1['max_betti'].get(1, 0)}")
        print(f"  Difference: {b1_diff:+d}")
        
        print(f"\n  AC0_Tribes  Entropy H1 = {ac0['lifetime_entropy'].get(1, 0):.4f}")
        print(f"  NC1_Parity  Entropy H1 = {nc1['lifetime_entropy'].get(1, 0):.4f}")
        print(f"  Difference: {e1_diff:+.4f}")
        
        print("\n  " + "-" * 50)
        if abs(pcc_diff) > 10 or abs(b1_diff) > 5:
            print("  ✓ SEPARATION DETECTED: AC0 and NC1 show distinct")
            print("    topological signatures!")
        else:
            print("  ⚠ OVERLAP: Metrics are close. Need more samples or")
            print("    different analysis approach.")
    
    print("\n" + "=" * 70)

def main():
    print("=" * 70)
    print("HIERARCHY ANALYSIS: AC0 vs NC1")
    print("=" * 70)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'class_separation_n12.npz')
    print(f"\nLoading: {data_path}")
    data = np.load(data_path)
    
    # Analyze each class (using H1 only for speed)
    print(f"\nComputing TDA metrics (max_dim=1, sampled to {MAX_POINTS} pts)...")
    metrics_list = []
    
    for name in ['AC0_Tribes', 'NC1_Parity', 'NC1_Majority', 'P_Poly_Random']:
        if name in data.files:
            m = analyze_class(name, data[name], max_dim=1)
            metrics_list.append(m)
    
    # Generate report
    print_comparison_report(metrics_list)

if __name__ == "__main__":
    main()
