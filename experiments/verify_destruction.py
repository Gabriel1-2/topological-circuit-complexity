"""
Topological Destruction Verification

Proves that XOR mixing destroys topological persistence.
"""

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ripser

N = 10
MAX_SAMPLES = 500

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def add_jitter(points, sigma=0.02):
    """Add jitter for TDA stability."""
    return points.astype(float) + np.random.normal(0, sigma, points.shape)

def compute_pcc(on_set, max_samples=MAX_SAMPLES):
    """Compute PCC (Total Persistence) for H1+."""
    if len(on_set) < 5:
        return 0.0
    
    if len(on_set) > max_samples:
        indices = np.random.choice(len(on_set), max_samples, replace=False)
        on_set = on_set[indices]
    
    pts = add_jitter(on_set)
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=1)
    dgms = results['dgms']
    
    pcc = 0.0
    for dim in range(1, len(dgms)):
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            if len(lifetimes) > 0:
                pcc += np.sum(lifetimes ** 2)
    
    return pcc

def run_experiment():
    print("=" * 60)
    print("TOPOLOGICAL DESTRUCTION VERIFICATION")
    print("=" * 60)
    
    coords = get_hypercube_coords(N)
    
    # Define island functions
    print(f"\n[1] Defining Island Functions (N={N})")
    print("-" * 40)
    
    # f1: Active when first bit = 0 (Left half)
    f1 = (coords[:, 0] == 0).astype(int)
    on_set_f1 = coords[f1 == 1]
    print(f"  f1: 'Left Island' (bit[0] = 0)")
    print(f"      On-Set size: {len(on_set_f1)}")
    
    # f2: Active when last bit = 0 (Another half)
    f2 = (coords[:, -1] == 0).astype(int)
    on_set_f2 = coords[f2 == 1]
    print(f"  f2: 'Right Island' (bit[{N-1}] = 0)")
    print(f"      On-Set size: {len(on_set_f2)}")
    
    # f_combined = f1 XOR f2
    f_combined = f1 ^ f2
    on_set_combined = coords[f_combined == 1]
    print(f"  f_combined: f1 XOR f2")
    print(f"      On-Set size: {len(on_set_combined)}")
    
    # Compute PCCs
    print(f"\n[2] Computing Topological Complexity (PCC)")
    print("-" * 40)
    
    pcc_f1 = compute_pcc(on_set_f1)
    print(f"  PCC(f1):        {pcc_f1:.4f}")
    
    pcc_f2 = compute_pcc(on_set_f2)
    print(f"  PCC(f2):        {pcc_f2:.4f}")
    
    pcc_combined = compute_pcc(on_set_combined)
    print(f"  PCC(f_combined): {pcc_combined:.4f}")
    
    # Analysis
    print(f"\n[3] Topological Destruction Analysis")
    print("-" * 40)
    
    sum_pcc = pcc_f1 + pcc_f2
    destruction = sum_pcc - pcc_combined
    destruction_ratio = destruction / sum_pcc if sum_pcc > 0 else 0
    
    print(f"\n  Sum of Individual PCCs: {sum_pcc:.4f}")
    print(f"  Combined PCC:           {pcc_combined:.4f}")
    print(f"\n  Destruction Amount:     {destruction:.4f}")
    print(f"  Destruction Ratio:      {destruction_ratio:.2%}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    
    if destruction > 0 and destruction_ratio > 0.1:
        print("\n  ✓ HYPOTHESIS CONFIRMED!")
        print("\n  XOR mixing DESTROYED topological persistence.")
        print(f"  The combined function lost {destruction_ratio:.1%} of the total")
        print("  topological mass compared to the sum of parts.")
        print("\n  Interpretation:")
        print("    The clustered 'islands' of f1 and f2 were fragmented")
        print("    by XOR into a more scattered, parity-like structure")
        print("    with fewer persistent topological features.")
    elif destruction > 0:
        print("\n  ~ WEAK DESTRUCTION")
        print(f"  Some topology was lost ({destruction_ratio:.1%}), but not dramatically.")
    else:
        print("\n  ✗ HYPOTHESIS REJECTED")
        print("  XOR mixing did NOT reduce total topological complexity.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_experiment()
