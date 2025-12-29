"""
Circuit Sweep Analysis

Computes PCC (Total Persistence) and Sensitivity for each circuit
in the sweep dataset for regression analysis.
"""

import os
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ripser

# Configuration
N = 10
MAX_SAMPLES = 300  # Subsample on-set for TDA

def hex_to_truth_table(hex_str):
    """Convert hex string back to boolean array."""
    n_bits = len(hex_str) * 4
    binary = bin(int(hex_str, 16))[2:].zfill(n_bits)
    return np.array([int(b) for b in binary], dtype=np.int8)

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def compute_sensitivity(truth_table, coords):
    """
    Compute average sensitivity of a boolean function.
    Sensitivity = average number of bit flips that flip the output.
    """
    n = coords.shape[1]
    total_sensitivity = 0
    
    for i in range(len(truth_table)):
        # Current output
        output = truth_table[i]
        
        # Count how many neighbors have different output
        sensitive_bits = 0
        for bit in range(n):
            # Flip bit 'bit' to get neighbor index
            neighbor_idx = i ^ (1 << (n - 1 - bit))
            if truth_table[neighbor_idx] != output:
                sensitive_bits += 1
        
        total_sensitivity += sensitive_bits
    
    return total_sensitivity / len(truth_table)

def add_jitter(points, sigma=0.02):
    """Add jitter for TDA stability."""
    return points.astype(float) + np.random.normal(0, sigma, points.shape)

def compute_pcc(on_set, max_samples=MAX_SAMPLES):
    """Compute PCC (Total Persistence) for H1+."""
    if len(on_set) < 5:
        return 0.0
    
    # Subsample
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

def analyze_sweep():
    print("=" * 60)
    print("CIRCUIT SWEEP ANALYSIS")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'circuit_sweep.csv')
    print(f"\nLoading: {data_path}")
    
    circuits = []
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            circuits.append({
                'depth': int(row['depth']),
                'size': int(row['size']),
                'sample': int(row['sample']),
                'truth_table_hex': row['truth_table_hex']
            })
    
    print(f"Loaded {len(circuits)} circuits")
    
    # Precompute coords
    coords = get_hypercube_coords(N)
    
    results = []
    
    print(f"\nAnalyzing circuits...")
    for i, circuit in enumerate(circuits):
        # Decode truth table
        tt = hex_to_truth_table(circuit['truth_table_hex'])
        
        # Get on-set
        on_set = coords[tt == 1]
        
        # Compute sensitivity
        sensitivity = compute_sensitivity(tt, coords)
        
        # Compute PCC
        pcc = compute_pcc(on_set)
        
        results.append({
            'depth': circuit['depth'],
            'size': circuit['size'],
            'sample': circuit['sample'],
            'sensitivity': sensitivity,
            'pcc': pcc,
            'on_set_size': len(on_set)
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i+1}/{len(circuits)} circuits...")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'sweep_results.csv')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['depth', 'size', 'sample', 'sensitivity', 'pcc', 'on_set_size'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved: {output_path}")
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    pccs = [r['pcc'] for r in results]
    sens = [r['sensitivity'] for r in results]
    
    print(f"\nPCC:         Mean={np.mean(pccs):.4f}, Std={np.std(pccs):.4f}")
    print(f"Sensitivity: Mean={np.mean(sens):.4f}, Std={np.std(sens):.4f}")
    
    # Correlation
    corr = np.corrcoef(pccs, sens)[0, 1]
    print(f"\nCorrelation (PCC, Sensitivity): {corr:.4f}")
    
    if corr < -0.3:
        print("  ✓ HYPOTHESIS SUPPORTED: High PCC correlates with LOW sensitivity!")
    elif corr > 0.3:
        print("  ✗ HYPOTHESIS REJECTED: High PCC correlates with HIGH sensitivity.")
    else:
        print("  ~ WEAK CORRELATION: No clear relationship.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_sweep()
