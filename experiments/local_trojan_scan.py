"""
Local Trojan Scanner

Uses random probing with local PCC to detect hidden structured regions.
Takes the MAXIMUM local PCC across all probes to find anomalies.
"""

import os
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import ripser

# Configuration
N = 12
N_PROBES = 100
HAMMING_RADIUS = 2
MIN_LOCAL_POINTS = 10

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def add_jitter(points, sigma=0.02):
    """Add small jitter for TDA stability."""
    return points.astype(float) + np.random.normal(0, sigma, points.shape)

def compute_local_pcc(coords, max_dim=1):
    """Compute PCC for a small local point cloud."""
    if len(coords) < 5:
        return 0.0
    
    # Limit size
    if len(coords) > 100:
        indices = np.random.choice(len(coords), 100, replace=False)
        coords = coords[indices]
    
    pts = add_jitter(coords)
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
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

def scan_function(on_set, all_coords, n_probes=N_PROBES, radius=HAMMING_RADIUS):
    """
    Probe random locations and return max local PCC.
    
    Args:
        on_set: Points where f(x) = 1
        all_coords: All hypercube coordinates
        n_probes: Number of random probes
        radius: Hamming radius for neighborhood
    
    Returns:
        max_local_pcc: Maximum PCC found across all probes
    """
    if len(on_set) == 0:
        return 0.0
    
    # Create set for fast lookup
    on_set_tuples = set(map(tuple, on_set))
    
    max_pcc = 0.0
    valid_probes = 0
    
    for _ in range(n_probes):
        # Pick random center from all coordinates
        center_idx = np.random.randint(len(all_coords))
        center = all_coords[center_idx]
        
        # Find neighbors within Hamming radius
        distances = np.sum(all_coords != center, axis=1)
        neighbor_mask = distances <= radius
        neighbors = all_coords[neighbor_mask]
        
        # Filter to local on-set
        local_on_set = np.array([n for n in neighbors if tuple(n) in on_set_tuples])
        
        if len(local_on_set) < MIN_LOCAL_POINTS:
            continue
        
        # Compute local PCC
        local_pcc = compute_local_pcc(local_on_set)
        max_pcc = max(max_pcc, local_pcc)
        valid_probes += 1
    
    return max_pcc

def run_local_scan():
    print("=" * 60)
    print("LOCAL TROJAN SCANNER")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  N = {N}, Probes = {N_PROBES}, Hamming Radius = {HAMMING_RADIUS}")
    print(f"  Min local points = {MIN_LOCAL_POINTS}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'trojan_data.npz')
    data = np.load(data_path, allow_pickle=True)
    
    benign_on_sets = data['benign_on_sets']
    infected_on_sets = data['infected_on_sets']
    
    print(f"\nLoaded {len(benign_on_sets)} benign and {len(infected_on_sets)} infected")
    
    # Precompute all coords
    all_coords = get_hypercube_coords(N)
    
    results = []
    
    # Scan benign
    print(f"\n[1] Scanning BENIGN functions...")
    benign_max_pccs = []
    for i, on_set in enumerate(benign_on_sets):
        max_pcc = scan_function(on_set, all_coords)
        benign_max_pccs.append(max_pcc)
        results.append({'type': 'Benign', 'sample_id': i, 'max_local_pcc': max_pcc})
        if (i + 1) % 10 == 0:
            print(f"    Scanned {i+1}/{len(benign_on_sets)}, Avg Max PCC: {np.mean(benign_max_pccs):.4f}")
    
    # Scan infected
    print(f"\n[2] Scanning INFECTED functions...")
    infected_max_pccs = []
    for i, on_set in enumerate(infected_on_sets):
        max_pcc = scan_function(on_set, all_coords)
        infected_max_pccs.append(max_pcc)
        results.append({'type': 'Infected', 'sample_id': i, 'max_local_pcc': max_pcc})
        if (i + 1) % 10 == 0:
            print(f"    Scanned {i+1}/{len(infected_on_sets)}, Avg Max PCC: {np.mean(infected_max_pccs):.4f}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trojan_scan_results.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'sample_id', 'max_local_pcc'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved: {output_path}")
    
    # Results
    print("\n" + "=" * 60)
    print("LOCAL SCAN RESULTS")
    print("=" * 60)
    
    m_benign = np.mean(benign_max_pccs)
    s_benign = np.std(benign_max_pccs)
    m_infected = np.mean(infected_max_pccs)
    s_infected = np.std(infected_max_pccs)
    
    print(f"\n{'Metric':<25} | {'Benign':<15} | {'Infected':<15}")
    print("-" * 60)
    print(f"{'Max Local PCC (Mean)':<25} | {m_benign:.4f}        | {m_infected:.4f}")
    print(f"{'Max Local PCC (Std)':<25} | {s_benign:.4f}        | {s_infected:.4f}")
    print(f"{'Max Local PCC (Max)':<25} | {np.max(benign_max_pccs):.4f}        | {np.max(infected_max_pccs):.4f}")
    
    # Signal-to-Noise
    diff = m_infected - m_benign
    pooled_std = np.sqrt((s_benign**2 + s_infected**2) / 2)
    snr = diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\n  Difference: {diff:+.4f}")
    print(f"  Cohen's d (SNR): {snr:+.4f}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("DETECTION VERDICT")
    print("=" * 60)
    
    if snr > 0.5:
        print("\n  ✓ TROJAN DETECTED: Local topology reveals hidden structure!")
    elif snr > 0.2:
        print("\n  ⚠ WEAK SIGNAL: Some anomaly detected, needs more probes.")
    else:
        print("\n  ✗ NO DETECTION: Trojan remains hidden.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_local_scan()
