"""
Trojan X-Ray Scanner

Uses TDA metrics to detect hidden structure in boolean functions.
Compares PCC and Max Lifespan between benign and infected samples.
"""

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ripser

# Configuration
MAX_SAMPLES = 500  # Subsample for speed

def add_jitter(points, sigma=0.02):
    """Add small jitter for TDA stability."""
    return points.astype(float) + np.random.normal(0, sigma, points.shape)

def compute_tda_metrics(coords, max_dim=1):
    """Compute PCC and Max Lifespan for a point cloud."""
    if len(coords) < 5:
        return {'pcc': 0.0, 'max_lifespan': 0.0, 'n_features': 0}
    
    # Subsample if needed
    if len(coords) > MAX_SAMPLES:
        indices = np.random.choice(len(coords), MAX_SAMPLES, replace=False)
        coords = coords[indices]
    
    # Jitter and compute distance
    pts = add_jitter(coords)
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    
    # Ripser
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
    dgms = results['dgms']
    
    # Compute metrics
    pcc = 0.0
    max_lifespan = 0.0
    n_features = 0
    
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            
            if len(lifetimes) > 0:
                # PCC for H1+
                if dim >= 1:
                    pcc += np.sum(lifetimes ** 2)
                
                # Max lifespan across all dimensions
                max_lifespan = max(max_lifespan, np.max(lifetimes))
                n_features += len(lifetimes)
    
    return {'pcc': pcc, 'max_lifespan': max_lifespan, 'n_features': n_features}

def scan_dataset():
    print("=" * 60)
    print("TROJAN X-RAY SCANNER")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'trojan_data.npz')
    print(f"\nLoading: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    benign_on_sets = data['benign_on_sets']
    infected_on_sets = data['infected_on_sets']
    
    print(f"Loaded {len(benign_on_sets)} benign and {len(infected_on_sets)} infected functions")
    
    # Scan benign
    print(f"\n[1] Scanning BENIGN functions...")
    benign_metrics = []
    for i, on_set in enumerate(benign_on_sets):
        metrics = compute_tda_metrics(on_set)
        benign_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"    Scanned {i+1}/{len(benign_on_sets)}")
    
    # Scan infected
    print(f"\n[2] Scanning INFECTED functions...")
    infected_metrics = []
    for i, on_set in enumerate(infected_on_sets):
        metrics = compute_tda_metrics(on_set)
        infected_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"    Scanned {i+1}/{len(infected_on_sets)}")
    
    # Extract arrays
    benign_pcc = np.array([m['pcc'] for m in benign_metrics])
    benign_lifespan = np.array([m['max_lifespan'] for m in benign_metrics])
    
    infected_pcc = np.array([m['pcc'] for m in infected_metrics])
    infected_lifespan = np.array([m['max_lifespan'] for m in infected_metrics])
    
    # Results
    print("\n" + "=" * 60)
    print("X-RAY SCAN RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} | {'Benign':<20} | {'Infected':<20} | {'Diff':<12}")
    print("-" * 75)
    
    # PCC
    m_benign_pcc = np.mean(benign_pcc)
    s_benign_pcc = np.std(benign_pcc)
    m_infected_pcc = np.mean(infected_pcc)
    s_infected_pcc = np.std(infected_pcc)
    diff_pcc = m_infected_pcc - m_benign_pcc
    
    print(f"{'PCC (Mean ± Std)':<20} | {m_benign_pcc:.4f} ± {s_benign_pcc:.4f}  | {m_infected_pcc:.4f} ± {s_infected_pcc:.4f}  | {diff_pcc:+.4f}")
    
    # Max Lifespan
    m_benign_life = np.mean(benign_lifespan)
    s_benign_life = np.std(benign_lifespan)
    m_infected_life = np.mean(infected_lifespan)
    s_infected_life = np.std(infected_lifespan)
    diff_life = m_infected_life - m_benign_life
    
    print(f"{'Max Lifespan (Mean)':<20} | {m_benign_life:.4f} ± {s_benign_life:.4f}  | {m_infected_life:.4f} ± {s_infected_life:.4f}  | {diff_life:+.4f}")
    
    # Signal-to-Noise Ratio
    print("\n" + "-" * 60)
    print("SIGNAL-TO-NOISE RATIO (Cohen's d)")
    print("-" * 60)
    
    pooled_std_pcc = np.sqrt((s_benign_pcc**2 + s_infected_pcc**2) / 2)
    snr_pcc = diff_pcc / pooled_std_pcc if pooled_std_pcc > 0 else 0
    
    pooled_std_life = np.sqrt((s_benign_life**2 + s_infected_life**2) / 2)
    snr_life = diff_life / pooled_std_life if pooled_std_life > 0 else 0
    
    print(f"\n  PCC SNR:          {snr_pcc:+.4f}")
    print(f"  Max Lifespan SNR: {snr_life:+.4f}")
    
    # Detection verdict
    print("\n" + "=" * 60)
    print("DETECTION VERDICT")
    print("=" * 60)
    
    if abs(snr_pcc) > 0.5 or abs(snr_life) > 0.5:
        print("\n  ✓ TROJAN DETECTED: Topological signature differs from benign!")
        print(f"    Best metric: {'PCC' if abs(snr_pcc) > abs(snr_life) else 'Max Lifespan'}")
    else:
        print("\n  ✗ NO DETECTION: Trojan is well-hidden from TDA.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    scan_dataset()
