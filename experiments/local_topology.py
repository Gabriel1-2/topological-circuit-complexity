"""
Topological Lie Detector

Measures local topological complexity around clean vs adversarial samples.
Hypothesis: Adversarial regions have "fractured" decision boundaries
leading to higher local PCC.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import ripser

# Configuration
N = 10
HIDDEN_SIZE = 128
N_SAMPLES = 300    # Points to sample in local neighborhood
RADIUS = 0.6       # Hyperball radius
NUM_TEST = 50      # Number of samples to test per class

class ParityMLP(nn.Module):
    """MLP for learning parity function."""
    def __init__(self, input_size=10, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

def sample_hyperball(center, n_samples, radius, dim):
    """
    Sample n_samples points uniformly inside a hyperball.
    Uses rejection sampling for simplicity.
    """
    samples = []
    while len(samples) < n_samples:
        # Sample from hypercube
        batch_size = n_samples * 2
        points = center + (np.random.uniform(-radius, radius, (batch_size, dim)))
        
        # Keep points within radius AND within [0, 1] bounds
        distances = np.linalg.norm(points - center, axis=1)
        valid_mask = (distances <= radius) & np.all(points >= 0, axis=1) & np.all(points <= 1, axis=1)
        valid_points = points[valid_mask]
        
        samples.extend(valid_points.tolist())
    
    return np.array(samples[:n_samples], dtype=np.float32)

def add_jitter(points, sigma=0.02):
    """Add small jitter for TDA stability."""
    return points + np.random.normal(0, sigma, points.shape)

def compute_local_pcc(coords, max_dim=1):
    """Compute PCC of local point cloud."""
    if len(coords) < 5:
        return 0.0
    
    # Sample if too large
    if len(coords) > 200:
        indices = np.random.choice(len(coords), 200, replace=False)
        coords = coords[indices]
    
    # Jitter and compute distance
    pts = add_jitter(coords.astype(float))
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    
    # Ripser
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
    dgms = results['dgms']
    
    # Total persistence for H1+
    total = 0.0
    for dim in range(1, len(dgms)):
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            total += np.sum(lifetimes ** 2)
    
    return total

def measure_local_complexity(center_point, model, n_samples=N_SAMPLES, radius=RADIUS):
    """
    Measure topological complexity of the local decision region.
    
    1. Sample points in hyperball around center
    2. Get model predictions
    3. Extract "local on-set" (predictions > 0.5)
    4. Compute PCC of local on-set
    """
    # Sample neighborhood
    neighbors = sample_hyperball(center_point, n_samples, radius, N)
    neighbors_tensor = torch.tensor(neighbors)
    
    # Get predictions
    with torch.no_grad():
        preds = model(neighbors_tensor).numpy()
    
    # Extract local on-set
    on_set_mask = preds > 0.5
    local_on_set = neighbors[on_set_mask]
    
    # Compute PCC
    if len(local_on_set) < 5:
        return 0.0, len(local_on_set), len(neighbors)
    
    pcc = compute_local_pcc(local_on_set)
    
    return pcc, len(local_on_set), len(neighbors)

def load_model():
    """Load trained model."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'parity_model.pt')
    model = ParityMLP(input_size=N, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def run_lie_detector():
    print("=" * 60)
    print("TOPOLOGICAL LIE DETECTOR")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Local samples: {N_SAMPLES}")
    print(f"  Hyperball radius: {RADIUS}")
    print(f"  Test samples per class: {NUM_TEST}")
    
    # Load model and data
    model = load_model()
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'adversarial_data.npz')
    data = np.load(data_path)
    clean_samples = data['clean_samples'][:NUM_TEST]
    adv_samples = data['adversarial_samples'][:NUM_TEST]
    
    print(f"\nLoaded {len(clean_samples)} clean and {len(adv_samples)} adversarial samples")
    
    results = []
    
    # Test clean samples
    print(f"\n[1] Analyzing CLEAN samples...")
    clean_pccs = []
    for i, sample in enumerate(clean_samples):
        pcc, on_set_size, total = measure_local_complexity(sample, model)
        clean_pccs.append(pcc)
        results.append({'type': 'Clean', 'sample_id': i, 'local_pcc': pcc, 'on_set_size': on_set_size})
        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(clean_samples)}, Avg PCC: {np.mean(clean_pccs):.4f}")
    
    # Test adversarial samples
    print(f"\n[2] Analyzing ADVERSARIAL samples...")
    adv_pccs = []
    for i, sample in enumerate(adv_samples):
        pcc, on_set_size, total = measure_local_complexity(sample, model)
        adv_pccs.append(pcc)
        results.append({'type': 'Adversarial', 'sample_id': i, 'local_pcc': pcc, 'on_set_size': on_set_size})
        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(adv_samples)}, Avg PCC: {np.mean(adv_pccs):.4f}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'detection_results.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'sample_id', 'local_pcc', 'on_set_size'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} | {'Clean':<12} | {'Adversarial':<12}")
    print("-" * 55)
    print(f"{'Mean Local PCC':<25} | {np.mean(clean_pccs):<12.4f} | {np.mean(adv_pccs):<12.4f}")
    print(f"{'Std Local PCC':<25} | {np.std(clean_pccs):<12.4f} | {np.std(adv_pccs):<12.4f}")
    print(f"{'Max Local PCC':<25} | {np.max(clean_pccs):<12.4f} | {np.max(adv_pccs):<12.4f}")
    print(f"{'Min Local PCC':<25} | {np.min(clean_pccs):<12.4f} | {np.min(adv_pccs):<12.4f}")
    
    # Hypothesis test
    separation = np.mean(adv_pccs) - np.mean(clean_pccs)
    print(f"\n  PCC Separation (Adv - Clean): {separation:+.4f}")
    
    if separation > 0:
        print("\n  ✓ HYPOTHESIS CONFIRMED: Adversarial regions show HIGHER local complexity!")
    else:
        print("\n  ✗ HYPOTHESIS REJECTED: Adversarial regions show LOWER local complexity.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_lie_detector()
