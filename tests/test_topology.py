
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from topology_calc import compute_persistence, calculate_pcc, betti_curves

def generate_hamming_circle(n_points=20, dim=20):
    """
    Generates points that form a loop in Hamming space.
    Strategy: A sliding window of 1s in a vector of 0s.
    """
    points = np.zeros((n_points, dim), dtype=int)
    window_size = 3
    
    for i in range(n_points):
        # Cyclically place 1s
        start = i % dim
        for j in range(window_size):
            idx = (start + j) % dim
            points[i, idx] = 1
            
    # Remove duplicates if any
    points = np.unique(points, axis=0)
    return points

def test_circle_topology():
    print("Generating Synthetic Hamming Circle...")
    # Generate larger circle to ensure sampling doesn't break it drastically
    # though here we are under the 2000 limit.
    points = generate_hamming_circle(n_points=50, dim=50)
    print(f"data shape: {points.shape}")
    
    print("Computing Persistence...")
    diagrams = compute_persistence(points, max_dim=1)
    
    # H0
    print(f"H0 features: {len(diagrams[0])}")
    
    # H1
    h1_dgm = diagrams[1]
    print(f"H1 features: {len(h1_dgm)}")
    if len(h1_dgm) > 0:
        lifetimes = h1_dgm[:, 1] - h1_dgm[:, 0]
        max_life = np.max(lifetimes)
        print(f"Max H1 lifetime: {max_life}")
        print(f"H1 Diagram:\n{h1_dgm}")
        
        # We expect at least one significant loop
        # Hamming distance is normalized (0-1), so 0.08 is significant for this small circle
        if max_life > 0.04:
            print("SUCCESS: Significant loop detected.")
        else:
            print("FAILURE: No significant loop detected.")
    else:
        print("FAILURE: No H1 features found.")

    print("\nCalculating PCC...")
    pcc = calculate_pcc(diagrams)
    print(f"Total Persistence (PCC): {pcc}")
    
    print("\nComputing Betti Curves...")
    curves = betti_curves(diagrams, n_steps=10)
    for dim, (xs, ys) in curves.items():
        print(f"Dim {dim} max betti: {np.max(ys)}")

if __name__ == "__main__":
    test_circle_topology()
