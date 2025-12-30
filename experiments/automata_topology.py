"""
The Topological Edge of Chaos

Generates all 256 Elementary Cellular Automata rules.
Computes PCC (Topological Complexity) and Shannon Entropy for each Space-Time diagram.
Goal: Find the topological signature of "Class 4" (Computation/Life).
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import ripser

# Configuration
WIDTH = 100
STEPS = 50
SAMPLES_PER_RULE = 1  # Speed up: 1 sample per rule
MAX_POINTS_FOR_TDA = 400  # Speed up: Smaller point cloud

def generate_ca(rule, width, steps, initial_state=None):
    """Generate a Space-Time diagram for a given EC rule."""
    grid = np.zeros((steps, width), dtype=int)
    
    if initial_state is None:
        grid[0] = np.random.randint(0, 2, width)
    else:
        grid[0] = initial_state
        
    # Pre-compute rule table
    rule_bin = np.array([int(x) for x in np.binary_repr(rule, width=8)][::-1])
    
    for t in range(steps - 1):
        # Rolling window for neighbors (left, center, right)
        # Using numpy fancy indexing or correlate is faster, but loop is clear
        left = np.roll(grid[t], 1)
        center = grid[t]
        right = np.roll(grid[t], -1)
        
        # Convert neighbor pattern to integer 0-7
        idx = 4 * left + 2 * center + right
        grid[t + 1] = rule_bin[idx]
        
    return grid

def compute_metrics(grid):
    """Compute Topological Complexity (PCC) and Shannon Entropy."""
    rows, cols = grid.shape
    
    # 1. Entropy (average row entropy)
    # Measures "disorder"
    row_means = np.mean(grid, axis=1) # Density over time
    # Binary entropy of the density
    # Avoid log(0)
    p = np.clip(row_means, 1e-9, 1 - 1e-9)
    avg_entropy = -np.mean(p * np.log2(p) + (1-p) * np.log2(1-p))

    # 2. Topological Complexity (PCC)
    # Measures "structure/holes" in spacetime
    
    # Get active points (y, x) coordinates
    # We treat Time as the Y axis
    y, x = np.where(grid == 1)
    points = np.column_stack((x, y))
    
    if len(points) < 5:
        return 0, 0  # Dead world
        
    # Subsample if needed (TDA is O(N^3))
    if len(points) > MAX_POINTS_FOR_TDA:
        indices = np.random.choice(len(points), MAX_POINTS_FOR_TDA, replace=False)
        points = points[indices]
    
    # Compute Persistent Homology
    # Use simple Euclidean distance in spacetime
    dist_matrix = squareform(pdist(points))
    
    try:
        res = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=1)
        dgms = res['dgms']
        
        pcc = 0.0
        if len(dgms) > 1:
            h1 = dgms[1]
            # H1 lifetime sum squared
            if len(h1) > 0:
                finite = h1[~np.isinf(h1[:, 1])]
                if len(finite) > 0:
                    lifetimes = finite[:, 1] - finite[:, 0]
                    pcc = np.sum(lifetimes ** 2)
    except:
        pcc = 0.0
        
    return pcc, avg_entropy

def run_experiment():
    print("=" * 60)
    print("THE TOPOLOGICAL EDGE OF CHAOS")
    print("=" * 60)
    print(f"Analyzing all 256 CA Rules...")
    
    results = []
    
    for rule in range(256):
        if rule % 32 == 0:
            print(f"Processing Rule {rule}...")
            
        rule_pccs = []
        rule_entropies = []
        
        for _ in range(SAMPLES_PER_RULE):
            grid = generate_ca(rule, WIDTH, STEPS)
            pcc, ent = compute_metrics(grid)
            rule_pccs.append(pcc)
            rule_entropies.append(ent)
            
        results.append({
            'rule': rule,
            'pcc_mean': np.mean(rule_pccs),
            'pcc_std': np.std(rule_pccs),
            'entropy_mean': np.mean(rule_entropies)
        })
        
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'automata_topology.csv')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rule', 'pcc_mean', 'pcc_std', 'entropy_mean'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\n✓ Saved: {output_path}")
    
    # Quick Check on Key Rules
    key_rules = [30, 90, 110, 184, 0, 255]
    print("\n" + "-" * 40)
    print("KEY RULES SNAPSHOT")
    print("-" * 40)
    print("Rule |   Class   |   PCC   | Entropy")
    print("-" * 40)
    
    res_dict = {r['rule']: r for r in results}
    
    for r in key_rules:
        if r in res_dict:
            d = res_dict[r]
            desc = "Unknown"
            if r == 30: desc = "Chaos"
            if r == 90: desc = "Fractal"
            if r == 110: desc = "COMPUTATION"
            if r == 184: desc = "Traffic"
            if r == 0: desc = "Death"
            
            print(f" {r:<3} | {desc:<9} | {d['pcc_mean']:>7.2f} | {d['entropy_mean']:.3f}")
            
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_experiment()
