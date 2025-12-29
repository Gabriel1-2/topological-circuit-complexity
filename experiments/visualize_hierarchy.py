"""
Hierarchy Manifold Visualization

Projects the 12-dimensional Hamming space to 3D using MDS,
visualizing the geometric structure of AC0_Tribes vs NC1_Parity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
})

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def sample_points(points, n_samples, seed=42):
    """Subsample points."""
    rng = np.random.default_rng(seed)
    if len(points) <= n_samples:
        return points
    indices = rng.choice(len(points), n_samples, replace=False)
    return points[indices]

def project_mds(on_set, off_set, n_components=3, max_points=500):
    """
    Project combined on/off sets to lower dimensions using MDS.
    Uses Hamming distance as the metric.
    """
    # Sample for speed
    on_sampled = sample_points(on_set, max_points // 2, seed=42)
    off_sampled = sample_points(off_set, max_points // 2, seed=43)
    
    # Combine
    combined = np.vstack([on_sampled, off_sampled])
    labels = np.array([1] * len(on_sampled) + [0] * len(off_sampled))
    
    # Compute Hamming distance matrix
    dist_matrix = squareform(pdist(combined, metric='hamming'))
    
    # MDS projection
    mds = MDS(n_components=n_components, dissimilarity='precomputed', 
              random_state=42, normalized_stress='auto', max_iter=300)
    coords_3d = mds.fit_transform(dist_matrix)
    
    return coords_3d, labels

def plot_3d_manifold(ax, coords, labels, title):
    """Plot 3D scatter with on/off coloring."""
    on_mask = labels == 1
    off_mask = labels == 0
    
    # Plot off-set (background) first
    ax.scatter(coords[off_mask, 0], coords[off_mask, 1], coords[off_mask, 2],
               c='#e74c3c', alpha=0.3, s=15, label='Off-Set (f=0)')
    
    # Plot on-set (foreground)
    ax.scatter(coords[on_mask, 0], coords[on_mask, 1], coords[on_mask, 2],
               c='#3498db', alpha=0.7, s=25, label='On-Set (f=1)')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('MDS Dim 1')
    ax.set_ylabel('MDS Dim 2')
    ax.set_zlabel('MDS Dim 3')
    ax.legend(loc='upper right', fontsize=8)

def main():
    print("=" * 60)
    print("HIERARCHY MANIFOLD VISUALIZATION")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'class_separation_n12.npz')
    print(f"\nLoading: {data_path}")
    data = np.load(data_path)
    
    n = 12
    all_coords = get_hypercube_coords(n)
    
    # Prepare figure
    fig = plt.figure(figsize=(14, 6))
    
    # AC0_Tribes
    print("\n[1] Projecting AC0_Tribes...")
    on_set_tribes = data['AC0_Tribes']
    # Get off-set (complement)
    on_set_tribes_set = set(map(tuple, on_set_tribes))
    off_set_tribes = np.array([c for c in all_coords if tuple(c) not in on_set_tribes_set])
    
    coords_tribes, labels_tribes = project_mds(on_set_tribes, off_set_tribes, max_points=600)
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3d_manifold(ax1, coords_tribes, labels_tribes, 'AC0_Tribes\n(DNF Clusters)')
    
    # NC1_Parity
    print("[2] Projecting NC1_Parity...")
    on_set_parity = data['NC1_Parity']
    on_set_parity_set = set(map(tuple, on_set_parity))
    off_set_parity = np.array([c for c in all_coords if tuple(c) not in on_set_parity_set])
    
    coords_parity, labels_parity = project_mds(on_set_parity, off_set_parity, max_points=600)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3d_manifold(ax2, coords_parity, labels_parity, 'NC1_Parity\n(Checkerboard Lattice)')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'hierarchy_manifold.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
