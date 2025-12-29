"""
Publication-Quality Figure Generator

Creates a side-by-side comparison of Threshold_k3 (Champion) vs Random (Baseline)
with jittered persistence diagrams and stacked barcodes.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from boolean_gen import generate_truth_table
from topology_calc import compute_persistence

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

def add_jitter(dgms, sigma=0.02):
    """Add Gaussian jitter to persistence diagrams for better visualization."""
    jittered = []
    for dgm in dgms:
        if len(dgm) > 0:
            noise = np.random.normal(0, sigma, dgm.shape)
            jittered.append(dgm + noise)
        else:
            jittered.append(dgm)
    return jittered

def plot_persistence_diagram(ax, dgms, title, colors=['#1f77b4', '#ff7f0e'], alpha=0.6):
    """Plot persistence diagram with jitter."""
    # Get max value for diagonal
    all_vals = []
    for dgm in dgms:
        if len(dgm) > 0:
            finite_mask = ~np.isinf(dgm[:, 1])
            if np.any(finite_mask):
                all_vals.extend(dgm[finite_mask].flatten())
    
    if not all_vals:
        ax.set_title(title)
        return
    
    max_val = max(all_vals) * 1.1
    min_val = min(0, min(all_vals))
    
    # Plot diagonal
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
    
    labels = ['H₀ (Components)', 'H₁ (Loops)']
    
    for dim, dgm in enumerate(dgms[:2]):  # Only H0 and H1
        if len(dgm) == 0:
            continue
        
        # Filter infinite points
        finite_mask = ~np.isinf(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        
        if len(finite_dgm) > 0:
            ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], 
                      c=colors[dim], alpha=alpha, s=25, 
                      label=labels[dim], edgecolors='none')
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

def plot_barcodes(ax, dgms, title, color='#1f77b4', max_bars=50):
    """Plot persistence barcodes (horizontal bars)."""
    bars = []
    
    for dim, dgm in enumerate(dgms[:2]):  # H0 and H1
        for birth, death in dgm:
            if not np.isinf(death):
                bars.append((birth, death, dim))
    
    # Sort by birth time
    bars.sort(key=lambda x: x[0])
    
    # Limit number of bars
    if len(bars) > max_bars:
        bars = bars[:max_bars]
    
    colors_dim = ['#2ecc71', '#e74c3c']  # Green for H0, Red for H1
    
    for i, (birth, death, dim) in enumerate(bars):
        ax.barh(i, death - birth, left=birth, height=0.8, 
                color=colors_dim[dim], alpha=0.7, edgecolor='none')
    
    ax.set_xlabel('Filtration Value')
    ax.set_ylabel('Feature Index')
    ax.set_title(title)
    ax.set_ylim(-1, len(bars))
    ax.grid(True, alpha=0.2, axis='x')
    
    # Legend
    h0_patch = mpatches.Patch(color='#2ecc71', label='H₀')
    h1_patch = mpatches.Patch(color='#e74c3c', label='H₁')
    ax.legend(handles=[h0_patch, h1_patch], loc='upper right')

def generate_publication_figure():
    print("Generating Publication-Quality Figure...")
    n = 8
    
    # Generate data
    print("  Computing Threshold_k3...")
    tt_thresh, coords_thresh = generate_truth_table(n, 'threshold', k=3)
    dgms_thresh = compute_persistence(coords_thresh, max_dim=1)
    
    print("  Computing Random...")
    tt_random, coords_random = generate_truth_table(n, 'random', seed=42)
    dgms_random = compute_persistence(coords_random, max_dim=1)
    
    # Add jitter
    dgms_thresh_jit = add_jitter(dgms_thresh, sigma=0.015)
    dgms_random_jit = add_jitter(dgms_random, sigma=0.015)
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    
    # Left side: Jittered Persistence Diagrams
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Plot both on same axes for comparison
    colors_thresh = ['#3498db', '#9b59b6']  # Blue shades for Threshold
    colors_random = ['#e67e22', '#c0392b']  # Orange/Red shades for Random
    
    # Plot Random first (background)
    for dim, dgm in enumerate(dgms_random_jit[:2]):
        if len(dgm) == 0:
            continue
        finite_mask = ~np.isinf(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        if len(finite_dgm) > 0:
            label = f'Random H{dim}' if dim < 2 else None
            ax1.scatter(finite_dgm[:, 0], finite_dgm[:, 1], 
                       c=colors_random[dim], alpha=0.4, s=20, 
                       label=label, edgecolors='none', marker='s')
    
    # Plot Threshold on top
    for dim, dgm in enumerate(dgms_thresh_jit[:2]):
        if len(dgm) == 0:
            continue
        finite_mask = ~np.isinf(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        if len(finite_dgm) > 0:
            label = f'Threshold H{dim}' if dim < 2 else None
            ax1.scatter(finite_dgm[:, 0], finite_dgm[:, 1], 
                       c=colors_thresh[dim], alpha=0.6, s=30, 
                       label=label, edgecolors='none', marker='o')
    
    # Diagonal
    all_vals = []
    for dgms in [dgms_thresh_jit, dgms_random_jit]:
        for dgm in dgms:
            if len(dgm) > 0:
                finite_mask = ~np.isinf(dgm[:, 1])
                if np.any(finite_mask):
                    all_vals.extend(dgm[finite_mask].flatten())
    
    max_val = max(all_vals) * 1.1 if all_vals else 1
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Birth', fontweight='bold')
    ax1.set_ylabel('Death', fontweight='bold')
    ax1.set_title('Persistence Diagrams: Threshold_k3 vs Random', fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    
    # Right side: Stacked Barcodes
    ax2_top = fig.add_subplot(2, 2, 2)
    ax2_bot = fig.add_subplot(2, 2, 4)
    
    plot_barcodes(ax2_top, dgms_thresh, 'Barcodes: Threshold_k3 (Champion)', max_bars=40)
    plot_barcodes(ax2_bot, dgms_random, 'Barcodes: Random (Baseline)', max_bars=40)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'topological_contrast.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    
    # Print stats
    print("\n  Statistics:")
    print(f"    Threshold_k3: {len(coords_thresh)} points in On-Set")
    print(f"    Random:       {len(coords_random)} points in On-Set")

if __name__ == "__main__":
    generate_publication_figure()
