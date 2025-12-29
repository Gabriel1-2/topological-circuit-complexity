"""
Learning Phase Transition Plot

Dual-axis visualization showing how topological complexity (PCC)
evolves as the neural network learns the Parity function.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

def load_training_logs(filepath):
    """Load training topology logs."""
    epochs, losses, accuracies, pccs = [], [], [], []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))
            accuracies.append(float(row['accuracy']))
            pccs.append(float(row['pcc']))
    
    return np.array(epochs), np.array(losses), np.array(accuracies), np.array(pccs)

def create_brain_scan_plot():
    print("=" * 60)
    print("LEARNING PHASE TRANSITION PLOT")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_topology.csv')
    print(f"\nLoading: {data_path}")
    epochs, losses, accuracies, pccs = load_training_logs(data_path)
    
    print(f"Loaded {len(epochs)} data points")
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left Y-axis: Accuracy (Blue)
    color1 = '#3498db'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Accuracy', color=color1, fontweight='bold')
    line1 = ax1.plot(epochs, accuracies, color=color1, linewidth=2.5, 
                     label='Accuracy', marker='o', markersize=4, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.45, 1.05)
    ax1.axhline(y=1.0, color=color1, linestyle='--', alpha=0.3, linewidth=1)
    
    # Right Y-axis: PCC (Red)
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('PCC (Topological Complexity)', color=color2, fontweight='bold')
    line2 = ax2.plot(epochs, pccs, color=color2, linewidth=2.5,
                     label='PCC', marker='s', markersize=4, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add annotations for key phases
    # Find peak PCC
    peak_idx = np.argmax(pccs)
    peak_epoch = epochs[peak_idx]
    peak_pcc = pccs[peak_idx]
    
    ax2.annotate(f'Peak Chaos\nPCC={peak_pcc:.1f}', 
                 xy=(peak_epoch, peak_pcc),
                 xytext=(peak_epoch + 50, peak_pcc + 10),
                 fontsize=9, color=color2,
                 arrowprops=dict(arrowstyle='->', color=color2, alpha=0.7))
    
    # Find convergence point (where accuracy hits ~99%)
    convergence_idx = np.where(accuracies >= 0.99)[0]
    if len(convergence_idx) > 0:
        conv_epoch = epochs[convergence_idx[0]]
        conv_acc = accuracies[convergence_idx[0]]
        conv_pcc = pccs[convergence_idx[0]]
        
        ax1.annotate(f'Convergence\nAcc={conv_acc:.2f}',
                     xy=(conv_epoch, conv_acc),
                     xytext=(conv_epoch + 80, conv_acc - 0.15),
                     fontsize=9, color=color1,
                     arrowprops=dict(arrowstyle='->', color=color1, alpha=0.7))
    
    # Phase regions
    ax1.axvspan(0, 30, alpha=0.1, color='yellow', label='Random Phase')
    ax1.axvspan(30, 100, alpha=0.1, color='orange', label='Learning Phase')
    ax1.axvspan(100, 500, alpha=0.1, color='green', label='Converged Phase')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=0.9)
    
    # Title
    plt.title('Neural Network "Brain Scan": Topology During Learning\n' + 
              '(MLP Learning N=10 Parity Function)', fontweight='bold', pad=15)
    
    # Grid
    ax1.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'brain_scan.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Print summary
    print("\n" + "-" * 40)
    print("PHASE TRANSITION SUMMARY")
    print("-" * 40)
    print(f"Peak PCC: {peak_pcc:.2f} at Epoch {peak_epoch}")
    print(f"Final PCC: {pccs[-1]:.2f}")
    print(f"PCC Collapse Ratio: {peak_pcc / pccs[-1]:.1f}x")
    print(f"Final Accuracy: {accuracies[-1]:.4f}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    create_brain_scan_plot()
