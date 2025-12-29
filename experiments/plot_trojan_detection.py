"""
Trojan Detection Visualization

Box plot comparing Max Local PCC distributions for Benign vs Infected functions.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

def load_results(filepath):
    """Load scan results."""
    benign_pcc = []
    infected_pcc = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pcc = float(row['max_local_pcc'])
            if row['type'] == 'Benign':
                benign_pcc.append(pcc)
            else:
                infected_pcc.append(pcc)
    
    return np.array(benign_pcc), np.array(infected_pcc)

def create_plot():
    print("=" * 60)
    print("TROJAN DETECTION VISUALIZATION")
    print("=" * 60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trojan_scan_results.csv')
    benign_pcc, infected_pcc = load_results(data_path)
    
    print(f"\nLoaded {len(benign_pcc)} benign and {len(infected_pcc)} infected samples")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Box plot
    box_data = [benign_pcc, infected_pcc]
    box_positions = [1, 2]
    box_colors = ['#3498db', '#e74c3c']
    
    bp = ax.boxplot(box_data, positions=box_positions, widths=0.5, 
                    patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set(color='#555', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='#555', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='white', linewidth=2)
    
    # Individual points
    for data, pos, color in zip(box_data, box_positions, box_colors):
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(pos + jitter, data, alpha=0.5, s=40, c=color, edgecolors='white', linewidth=0.5)
    
    ax.set_xticklabels(['Benign\n(Pure Random)', 'Infected\n(Hidden Tribes)'], fontweight='bold')
    ax.set_ylabel('Max Local PCC Score', fontweight='bold')
    ax.set_title('Local Topology Trojan Detection\n(Random Probing Method)', fontweight='bold', pad=15)
    
    # Statistics
    m_benign = np.mean(benign_pcc)
    m_infected = np.mean(infected_pcc)
    s_benign = np.std(benign_pcc)
    s_infected = np.std(infected_pcc)
    diff = m_infected - m_benign
    
    pooled_std = np.sqrt((s_benign**2 + s_infected**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    t_stat, p_value = stats.ttest_ind(benign_pcc, infected_pcc)
    
    stats_text = (f"Benign: μ = {m_benign:.2f}\n"
                  f"Infected: μ = {m_infected:.2f}\n"
                  f"Cohen's d = {cohens_d:.2f}\n"
                  f"p-value = {p_value:.2e}")
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Significance
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."
    
    max_val = max(np.max(benign_pcc), np.max(infected_pcc))
    ax.plot([1, 1, 2, 2], [max_val + 0.3, max_val + 0.5, max_val + 0.5, max_val + 0.3], 
            color='black', linewidth=1.5)
    ax.text(1.5, max_val + 0.7, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    # Detection threshold line
    threshold = (m_benign + m_infected) / 2
    ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(2.3, threshold, f'Threshold\n({threshold:.2f})', fontsize=9, va='center', color='green')
    
    ax.set_ylim(1.5, max_val + 1.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'trojan_detection_local.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print(f"\nCohen's d: {cohens_d:.4f}")
    print(f"p-value: {p_value:.2e}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    create_plot()
