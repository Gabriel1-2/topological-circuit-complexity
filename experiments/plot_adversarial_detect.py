"""
Adversarial Detection Visualization

Box plot comparing Local PCC distributions for Clean vs Adversarial samples.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

def load_results(filepath):
    """Load detection results."""
    clean_pcc = []
    adv_pcc = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pcc = float(row['local_pcc'])
            if row['type'] == 'Clean':
                clean_pcc.append(pcc)
            else:
                adv_pcc.append(pcc)
    
    return np.array(clean_pcc), np.array(adv_pcc)

def create_detection_plot():
    print("=" * 60)
    print("ADVERSARIAL DETECTION VISUALIZATION")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'detection_results.csv')
    clean_pcc, adv_pcc = load_results(data_path)
    
    print(f"\nLoaded {len(clean_pcc)} clean and {len(adv_pcc)} adversarial samples")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Box plot
    box_data = [clean_pcc, adv_pcc]
    box_positions = [1, 2]
    box_colors = ['#3498db', '#e74c3c']
    
    bp = ax.boxplot(box_data, positions=box_positions, widths=0.5, 
                    patch_artist=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(color='#555', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='#555', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='white', linewidth=2)
    
    # Add individual points (jittered)
    for i, (data, pos, color) in enumerate(zip(box_data, box_positions, box_colors)):
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(pos + jitter, data, alpha=0.5, s=30, c=color, edgecolors='white', linewidth=0.5)
    
    # Labels
    ax.set_xticklabels(['Clean\nSamples', 'Adversarial\nSamples'], fontweight='bold')
    ax.set_ylabel('Local PCC Score', fontweight='bold')
    ax.set_title('Topological Lie Detector: Local Complexity Comparison\n' +
                 '(Higher PCC = More "Fractured" Decision Boundary)', fontweight='bold', pad=15)
    
    # Add statistics annotation
    mean_clean = np.mean(clean_pcc)
    mean_adv = np.mean(adv_pcc)
    separation = mean_adv - mean_clean
    
    # T-test
    t_stat, p_value = stats.ttest_ind(clean_pcc, adv_pcc)
    
    stats_text = (f"Clean: μ = {mean_clean:.4f}\n"
                  f"Adv: μ = {mean_adv:.4f}\n"
                  f"Separation: {separation:+.4f}\n"
                  f"t-stat: {t_stat:.2f}\n"
                  f"p-value: {p_value:.2e}")
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add significance stars if p < 0.05
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."
    
    max_val = max(np.max(clean_pcc), np.max(adv_pcc))
    ax.plot([1, 1, 2, 2], [max_val + 0.02, max_val + 0.04, max_val + 0.04, max_val + 0.02], 
            color='black', linewidth=1.5)
    ax.text(1.5, max_val + 0.05, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    ax.set_ylim(-0.02, max_val + 0.12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'adversarial_detection.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Print summary
    print("\n" + "-" * 40)
    print("STATISTICAL SUMMARY")
    print("-" * 40)
    print(f"Clean Mean:       {mean_clean:.4f}")
    print(f"Adversarial Mean: {mean_adv:.4f}")
    print(f"Separation:       {separation:+.4f}")
    print(f"Effect Size:      {separation / np.std(clean_pcc):.2f}σ")
    print(f"t-statistic:      {t_stat:.4f}")
    print(f"p-value:          {p_value:.2e}")
    
    if p_value < 0.05:
        print("\n✓ Statistically significant difference (p < 0.05)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    create_detection_plot()
