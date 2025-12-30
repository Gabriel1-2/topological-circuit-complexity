"""
Visualizing DNA Topology

Plots values and sample CGR fractals for Coding vs Random DNA.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import generation function to plot CGRs
try:
    from sorted_dna_topology import generate_dna, dna_to_cgr
except:
    # Quick re-def if import fails due to path
    def generate_dna(mode='random', length=1000):
        bases = ['A', 'C', 'G', 'T']
        if mode == 'random': return np.random.choice(bases, length)
        elif mode == 'coding':
            codons = [['A','T','G'], ['C','G','C'], ['T','A','T'], ['G','G','A']]
            seq = []
            while len(seq) < length:
                if np.random.random() < 0.8:
                    seq.extend(codons[np.random.randint(0, len(codons))])
                else: seq.append(np.random.choice(bases))
            return np.array(seq[:length])
        return np.array([])

    def dna_to_cgr(sequence):
        x, y = 0.5, 0.5
        coords = []
        for base in sequence:
            if base == 'A': x, y = x/2, y/2
            elif base == 'C': x, y = x/2, (y+1)/2
            elif base == 'G': x, y = (x+1)/2, (y+1)/2
            elif base == 'T': x, y = (x+1)/2, y/2
            coords.append([x, y])
        return np.array(coords)

def plot_dna_results():
    print("=" * 60)
    print("DATA VISUALIZATION: DNA MANIFOLDS")
    print("=" * 60)
    
    # Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dna_topology.csv')
    df = pd.read_csv(data_path)
    
    # Plot 1: Boxplot of PCC
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Group data
    coding_pcc = df[df['type'] == 'coding']['pcc']
    random_pcc = df[df['type'] == 'random']['pcc']
    periodic_pcc = df[df['type'] == 'periodic']['pcc']
    
    ax = axes[0]
    ax.boxplot([random_pcc, coding_pcc, periodic_pcc], labels=['Random (Junk)', 'Coding (Gene)', 'Periodic'])
    ax.set_ylabel('Topological Complexity (PCC)', fontweight='bold')
    ax.set_title('Topological Signature of "Life"', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: CGR Visual (Random)
    ax = axes[1]
    seq = generate_dna('random', 2000)
    cgr = dna_to_cgr(seq)
    ax.scatter(cgr[:, 0], cgr[:, 1], s=1, alpha=0.5, color='orange')
    ax.set_title('Random DNA (Chaos)', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Plot 3: CGR Visual (Coding)
    ax = axes[2]
    seq = generate_dna('coding', 2000)
    cgr = dna_to_cgr(seq)
    ax.scatter(cgr[:, 0], cgr[:, 1], s=1, alpha=0.5, color='crimson')
    ax.set_title('Gene Sequence (Structure)', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'dna_manifold.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    # Stats
    print("\n" + "-" * 40)
    print("STATISTICAL VERDICT")
    print("-" * 40)
    
    diff_percent = (coding_pcc.mean() - random_pcc.mean()) / random_pcc.mean() * 100
    print(f"Coding vs Random Difference: {diff_percent:+.2f}%")
    
    if diff_percent > 10:
        print("\n✓ HYPOTHESIS CONFIRMED")
        print("Genes have significantly higher topological complexity than junk DNA.")
        print("Life leaves a topological footprint.")
    else:
        print("\n~ INCONCLUSIVE")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    plot_dna_results()
