"""
Visualizing the Edge of Chaos

Creates "The Complexity Map": Topological Complexity (PCC) vs Entropy.
Identifies if "Class 4" rules (like 110) occupy a distinct region.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_edge_of_chaos():
    print("=" * 60)
    print("VISUALIZING THE EDGE OF CHAOS")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'automata_topology.csv')
    df = pd.read_csv(data_path)
    
    # Define Wolfram Classes (approximate)
    # Class 1: Homogeneous (Death)
    class_1 = [0, 8, 32, 40, 128, 136, 160, 168]
    # Class 2: Periodicity
    class_2 = [108, 1, 19, 23, 50, 77]
    # Class 3: Chaos
    class_3 = [30, 45, 90, 105, 150]
    # Class 4: Complexity (Edge of Chaos)
    class_4 = [110, 124, 137, 193] # 110 is the famous one
    
    # Assign colors
    colors = []
    sizes = []
    labels = []
    
    for r in df['rule']:
        if r in class_4:
            colors.append('crimson') # Computation (Red)
            sizes.append(150)
            labels.append('Class 4 (Life)')
        elif r in class_3:
            colors.append('orange') # Chaos (Orange)
            sizes.append(80)
            labels.append('Class 3 (Chaos)')
        elif r in class_1:
            colors.append('black') # Death (Black)
            sizes.append(50)
            labels.append('Class 1 (Order)')
        elif r in class_2:
            colors.append('blue')  # Periodic (Blue)
            sizes.append(50)
            labels.append('Class 2 (Periodic)')
        else:
            colors.append('gray')  # Others
            sizes.append(30)
            labels.append('Unknown')
            
    df['color'] = colors
    df['size'] = sizes
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(df['entropy_mean'], df['pcc_mean'], 
                         c=df['color'], s=df['size'], alpha=0.7, edgecolors='white')
    
    # Annotate key rules
    for r in class_4 + class_3 + [0, 90]:
        row = df[df['rule'] == r].iloc[0]
        ax.annotate(str(r), (row['entropy_mean'], row['pcc_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Entropy (Disorder)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Topological Complexity (PCC)', fontsize=12, fontweight='bold')
    ax.set_title('The Complexity Map: Searching for the Shape of Life', fontsize=14, fontweight='bold')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=12)]
    ax.legend(custom_lines, ['Order', 'Periodic', 'Chaos', 'COMPUTATION (Rule 110)'], loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # Interpret Zones
    ax.text(0.1, ax.get_ylim()[1]*0.1, "Zone of Death\n(Low Entropy, No Topology)", 
            color='black', alpha=0.6, ha='center')
    
    ax.text(0.9, ax.get_ylim()[1]*0.1, "Zone of Chaos\n(High Entropy, Fragmented Topology)", 
            color='orange', alpha=0.6, ha='center')
            
    ax.text(0.5, ax.get_ylim()[1]*0.9, "THE EDGE OF CHAOS\n(Medium Entropy, PEAK Topology)", 
            color='crimson', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8))
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'edge_of_chaos.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    plot_edge_of_chaos()
