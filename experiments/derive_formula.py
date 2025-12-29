"""
Derive the Law of Topological Complexity

Fits a power-law model to find how PCC scales with circuit parameters.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'figure.dpi': 150,
})

def load_results(filepath):
    """Load sweep results."""
    data = {'depth': [], 'size': [], 'sensitivity': [], 'pcc': []}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['depth'].append(int(row['depth']))
            data['size'].append(int(row['size']))
            data['sensitivity'].append(float(row['sensitivity']))
            data['pcc'].append(float(row['pcc']))
    
    return {k: np.array(v) for k, v in data.items()}

def print_correlation_matrix(data):
    """Print correlation matrix."""
    print("\n" + "-" * 50)
    print("CORRELATION MATRIX")
    print("-" * 50)
    
    keys = ['depth', 'size', 'sensitivity', 'pcc']
    matrix = np.zeros((4, 4))
    
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            matrix[i, j] = np.corrcoef(data[k1], data[k2])[0, 1]
    
    print(f"\n{'':>15} {'Depth':>10} {'Size':>10} {'Sens':>10} {'PCC':>10}")
    print("-" * 55)
    for i, k in enumerate(keys):
        row = f"{k.capitalize():>15}"
        for j in range(4):
            row += f" {matrix[i,j]:>10.4f}"
        print(row)

def fit_power_law(data):
    """Fit log-linear regression for power-law exponents."""
    print("\n" + "-" * 50)
    print("POWER-LAW REGRESSION")
    print("-" * 50)
    
    # Filter non-positive values
    mask = (data['pcc'] > 0) & (data['sensitivity'] > 0)
    
    log_pcc = np.log(data['pcc'][mask])
    log_size = np.log(data['size'][mask])
    log_depth = np.log(data['depth'][mask])
    log_sens = np.log(data['sensitivity'][mask])
    
    # Fit: log(PCC) = a*log(Size) + b*log(Depth) + c*log(Sens) + k
    X = np.column_stack([log_size, log_depth, log_sens])
    model = LinearRegression()
    model.fit(X, log_pcc)
    
    a, b, c = model.coef_
    k = model.intercept_
    
    print(f"\n  log(PCC) = {a:.4f}*log(Size) + {b:.4f}*log(Depth) + {c:.4f}*log(Sens) + {k:.4f}")
    print(f"\n  R² = {model.score(X, log_pcc):.4f}")
    
    # Also fit without sensitivity
    X2 = np.column_stack([log_size, log_depth])
    model2 = LinearRegression()
    model2.fit(X2, log_pcc)
    
    a2, b2 = model2.coef_
    k2 = model2.intercept_
    
    print(f"\n  Simplified (no Sensitivity):")
    print(f"  log(PCC) = {a2:.4f}*log(Size) + {b2:.4f}*log(Depth) + {k2:.4f}")
    print(f"  R² = {model2.score(X2, log_pcc):.4f}")
    
    # Convert to power law
    K = np.exp(k2)
    print(f"\n  === THE LAW OF TOPOLOGICAL COMPLEXITY ===")
    print(f"  PCC ≈ {K:.4f} × Size^{a2:.4f} × Depth^{b2:.4f}")
    
    return {'a': a2, 'b': b2, 'K': K, 'r2': model2.score(X2, log_pcc)}

def create_3d_plot(data, formula, output_path):
    """Create 3D scatter plot of PCC landscape."""
    print("\n  Creating 3D phase plot...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by PCC
    sc = ax.scatter(data['size'], data['depth'], data['pcc'],
                    c=data['pcc'], cmap='viridis', s=15, alpha=0.6)
    
    ax.set_xlabel('Circuit Size (gates)', fontweight='bold')
    ax.set_ylabel('Circuit Depth (layers)', fontweight='bold')
    ax.set_zlabel('PCC (Total Persistence)', fontweight='bold')
    
    ax.set_title('The Landscape of Topological Complexity\n' +
                 f"PCC ≈ {formula['K']:.2f} × Size^{formula['a']:.2f} × Depth^{formula['b']:.2f}  (R²={formula['r2']:.3f})",
                 fontweight='bold', pad=20)
    
    plt.colorbar(sc, ax=ax, shrink=0.6, label='PCC')
    
    # Add formula box
    formula_text = (f"The Law:\n"
                    f"PCC ∝ Size^{formula['a']:.2f}\n"
                    f"PCC ∝ Depth^{formula['b']:.2f}")
    ax.text2D(0.02, 0.98, formula_text, transform=ax.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")

def main():
    print("=" * 60)
    print("DERIVING THE LAW OF TOPOLOGICAL COMPLEXITY")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'sweep_results.csv')
    print(f"\nLoading: {data_path}")
    data = load_results(data_path)
    print(f"Loaded {len(data['pcc'])} samples")
    
    # Correlation matrix
    print_correlation_matrix(data)
    
    # Power-law regression
    formula = fit_power_law(data)
    
    # 3D plot
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'pcc_law.png')
    create_3d_plot(data, formula, output_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  THE LAW OF TOPOLOGICAL COMPLEXITY:")
    print(f"\n      PCC ≈ {formula['K']:.4f} × Size^{formula['a']:.4f} × Depth^{formula['b']:.4f}")
    print(f"\n  Interpretation:")
    if formula['a'] > 0:
        print(f"    • Size (gates):  MORE gates → HIGHER PCC (a = +{formula['a']:.2f})")
    else:
        print(f"    • Size (gates):  MORE gates → LOWER PCC (a = {formula['a']:.2f})")
    if formula['b'] > 0:
        print(f"    • Depth (layers): MORE layers → HIGHER PCC (b = +{formula['b']:.2f})")
    else:
        print(f"    • Depth (layers): MORE layers → LOWER PCC (b = {formula['b']:.2f})")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
