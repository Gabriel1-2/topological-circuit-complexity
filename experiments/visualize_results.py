
import sys
import os
import zlib
import numpy as np
import matplotlib.pyplot as plt
import persim

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from boolean_gen import generate_truth_table
from topology_calc import compute_persistence, calculate_pcc

def approximation_circuit_complexity(truth_table):
    """
    Approximates circuit complexity using Lempel-Ziv complexity 
    (compressed size of the truth table).
    """
    data = truth_table.astype(np.uint8).tobytes()
    return len(zlib.compress(data))

def add_noise(truth_table, noise_level=0.05):
    """Flips bits with probability noise_level."""
    mask = np.random.random(truth_table.shape) < noise_level
    noisy_tt = truth_table.copy()
    noisy_tt[mask] = 1 - noisy_tt[mask]
    return noisy_tt

def run_experiment_n8():
    print("Running Experiment N=8...")
    n = 8
    
    # 1. Generate core functions
    funcs = ['random', 'majority', 'parity']
    results = {}
    
    print(f"Generating data for {funcs}...")
    for f_type in funcs:
        tt, coords = generate_truth_table(n, f_type, seed=42)
        print(f"  {f_type}: {len(coords)} points in On-Set")
        dgms = compute_persistence(coords, max_dim=1)
        results[f_type] = dgms

    # 2. Plot 1: Persistence Barcodes
    print("Generating Plot 1: Persistence Barcodes...")
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, f_type in enumerate(funcs):
        ax = axes[i]
        dgms = results[f_type]
        persim.plot_diagrams(dgms, ax=ax, title=f"{f_type.capitalize()} (N={n})")
        
    plt.tight_layout()
    # Save temporarily to keep in memory or specific file?
    # We will combine plot 2 or save separately?
    # User asked for "A PNG image named pcc_comparison_n8.png showing the barcodes".
    # Wait, Plot 2 is also requested.
    # Maybe 2 separate images or one big one?
    # "Deliverable: A PNG image named pcc_comparison_n8.png showing the barcodes."
    # It explicitly mentions "showing the barcodes".
    # But request also asks for "Plot 2: PCC Correlation".
    # Usually "Deliverable" implies the final artifact.
    # I will make a combined figure with 2 rows.
    # Row 1: Barcodes (3 subplots)
    # Row 2: PCC Scatter (1 subplot, spanning width)
    
    plt.close(fig1) # clear previous setup
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Row 1: Barcodes
    for i, f_type in enumerate(funcs):
        ax = fig.add_subplot(gs[0, i])
        dgms = results[f_type]
        persim.plot_diagrams(dgms, ax=ax, title=f"{f_type.capitalize()} (N={n})")

    # 3. Experiment 2: PCC Correlation
    print("Generating Plot 2: PCC Correlation...")
    
    random_pccs = []
    random_complexities = []
    
    structured_pccs = []
    structured_complexities = []
    
    # Generate 50 Random
    print("  Simulating 50 Random functions...")
    for _ in range(50):
        tt, coords = generate_truth_table(n, 'random')
        dgms = compute_persistence(coords, max_dim=1)
        pcc = calculate_pcc(dgms)
        comp = approximation_circuit_complexity(tt)
        
        random_pccs.append(pcc)
        random_complexities.append(comp)
        
    # Generate 50 Structured (Noisy Majority/Parity)
    print("  Simulating 50 Structured functions (Noisy)...")
    base_types = ['majority', 'parity']
    for _ in range(50):
        b_type = np.random.choice(base_types)
        # Get base
        tt_base, _ = generate_truth_table(n, b_type)
        # Add noise
        tt_noisy = add_noise(tt_base, noise_level=0.05)
        
        # Get coords for noisy
        # We need to extract coords from noisy tt manually
        # Re-use logic from boolean_gen or just do it here
        # Indices where tt is 1
        indices = np.where(tt_noisy == 1)[0]
        # We need hypercube coords
        # Let's import get_hypercube_coords or re-implement
        # It's hidden in boolean_gen.
        # I'll just rely on generate_truth_table returning coords matching the truth table?
        # No, generate_truth_table returns coords matching the generated function.
        # I supplied 'random' above.
        # Here I have modified tt.
        # I need to convert indices to coords.
        # I'll modify boolean_gen.py to expose get_hypercube_coords OR
        # copy the simple logic here.
        pass
    
    # helper for coords
    # (1 << np.arange(n)[::-1])
    masks = 1 << np.arange(n)[::-1] # 8 bits
    all_indices = np.arange(2**n, dtype=int)
    all_coords = ((all_indices[:, None] & masks) > 0).astype(int)
    
    # Re-run structured loop with correct coords
    filtered_structured_complexities = []
    filtered_structured_pccs = []
    
    for _ in range(50):
        b_type = np.random.choice(base_types)
        tt_base, _ = generate_truth_table(n, b_type)
        tt_noisy = add_noise(tt_base, noise_level=0.02) # Low noise to keep structure
        
        indices = np.where(tt_noisy == 1)[0]
        coords = all_coords[indices]
        
        dgms = compute_persistence(coords, max_dim=1)
        pcc = calculate_pcc(dgms)
        comp = approximation_circuit_complexity(tt_noisy)
        
        filtered_structured_pccs.append(pcc)
        filtered_structured_complexities.append(comp)

    # Plot Scatter
    ax_scatter = fig.add_subplot(gs[1, :]) # Span all columns
    
    ax_scatter.scatter(random_complexities, random_pccs, c='red', alpha=0.6, label='Random')
    ax_scatter.scatter(filtered_structured_complexities, filtered_structured_pccs, c='blue', alpha=0.6, label='Structured (Noisy)')
    
    ax_scatter.set_xlabel('Approximate Circuit Complexity (Compressed Size)')
    ax_scatter.set_ylabel('PCC Score (Total Persistence)')
    ax_scatter.set_title('PCC vs Complexity Correlation')
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'pcc_comparison_n8.png')
    plt.savefig(output_path)
    print(f"Analysis complete. Plot saved to: {output_path}")

if __name__ == "__main__":
    run_experiment_n8()
