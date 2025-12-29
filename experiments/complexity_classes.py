"""
Complexity Classes Generator

Generates On-Sets for boolean functions from different complexity classes:
- AC0: Tribes (DNF of small-width ANDs)
- NC1: Parity, Majority
- P/poly: Random functions

Used for Topological Complexity Class Separation experiments.
"""

import numpy as np
import os

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def tribes_function(coords, width=3):
    """
    AC0 Tribes function: OR of disjoint ANDs.
    
    Splits n bits into groups of 'width'. Returns 1 if ANY group has all 1s.
    Example for n=12, width=3: (x0&x1&x2) | (x3&x4&x5) | (x6&x7&x8) | (x9&x10&x11)
    
    This is a canonical AC0 function - computable by constant-depth circuits.
    """
    n = coords.shape[1]
    num_groups = n // width
    
    # Reshape to group bits
    # Only use bits that fit evenly into groups
    usable_bits = num_groups * width
    grouped = coords[:, :usable_bits].reshape(coords.shape[0], num_groups, width)
    
    # AND within each group (product along width axis)
    group_ands = np.prod(grouped, axis=2)  # Shape: (2^n, num_groups)
    
    # OR across groups (max = 1 if any group is all 1s)
    result = np.max(group_ands, axis=1)
    
    return result.astype(np.int8)

def parity_function(coords):
    """
    NC1 Parity function: XOR of all bits.
    
    Returns 1 if Hamming weight is odd.
    This is the canonical NC1-complete function (cannot be computed by AC0).
    """
    hw = np.sum(coords, axis=1)
    return (hw % 2).astype(np.int8)

def majority_function(coords):
    """
    NC1 Majority function: 1 if more than half the bits are 1.
    
    This is TC0, between AC0 and NC1 in the hierarchy.
    """
    n = coords.shape[1]
    hw = np.sum(coords, axis=1)
    return (hw > n / 2).astype(np.int8)

def random_function(n, seed=42):
    """
    P/poly Random function: completely random truth table.
    
    Represents the "hardest" functions - likely exponential circuit complexity.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=2**n, dtype=np.int8)

def generate_class_data(n, tribes_width=3, random_seed=42):
    """
    Generate On-Sets for all complexity classes.
    
    Args:
        n: Number of bits.
        tribes_width: Width of AND groups for Tribes (default 3).
        random_seed: Seed for random function.
    
    Returns:
        dict: Mapping class name -> On-Set coordinates (M x n array)
    """
    print(f"Generating complexity class data for N={n}...")
    
    coords = get_hypercube_coords(n)
    
    # Generate truth tables
    print("  [1/4] AC0_Tribes...")
    tt_tribes = tribes_function(coords, width=tribes_width)
    
    print("  [2/4] NC1_Parity...")
    tt_parity = parity_function(coords)
    
    print("  [3/4] NC1_Majority...")
    tt_majority = majority_function(coords)
    
    print("  [4/4] P_Poly_Random...")
    tt_random = random_function(n, seed=random_seed)
    
    # Extract On-Sets (coordinates where f(x) = 1)
    data = {
        'AC0_Tribes': coords[tt_tribes == 1],
        'NC1_Parity': coords[tt_parity == 1],
        'NC1_Majority': coords[tt_majority == 1],
        'P_Poly_Random': coords[tt_random == 1],
    }
    
    # Print statistics
    print("\n  Class Statistics:")
    print(f"  {'Class':<20} | {'On-Set Size':<12} | {'Fraction':<10}")
    print("  " + "-" * 50)
    for name, on_set in data.items():
        frac = len(on_set) / (2**n)
        print(f"  {name:<20} | {len(on_set):<12} | {frac:.4f}")
    
    return data

def save_class_data(data, filepath):
    """Save class data to NPZ file."""
    np.savez_compressed(filepath, **{k: v for k, v in data.items()})
    print(f"\n  Saved to: {filepath}")

def main():
    n = 12
    
    print("=" * 60)
    print("COMPLEXITY CLASS SEPARATION DATA GENERATOR")
    print("=" * 60)
    
    # Generate data
    data = generate_class_data(n, tribes_width=3, random_seed=42)
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'class_separation_n{n}.npz')
    save_class_data(data, output_path)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    # Verify
    print("\nVerification (loading saved file):")
    loaded = np.load(output_path)
    for key in loaded.files:
        print(f"  {key}: shape = {loaded[key].shape}")

if __name__ == "__main__":
    main()
