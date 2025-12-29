"""
Trojan Simulation for Topological Malware Detection

Creates benign (pure random) and infected (random + hidden structured payload)
boolean functions to test if TDA can detect hidden local structure.
"""

import os
import numpy as np

# Configuration
N = 12
TROJAN_TRIGGER_BITS = 4  # First 4 bits must be '1111' to activate trojan
NUM_FUNCTIONS = 50

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def generate_random_function(n, seed=None):
    """Generate purely random truth table."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=2**n, dtype=np.int8)

def tribes_pattern(coords, width=3):
    """
    Compute Tribes pattern for given coordinates.
    Returns 1 if any group of 'width' consecutive bits are all 1.
    """
    n = coords.shape[1]
    num_groups = n // width
    usable_bits = num_groups * width
    
    if usable_bits == 0:
        return np.zeros(len(coords), dtype=np.int8)
    
    grouped = coords[:, :usable_bits].reshape(len(coords), num_groups, width)
    group_ands = np.prod(grouped, axis=2)
    return np.max(group_ands, axis=1).astype(np.int8)

def inject_trojan(n=N, trigger_bits=TROJAN_TRIGGER_BITS, seed=None):
    """
    Create an infected function with hidden Tribes payload.
    
    - Base: Random noise everywhere
    - Trojan: When first 'trigger_bits' are all 1, follow Tribes pattern
              on the remaining bits
    
    Returns:
        truth_table: The infected truth table
        trojan_region_size: Number of inputs in the trojan region
    """
    rng = np.random.default_rng(seed)
    
    # Generate base random function
    truth_table = rng.integers(0, 2, size=2**n, dtype=np.int8)
    
    # Get all coordinates
    coords = get_hypercube_coords(n)
    
    # Find trojan region: where first 'trigger_bits' are all 1
    trigger_mask = np.all(coords[:, :trigger_bits] == 1, axis=1)
    trojan_indices = np.where(trigger_mask)[0]
    
    # Get remaining bits for trojan region
    remaining_coords = coords[trigger_mask, trigger_bits:]
    
    # Apply Tribes pattern to the trojan region
    trojan_pattern = tribes_pattern(remaining_coords, width=2)  # Width 2 for smaller region
    
    # Inject trojan (overwrite random values in trojan region)
    truth_table[trojan_indices] = trojan_pattern
    
    return truth_table, len(trojan_indices)

def generate_datasets():
    print("=" * 60)
    print("TROJAN SIMULATION")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  N = {N} bits (2^{N} = {2**N} truth table size)")
    print(f"  Trigger = first {TROJAN_TRIGGER_BITS} bits all 1")
    print(f"  Functions per class = {NUM_FUNCTIONS}")
    
    coords = get_hypercube_coords(N)
    
    # Benign dataset
    print(f"\n[1] Generating {NUM_FUNCTIONS} BENIGN functions...")
    benign_on_sets = []
    benign_sizes = []
    
    for i in range(NUM_FUNCTIONS):
        tt = generate_random_function(N, seed=i * 1000)
        on_set = coords[tt == 1]
        benign_on_sets.append(on_set)
        benign_sizes.append(len(on_set))
    
    print(f"    Avg On-Set size: {np.mean(benign_sizes):.1f} ± {np.std(benign_sizes):.1f}")
    
    # Infected dataset
    print(f"\n[2] Generating {NUM_FUNCTIONS} INFECTED functions...")
    infected_on_sets = []
    infected_sizes = []
    trojan_region_sizes = []
    
    for i in range(NUM_FUNCTIONS):
        tt, trojan_size = inject_trojan(N, TROJAN_TRIGGER_BITS, seed=i * 2000)
        on_set = coords[tt == 1]
        infected_on_sets.append(on_set)
        infected_sizes.append(len(on_set))
        trojan_region_sizes.append(trojan_size)
    
    print(f"    Avg On-Set size: {np.mean(infected_sizes):.1f} ± {np.std(infected_sizes):.1f}")
    print(f"    Trojan region size: {trojan_region_sizes[0]} inputs")
    
    # Check similarity
    size_diff = abs(np.mean(benign_sizes) - np.mean(infected_sizes))
    print(f"\n    On-Set size difference: {size_diff:.1f} (should be small)")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'trojan_data.npz')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to object array for variable-length arrays
    np.savez_compressed(
        output_path,
        benign_on_sets=np.array(benign_on_sets, dtype=object),
        infected_on_sets=np.array(infected_on_sets, dtype=object),
        benign_sizes=np.array(benign_sizes),
        infected_sizes=np.array(infected_sizes),
        n=N,
        trigger_bits=TROJAN_TRIGGER_BITS
    )
    
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TROJAN INJECTION SUMMARY")
    print("=" * 60)
    print(f"\nTrojan Mechanism:")
    print(f"  Trigger: When bits [0:{TROJAN_TRIGGER_BITS}] = '{'1'*TROJAN_TRIGGER_BITS}'")
    print(f"  Payload: Tribes pattern on remaining {N-TROJAN_TRIGGER_BITS} bits")
    print(f"  Region Size: {trojan_region_sizes[0]} / {2**N} = {100*trojan_region_sizes[0]/2**N:.1f}%")
    
    print(f"\nCamouflage Check:")
    print(f"  Benign avg size:   {np.mean(benign_sizes):.1f}")
    print(f"  Infected avg size: {np.mean(infected_sizes):.1f}")
    print(f"  Difference: {size_diff:.1f} ({100*size_diff/np.mean(benign_sizes):.1f}%)")
    
    if size_diff / np.mean(benign_sizes) < 0.05:
        print("\n  ✓ GOOD CAMOUFLAGE: Infected looks similar to benign by simple counting!")
    else:
        print("\n  ⚠ WEAK CAMOUFLAGE: Size difference may be detectable")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    generate_datasets()
