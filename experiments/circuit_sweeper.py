"""
Circuit Sweeper

Generates a large dataset of random boolean circuits with varying
depth and size for regression analysis of topological complexity.
"""

import os
import csv
import numpy as np

# Configuration
N_INPUTS = 10
DEPTH_RANGE = range(2, 9)      # 2 to 8
SIZE_RANGE = range(10, 101, 10) # 10, 20, ..., 100
SAMPLES_PER_CONFIG = 20

def get_hypercube_coords(n):
    """Generate all 2^n binary coordinates."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    return ((indices[:, None] & masks) > 0).astype(np.int8)

def truth_table_to_hex(tt):
    """Convert boolean truth table to hex string."""
    tt_int = int(''.join(map(str, tt.astype(int))), 2)
    return hex(tt_int)[2:].zfill(len(tt) // 4)

class Wire:
    """Represents a wire in the circuit."""
    def __init__(self, values):
        self.values = values  # Truth table values for this wire

class RandomCircuit:
    """Random boolean circuit generator."""
    
    def __init__(self, n_inputs, depth, size, seed=None):
        self.n_inputs = n_inputs
        self.depth = depth
        self.size = size
        self.rng = np.random.default_rng(seed)
        
        # All coordinates for evaluation
        self.coords = get_hypercube_coords(n_inputs)
        
        # Input wires
        self.wires = [Wire(self.coords[:, i]) for i in range(n_inputs)]
        
    def build(self):
        """Build the circuit."""
        # Distribute gates across layers
        gates_per_layer = self._distribute_gates()
        
        for layer_idx, n_gates in enumerate(gates_per_layer):
            layer_wires = []
            
            for _ in range(n_gates):
                gate_type = self.rng.choice(['AND', 'OR', 'NOT'])
                
                if gate_type == 'NOT':
                    # Unary gate
                    input_wire = self.rng.choice(self.wires)
                    output = 1 - input_wire.values
                else:
                    # Binary gate
                    input_wires = self.rng.choice(self.wires, size=2, replace=True)
                    if gate_type == 'AND':
                        output = input_wires[0].values & input_wires[1].values
                    else:  # OR
                        output = input_wires[0].values | input_wires[1].values
                
                layer_wires.append(Wire(output))
            
            # Add layer wires to available pool
            self.wires.extend(layer_wires)
        
        # Final output: last wire
        return self.wires[-1].values
    
    def _distribute_gates(self):
        """Distribute gates across layers."""
        # Ensure at least 1 gate per layer
        gates_per_layer = [1] * self.depth
        remaining = self.size - self.depth
        
        if remaining > 0:
            # Randomly distribute remaining gates
            extra = self.rng.multinomial(remaining, [1/self.depth] * self.depth)
            gates_per_layer = [g + e for g, e in zip(gates_per_layer, extra)]
        
        return gates_per_layer

def generate_circuit(n_inputs, depth, size, seed=None):
    """Generate a random circuit and return its truth table."""
    circuit = RandomCircuit(n_inputs, depth, size, seed)
    return circuit.build()

def run_sweep():
    print("=" * 60)
    print("CIRCUIT SWEEPER")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  N inputs: {N_INPUTS}")
    print(f"  Depth: {list(DEPTH_RANGE)}")
    print(f"  Size: {list(SIZE_RANGE)}")
    print(f"  Samples per config: {SAMPLES_PER_CONFIG}")
    
    total_configs = len(list(DEPTH_RANGE)) * len(list(SIZE_RANGE))
    total_circuits = total_configs * SAMPLES_PER_CONFIG
    print(f"  Total circuits: {total_circuits}")
    
    results = []
    circuit_count = 0
    
    for depth in DEPTH_RANGE:
        for size in SIZE_RANGE:
            for sample in range(SAMPLES_PER_CONFIG):
                seed = depth * 10000 + size * 100 + sample
                
                try:
                    tt = generate_circuit(N_INPUTS, depth, size, seed)
                    tt_hex = truth_table_to_hex(tt)
                    
                    results.append({
                        'depth': depth,
                        'size': size,
                        'sample': sample,
                        'truth_table_hex': tt_hex
                    })
                    
                    circuit_count += 1
                    
                except Exception as e:
                    print(f"  Error at depth={depth}, size={size}, sample={sample}: {e}")
            
            if circuit_count % 100 == 0:
                print(f"  Generated {circuit_count}/{total_circuits} circuits...")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'circuit_sweep.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['depth', 'size', 'sample', 'truth_table_hex'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Total circuits: {len(results)}")
    
    # Stats
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    for depth in DEPTH_RANGE:
        depth_circuits = [r for r in results if r['depth'] == depth]
        print(f"  Depth {depth}: {len(depth_circuits)} circuits")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_sweep()
