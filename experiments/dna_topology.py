"""
The Topology of Life: DNA Analysis

Simulates "Coding" sequences (structured, repetitive motifs) vs "Junk" sequences (random).
Converts to Chaos Game Representation (CGR) and computes PCC.
"""

import os
import csv
import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
import ripser

# Configuration
SEQ_LENGTH = 1000
NUM_SAMPLES = 50
MAX_POINTS = 500  # Subsample for TDA

def generate_dna(mode='random', length=1000):
    bases = ['A', 'C', 'G', 'T']
    
    if mode == 'random':
        # "Junk" DNA - Independent Random
        return np.random.choice(bases, length)
    
    elif mode == 'coding':
        # "Gene-like" - Has structure, repeats, motifs
        # Simple simulation: Repeating codons + some noise
        codons = [['A','T','G'], ['C','G','C'], ['T','A','T'], ['G','G','A']]
        seq = []
        while len(seq) < length:
            if np.random.random() < 0.8: # 80% structure
                codon = codons[np.random.randint(0, len(codons))]
                seq.extend(codon)
            else: # 20% mutation/noise
                seq.append(np.random.choice(bases))
        return np.array(seq[:length])
    
    elif mode == 'periodic':
        # "Satellite" DNA - Simple Repeats (e.g. CAGCAGCAG)
        motif = ['C', 'A', 'G']
        seq = []
        for _ in range(length // 3 + 1):
            seq.extend(motif)
        return np.array(seq[:length])

def dna_to_cgr(sequence):
    """
    Chaos Game Representation (CGR)
    Converts DNA sequence to [x, y] coordinates.
    A=(0,0), C=(0,1), G=(1,1), T=(1,0)
    """
    x, y = 0.5, 0.5
    coords = []
    
    for base in sequence:
        if base == 'A':
            x, y = x/2, y/2
        elif base == 'C':
            x, y = x/2, (y+1)/2
        elif base == 'G':
            x, y = (x+1)/2, (y+1)/2
        elif base == 'T':
            x, y = (x+1)/2, y/2
        coords.append([x, y])
        
    return np.array(coords)

def compute_pcc(coords):
    """Compute H1 Persistence of the CGR point cloud."""
    if len(coords) > MAX_POINTS:
        idx = np.random.choice(len(coords), MAX_POINTS, replace=False)
        coords = coords[idx]
        
    # TDA
    dist = squareform(pdist(coords))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = ripser.ripser(dist, distance_matrix=True, maxdim=1)
            dgms = res['dgms']
            pcc = 0.0
            if len(dgms) > 1:
                h1 = dgms[1]
                finite = h1[~np.isinf(h1[:, 1])]
                if len(finite) > 0:
                    lifetimes = finite[:, 1] - finite[:, 0]
                    pcc = np.sum(lifetimes ** 2)
            return pcc
        except:
            return 0.0

def run_experiment():
    print("=" * 60)
    print("THE TOPOLOGY OF LIFE (DNA ANALYSIS)")
    print("=" * 60)
    
    results = []
    
    modes = ['random', 'coding', 'periodic']
    
    for mode in modes:
        print(f"Processing {mode} DNA ({NUM_SAMPLES} samples)...")
        pcc_values = []
        
        for _ in range(NUM_SAMPLES):
            # Generate
            seq = generate_dna(mode, SEQ_LENGTH)
            # Encode
            cgr = dna_to_cgr(seq)
            # Analyze
            pcc = compute_pcc(cgr)
            
            pcc_values.append(pcc)
            
            results.append({
                'type': mode,
                'pcc': pcc
            })
            
        print(f"  Avg PCC: {np.mean(pcc_values):.4f}")
        
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dna_topology.csv')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'pcc'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\n✓ Saved: {output_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_experiment()
