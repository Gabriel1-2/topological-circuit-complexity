"""
Boolean Function Generator Module

This module provides tools to generate truth tables and On-Sets (vertices where f(x)=1)
for various boolean functions. This data is intended for Topological Data Analysis (TDA)
on the Boolean Hypercube.
"""

import numpy as np
import networkx as nx
from itertools import combinations

def get_hypercube_coords(n):
    """
    Generates all 2^n binary coordinates of the boolean hypercube.
    
    Args:
        n (int): Number of variables/bits.
        
    Returns:
        np.ndarray: A (2^n, n) array of binary coordinates.
    """
    # Create numbers 0 to 2^n - 1
    indices = np.arange(2**n, dtype=int)
    # Convert to bits using broadcasting and bitwise AND
    # (1 << np.arange(n)[::-1]) creates masks [2^(n-1), ..., 2^0]
    masks = 1 << np.arange(n)[::-1]
    coords = ((indices[:, None] & masks) > 0).astype(int)
    return coords

def _is_clique(adj_bits, n_bits, k):
    """
    Helper to check if a graph represented by adj_bits has a clique of size k.
    Assumes adj_bits represents edges of an undirected graph.
    """
    # Determine number of vertices v
    # n_bits = v * (v - 1) / 2
    # 2 * n_bits = v^2 - v => v^2 - v - 2*n_bits = 0
    # v = (1 + sqrt(1 + 8*n_bits)) / 2
    discriminant = 1 + 8 * n_bits
    root = int(np.sqrt(discriminant))
    if root * root != discriminant:
        raise ValueError(f"Input n={n_bits} does not correspond to a complete graph edge count (n = v(v-1)/2).")
    
    v = (1 + root) // 2
    if v * (v - 1) // 2 != n_bits:
         raise ValueError(f"Input n={n_bits} is not valid for adjacency size.")

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(v))
    
    edge_idx = 0
    # Use standard ordering of edges: (0,1), (0,2)... (0,v-1), (1,2)...
    # This must be consistent.
    for i in range(v):
        for j in range(i + 1, v):
            if adj_bits[edge_idx] == 1:
                G.add_edge(i, j)
            edge_idx += 1
            
    # Check for k-clique
    # For small graphs, this is acceptable. Optimized clique algorithms exist but this is generic.
    from networkx.algorithms.clique import find_cliques
    # Simple check: max clique size >= k
    # Or specifically does it contain a k-clique? (Same thing)
    if k > v:
        return 0
        
    # Check if max clique size >= k
    # graph_clique_number is explicitly the size of the largest clique
    # For very large graphs this is slow, but we are limited by 2^n table generation anyway (n=15 max).
    # If n=15 -> v=6. This is trivial for NetworkX.
    max_clique_len = max((len(c) for c in find_cliques(G)), default=0)
    return 1 if max_clique_len >= k else 0

def generate_truth_table(n, func_type, **kwargs):
    """
    Generates the truth table and On-Set coordinates for a specified boolean function.
    
    Args:
        n (int): Number of variables (bits).
        func_type (str): Type of function ('random', 'majority', 'parity', 'clique').
        **kwargs: Additional arguments for specific types:
            - clique: 'k' (int, default=3) size of clique to detect.
            - seed: (int) for random generation.
            
    Returns:
        tuple: (truth_table, on_set_coords)
            truth_table (np.ndarray): 1D array of size 2^n in lexicographical order.
            on_set_coords (np.ndarray): (M, n) array of coordinates where f(x) = 1.
    """
    coords = get_hypercube_coords(n)
    num_rows = coords.shape[0]
    truth_table = np.zeros(num_rows, dtype=int)
    
    if func_type == 'random':
        seed = kwargs.get('seed', None)
        rng = np.random.default_rng(seed)
        truth_table = rng.integers(0, 2, size=num_rows)
        
    elif func_type == 'majority':
        # 1 if HammingWeight > n/2
        hw = np.sum(coords, axis=1)
        truth_table = (hw > (n / 2)).astype(int)
        
    elif func_type == 'parity':
        # 1 if HammingWeight is odd
        hw = np.sum(coords, axis=1)
        truth_table = (hw % 2).astype(int)
        
    elif func_type == 'threshold':
        # 1 if HammingWeight >= k (Linear Threshold Function)
        k = kwargs.get('k', n // 2)
        hw = np.sum(coords, axis=1)
        truth_table = (hw >= k).astype(int)
        
    elif func_type == 'clique':
        k = kwargs.get('k', 3)
        # Iterate over all possible graphs (rows in coords)
        # This is the slow part, loop in python, but necessary for complex graph property
        # n is small enough (<20 probably)
        for i in range(num_rows):
            truth_table[i] = _is_clique(coords[i], n, k)
            
    else:
        raise ValueError(f"Unknown function type: {func_type}")
        
    # Extract On-Set (coordinates where truth_table is 1)
    on_set_indices = np.where(truth_table == 1)[0]
    on_set_coords = coords[on_set_indices]
    
    return truth_table, on_set_coords
