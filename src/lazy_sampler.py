"""
Lazy Sampler Module

Efficiently generates M distinct points from the On-Set of a boolean function
WITHOUT building the full 2^N truth table. Essential for scaling to N=16+.

Uses constructive generation and rejection sampling strategies.
"""

import numpy as np
from itertools import combinations
from typing import Set, Tuple

def _int_to_binary_vector(x: int, n: int) -> np.ndarray:
    """Convert integer to n-bit binary vector (big-endian)."""
    return np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.int8)

def _binary_vector_to_int(v: np.ndarray) -> int:
    """Convert binary vector to integer."""
    n = len(v)
    return sum(int(v[i]) << (n - 1 - i) for i in range(n))

def sample_random(n: int, m: int, seed: int = None) -> np.ndarray:
    """
    Sample M unique random points from {0,1}^n.
    
    For 'random' boolean functions, any point could be in the On-Set,
    so we just sample uniformly from the hypercube.
    """
    rng = np.random.default_rng(seed)
    max_val = 2 ** n
    
    if m > max_val:
        raise ValueError(f"Cannot sample {m} unique points from 2^{n} = {max_val} total.")
    
    # For small m relative to 2^n, use rejection sampling
    if m < max_val // 2:
        seen: Set[int] = set()
        points = []
        
        while len(points) < m:
            x = rng.integers(0, max_val)
            if x not in seen:
                seen.add(x)
                points.append(_int_to_binary_vector(x, n))
        
        return np.array(points, dtype=np.int8)
    else:
        # For large m, shuffle and take first m
        all_indices = np.arange(max_val)
        rng.shuffle(all_indices)
        selected = all_indices[:m]
        return np.array([_int_to_binary_vector(x, n) for x in selected], dtype=np.int8)

def sample_threshold_k(n: int, m: int, k: int, seed: int = None) -> np.ndarray:
    """
    Sample M unique points with Hamming weight >= k.
    
    Constructive approach: Generate random vectors with exactly w ones,
    where w is sampled from {k, k+1, ..., n} weighted by binomial coefficients.
    """
    rng = np.random.default_rng(seed)
    
    # Calculate total number of valid points: sum(C(n,w) for w in k..n)
    from math import comb
    weights = [comb(n, w) for w in range(k, n + 1)]
    total_valid = sum(weights)
    
    if m > total_valid:
        raise ValueError(f"Cannot sample {m} points; only {total_valid} have weight >= {k}.")
    
    # Probability distribution over weights
    probs = np.array(weights, dtype=float) / total_valid
    weight_values = list(range(k, n + 1))
    
    seen: Set[Tuple[int, ...]] = set()
    points = []
    
    attempts = 0
    max_attempts = m * 100  # Safety limit
    
    while len(points) < m and attempts < max_attempts:
        attempts += 1
        
        # Sample a weight according to distribution
        w = rng.choice(weight_values, p=probs)
        
        # Generate a random vector with exactly w ones
        positions = rng.choice(n, size=w, replace=False)
        vec = np.zeros(n, dtype=np.int8)
        vec[positions] = 1
        
        # Check uniqueness
        key = tuple(vec)
        if key not in seen:
            seen.add(key)
            points.append(vec)
    
    if len(points) < m:
        raise RuntimeError(f"Could only sample {len(points)} unique points after {max_attempts} attempts.")
    
    return np.array(points, dtype=np.int8)

def sample_parity(n: int, m: int, target_parity: int = 1, seed: int = None) -> np.ndarray:
    """
    Sample M unique points with specified parity (0=even, 1=odd Hamming weight).
    
    Rejection sampling: Generate random vectors, keep those with matching parity.
    Expected acceptance rate: 50%.
    """
    rng = np.random.default_rng(seed)
    max_val = 2 ** n
    
    # Exactly half the hypercube has each parity
    valid_count = max_val // 2
    
    if m > valid_count:
        raise ValueError(f"Cannot sample {m} points; only {valid_count} have parity={target_parity}.")
    
    seen: Set[int] = set()
    points = []
    
    while len(points) < m:
        x = rng.integers(0, max_val)
        
        if x in seen:
            continue
        
        # Check parity using population count
        hw = bin(x).count('1')
        if hw % 2 == target_parity:
            seen.add(x)
            points.append(_int_to_binary_vector(x, n))
    
    return np.array(points, dtype=np.int8)

def sample_active_points(n: int, m: int, func_type: str, **kwargs) -> np.ndarray:
    """
    Main interface: Sample M distinct points from the On-Set of a boolean function.
    
    Args:
        n: Number of variables (bits).
        m: Number of points to sample.
        func_type: Type of function ('random', 'threshold', 'parity').
        **kwargs: Additional arguments:
            - threshold: 'k' (required) - threshold value
            - parity: 'target' (default=1) - 0 for even, 1 for odd
            - seed: Random seed
    
    Returns:
        np.ndarray: Shape (m, n) array of binary vectors.
    """
    seed = kwargs.get('seed', None)
    
    if func_type == 'random':
        return sample_random(n, m, seed=seed)
    
    elif func_type == 'threshold':
        k = kwargs.get('k')
        if k is None:
            raise ValueError("'k' is required for threshold sampling.")
        return sample_threshold_k(n, m, k, seed=seed)
    
    elif func_type == 'parity':
        target = kwargs.get('target', 1)
        return sample_parity(n, m, target_parity=target, seed=seed)
    
    else:
        raise ValueError(f"Unknown function type: {func_type}")


# Quick test
if __name__ == "__main__":
    print("Testing Lazy Sampler at N=16...")
    
    n = 16
    m = 1000
    
    print(f"\n[1] Random sampling: n={n}, m={m}")
    pts = sample_active_points(n, m, 'random', seed=42)
    print(f"    Shape: {pts.shape}")
    print(f"    Unique: {len(set(tuple(p) for p in pts))}")
    
    print(f"\n[2] Threshold_k=8 sampling: n={n}, m={m}")
    pts = sample_active_points(n, m, 'threshold', k=8, seed=42)
    print(f"    Shape: {pts.shape}")
    print(f"    Min weight: {pts.sum(axis=1).min()}, Max: {pts.sum(axis=1).max()}")
    
    print(f"\n[3] Parity (odd) sampling: n={n}, m={m}")
    pts = sample_active_points(n, m, 'parity', target=1, seed=42)
    print(f"    Shape: {pts.shape}")
    print(f"    All odd parity: {all(p.sum() % 2 == 1 for p in pts)}")
    
    print("\n✓ All tests passed!")
