"""
Topology Analysis Engine
Performs Persistent Homology analysis on boolean/binary data using Hamming distance.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import ripser
from typing import List, Dict, Tuple, Optional

def get_distance_matrix(points: np.ndarray, metric: str = 'hamming', max_points: int = 2000) -> np.ndarray:
    """
    Computes the pairwise distance matrix for a set of points.
    
    Args:
        points: (N, D) array of data points.
        metric: Distance metric to use (default: 'hamming').
        max_points: Maximum number of points to use. If N > max_points, 
                   random sampling is performed.
    
    Returns:
        (M, M) distance matrix where M = min(N, max_points).
    """
    N = points.shape[0]
    
    if N > max_points:
        # Sampling Mode
        indices = np.random.choice(N, max_points, replace=False)
        sampled_points = points[indices]
    else:
        sampled_points = points
        
    # pdist returns condensed distance matrix, squareform converts to square matrix
    dists = pdist(sampled_points, metric=metric)
    return squareform(dists)

def compute_persistence(points: np.ndarray, max_dim: int = 2) -> List[np.ndarray]:
    """
    Computes persistent homology using the Vietoris-Rips filtration.
    
    Args:
        points: (N, D) array of boolean/binary data.
        max_dim: Maximum homology dimension to compute (default: 2).
        
    Returns:
        List of persistence diagrams [H0, H1, H2, ...]
        Each diagram is an (k, 2) array of (birth, death) pairs.
    """
    # Precompute distance matrix with Hamming distance
    # We must do this because Ripser defaults to Euclidean if passed raw points
    dist_matrix = get_distance_matrix(points, metric='hamming')
    
    # Run Ripser with precomputed distance matrix
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
    
    return results['dgms']

def calculate_pcc(diagrams: List[np.ndarray], skip_h0: bool = True) -> float:
    """
    Computes Total Persistence (PCC - Persistent Cycle Complexity).
    Calculated as sum((death - birth)^2) for features.
    
    Args:
        diagrams: List of persistence diagrams [H0, H1, H2, ...]
        skip_h0: If True, skip H0 (connected components) and focus on H1+ (loops/voids).
                 H0 is dominated by component merging which doesn't reflect circuit complexity.
    
    Returns:
        Total persistence score.
    """
    total_persistence = 0.0
    
    start_dim = 1 if skip_h0 else 0
    
    for dim in range(start_dim, len(diagrams)):
        dgm = diagrams[dim]
        
        # Skip empty diagrams
        if dgm.shape[0] == 0:
            continue
            
        # Filter out features with infinite death
        finite_features = dgm[~np.isinf(dgm[:, 1])]
        
        if len(finite_features) > 0:
            lifetimes = finite_features[:, 1] - finite_features[:, 0]
            total_persistence += np.sum(lifetimes ** 2)
            
    return total_persistence

def calculate_pcc_normalized(diagrams: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes multiple persistence-based metrics for robust comparison.
    Returns a dictionary of metrics that work better at different scales.
    
    Metrics:
        - h0_entropy: Entropy of H0 lifetimes (measures clustering structure)
        - h0_mean_lifetime: Average component lifetime  
        - h0_max_lifetime: Maximum component lifetime
        - h1_total: Standard H1 total persistence
        - total_features: Count of all finite features
    """
    metrics = {
        'h0_entropy': 0.0,
        'h0_mean_lifetime': 0.0,
        'h0_max_lifetime': 0.0,
        'h1_total': 0.0,
        'total_features': 0
    }
    
    for dim, dgm in enumerate(diagrams[:2]):  # H0 and H1 only
        if dgm.shape[0] == 0:
            continue
            
        finite_features = dgm[~np.isinf(dgm[:, 1])]
        
        if len(finite_features) == 0:
            continue
            
        lifetimes = finite_features[:, 1] - finite_features[:, 0]
        lifetimes = lifetimes[lifetimes > 0]  # Filter zero lifetimes
        
        if len(lifetimes) == 0:
            continue
        
        metrics['total_features'] += len(lifetimes)
        
        if dim == 0:
            # H0 metrics
            metrics['h0_mean_lifetime'] = float(np.mean(lifetimes))
            metrics['h0_max_lifetime'] = float(np.max(lifetimes))
            
            # Entropy of normalized lifetimes
            probs = lifetimes / lifetimes.sum()
            probs = probs[probs > 0]
            metrics['h0_entropy'] = float(-np.sum(probs * np.log(probs)))
            
        elif dim == 1:
            # H1 total persistence  
            metrics['h1_total'] = float(np.sum(lifetimes ** 2))
    
    return metrics

def betti_curves(diagrams: List[np.ndarray], n_steps: int = 100) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes Betti numbers (count of active features) over a range of scales.
    
    Args:
        diagrams: List of persistence diagrams.
        n_steps: Number of steps in the filtration range.
        
    Returns:
        Dictionary mapping dimension -> (xs, ys)
        where xs are filtration values and ys are Betti numbers.
    """
    curves = {}
    
    # Find global min/max for filtration range to align curves
    # Filter out infinity for max calculation
    all_births = []
    all_deaths = []
    
    for dgm in diagrams:
        if dgm.shape[0] > 0:
            all_births.append(dgm[:, 0].min())
            finite_deaths = dgm[~np.isinf(dgm[:, 1])][:, 1]
            if len(finite_deaths) > 0:
                all_deaths.append(finite_deaths.max())
    
    if not all_births:
        return {}
        
    min_val = min(all_births)
    # If no finite deaths, pick a reasonable max (e.g. max birth + padding)
    max_val = max(all_deaths) if all_deaths else min_val + 1.0
    
    xs = np.linspace(min_val, max_val, n_steps)
    
    for dim, dgm in enumerate(diagrams):
        ys = np.zeros_like(xs, dtype=int)
        
        for i, x in enumerate(xs):
            # Count features born before x and dead after x
            # alive: birth <= x AND death > x
            # Treat infinite death as always > x
            alive_mask = (dgm[:, 0] <= x) & (dgm[:, 1] > x)
            ys[i] = np.sum(alive_mask)
            
        curves[dim] = (xs, ys)
        
    return curves
