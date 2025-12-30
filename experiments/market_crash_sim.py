"""
Financial Topology: Market Crash Simulation

Simulates a regime change from 'Normal' (Sector-based correlation) 
to 'Crash' (Global Panic correlation) and tracks the Rolling PCC.

Hypothesis: PCC will drop significantly during a crash as the complex
sector structure collapses into a single hypersphere of panic.
"""

import os
import csv
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import ripser

# Configuration
N_STOCKS = 50
N_DAYS = 1000
CRASH_START = 700
SECTORS = 5
STOCKS_PER_SECTOR = 10

# Correlation Params
R_INTRA_SECTOR = 0.6
R_INTER_SECTOR = 0.1
R_PANIC = 0.9

# Volatility Params
VOL_NORMAL = 0.01
VOL_CRASH = 0.02

# TDA Params
WINDOW_SIZE = 50
STEP_SIZE = 10

def generate_covariance_matrix(regime='normal'):
    """Generate covariance matrix based on regime."""
    corr = np.zeros((N_STOCKS, N_STOCKS))
    
    if regime == 'normal':
        # Block diagonal structure
        for i in range(N_STOCKS):
            for j in range(N_STOCKS):
                if i == j:
                    corr[i, j] = 1.0
                elif (i // STOCKS_PER_SECTOR) == (j // STOCKS_PER_SECTOR):
                    corr[i, j] = R_INTRA_SECTOR
                else:
                    corr[i, j] = R_INTER_SECTOR
        vol = VOL_NORMAL
    else:
        # Panic mode: global high correlation
        corr[:] = R_PANIC
        np.fill_diagonal(corr, 1.0)
        vol = VOL_CRASH
        
    # Covariance = Correlation * Vol * Vol
    cov = corr * (vol ** 2)
    return cov

def compute_rolling_pcc(returns, window_size, step_size):
    """Compute PCC on rolling correlation matrices."""
    results = []
    
    n_windows = (len(returns) - window_size) // step_size
    
    print(f"Processing {n_windows} windows...")
    
    for i in range(0, len(returns) - window_size, step_size):
        # Get window
        window = returns.iloc[i : i + window_size]
        current_day = i + window_size
        
        # Compute Correlation Matrix
        corr_matrix = window.corr().values
        
        # Convert to Distance Matrix: d = sqrt(2(1-r))
        # Clip to ensure non-negative before sqrt (numerical noise)
        dist_matrix = np.sqrt(2 * np.clip(1 - corr_matrix, 0, 2))
        
        # PCC Calculation
        # Use distance_matrix=True
        res = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=1)
        dgms = res['dgms']
        
        pcc = 0.0
        # Sum H1 persistence
        if len(dgms) > 1:
            h1 = dgms[1]
            finite = h1[~np.isinf(h1[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                pcc = np.sum(lifetimes ** 2)
        
        results.append({
            'day': current_day,
            'pcc': pcc
        })
        
        if i % 100 == 0:
            print(f"  Day {current_day} PCC: {pcc:.4f}")
            
    return results

def run_simulation():
    print("=" * 60)
    print("MARKET CRASH SIMULATION")
    print("=" * 60)
    
    # 1. Generate Data
    print("\n[1] Generating Market Data...")
    
    # Normal Regime
    cov_normal = generate_covariance_matrix('normal')
    returns_normal = np.random.multivariate_normal(
        np.zeros(N_STOCKS), cov_normal, CRASH_START
    )
    
    # Crash Regime
    cov_crash = generate_covariance_matrix('crash')
    returns_crash = np.random.multivariate_normal(
        np.zeros(N_STOCKS), cov_crash, N_DAYS - CRASH_START
    )
    
    # Combine
    all_returns = np.vstack([returns_normal, returns_crash])
    returns_df = pd.DataFrame(all_returns, columns=[f'Stock_{i}' for i in range(N_STOCKS)])
    
    # Calculate Prices
    prices = 100 * np.cumprod(1 + all_returns, axis=0)
    avg_price = np.mean(prices, axis=1)
    
    # 2. Run TDA
    print("\n[2] Running Topological Analysis (Rolling Window)...")
    pcc_results = compute_rolling_pcc(returns_df, WINDOW_SIZE, STEP_SIZE)
    
    # 3. Save Results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_crash.csv')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['day', 'average_price', 'pcc', 'regime'])
        writer.writeheader()
        
        for res in pcc_results:
            day = res['day']
            regime = 'Normal' if day < CRASH_START else 'Crash'
            # Get avg price at this day
            price = avg_price[day-1]
            
            writer.writerow({
                'day': day,
                'average_price': price,
                'pcc': res['pcc'],
                'regime': regime
            })
            
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    normal_pccs = [r['pcc'] for r in pcc_results if r['day'] < CRASH_START]
    crash_pccs = [r['pcc'] for r in pcc_results if r['day'] >= CRASH_START]
    
    avg_normal = np.mean(normal_pccs)
    avg_crash = np.mean(crash_pccs)
    
    print("\n" + "=" * 60)
    print("FINANCIAL TOPOLOGY SUMMARY")
    print("=" * 60)
    print(f"  Avg PCC (Normal): {avg_normal:.4f}")
    print(f"  Avg PCC (Crash):  {avg_crash:.4f}")
    print(f"  Change:           {avg_crash - avg_normal:.4f}")
    
    if avg_crash < avg_normal:
        print("\n  ✓ HYPOTHESIS CONFIRMED: Market Crash destroys topological complexity!")
        print("    Sector independence collapses into global panic.")
    else:
        print("\n  ✗ HYPOTHESIS REJECTED.")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_simulation()
