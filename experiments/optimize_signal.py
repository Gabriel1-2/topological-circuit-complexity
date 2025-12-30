"""
Signal Optimizer

Grid Search for Topological Alpha Strategy.
Optimizes: PCC Percentile Threshold & Smoothing Window.
Metric: Sharpe Ratio.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
THRESHOLDS = range(5, 55, 5) # 5, 10, ..., 50
WINDOWS = range(1, 15, 2)     # 1, 3, 5, ..., 13
TC = 0.001
RFR = 0.03

def backtest_params(df, threshold, smooth_window):
    """Fast vectorized backtest for a single parameter set."""
    # Smooth PCC
    if smooth_window > 1:
        pcc_smooth = df['pcc'].rolling(window=smooth_window).mean()
    else:
        pcc_smooth = df['pcc']
        
    # Percentile (Rolling)
    # Use smaller window for speed in loop if needed, but keeping 252 for consistency
    pcc_pct = pcc_smooth.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    
    # Signal
    signal = (pcc_pct >= threshold).astype(int)
    position = signal.shift(1).fillna(1)
    
    # Returns
    trade = position.diff().fillna(0).abs()
    daily_cash_ret = (1 + RFR) ** (1/252) - 1
    
    spy_ret = df['spy_price'].pct_change().fillna(0)
    
    raw_ret = position * spy_ret + (1 - position) * daily_cash_ret
    net_ret = raw_ret - (trade * TC)
    
    # Sharpe
    valid_ret = net_ret.iloc[252:] # Skip warmup
    if len(valid_ret) == 0: return 0
    
    excess = valid_ret - daily_cash_ret
    if valid_ret.std() == 0: return 0
    sharpe = (excess.mean() / valid_ret.std()) * np.sqrt(252)
    
    return sharpe

def run_optimization():
    print("=" * 60)
    print("SIGNAL OPTIMIZATION: GRID SEARCH")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Testing {len(THRESHOLDS) * len(WINDOWS)} combinations...")
    
    results = np.zeros((len(WINDOWS), len(THRESHOLDS)))
    
    # Iterate
    for i, w in enumerate(WINDOWS):
        for j, t in enumerate(THRESHOLDS):
            sharpe = backtest_params(df, t, w)
            results[i, j] = sharpe
            
    # Find best
    best_idx = np.unravel_index(results.argmax(), results.shape)
    best_w = WINDOWS[best_idx[0]]
    best_t = THRESHOLDS[best_idx[1]]
    best_sharpe = results.max()
    
    print("\nOPTIMIZATION RESULTS")
    print("-" * 40)
    print(f"Best Sharpe Ratio: {best_sharpe:.3f}")
    print(f"Optimal Window:    {best_w}")
    print(f"Optimal Threshold: {best_t}th percentile")
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".2f", cmap='viridis',
                xticklabels=THRESHOLDS, yticklabels=WINDOWS)
    plt.xlabel('PCC Threshold Percentile', fontweight='bold')
    plt.ylabel('Smoothing Window (Days)', fontweight='bold')
    plt.title(f'Sharpe Ratio Heatmap (Best: {best_sharpe:.2f})', fontweight='bold')
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'parameter_heatmap.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_optimization()
