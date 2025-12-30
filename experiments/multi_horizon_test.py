"""
Multi-Horizon Leading Indicator Test

Tests PCC predictive power across multiple forward windows (7, 14, 30, 60 days).
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

FORWARD_WINDOWS = [7, 14, 30, 60]

def run_multi_horizon_test():
    print("=" * 60)
    print("MULTI-HORIZON LEADING INDICATOR TEST")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['spy_return'] = np.log(df['spy_price'] / df['spy_price'].shift(1))
    
    print(f"\nLoaded {len(df)} data points")
    
    results = []
    
    for window in FORWARD_WINDOWS:
        print(f"\n--- Forward Window: {window} Days ---")
        
        # Calculate forward volatility
        forward_vol = []
        for i in range(len(df)):
            future_returns = df['spy_return'].iloc[i+1 : i+1+window]
            if len(future_returns) >= window // 2:
                fwd_vol = future_returns.std() * np.sqrt(252)
            else:
                fwd_vol = np.nan
            forward_vol.append(fwd_vol)
        
        df[f'forward_vol_{window}'] = forward_vol
        
        # Correlation
        valid = df.dropna(subset=['pcc', f'forward_vol_{window}'])
        pcc = valid['pcc'].values
        fwd_vol = valid[f'forward_vol_{window}'].values
        
        pearson_r, pearson_p = pearsonr(pcc, fwd_vol)
        spearman_r, spearman_p = spearmanr(pcc, fwd_vol)
        
        # Decile spread
        valid = valid.copy()
        valid['decile'] = pd.qcut(valid['pcc'], 10, labels=False, duplicates='drop')
        low_vol = valid[valid['decile'] <= 2][f'forward_vol_{window}'].mean()
        high_vol = valid[valid['decile'] >= 7][f'forward_vol_{window}'].mean()
        spread = low_vol - high_vol
        
        print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"  Decile Spread: {spread*100:+.2f}%")
        
        results.append({
            'window': window,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'low_pcc_vol': low_vol,
            'high_pcc_vol': high_vol,
            'spread': spread
        })
    
    # Summary Table
    print("\n" + "=" * 60)
    print("SUMMARY: PCC Predictive Power Across Horizons")
    print("=" * 60)
    print(f"\n{'Window':>8} | {'Pearson r':>10} | {'p-value':>10} | {'Spread':>10}")
    print("-" * 50)
    for r in results:
        sig = "*" if r['pearson_p'] < 0.05 else ""
        print(f"  {r['window']:>4} d  |   {r['pearson_r']:>7.4f}  |  {r['pearson_p']:>8.4f}  |  {r['spread']*100:>+6.2f}%{sig}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    windows = [r['window'] for r in results]
    correlations = [r['pearson_r'] for r in results]
    spreads = [r['spread'] * 100 for r in results]
    
    ax = axes[0]
    ax.bar(windows, correlations, color='steelblue', edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Forward Window (Days)', fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontweight='bold')
    ax.set_title('PCC-Volatility Correlation by Horizon', fontweight='bold')
    ax.set_xticks(windows)
    
    ax = axes[1]
    ax.bar(windows, spreads, color='crimson', edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Forward Window (Days)', fontweight='bold')
    ax.set_ylabel('Decile Spread (%)', fontweight='bold')
    ax.set_title('Low PCC - High PCC Volatility Spread', fontweight='bold')
    ax.set_xticks(windows)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'multi_horizon_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_multi_horizon_test()
