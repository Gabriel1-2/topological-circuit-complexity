"""
Cross-Asset Topology Analysis

Analyzes the topological structure of multi-asset correlations.
Hypothesis: Low cross-asset PCC predicts "diversification failure" 
(when stocks, bonds, gold all move together).
"""

import os
import csv
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
import ripser
import matplotlib.pyplot as plt

# Configuration
START_DATE = "2005-01-01"
END_DATE = "2023-12-31"
WINDOW_SIZE = 30
STEP_SIZE = 5

# Cross-Asset Universe
ASSETS = {
    'SPY': 'S&P 500 (Equities)',
    'TLT': '20+ Year Treasury (Bonds)',
    'GLD': 'Gold',
    'UUP': 'US Dollar Index',
    'HYG': 'High Yield Corporate Bonds',
    'EEM': 'Emerging Markets',
    'VNQ': 'Real Estate (REITs)',
    'USO': 'Oil'
}

def fetch_cross_asset_data():
    print(f"Fetching {len(ASSETS)} assets...")
    tickers = list(ASSETS.keys())
    
    raw_data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)
    
    # Get Close prices
    if 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        # Handle MultiIndex
        data = raw_data.xs('Close', axis=1, level=0)
    
    # Calculate returns
    returns = np.log(data / data.shift(1)).dropna()
    
    # Filter to assets that exist
    available = [t for t in tickers if t in returns.columns]
    print(f"Available assets: {available}")
    
    return returns[available], data

def compute_cross_asset_pcc(returns, window_size, step_size):
    results = []
    timestamps = returns.index
    n_windows = (len(returns) - window_size) // step_size
    
    print(f"\nProcessing {n_windows} windows...")
    
    for i in range(0, len(returns) - window_size, step_size):
        window = returns.iloc[i : i + window_size]
        current_date = timestamps[i + window_size]
        
        # Correlation matrix
        corr = window.corr().values
        
        # Distance: d = sqrt(2(1-r))
        dist = np.sqrt(2 * np.clip(1 - corr, 0, 2))
        dist[np.isnan(dist)] = 0
        
        # PCC
        res = ripser.ripser(dist, distance_matrix=True, maxdim=1)
        dgms = res['dgms']
        
        pcc = 0.0
        if len(dgms) > 1:
            h1 = dgms[1]
            finite = h1[~np.isinf(h1[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                pcc = np.sum(lifetimes ** 2)
        
        # Average correlation (baseline comparison)
        upper_tri = corr[np.triu_indices_from(corr, k=1)]
        avg_corr = np.mean(upper_tri)
        
        results.append({
            'date': current_date,
            'pcc': pcc,
            'avg_corr': avg_corr
        })
        
        if i % 200 == 0:
            print(f"  {current_date.date()}: PCC={pcc:.4f}, AvgCorr={avg_corr:.3f}")
            
    return results

def compute_all_asset_drawdown(prices, window=30):
    """Calculate forward max drawdown across ALL assets."""
    drawdowns = []
    
    for i in range(len(prices)):
        future_prices = prices.iloc[i+1 : i+1+window]
        if len(future_prices) < window // 2:
            drawdowns.append(np.nan)
            continue
        
        # Calculate returns for each asset
        asset_returns = (future_prices.iloc[-1] / future_prices.iloc[0] - 1)
        
        # "Diversification failure" = ALL assets down
        all_down = (asset_returns < 0).all()
        min_return = asset_returns.min()
        
        drawdowns.append(min_return if all_down else 0)
    
    return drawdowns

def run_analysis():
    print("=" * 60)
    print("CROSS-ASSET TOPOLOGY ANALYSIS")
    print("=" * 60)
    print("Does topological simplification predict diversification failure?")
    
    # Fetch data
    returns, prices = fetch_cross_asset_data()
    
    # Compute PCC
    pcc_data = compute_cross_asset_pcc(returns, WINDOW_SIZE, STEP_SIZE)
    
    # Convert to DataFrame
    df = pd.DataFrame(pcc_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate forward "all-asset drawdown"
    print("\nCalculating forward all-asset drawdowns...")
    
    # Align prices with PCC dates
    price_df = prices.copy()
    
    forward_dd = []
    for _, row in df.iterrows():
        date = row['date']
        idx = price_df.index.get_indexer([date], method='nearest')[0]
        
        future = price_df.iloc[idx+1 : idx+31]
        if len(future) < 15:
            forward_dd.append(np.nan)
            continue
        
        # Returns for each asset
        start_prices = price_df.iloc[idx]
        end_prices = future.iloc[-1]
        asset_returns = (end_prices / start_prices - 1)
        
        # Count how many assets are down
        n_down = (asset_returns < 0).sum()
        worst_return = asset_returns.min()
        
        forward_dd.append(worst_return if n_down >= len(asset_returns) - 1 else 0)
    
    df['forward_drawdown'] = forward_dd
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cross_asset_topology.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    # Analysis
    print("\n" + "-" * 40)
    print("CORRELATION ANALYSIS")
    print("-" * 40)
    
    valid = df.dropna()
    if len(valid) > 0:
        r, p = pearsonr(valid['pcc'], valid['forward_drawdown'])
        print(f"\n  PCC vs Forward Drawdown: r={r:.4f}, p={p:.4f}")
        
        # Decile analysis
        valid = valid.copy()
        valid['pcc_decile'] = pd.qcut(valid['pcc'], 5, labels=False, duplicates='drop')
        
        print("\n  PCC Quintile | Avg Forward Drawdown")
        print("  " + "-" * 35)
        for q, grp in valid.groupby('pcc_decile'):
            avg_dd = grp['forward_drawdown'].mean()
            print(f"      {q}       |   {avg_dd*100:+.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: PCC over time
    ax = axes[0]
    ax.plot(df['date'], df['pcc'], color='crimson', linewidth=1, label='Cross-Asset PCC')
    ax.axhline(df['pcc'].mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
    
    # Highlight crisis periods
    crisis_dates = [
        ('2008-09-15', '2008 Crisis'),
        ('2020-03-01', 'COVID-19'),
        ('2022-01-01', '2022 Selloff')
    ]
    for date_str, label in crisis_dates:
        try:
            crisis = pd.to_datetime(date_str)
            ax.axvline(x=crisis, color='black', linestyle=':', alpha=0.7)
            ax.text(crisis, ax.get_ylim()[1]*0.9, f' {label}', fontsize=9)
        except:
            pass
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Cross-Asset PCC', fontweight='bold')
    ax.set_title('Cross-Asset Topological Complexity (SPY, TLT, GLD, UUP, HYG, EEM, VNQ, USO)', 
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PCC vs Avg Correlation
    ax = axes[1]
    ax.scatter(df['avg_corr'], df['pcc'], alpha=0.5, s=15, c='steelblue')
    ax.set_xlabel('Average Cross-Asset Correlation', fontweight='bold')
    ax.set_ylabel('Topological Complexity (PCC)', fontweight='bold')
    ax.set_title('Correlation vs Topology: Are They Redundant?', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'cross_asset_crash.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_analysis()
