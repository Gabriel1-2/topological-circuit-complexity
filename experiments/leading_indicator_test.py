"""
Leading Indicator Test

Tests if current Topological Complexity (PCC) predicts future market volatility.
Hypothesis: Low PCC today → High volatility in the next 30 days.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Configuration
FORWARD_WINDOW = 30  # Days to look ahead for volatility

def run_leading_indicator_test():
    print("=" * 60)
    print("LEADING INDICATOR TEST")
    print("=" * 60)
    print(f"Does Low PCC Today → High Volatility Tomorrow?")
    print(f"Forward Window: {FORWARD_WINDOW} days")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\nLoaded {len(df)} data points")
    
    # Calculate forward volatility (realized vol of next N days)
    # We need SPY returns, so compute from prices
    df['spy_return'] = np.log(df['spy_price'] / df['spy_price'].shift(1))
    
    # Rolling forward volatility (std of next N returns)
    forward_vol = []
    for i in range(len(df)):
        future_returns = df['spy_return'].iloc[i+1 : i+1+FORWARD_WINDOW]
        if len(future_returns) >= FORWARD_WINDOW // 2:  # Need at least half window
            fwd_vol = future_returns.std() * np.sqrt(252)  # Annualized
        else:
            fwd_vol = np.nan
        forward_vol.append(fwd_vol)
    
    df['forward_vol'] = forward_vol
    
    # Drop NaNs
    analysis_df = df.dropna(subset=['pcc', 'forward_vol'])
    print(f"Valid samples for analysis: {len(analysis_df)}")
    
    # Correlation Analysis
    print("\n" + "-" * 40)
    print("CORRELATION ANALYSIS")
    print("-" * 40)
    
    pcc = analysis_df['pcc'].values
    fwd_vol = analysis_df['forward_vol'].values
    
    # Pearson
    pearson_r, pearson_p = pearsonr(pcc, fwd_vol)
    print(f"\n  Pearson Correlation:  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    
    # Spearman (rank-based, more robust)
    spearman_r, spearman_p = spearmanr(pcc, fwd_vol)
    print(f"  Spearman Correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
    
    # Decile Analysis
    print("\n" + "-" * 40)
    print("DECILE ANALYSIS")
    print("-" * 40)
    print("(Comparing future volatility when PCC is in different quantiles)")
    
    analysis_df = analysis_df.copy()
    analysis_df['pcc_decile'] = pd.qcut(analysis_df['pcc'], 10, labels=False, duplicates='drop')
    
    decile_stats = analysis_df.groupby('pcc_decile')['forward_vol'].agg(['mean', 'std', 'count'])
    print("\n  Decile | Avg Forward Vol | Samples")
    print("  " + "-" * 40)
    for decile, row in decile_stats.iterrows():
        print(f"     {decile}   |   {row['mean']*100:.2f}%       |   {int(row['count'])}")
    
    # Low vs High PCC comparison
    low_pcc_vol = analysis_df[analysis_df['pcc_decile'] <= 2]['forward_vol'].mean()
    high_pcc_vol = analysis_df[analysis_df['pcc_decile'] >= 7]['forward_vol'].mean()
    
    print(f"\n  Low PCC (Deciles 0-2) Avg Forward Vol:  {low_pcc_vol*100:.2f}%")
    print(f"  High PCC (Deciles 7-9) Avg Forward Vol: {high_pcc_vol*100:.2f}%")
    print(f"  Difference: {(low_pcc_vol - high_pcc_vol)*100:+.2f}%")
    
    # Visualization
    print("\n  Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(pcc, fwd_vol * 100, alpha=0.5, s=15, c='steelblue')
    
    # Trend line
    z = np.polyfit(pcc, fwd_vol * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pcc.min(), pcc.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r={pearson_r:.3f})')
    
    ax.set_xlabel('Current PCC (Topological Complexity)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Forward {FORWARD_WINDOW}-Day Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title(f"Leading Indicator Test: PCC vs Future Volatility\n" +
                 f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})",
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'leading_indicator.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if pearson_r < -0.1 and pearson_p < 0.05:
        print("\n  ✓ HYPOTHESIS CONFIRMED!")
        print("  Low PCC (simplified market) PREDICTS higher future volatility.")
        print("  PCC is a LEADING INDICATOR for market stress.")
    elif pearson_r > 0.1 and pearson_p < 0.05:
        print("\n  ✗ HYPOTHESIS REJECTED (Inverted)")
        print("  High PCC predicts higher future volatility.")
    else:
        print("\n  ~ INCONCLUSIVE")
        print("  No statistically significant predictive relationship found.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_leading_indicator_test()
