"""
VIX Backtest: Topological Alpha Strategy

Simple backtest: Go long VIX when PCC is in lowest decile (expect volatility spike).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
PCC_THRESHOLD_PERCENTILE = 20  # Go long when PCC below this percentile
HOLDING_PERIOD = 30  # Days to hold position

def run_backtest():
    print("=" * 60)
    print("VIX BACKTEST: TOPOLOGICAL ALPHA STRATEGY")
    print("=" * 60)
    print(f"\nStrategy: Go LONG volatility when PCC < {PCC_THRESHOLD_PERCENTILE}th percentile")
    print(f"Holding Period: {HOLDING_PERIOD} days")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate forward returns (proxy for VIX gains during high vol)
    # Use absolute SPY returns as volatility proxy (higher abs return = VIX profits)
    df['spy_return'] = np.log(df['spy_price'] / df['spy_price'].shift(1))
    
    # Calculate forward realized volatility as our "VIX return" proxy
    forward_vol = []
    for i in range(len(df)):
        future_returns = df['spy_return'].iloc[i+1 : i+1+HOLDING_PERIOD]
        if len(future_returns) >= HOLDING_PERIOD // 2:
            fwd_vol = future_returns.std() * np.sqrt(252)
        else:
            fwd_vol = np.nan
        forward_vol.append(fwd_vol)
    df['forward_vol'] = forward_vol
    
    # Rolling PCC percentile
    df['pcc_percentile'] = df['pcc'].rolling(252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    
    # Generate signals
    df['signal'] = (df['pcc_percentile'] < PCC_THRESHOLD_PERCENTILE).astype(int)
    
    # Calculate returns
    df['strategy_vol'] = df['signal'] * df['forward_vol']  # Vol captured when signaling
    df['baseline_vol'] = df['forward_vol']  # Always in market
    
    # Drop NaNs
    analysis = df.dropna(subset=['strategy_vol', 'baseline_vol', 'pcc_percentile'])
    
    # Performance metrics
    signal_days = analysis[analysis['signal'] == 1]
    no_signal_days = analysis[analysis['signal'] == 0]
    
    print("\n" + "-" * 40)
    print("PERFORMANCE METRICS")
    print("-" * 40)
    
    avg_vol_when_signal = signal_days['forward_vol'].mean() if len(signal_days) > 0 else 0
    avg_vol_no_signal = no_signal_days['forward_vol'].mean() if len(no_signal_days) > 0 else 0
    
    print(f"\n  Total Trading Days: {len(analysis)}")
    print(f"  Signal Days: {len(signal_days)} ({len(signal_days)/len(analysis)*100:.1f}%)")
    print(f"\n  Avg Forward Vol (Signal ON):  {avg_vol_when_signal*100:.2f}%")
    print(f"  Avg Forward Vol (Signal OFF): {avg_vol_no_signal*100:.2f}%")
    print(f"  Edge: {(avg_vol_when_signal - avg_vol_no_signal)*100:+.2f}%")
    
    # Hit rate: How often does low PCC actually precede high vol?
    median_vol = analysis['forward_vol'].median()
    if len(signal_days) > 0:
        hit_rate = (signal_days['forward_vol'] > median_vol).mean()
    else:
        hit_rate = 0
    print(f"\n  Hit Rate (Vol > Median when Signal): {hit_rate*100:.1f}%")
    
    # Equity curve (cumulative excess volatility captured)
    analysis = analysis.copy()
    analysis['excess_vol'] = (analysis['signal'] * (analysis['forward_vol'] - analysis['forward_vol'].mean()))
    analysis['cumulative_edge'] = analysis['excess_vol'].cumsum()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Signal overlay
    ax = axes[0]
    ax.plot(analysis['date'], analysis['spy_price'], color='blue', linewidth=1, label='S&P 500')
    
    # Highlight signal regions
    signal_dates = analysis[analysis['signal'] == 1]['date']
    for d in signal_dates:
        ax.axvline(x=d, color='red', alpha=0.1, linewidth=2)
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('S&P 500 Price', fontweight='bold')
    ax.set_title('PCC Signal Overlay (Red = Low PCC, Expect High Vol)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative edge
    ax = axes[1]
    ax.plot(analysis['date'], analysis['cumulative_edge'] * 100, color='green', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(analysis['date'], 0, analysis['cumulative_edge'] * 100, 
                    where=analysis['cumulative_edge'] > 0, alpha=0.3, color='green')
    ax.fill_between(analysis['date'], 0, analysis['cumulative_edge'] * 100, 
                    where=analysis['cumulative_edge'] <= 0, alpha=0.3, color='red')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Cumulative Edge (%)', fontweight='bold')
    ax.set_title('Cumulative Volatility Edge from PCC Signal', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'vix_backtest.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    edge = avg_vol_when_signal - avg_vol_no_signal
    if edge > 0 and hit_rate > 0.5:
        print("\n  ✓ STRATEGY HAS EDGE!")
        print(f"  The PCC signal identifies periods of {edge*100:.1f}% higher volatility.")
        print(f"  Hit rate of {hit_rate*100:.0f}% confirms predictive value.")
    else:
        print("\n  ~ Strategy has weak or no edge.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_backtest()
