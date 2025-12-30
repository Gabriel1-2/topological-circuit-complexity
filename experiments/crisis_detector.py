"""
Real-Time Crisis Detector

Monitors cross-asset topology and triggers alerts when PCC spikes,
indicating diversification failure is occurring NOW.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import squareform
import ripser
import matplotlib.pyplot as plt

# Configuration
LOOKBACK_DAYS = 60
WINDOW_SIZE = 20
ALERT_THRESHOLD_PERCENTILE = 90  # Alert when PCC > 90th percentile

ASSETS = ['SPY', 'TLT', 'GLD', 'UUP', 'HYG', 'EEM', 'VNQ']

def fetch_recent_data(days=LOOKBACK_DAYS):
    """Fetch recent data for crisis detection."""
    print(f"Fetching last {days} days of data...")
    
    raw = yf.download(ASSETS, period=f'{days}d', progress=False)
    
    if 'Close' in raw.columns:
        prices = raw['Close']
    else:
        prices = raw.xs('Close', axis=1, level=0)
    
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns, prices

def compute_pcc(returns_window):
    """Compute PCC for a single window."""
    corr = returns_window.corr().values
    dist = np.sqrt(2 * np.clip(1 - corr, 0, 2))
    dist[np.isnan(dist)] = 0
    
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

def run_crisis_detector():
    print("=" * 60)
    print("REAL-TIME CRISIS DETECTOR")
    print("=" * 60)
    print(f"Monitoring: {', '.join(ASSETS)}")
    print(f"Alert Threshold: PCC > {ALERT_THRESHOLD_PERCENTILE}th percentile")
    
    # Fetch data
    returns, prices = fetch_recent_data()
    
    # Compute rolling PCC
    pcc_values = []
    dates = []
    
    for i in range(WINDOW_SIZE, len(returns)):
        window = returns.iloc[i-WINDOW_SIZE:i]
        pcc = compute_pcc(window)
        pcc_values.append(pcc)
        dates.append(returns.index[i])
    
    df = pd.DataFrame({'date': dates, 'pcc': pcc_values})
    
    # Calculate threshold from historical data
    # Load historical baseline
    hist_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cross_asset_topology.csv')
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        threshold = np.percentile(hist['pcc'].dropna(), ALERT_THRESHOLD_PERCENTILE)
        print(f"\nHistorical {ALERT_THRESHOLD_PERCENTILE}th percentile PCC: {threshold:.4f}")
    else:
        threshold = np.percentile(pcc_values, ALERT_THRESHOLD_PERCENTILE)
        print(f"\nUsing recent {ALERT_THRESHOLD_PERCENTILE}th percentile: {threshold:.4f}")
    
    # Current status
    current_pcc = pcc_values[-1]
    current_date = dates[-1]
    
    print("\n" + "-" * 40)
    print("CURRENT STATUS")
    print("-" * 40)
    print(f"\n  Date: {current_date.date()}")
    print(f"  Current PCC: {current_pcc:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    # Alert logic
    if current_pcc > threshold:
        alert_level = "🚨 CRISIS ALERT"
        status_color = 'red'
        message = "Cross-asset correlations are spiking. Diversification may be failing."
    elif current_pcc > threshold * 0.7:
        alert_level = "⚠️ ELEVATED"
        status_color = 'orange'
        message = "Topology is elevated. Monitor closely."
    else:
        alert_level = "✅ NORMAL"
        status_color = 'green'
        message = "Cross-asset topology is healthy. Diversification intact."
    
    print(f"\n  Status: {alert_level}")
    print(f"  {message}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['date'], df['pcc'], color='crimson', linewidth=2, label='Cross-Asset PCC')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Alert Threshold ({ALERT_THRESHOLD_PERCENTILE}th pct)')
    ax.axhline(threshold * 0.7, color='orange', linestyle='--', linewidth=1, label='Warning Level')
    
    # Highlight current
    ax.scatter([current_date], [current_pcc], s=200, c=status_color, zorder=5, edgecolors='black')
    ax.annotate(f'NOW: {current_pcc:.4f}', (current_date, current_pcc), 
                textcoords='offset points', xytext=(10, 10), fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Cross-Asset PCC', fontweight='bold')
    ax.set_title(f'Real-Time Crisis Detector | Status: {alert_level}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'crisis_detector.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)
    
    return current_pcc, threshold, alert_level

if __name__ == "__main__":
    run_crisis_detector()
