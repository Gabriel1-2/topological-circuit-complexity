"""
Real-World Market Topology Visualization

Visualizes the S&P 500 Price vs Topological Complexity (2000-2023).
Does PCC signal the 2008 and 2020 crashes?
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_real_market():
    print("=" * 60)
    print("REAL MARKET TOPOLOGY VISUALIZATION")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create plot
    print("Creating dual-axis plot...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot Price (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('S&P 500 (SPY)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['date'], df['spy_price'], color=color, linewidth=1.5, label='S&P 500')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create twin axis
    ax2 = ax1.twinx()
    
    # Plot PCC (Right Axis)
    color = 'tab:red'
    ax2.set_ylabel('Topological Complexity (PCC)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(df['date'], df['pcc'], color=color, linewidth=1, linestyle='-', alpha=0.7, label='Rolling PCC')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight Major Crashes
    # 2008 Financial Crisis (Lehman Brothers Sept 2008)
    crisis_2008 = pd.to_datetime('2008-09-15')
    plt.axvline(x=crisis_2008, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.text(crisis_2008, ax2.get_ylim()[1]*0.9, ' 2008 CRISIS', 
             color='black', fontweight='bold', fontsize=10, ha='left')
             
    # 2020 COVID Crash (March 2020)
    covid_2020 = pd.to_datetime('2020-03-01')
    plt.axvline(x=covid_2020, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.text(covid_2020, ax2.get_ylim()[1]*0.9, ' COVID-19', 
             color='black', fontweight='bold', fontsize=10, ha='left')
    
    # Title
    plt.title("The Topological Alpha: S&P 500 (2000-2023)\n" +
              "Does Topological Collapse Predict Market Crashes?", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'real_market_crash.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    plot_real_market()
