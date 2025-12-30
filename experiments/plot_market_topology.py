"""
Financial Topology Visualization

Visualizes the relationship between Market Price and Topological Complexity (PCC).
Shows how PCC collapses when the market regime shifts to panic.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
CRASH_START = 700

def plot_market_signal():
    print("=" * 60)
    print("MARKET TOPOLOGY VISUALIZATION")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_crash.csv')
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    
    # Create plot
    print("Creating dual-axis plot...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Price (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Market Index (Price)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['day'], df['average_price'], color=color, linewidth=2, label='Market Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create twin axis
    ax2 = ax1.twinx()
    
    # Plot PCC (Right Axis)
    color = 'tab:red'
    ax2.set_ylabel('Topological Complexity (PCC)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(df['day'], df['pcc'], color=color, linewidth=2, linestyle='-', alpha=0.8, label='Rolling PCC')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight Crash Regime
    plt.axvline(x=CRASH_START, color='black', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(CRASH_START + 10, ax2.get_ylim()[1]*0.95, 'CRASH REJIMES START', 
             color='black', fontweight='bold', rotation=0)
    
    # Add annotations
    # Annotate Normal Regime
    normal_pcc = df[df['day'] < CRASH_START]['pcc'].mean()
    crash_pcc = df[df['day'] >= CRASH_START]['pcc'].mean()
    
    plt.text(200, ax2.get_ylim()[1]*0.8, f"Normal Regime\nHigh Complexity\nAvg PCC: {normal_pcc:.3f}", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='tab:blue'),
             ha='center')
             
    plt.text(850, ax2.get_ylim()[1]*0.2, f"Panic Regime\nLow Complexity\nAvg PCC: {crash_pcc:.3f}", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='tab:red'),
             ha='center')
    
    # Title
    plt.title("Topological Alpha: Market Complexity vs. Price Action\n" +
              "Structural Collapse Warning Signal", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'market_crash_signal.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    plot_market_signal()
