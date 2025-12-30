"""
Enhanced Backtester: Topo-Trend Strategy

Strategy:
- Signal CASH only if:
    1. PCC < Threshold (Fragile Structure)
    2. AND Price < SMA_50 (Trend is Weak)
- Else: Long SPY

Rationale: Avoid exits during "melt-ups" where topology is simple but price is strong.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE_ANNUAL = 0.03
PCC_THRESHOLD_PERCENTILE = 20  # Optimized from previous step (or use 5 if safer)
SMA_WINDOW = 50

def calculate_metrics(daily_returns, risk_free_rate=0.0):
    """Calculate professional performance metrics."""
    total_return = (1 + daily_returns).prod() - 1
    n_days = len(daily_returns)
    cagr = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    daily_rfr = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = daily_returns - daily_rfr
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'CAGR': cagr,
        'Vol': ann_vol,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'Calmar': calmar
    }

def run_enhanced_backtest():
    print("=" * 60)
    print("ENHANCED STRATEGY: TOPO-TREND (PCC + SMA)")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Indicators
    # PCC Signal
    df['pcc_smooth'] = df['pcc'].rolling(window=3).mean()
    df['pcc_quantile'] = df['pcc_smooth'].rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    df['topo_signal'] = (df['pcc_quantile'] < PCC_THRESHOLD_PERCENTILE) # True = Fragile
    
    # Trend Signal
    df['sma'] = df['spy_price'].rolling(window=SMA_WINDOW).mean()
    df['trend_signal'] = (df['spy_price'] < df['sma']) # True = Downtrend
    
    # Combined Signal: Cash if Fragile AND Downtrend
    # Invert logic: Equity if NOT (Fragile AND Downtrend)
    # i.e., Stay long if Fragile but Uptrending (Bubble surfing)
    df['signal'] = (~(df['topo_signal'] & df['trend_signal'])).astype(int)
    
    df['position'] = df['signal'].shift(1).fillna(1)
    
    # 2. Backtest
    # Costs
    df['trade'] = df['position'].diff().abs().fillna(0)
    cost_penalty = df['trade'] * TRANSACTION_COST
    
    # Returns
    daily_cash_ret = (1 + RISK_FREE_RATE_ANNUAL) ** (1/252) - 1
    df['spy_ret'] = df['spy_price'].pct_change().fillna(0)
    
    raw_ret = df['position'] * df['spy_ret'] + (1 - df['position']) * daily_cash_ret
    net_ret = raw_ret - cost_penalty
    
    df['strategy_equity'] = INITIAL_CAPITAL * (1 + net_ret).cumprod()
    
    # Benchmark
    benchmark_curve = (INITIAL_CAPITAL / df['spy_price'].iloc[0]) * df['spy_price']
    
    # Metrics
    valid_df = df.dropna(subset=['pcc_quantile', 'sma'])
    strat_metrics = calculate_metrics(net_ret.loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    bench_metrics = calculate_metrics(df['spy_ret'].loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    
    print("\nMETRICS COMPARISON")
    print("-" * 65)
    print(f"{'Metric':<15} | {'Benchmark':<15} | {'Topo-Trend (+SMA)':<18}")
    print("-" * 65)
    print(f"{'CAGR':<15} | {bench_metrics['CAGR']*100:>14.2f}% | {strat_metrics['CAGR']*100:>17.2f}%")
    print(f"{'Volatility':<15} | {bench_metrics['Vol']*100:>14.2f}% | {strat_metrics['Vol']*100:>17.2f}%")
    print(f"{'Max Drawdown':<15} | {bench_metrics['MaxDD']*100:>14.2f}% | {strat_metrics['MaxDD']*100:>17.2f}%")
    print(f"{'Sharpe Ratio':<15} | {bench_metrics['Sharpe']:>15.2f} | {strat_metrics['Sharpe']:>18.2f}")
    print("-" * 65)
    
    # Count trades
    n_trades = df['trade'].sum()
    print(f"\n  Total Trades: {int(n_trades)}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valid_df['date'], benchmark_curve.loc[valid_df.index], label='Benchmark (SPY)', color='gray', alpha=0.6)
    ax.plot(valid_df['date'], df['strategy_equity'].loc[valid_df.index], label=f'Topo-Trend Strategy (Sharpe {strat_metrics["Sharpe"]:.2f})', color='green', linewidth=2)
    
    ax.set_ylabel('Portfolio Value ($)', fontweight='bold')
    ax.set_title('Topo-Trend Strategy: Combining Topology with Trend Following', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'equity_curve_enhanced.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_enhanced_backtest()
