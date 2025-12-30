"""
Professional Backtester: Topological Alpha Strategy

Simulates a "Regime Switching" portfolio:
- High PCC (Safe): Long SPY (100% Equity)
- Low PCC (Fragile): Cash (0% Equity)

Features:
- Transaction Costs (0.1% per trade)
- Risk-Free Rate (3% annualized on cash)
- Metric Calculation: Sharpe, Sortino, Max Drawdown
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # 0.1% slippage + comms
RISK_FREE_RATE_ANNUAL = 0.03
PCC_THRESHOLD_PERCENTILE = 20  # Go to cash if PCC < 20th percentile
SIGNAL_SMOOTHING = 3  # Smooth PCC over 3 windows to reduce churn

def calculate_metrics(daily_returns, risk_free_rate=0.0):
    """Calculate professional performance metrics."""
    # Annualized Return
    total_return = (1 + daily_returns).prod() - 1
    n_days = len(daily_returns)
    cagr = (1 + total_return) ** (252 / n_days) - 1
    
    # Volatility
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    # Sharpe (Excess return / Vol)
    # Adjust RFR for daily
    daily_rfr = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = daily_returns - daily_rfr
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # Sortino (Excess return / Downside Vol)
    downside_returns = excess_returns.copy()
    downside_returns[downside_returns > 0] = 0
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    # Max Drawdown
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    
    # Calmar (CAGR / MaxDD)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'CAGR': cagr,
        'Vol': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MaxDD': max_dd,
        'Calmar': calmar
    }

def run_pro_backtest():
    print("=" * 60)
    print("INSTITUTIONAL VALIDATION: TOPOLOGICAL SWITCH")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run real_market_topology.py first.")
        return

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Signal Generation
    # Smooth PCC to prevent fake-outs
    df['pcc_smooth'] = df['pcc'].rolling(window=SIGNAL_SMOOTHING).mean()
    
    # Determine Threshold (Expanding window to avoid look-ahead bias would be best,
    # but for this validation we use rolling historical window)
    # Using 1-year rolling percentile to adapt to regime changes
    df['pcc_quantile'] = df['pcc_smooth'].rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    
    # Logic: If PCC < Threshold, Signal = CASH (0). Else EQUITY (1).
    # We lag signal by 1 day (Signal today executes tomorrow) to be realistic
    df['signal'] = (df['pcc_quantile'] >= PCC_THRESHOLD_PERCENTILE).astype(int)
    df['position'] = df['signal'].shift(1).fillna(1) # Start invested
    
    # 2. Portfolio Simulation
    cash = INITIAL_CAPITAL
    shares = 0
    equity_curve = []
    
    # Benchmark
    benchmark_curve = (INITIAL_CAPITAL / df['spy_price'].iloc[0]) * df['spy_price']
    
    # Iterate (Vectorized simulation is hard with transaction costs on switch)
    current_pos = 0 # 0=Cash, 1=Equity
    
    # Initial allocation (Assume we start with Signal)
    if df['position'].iloc[0] == 1:
        shares = cash / df['spy_price'].iloc[0]
        cash = 0
        current_pos = 1
    
    portfolio_values = []
    
    # Detect switches
    # diff = 1 (0->1 Buy), diff = -1 (1->0 Sell)
    df['trade'] = df['position'].diff().fillna(0)
    
    # We need to simulate day by day to handle cash interest and costs accurately
    # But for speed, we can calculate costs on trade days
    
    # Simplified Vectorized Approach with Cost Penalty
    # Daily Returns of SPY
    df['spy_ret'] = df['spy_price'].pct_change().fillna(0)
    
    # Strategy Returns
    # Return = (Position_yesterday * SPY_Return) + (Cash_yesterday * Cash_Return) - Costs
    # Cash Return daily
    daily_cash_ret = (1 + RISK_FREE_RATE_ANNUAL) ** (1/252) - 1
    
    strategy_returns = []
    
    # Since position is already shifted (signal yesterday determines position today),
    # Strategy ReturnToday = PositionToday * SPY_ReturnToday + (1-PositionToday)*CashReturn
    
    # Switch Costs: If trade != 0, we pay cost.
    # Cost applies to the *entire capital reallocated*.
    # Approximation: Cost = 0.1% * 1 (full rotation) on switch days
    cost_penalty = df['trade'].abs() * TRANSACTION_COST
    
    # Raw Strategy Return (before cost)
    raw_ret = df['position'] * df['spy_ret'] + (1 - df['position']) * daily_cash_ret
    
    # Net Return
    net_ret = raw_ret - cost_penalty
    
    # Equity Curve
    df['strategy_equity'] = INITIAL_CAPITAL * (1 + net_ret).cumprod()
    df['benchmark_equity'] = benchmark_curve
    
    # 3. Validation Metrics
    # Filter to period with valid signals
    valid_df = df.dropna(subset=['pcc_quantile'])
    
    strat_metrics = calculate_metrics(net_ret.loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    bench_metrics = calculate_metrics(df['spy_ret'].loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    
    print("\nMETRICS COMPARISON")
    print("-" * 65)
    print(f"{'Metric':<15} | {'Benchmark (SPY)':<18} | {'Topology Strategy':<18}")
    print("-" * 65)
    print(f"{'CAGR':<15} | {bench_metrics['CAGR']*100:>17.2f}% | {strat_metrics['CAGR']*100:>17.2f}%")
    print(f"{'Volatility':<15} | {bench_metrics['Vol']*100:>17.2f}% | {strat_metrics['Vol']*100:>17.2f}%")
    print(f"{'Max Drawdown':<15} | {bench_metrics['MaxDD']*100:>17.2f}% | {strat_metrics['MaxDD']*100:>17.2f}%")
    print(f"{'Sharpe Ratio':<15} | {bench_metrics['Sharpe']:>18.2f} | {strat_metrics['Sharpe']:>18.2f}")
    print(f"{'Calmar Ratio':<15} | {bench_metrics['Calmar']:>18.2f} | {strat_metrics['Calmar']:>18.2f}")
    print("-" * 65)
    
    # 4. Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Equity Curve
    ax = axes[0]
    ax.plot(valid_df['date'], valid_df['benchmark_equity'], label=f'Benchmark (SPY) - Sharpe {bench_metrics["Sharpe"]:.2f}', color='gray', alpha=0.7)
    ax.plot(valid_df['date'], valid_df['strategy_equity'], label=f'Topological Switch - Sharpe {strat_metrics["Sharpe"]:.2f}', color='blue', linewidth=2)
    
    # Highlight Defensive Periods (Cash)
    # Find segments where position == 0
    # We fill intervals
    y_min, y_max = ax.get_ylim()
    ax.fill_between(valid_df['date'], y_min, y_max, where=(valid_df['position'] == 0), 
                    color='red', alpha=0.1, label='Defensive (Cash) Mode')
    
    ax.set_ylabel('Portfolio Value ($)', fontweight='bold')
    ax.set_title('Institutional Validation: Equity Curve with Transaction Costs', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log') # Log scale is standard for long backtests
    
    # Drawdown Curve
    ax = axes[1]
    
    def get_dd(equity):
        peak = equity.cummax()
        return (equity - peak) / peak
        
    bench_dd = get_dd(valid_df['benchmark_equity'])
    strat_dd = get_dd(valid_df['strategy_equity'])
    
    ax.fill_between(valid_df['date'], bench_dd * 100, 0, color='gray', alpha=0.3, label='Benchmark DD')
    ax.plot(valid_df['date'], strat_dd * 100, color='blue', linewidth=1, label='Strategy DD')
    
    ax.set_ylabel('Drawdown (%)', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'equity_curve_pro.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("INSTITUTIONAL VERDICT")
    print("=" * 60)
    
    if strat_metrics['Sharpe'] > 1.0 and strat_metrics['MaxDD'] > bench_metrics['MaxDD']: 
        # Note: MaxDD is negative, so Strat > Bench means Strat is closer to 0 (smaller loss)
        print("\n🏆 PASSED: HEDGE FUND QUALITY")
        print("Strategy has Sharpe > 1.0 and reduced Max Drawdown.")
    elif strat_metrics['Sharpe'] > bench_metrics['Sharpe']:
        print("\n✓ PASSED: ALPHA GENERATED")
        print("Strategy improves risk-adjusted returns over Benchmark.")
    else:
        print("\nFAILED")
        print("Strategy fails to beat Buy & Hold after costs.")

if __name__ == "__main__":
    run_pro_backtest()
