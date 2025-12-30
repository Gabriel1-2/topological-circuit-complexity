"""
Antifragile Backtester: Long/Short Topological Alpha

Strategy:
- Regime 1 (Safe): PCC High OR Uptrend -> 130% Long SPY (Leveraged Alpha)
- Regime 2 (Crash): PCC Low AND Downtrend -> -30% Short SPY (Profit from Chaos)

Features:
- Margin Interest (Borrow cost for leverage: Risk Free Rate + 1%)
- Short Stock Borrow Cost (1% annual)
- Transaction Costs (0.1%)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE_ANNUAL = 0.03
MARGIN_RATE_ANNUAL = 0.04 # Cost to borrow cash for leverage
SHORT_BORROW_RATE = 0.01  # Cost to borrow stock for shorting

# Signal Params
PCC_THRESHOLD_PERCENTILE = 20
SMA_WINDOW = 50

# Leverage Params
LEVERAGE_LONG = 1.3  # 130% Long
LEVERAGE_SHORT = -0.3 # 30% Short (Net Short)

def calculate_metrics(daily_returns, risk_free_rate=0.0):
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
    sortino = (excess_returns.mean() / excess_returns[excess_returns<0].std()) * np.sqrt(252)
    
    return {
        'CAGR': cagr,
        'Vol': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MaxDD': max_dd
    }

def run_antifragile_backtest():
    print("=" * 60)
    print("ANTIFRAGILE STRATEGY: LONG/SHORT LEVERAGE")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Indicators
    df['pcc_smooth'] = df['pcc'].rolling(window=3).mean()
    df['pcc_quantile'] = df['pcc_smooth'].rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    df['topo_signal'] = (df['pcc_quantile'] < PCC_THRESHOLD_PERCENTILE) # True = Fragile
    
    df['sma'] = df['spy_price'].rolling(window=SMA_WINDOW).mean()
    df['trend_signal'] = (df['spy_price'] < df['sma']) # True = Downtrend
    
    # Combined Signal: CRASH = Fragile AND Downtrend
    # In Crash Regime: Short
    # In Safe Regime: Leveraged Long
    df['crash_mode'] = (df['topo_signal'] & df['trend_signal'])
    
    # Target Position
    df['target_exposure'] = np.where(df['crash_mode'], LEVERAGE_SHORT, LEVERAGE_LONG)
    
    # Shift by 1 day to trade at open next day
    df['position'] = df['target_exposure'].shift(1).fillna(LEVERAGE_LONG)
    
    # 2. Backtest with Margin calculation
    # Daily SPY Return
    df['spy_ret'] = df['spy_price'].pct_change().fillna(0)
    
    # Costs rates (daily)
    daily_margin_cost = (1 + MARGIN_RATE_ANNUAL) ** (1/252) - 1
    daily_short_cost = (1 + SHORT_BORROW_RATE) ** (1/252) - 1
    daily_cash_return = (1 + RISK_FREE_RATE_ANNUAL) ** (1/252) - 1

    # Calculation
    # Equity Change = (Position * SPY_Ret) + (Cash_Balance * Cash_Rate) - Borrow_Costs
    # Cash Balance = 1 - Position
    # If Position > 1 (Leveraged): Cash Balance is Negative (Borrowing), we pay Margin Rate
    # If Position < 0 (Short): Cash Balance is > 1 (Short Proceeds + Capital), we earn Cash Rate, but pay Short Borrow Cost on Position
    
    # This is complex to vectorize perfectly, let's approximate:
    # Gross Return = Position * SPY_Ret
    gross_ret = df['position'] * df['spy_ret']
    
    # Financing Cost
    # If Pos > 1: Pay margin on (Pos - 1)
    # If Pos < 0: Pay short borrow on abs(Pos), Earn interest on (1 + abs(Pos)) cash - Wait, retail usually earns 0 on short proceeds, let's assume institutional earns RFR on 1.0 capital always + interest on short proceeds (minus haircut).
    # Simplified Institutional:
    # You always have 1.0 Equity earning RFR (if unencumbered) or paying cost.
    # Actually simpler: Excess Return = Position * (SPY - Financing).
    # Financing = RFR (or Margin Rate).
    
    # Let's do components:
    financing_cost = pd.Series(0.0, index=df.index)
    
    # Leveraged Longs (>1.0)
    # Borrow (Pos - 1.0) at Margin Rate
    lev_mask = df['position'] > 1.0
    financing_cost[lev_mask] = (df['position'][lev_mask] - 1.0) * daily_margin_cost
    
    # Shorts (<0.0)
    # Pay Borrow Cost on abs(Pos)
    # Earn RFR on Collateral (1.0) - simplified
    # Gross = Pos * SPY_Ret
    # Net = Gross + RFR - Short_Cost
    short_mask = df['position'] < 0.0
    financing_cost[short_mask] = df['position'][short_mask].abs() * daily_short_cost
    
    # Basic return on capital (RFR) - assumes cash is fully collateralizing
    base_yield = daily_cash_return 
    
    # Strategy Daily Return
    # Ret = (Pos * StockRet) + BaseYield - FinancingCost
    raw_ret = gross_ret + base_yield - financing_cost
    
    # Transaction Costs
    df['trade'] = df['position'].diff().abs().fillna(0)
    # Cost applies to notional traded
    trade_cost = df['trade'] * TRANSACTION_COST
    
    net_ret = raw_ret - trade_cost
    
    # Equity Curve
    df['strategy_equity'] = INITIAL_CAPITAL * (1 + net_ret).cumprod()
    benchmark_curve = (INITIAL_CAPITAL / df['spy_price'].iloc[0]) * df['spy_price']
    
    # 3. Metrics
    valid_df = df.dropna(subset=['target_exposure'])
    strat_metrics = calculate_metrics(net_ret.loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    bench_metrics = calculate_metrics(df['spy_ret'].loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    
    print("\nMETRICS COMPARISON")
    print("-" * 65)
    print(f"{'Metric':<15} | {'Benchmark':<15} | {'Antifragile':<18}")
    print("-" * 65)
    print(f"{'CAGR':<15} | {bench_metrics['CAGR']*100:>14.2f}% | {strat_metrics['CAGR']*100:>17.2f}%")
    print(f"{'Volatility':<15} | {bench_metrics['Vol']*100:>14.2f}% | {strat_metrics['Vol']*100:>17.2f}%")
    print(f"{'Max Drawdown':<15} | {bench_metrics['MaxDD']*100:>14.2f}% | {strat_metrics['MaxDD']*100:>17.2f}%")
    print(f"{'Sharpe Ratio':<15} | {bench_metrics['Sharpe']:>15.2f} | {strat_metrics['Sharpe']:>18.2f}")
    print(f"{'Sortino Ratio':<15} | {0.0:>15.2f} | {strat_metrics['Sortino']:>18.2f}")
    print("-" * 65)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Equity
    ax = axes[0]
    ax.plot(valid_df['date'], benchmark_curve.loc[valid_df.index], label='Benchmark', color='gray', alpha=0.5)
    ax.plot(valid_df['date'], df['strategy_equity'].loc[valid_df.index], label=f'Antifragile (Sharpe {strat_metrics["Sharpe"]:.2f})', color='purple', linewidth=2)
    
    # Highlight Short Periods
    y_min, y_max = ax.get_ylim()
    ax.fill_between(valid_df['date'], y_min, y_max, where=(valid_df['position'] < 0), 
                    color='red', alpha=0.15, label='Short Exposure')
    
    ax.set_yscale('log')
    ax.set_title('Antifragile Strategy: Leveraged Long + Crisis Shorting', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Leverage Plot
    ax = axes[1]
    ax.plot(valid_df['date'], valid_df['position'], color='black', linewidth=1, label='Exposure (1.3 to -0.3)')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_ylabel('Net Exposure', fontweight='bold')
    ax.set_title('Position Sizing', fontweight='bold')
    ax.fill_between(valid_df['date'], valid_df['position'], 0, where=(valid_df['position']>0), color='green', alpha=0.3)
    ax.fill_between(valid_df['date'], valid_df['position'], 0, where=(valid_df['position']<0), color='red', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'equity_curve_antifragile.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

if __name__ == "__main__":
    run_antifragile_backtest()
