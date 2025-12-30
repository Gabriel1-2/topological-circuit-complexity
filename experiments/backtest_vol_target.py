"""
Volatility-Scaled Antifragile Backtester

Strategy:
- Volatility Targeting: Position Size = Target_Vol / Realized_Vol
- Signal: Topo-Trend (Long in Safe, Short in Crash)
- Goal: Capture crash profits while reducing exposure during high-volatility chaos.

Features:
- Dynamic Leverage (Capped at 2.0x)
- Shorting during crashes
- Transaction Costs & Margin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE_ANNUAL = 0.03
MARGIN_RATE_ANNUAL = 0.04
SHORT_BORROW_RATE = 0.01

# Signal Params
PCC_THRESHOLD_PERCENTILE = 20
SMA_WINDOW = 50

# Vol Control Params
TARGET_VOL = 0.15  # 15% Annualized Volatility
VOL_LOOKBACK = 20  # 20-Day Realized Volatility
MAX_LEVERAGE = 2.0 # Cap leverage at 2x

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

def run_vol_target_backtest():
    print("=" * 60)
    print("VOLATILITY-SCALED ANTIFRAGILE STRATEGY")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Volatility Targeting
    df['spy_ret'] = df['spy_price'].pct_change().fillna(0)
    
    # Realized Vol (Annualized)
    df['realized_vol'] = df['spy_ret'].rolling(window=VOL_LOOKBACK).std() * np.sqrt(252)
    
    # Volatility Scalar
    # Avoid div by zero, clip thresholds
    df['vol_scalar'] = (TARGET_VOL / df['realized_vol']).replace([np.inf, -np.inf], 1.0)
    
    # Cap Leverage
    df['vol_scalar'] = df['vol_scalar'].clip(0, MAX_LEVERAGE)
    
    # 2. Indicators
    df['pcc_smooth'] = df['pcc'].rolling(window=3).mean()
    df['pcc_quantile'] = df['pcc_smooth'].rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    df['topo_signal'] = (df['pcc_quantile'] < PCC_THRESHOLD_PERCENTILE) # True = Fragile
    
    df['sma'] = df['spy_price'].rolling(window=SMA_WINDOW).mean()
    df['trend_signal'] = (df['spy_price'] < df['sma']) # True = Downtrend
    
    # Regime Logic
    # Crash Mode: Fragile AND Downtrend -> Short
    # Safe Mode: Else -> Long
    df['crash_mode'] = (df['topo_signal'] & df['trend_signal'])
    
    # Target Exposure
    # If Crash: Short scaled by Vol Scalar
    # If Safe: Long scaled by Vol Scalar
    df['dir_scalar'] = np.where(df['crash_mode'], -1.0, 1.0)
    
    df['target_exposure'] = df['dir_scalar'] * df['vol_scalar']
    
    # Shift to trade at open next day
    df['position'] = df['target_exposure'].shift(1).fillna(0) # Start flat
    
    # 3. Backtest
    
    # Financing Costs (See backtest_antifragile.py logic)
    daily_margin_cost = (1 + MARGIN_RATE_ANNUAL) ** (1/252) - 1
    daily_short_cost = (1 + SHORT_BORROW_RATE) ** (1/252) - 1
    daily_cash_return = (1 + RISK_FREE_RATE_ANNUAL) ** (1/252) - 1
    
    financing_cost = pd.Series(0.0, index=df.index)
    
    # Long Leverage (>1.0)
    lev_mask = df['position'] > 1.0
    financing_cost[lev_mask] = (df['position'][lev_mask] - 1.0) * daily_margin_cost
    
    # Shorts (<0.0)
    short_mask = df['position'] < 0.0
    financing_cost[short_mask] = df['position'][short_mask].abs() * daily_short_cost
    
    # Returns
    gross_ret = df['position'] * df['spy_ret']
    base_yield = np.where(df['position'] < 1.0, daily_cash_return, 0.0) # Earn yield on unused cash
    # Institutional Simplified: Earn yield on 100% equity always, pay Short cost / Margin cost
    # Let's stick to the previous conservative approx:
    # Ret = Pos*Spy + (1-Abs(Pos))*Yield -- no that's wrong for leverage
    # Standard: Total Return = (Capital + PnL)/Capital - 1
    # PnL = Notional * Return - Costs
    # Notional = Position * Capital
    # PnL_Gross = Position * SpyRet
    # Interest:
    # If Pos=1.5: Borrow 0.5. Pay 0.5*Margin. Earn 0 on 1.0 (Assume invested).
    # If Pos=-0.5: Short 0.5. Pay 0.5*StockBorrow. Earn Yield on 1.0 Capital + 0.5 Proceeds.
    
    # Let's refinace carefully:
    # 1. Asset PnL
    asset_pnl = df['position'] * df['spy_ret']
    
    # 2. Interest PnL
    # Cash Balance
    # Long 1.5 -> Cash = -0.5
    # Short 0.5 -> Cash = 1.0 (Capital) + 0.5 (Proceeds) = 1.5
    cash_bal = 1.0 - df['position']
    # Adjust for shorts: In retail/pro, usually yield on short proceeds is partial.
    # Let's assume we earn RFR on Capital (1.0) minus leverage costs or plus short rebates.
    # Simple Institutional Model:
    # Return = RFR + Exposure * (AssetRet - RFR) [CAPM style]
    # Adjusted for costs:
    # Long: RFR + Pos*(SpyRet - RFR) - max(0, Pos-1)*Spread
    # Short: RFR + Pos*(SpyRet - RFR) - abs(Pos)*StockBorrowSpread
    
    # Using Excess Return formulation
    excess_spy = df['spy_ret'] - daily_cash_return
    
    # Cost Spreads
    margin_spread = MARGIN_RATE_ANNUAL - RISK_FREE_RATE_ANNUAL # Extra cost above RFR
    daily_margin_spread = (1 + margin_spread)**(1/252) - 1
    
    # Strategy Return (Excess over RFR)
    base_excess = df['position'] * excess_spy
    
    # Cost Penalties
    lev_cost = np.maximum(0, df['position'] - 1) * daily_margin_spread
    short_cost = np.maximum(0, -df['position']) * daily_short_cost # Full borrow cost
    
    net_excess = base_excess - lev_cost - short_cost
    
    # Total Return = RFR + Net Excess
    total_ret = daily_cash_return + net_excess
    
    # Transaction Costs
    df['trade'] = df['position'].diff().abs().fillna(0)
    trade_cost = df['trade'] * TRANSACTION_COST
    
    final_ret = total_ret - trade_cost
    
    # Checking for NaN
    final_ret = final_ret.fillna(0)
    
    # Equity
    df['strategy_equity'] = INITIAL_CAPITAL * (1 + final_ret).cumprod()
    benchmark_curve = (INITIAL_CAPITAL / df['spy_price'].iloc[0]) * df['spy_price']
    
    # Metrics
    valid_df = df.dropna(subset=['target_exposure'])
    strat_metrics = calculate_metrics(final_ret.loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    bench_metrics = calculate_metrics(df['spy_ret'].loc[valid_df.index], RISK_FREE_RATE_ANNUAL)
    
    print("\nMETRICS COMPARISON")
    print("-" * 65)
    print(f"{'Metric':<15} | {'Benchmark':<15} | {'Vol-Scaled Antifragile':<24}")
    print("-" * 65)
    print(f"{'CAGR':<15} | {bench_metrics['CAGR']*100:>14.2f}% | {strat_metrics['CAGR']*100:>17.2f}%")
    print(f"{'Volatility':<15} | {bench_metrics['Vol']*100:>14.2f}% | {strat_metrics['Vol']*100:>17.2f}%")
    print(f"{'Max Drawdown':<15} | {bench_metrics['MaxDD']*100:>14.2f}% | {strat_metrics['MaxDD']*100:>17.2f}%")
    print(f"{'Sharpe Ratio':<15} | {bench_metrics['Sharpe']:>15.2f} | {strat_metrics['Sharpe']:>18.2f}")
    print(f"{'Sortino Ratio':<15} | {0.0:>15.2f} | {strat_metrics['Sortino']:>18.2f}")
    print("-" * 65)
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Equity
    ax = axes[0]
    ax.plot(valid_df['date'], benchmark_curve.loc[valid_df.index], label='Benchmark', color='gray', alpha=0.5)
    ax.plot(valid_df['date'], df['strategy_equity'].loc[valid_df.index], label=f'Vol-Scaled (Sharpe {strat_metrics["Sharpe"]:.2f})', color='crimson', linewidth=2)
    ax.set_yscale('log')
    ax.set_title('Volatility-Scaled Antifragile Strategy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Exposure
    ax = axes[1]
    ax.plot(valid_df['date'], valid_df['position'], color='black', linewidth=1, label='Net Exposure')
    ax.fill_between(valid_df['date'], valid_df['position'], 0, where=(valid_df['position']>0), color='green', alpha=0.3)
    ax.fill_between(valid_df['date'], valid_df['position'], 0, where=(valid_df['position']<0), color='red', alpha=0.3)
    ax.set_ylabel('Leverage', fontweight='bold')
    ax.set_title('Dynamic Position Sizing (Vol Target 15%)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Volatility
    ax = axes[2]
    ax.plot(valid_df['date'], valid_df['realized_vol']*100, color='orange', label='Market Volatility (Annotated)')
    ax.axhline(15, color='black', linestyle='--', label='Target Vol 15%')
    ax.set_ylabel('Ann. Volatility %', fontweight='bold')
    ax.set_title('Market Regime', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'equity_curve_vol_target.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

if __name__ == "__main__":
    run_vol_target_backtest()
