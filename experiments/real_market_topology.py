"""
Real-World Financial Topology

Validates topological signals on historical S&P 500 data (2000-2023).
Tracks PCC evolution through Dot-com, 2008 Crisis, and COVID-19.
"""

import os
import csv
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
import ripser

# Configuration
START_DATE = "2000-01-01"
END_DATE = "2023-01-01"
WINDOW_SIZE = 50
STEP_SIZE = 5

# Representative S&P 500 Basket (30 Stocks across sectors)
TICKERS = [
    # Tech
    'AAPL', 'MSFT', 'INTC', 'CSCO', 'ORCL', 'IBM',
    # Finance
    'JPM', 'BAC', 'C', 'WFC', 'AXP', 'GS',
    # Industrial
    'GE', 'BA', 'CAT', 'MMM', 'HON',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB',
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD',
    # Healthcare
    'PFE', 'MRK', 'JNJ', 'ABT', 'LLY'
]
MARKET_INDEX = 'SPY'

def fetch_data():
    print(f"Fetching data for {len(TICKERS)} stocks + {MARKET_INDEX}...")
    print(f"Range: {START_DATE} to {END_DATE}")
    
    # Download returns
    tickers = TICKERS + [MARKET_INDEX]
    # Use auto_adjust=False to ensure we get explicit columns, or check what is returned
    raw_data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)
    
    # Debug: Print columns to understand structure
    print("Columns available:", raw_data.columns)
    
    # Try to get adjusted close, fallback to close
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        print("Warning: 'Adj Close' not found, using 'Close'")
        data = raw_data['Close']
    else:
        # Handle case where columns might be MultiIndex but not in expected format
        # Or checking if it's already just the prices if only one ticker (unlikely here)
        raise KeyError(f"Could not find 'Adj Close' or 'Close' in data. Columns: {raw_data.columns}")

    # Calculate log returns
    returns = np.log(data / data.shift(1)).dropna()
    
    # Separate component returns and market price
    # Ensure they exist in columns
    available_tickers = [t for t in TICKERS if t in data.columns]
    if MARKET_INDEX in data.columns:
        spy_price = data[MARKET_INDEX]
    else:
        # Fallback if SPY failed
        print(f"Warning: {MARKET_INDEX} not in data")
        spy_price = pd.Series(index=data.index, data=0)

    stock_returns = returns[available_tickers]
    
    print(f"Data shape: {stock_returns.shape}")
    return stock_returns, spy_price

def compute_rolling_pcc(returns, window_size, step_size):
    results = []
    n_windows = (len(returns) - window_size) // step_size
    print(f"\nProcessing {n_windows} windows...")
    
    timestamps = returns.index
    
    for i in range(0, len(returns) - window_size, step_size):
        window = returns.iloc[i : i + window_size]
        current_date = timestamps[i + window_size]
        
        # Correlation
        corr = window.corr().values
        
        # Distance: d = sqrt(2(1-r))
        dist = np.sqrt(2 * np.clip(1 - corr, 0, 2))
        dist[np.isnan(dist)] = 0  # Handle potential NaNs
        
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
        
        results.append({
            'date': current_date,
            'pcc': pcc
        })
        
        if i % 100 == 0:
            print(f"  {current_date.date()}: PCC={pcc:.4f}")
            
    return results

def run_analysis():
    print("=" * 60)
    print("REAL-WORLD MARKET TOPOLOGY")
    print("=" * 60)
    
    # 1. Fetch
    returns, prices = fetch_data()
    
    # 2. Analyze
    pcc_data = compute_rolling_pcc(returns, WINDOW_SIZE, STEP_SIZE)
    
    # 3. Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_market_crash.csv')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'pcc', 'spy_price'])
        writer.writeheader()
        
        for res in pcc_data:
            date = res['date']
            # Get closest price
            try:
                price = prices.loc[date]
                if isinstance(price, pd.Series): # Handle duplicate index if any
                    price = price.iloc[0]
            except KeyError:
                price = 0.0 
                
            writer.writerow({
                'date': date.strftime('%Y-%m-%d'),
                'pcc': res['pcc'],
                'spy_price': float(price) # Ensure float
            })
            
    print(f"\n✓ Saved: {output_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_analysis()
