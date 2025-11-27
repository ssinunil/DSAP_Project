"""
Download and load Swiss stock data.
"""

import os
import yfinance as yf
import pandas as pd


def download_swiss_stocks(tickers, start_date="2018-01-01", end_date="2024-12-31"):

    """Download historical data for Swiss stocks."""
    
    print(f"Downloading data for {len(tickers)} stocks...")
    
    all_data = {}
    
    for ticker in tickers:
        print(f"  Downloading {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            all_data[ticker] = data
            print(f"    Success: {len(data)} days")
        else:
            print("    Failed: No data found")
    
    return all_data


def save_raw_data(data_dict, output_dir="data/raw"):
    """Save data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving data to {output_dir}...")
    
    for ticker, data in data_dict.items():
        clean_ticker = ticker.replace('.', '_')
        filepath = os.path.join(output_dir, f"{clean_ticker}.csv")
        data.to_csv(filepath)
        print(f"  Saved {clean_ticker}.csv")
    
    print("All data saved!")


def load_raw_data(ticker, data_dir="data/raw"):
    """Load data from CSV file."""
    clean_ticker = ticker.replace('.', '_')
    filepath = os.path.join(data_dir, f"{clean_ticker}.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    return pd.read_csv(filepath, index_col=0, parse_dates=True)
