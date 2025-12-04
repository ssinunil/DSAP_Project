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
    
    # Load CSV with proper date parsing, skip bad rows
    df = pd.read_csv(filepath, index_col=0, parse_dates=True, 
                     on_bad_lines='skip')
    
    # Remove any rows where index is not a valid date
    df = df[df.index.notna()]
    
    # Remove rows with 'Ticker' or other non-date values in index
    if df.index.dtype == 'object':
        # If index is still object type, filter out non-numeric data
        df = df[~df.index.astype(str).str.contains('Ticker|Symbol|Date', case=False, na=False)]
        # Try to convert to datetime again
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]
    
    # Convert price columns to numeric (in case they're strings)
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    return df
