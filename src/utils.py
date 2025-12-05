"""
Utility functions for data exploration and visualization.
"""
import os
import matplotlib.pyplot as plt

def explore_data(df, ticker_name):
    """Display comprehensive statistics about the data."""
    print(f"\n{'='*50}")
    print(f"Data Exploration - {ticker_name}")
    print(f"{'='*50}")
    
    print("\nDataset Dimensions:")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    
    print("\nDate Range:")
    print(f"  From: {df.index.min().date()}")
    print(f"  To: {df.index.max().date()}")
    print(f"  Duration: {(df.index.max() - df.index.min()).days} days")
    
    print("\nClosing Price Statistics:")
    print(f"  Mean: {df['Close'].mean():.2f} CHF")
    print(f"  Min: {df['Close'].min():.2f} CHF")
    print(f"  Max: {df['Close'].max():.2f} CHF")
    print(f"  Std Dev: {df['Close'].std():.2f} CHF")
    
    # Calculate returns
    returns = df['Close'].pct_change().dropna()
    print("\nDaily Returns Statistics:")
    print(f"  Mean: {returns.mean()*100:.3f}%")
    print(f"  Std Dev: {returns.std()*100:.3f}%")
    print(f"  Min: {returns.min()*100:.2f}%")
    print(f"  Max: {returns.max()*100:.2f}%")
    
    print("\nMissing Values:")
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  Total: {missing}")
    else:
        print("  None - Dataset is complete")

def plot_price_history(df, ticker_name, output_dir="data/plots"):
    """
    Plot stock price history.
    
    Creates a view of price evolution over time.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    
    plt.plot(df.index, df['Close'], linewidth=1.5, 
            color='blue', label='Closing Price')
    plt.title(f'Price History - {ticker_name}', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (CHF)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    
    filepath = os.path.join(output_dir, f'06_price_history_{ticker_name.replace(".", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def save_results(results_df, filepath="data/processed/results.csv"):
    """
    Save results to CSV file.
    
    Saves model comparison results for later analysis.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")
