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


def plot_price_history(df, ticker_name):
    """
    Plot stock price history.
    
    Creates a view of price evolution over time.
    """
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
    plt.show()


def plot_returns_distribution(df, ticker_name):
    """
    Visualize the distribution of daily returns.
    
    Shows histogram and box plot to understand return patterns.
    """
    returns = df['Close'].pct_change().dropna() * 100  # Convert to percentage
    
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(returns, bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', 
                   linewidth=2, label='Zero Return')
    axes[0].set_title(f'Returns Distribution - {ticker_name}', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Daily Return (%)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[1].boxplot(returns, vert=True)
    axes[1].set_title(f'Returns Box Plot - {ticker_name}', 
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Daily Return (%)', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'Mean: {returns.mean():.3f}%\n'
    stats_text += f'Std: {returns.std():.3f}%\n'
    stats_text += f'Min: {returns.min():.2f}%\n'
    stats_text += f'Max: {returns.max():.2f}%'
    
    axes[1].text(1.15, returns.median(), stats_text,
                fontsize=10, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def save_results(results_df, filepath="data/processed/results.csv"):
    """
    Save results to CSV file.
    
    Saves model comparison results for later analysis.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")
