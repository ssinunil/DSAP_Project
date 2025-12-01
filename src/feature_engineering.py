"""
Create features from raw stock data.
"""

def create_lag_features(df, lags=None):
    """
    Create lag features (previous day prices).
    
    Example: lag_1 = yesterday's price, lag_5 = price from 5 days ago
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    df = df.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    
    return df


def create_technical_indicators(df):
    """Create technical indicators (moving averages, volatility, etc.)."""
    df = df.copy()
    
    # Daily return (percentage change)
    df['daily_return'] = df['Close'].pct_change()
    
    # Moving averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    
    # Volatility (standard deviation of returns)
    df['volatility_10'] = df['daily_return'].rolling(window=10).std()
    
    return df


def create_target(df):
    """Create target variable (tomorrow's price)."""
    df = df.copy()
    df['target'] = df['Close'].shift(-1)  # Tomorrow's closing price
    return df


def prepare_features(df, lags=None):
    """
    Complete pipeline: create all features and prepare data for ML.
    
    Returns X (features) and y (target).
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    print("Creating features...")
    
    # Create all features
    df = create_lag_features(df, lags=lags)
    df = create_technical_indicators(df)
    df = create_target(df)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Separate features (X) from target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"  Dataset ready: {len(X)} rows, {X.shape[1]} features")
    
    return X, y
