"""
Feature engineering for stock prediction.
"""

def create_lag_features(df, n_lags=5):
    """
    Create percentage change lag features.
    """
    for i in range(1, n_lags + 1):
        df[f'return_lag_{i}'] = df['Close'].pct_change(i)
    
    return df

def create_technical_indicators(df):
    """
    Create technical indicators based on percentage changes.
    """

    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    
    df['distance_ma5_pct'] = (df['Close'] - df['ma_5']) / df['ma_5']
    df['distance_ma20_pct'] = (df['Close'] - df['ma_20']) / df['ma_20']
    
    df['volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
    df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    if 'Volume' in df.columns:
        df['volume_change'] = df['Volume'].pct_change()
    
    df = df.drop(['ma_5', 'ma_20'], axis=1)
    
    return df

def create_target(df):
    """
    Create target: PERCENTAGE CHANGE for next day.
    
    CHANGEMENT CLÉ: On prédit le % de changement, pas le prix absolu!
    """
    df['target'] = df['Close'].pct_change().shift(-1)
    
    df['current_price'] = df['Close']
    
    return df

def prepare_features(df):
    """
    Prepare all features and target variable.
    
    NOUVELLE APPROCHE: Prédit le changement (%), pas le prix absolu.
    """
    print("Creating features...")
    
    # Create lag features (returns)
    df = create_lag_features(df, n_lags=5)
    
    # Create technical indicators
    df = create_technical_indicators(df)
    
    # Create target (% change)
    df = create_target(df)
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Separate features and target
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'current_price', 'Close', 'High', 'Low', 
                                     'Open', 'Adj Close', 'Volume']]
    
    X = df[feature_columns]
    y = df['target']
    
    import numpy as np
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    current_prices = df['current_price'][valid_mask]
    
    print(f"  Dataset ready: {len(X)} rows, {len(feature_columns)} features")
    print("  Target: Predicting % change (not absolute price)")
    
    return X, y, current_prices
