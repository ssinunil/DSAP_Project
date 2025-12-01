"""
Machine learning models for stock prediction.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def split_train_test(X, y, train_end_date='2022-12-31'):
    """Split data by date: before train_end_date = train, after = test."""
    train_mask = X.index <= train_end_date
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    print("\nData split:")
    print(f"  Training: {len(X_train)} rows (up to {train_end_date})")
    print(f"  Testing: {len(X_test)} rows (after {train_end_date})")
    
    return X_train, X_test, y_train, y_test


def create_models():
    """Create the 3 models."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    return models


def train_models(models, X_train, y_train):
    """Train all models."""
    print("\nTraining models...")
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
    
    print("All models trained!")
    return models


def predict(models, X_test):
    """Make predictions with all models."""
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    
    return predictions
