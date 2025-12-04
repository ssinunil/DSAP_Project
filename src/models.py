"""
Machine learning models for stock prediction.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
        'RandomForest': RandomForestRegressor(
            n_estimators=200,       # More trees
            max_depth=None,         # No depth limit (let trees grow)
            min_samples_split=5,    # Minimum samples to split
            min_samples_leaf=2,     # Minimum samples in leaf
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,       # More iterations
            max_depth=7,            # Deeper trees
            learning_rate=0.05,     # Slower learning (more stable)
            subsample=0.8,          # Use 80% of data per tree
            colsample_bytree=0.8,   # Use 80% of features per tree
            random_state=42
        )
    }
    return models


def train_models(models, X_train, y_train):
    """Train all models."""
    print("\nTraining models...")
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
    
    print("All models trained!")
    return models, scaler


def predict(models, X_test, scaler):
    """Make predictions with all models."""
    # Scale test data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(X_test_scaled)
    
    return predictions
