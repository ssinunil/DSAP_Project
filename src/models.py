"""
Machine learning models for stock prediction.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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
    """
    Create the 3 models with AGGRESSIVE hyperparameters.
    
    CORRECTION: Hyperparamètres beaucoup plus agressifs pour forcer
    les modèles à capturer la vraie variance des prix.
    """
    models = {
        'Ridge': Ridge(alpha=0.1),  
        
        'RandomForest': RandomForestRegressor(
            n_estimators=500,           
            max_depth=None,             
            min_samples_split=2,        
            min_samples_leaf=1,         
            max_features='sqrt',        
            bootstrap=True,             
            random_state=42,
            n_jobs=-1                   
        ),
        
        
        'XGBoost': XGBRegressor(
            n_estimators=500,           
            max_depth=8,                
            learning_rate=0.1,          
            min_child_weight=1,         
            subsample=0.9,              
            colsample_bytree=0.9,      
            gamma=0,                    
            reg_alpha=0,                
            reg_lambda=0.1,             
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def train_models(models, X_train, y_train):
    """
    Train all models with automatic feature normalization.
    """
    print("\nTraining models...")
    
    print("  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        index=X_train.index,
        columns=X_train.columns
    )
    
    print("  ✓ Features normalized (mean≈0, std≈1)")
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
    
    print("All models trained!")
    
    return models, scaler


def predict(models, X_test, scaler):
    """
    Make predictions with all models.
    """
    predictions = {}
    
    X_test_scaled = scaler.transform(X_test)
    
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        index=X_test.index,
        columns=X_test.columns
    )
    
    for name, model in models.items():
        predictions[name] = model.predict(X_test_scaled)
    
    return predictions
