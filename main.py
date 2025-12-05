"""
Main script for Swiss stock price prediction project.
"""
# pylint: disable=import-error

import sys
import warnings

sys.path.append('src')

from data_loading import download_swiss_stocks, save_raw_data, load_raw_data
from feature_engineering import prepare_features
from models import split_train_test, create_models, train_models, predict
from evaluation import (evaluate_models, plot_predictions, plot_prediction_errors,
                       plot_error_distribution, plot_model_comparison,
                       plot_actual_vs_predicted_scatter, create_evaluation_report,
                       save_evaluation_results)
from utils import explore_data, plot_price_history

warnings.filterwarnings('ignore')


def main():
    """Main function that runs the entire pipeline."""
    
    print("="*70)
    print("SWISS STOCK PRICE PREDICTION PROJECT")
    print("="*70)
    
    # ========== CONFIGURATION ==========
    TICKERS = ['NESN.SW', 'UBSG.SW', 'NOVN.SW', 'ROG.SW', 'ABBN.SW']
    SELECTED_TICKER = 'NESN.SW'
    
    START_DATE = "2018-01-01"
    END_DATE = "2024-12-31"
    TRAIN_END_DATE = "2022-12-31"
    
    
    # ========== STEP 1: DOWNLOAD DATA ==========
    print("\n" + "="*70)
    print("STEP 1: Download Data")
    print("="*70)
    
    all_data = download_swiss_stocks(TICKERS, START_DATE, END_DATE)
    save_raw_data(all_data, output_dir="data/raw")
    
    
    # ========== STEP 2: LOAD AND EXPLORE DATA ==========
    print("\n" + "="*70)
    print("STEP 2: Load and Explore Data")
    print("="*70)
    
    df = load_raw_data(SELECTED_TICKER, data_dir="data/raw")
    explore_data(df, SELECTED_TICKER)
    plot_price_history(df, SELECTED_TICKER)
    
    
    # ========== STEP 3: CREATE FEATURES ==========
    print("\n" + "="*70)
    print("STEP 3: Create Features")
    print("="*70)
    
    X, y = prepare_features(df, lags=[1, 2, 3, 5, 10])
    
    print("\nFeatures created:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")
    
    
    # ========== STEP 4: SPLIT DATA ==========
    print("\n" + "="*70)
    print("STEP 4: Split Data")
    print("="*70)
    
    X_train, X_test, y_train, y_test = split_train_test(X, y, TRAIN_END_DATE)
    
    
    # ========== STEP 5: TRAIN MODELS ==========
    print("\n" + "="*70)
    print("STEP 5: Train Models")
    print("="*70)
    
    models = create_models()
    models = train_models(models, X_train, y_train)
    
    
    # ========== STEP 6: MAKE PREDICTIONS ==========
    print("\n" + "="*70)
    print("STEP 6: Make Predictions")
    print("="*70)
    
    predictions = predict(models, X_test)
    print("Predictions generated for all models")
    
    
    # ========== STEP 7: EVALUATE MODELS ==========
    print("\n" + "="*70)
    print("STEP 7: Evaluate Models")
    print("="*70)
    
    results_df = evaluate_models(predictions, y_test)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    save_evaluation_results(results_df)
    
    
    # ========== STEP 8: CREATE VISUALIZATIONS ==========
    print("\n" + "="*70)
    print("STEP 8: Create Visualizations")
    print("="*70)
    
    # 1. Predictions vs Actual
    print("\nCreating predictions plot...")
    plot_predictions(y_test, predictions, SELECTED_TICKER)
    
    # 2. Prediction Errors Over Time
    print("Creating prediction errors plot...")
    plot_prediction_errors(y_test, predictions, SELECTED_TICKER)
    
    # 3. Error Distribution
    print("Creating error distribution plot...")
    plot_error_distribution(y_test, predictions)
    
    # 4. Model Comparison
    print("Creating model comparison plot...")
    plot_model_comparison(results_df)
    
    # 5. Actual vs Predicted Scatter
    print("Creating scatter plots...")
    plot_actual_vs_predicted_scatter(y_test, predictions)
    
    
    # ========== STEP 9: GENERATE REPORT ==========
    print("\n" + "="*70)
    print("STEP 9: Generate Comprehensive Report")
    print("="*70)
    
    create_evaluation_report(results_df, y_test, SELECTED_TICKER)
    
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nStock analyzed: {SELECTED_TICKER}")
    print(f"Training period: {START_DATE} to {TRAIN_END_DATE}")
    print(f"Test period: {TRAIN_END_DATE} to {END_DATE}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Display best model
    best_model = results_df.loc[results_df['MAPE'].idxmin(), 'Model']
    best_mape = results_df.loc[results_df['MAPE'].idxmin(), 'MAPE']
    best_r2 = results_df.loc[results_df['MAPE'].idxmin(), 'R2']
    
    print(f"\nBest Model: {best_model}")
    print(f"  - MAPE: {best_mape:.2f}%")
    print(f"  - R²: {best_r2:.4f} ({best_r2*100:.1f}% variance explained)")
    
    print("\n" + "="*70)
    print("All results saved in data/processed/")
    print("="*70)


if __name__ == "__main__":
    main()
