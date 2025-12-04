"""
Evaluate model performance and create visualizations.
"""

import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive performance metrics.
    
    Returns a dictionary with MAE, RMSE, R², MAPE, and additional metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    mean_error = np.mean(y_true - y_pred)  # Bias (positive = overestimation)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Max_Error': max_error,
        'Bias': mean_error
    }


def evaluate_models(predictions, y_test):
    """
    Evaluate all models and return detailed results table.
    
    Prints metrics for each model and returns a comprehensive DataFrame.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    results = []
    
    for model_name, pred in predictions.items():
        metrics = calculate_metrics(y_test, pred)
        metrics['Model'] = model_name
        results.append(metrics)
        
        print(f"\n{model_name}:")
        print(f"  MAE:        {metrics['MAE']:.2f} CHF")
        print(f"  RMSE:       {metrics['RMSE']:.2f} CHF")
        print(f"  R²:         {metrics['R2']:.4f}")
        print(f"  MAPE:       {metrics['MAPE']:.2f}%")
        print(f"  Max Error:  {metrics['Max_Error']:.2f} CHF")
        print(f"  Bias:       {metrics['Bias']:.2f} CHF")
    
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'MAE', 'RMSE', 'R2', 'MAPE', 'Max_Error', 'Bias']]
    
    # Identify best model
    best_model = results_df.loc[results_df['MAPE'].idxmin(), 'Model']
    print("\n" + "="*60)
    print(f"Best Model (lowest MAPE): {best_model}")
    print("="*60)
    
    return results_df


def plot_predictions(y_test, predictions, ticker_name, save_path=None):
    """
    Plot predictions vs actual values for all models.
    
    Creates a line plot comparing actual prices with predicted prices.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot actual values
    plt.plot(y_test.index, y_test.values, label='Actual', 
             color='black', linewidth=2.5, alpha=0.8)
    
    # Plot predictions from each model
    colors = ['blue', 'red', 'green']
    linestyles = ['--', '-.', ':']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        plt.plot(y_test.index, pred, label=f'{model_name}',
                color=colors[i], linestyle=linestyles[i], 
                linewidth=1.5, alpha=0.7)
    
    plt.title(f'Stock Price Predictions vs Actual Values - {ticker_name}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Closing Price (CHF)', fontsize=13)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved: {save_path}")
    
    plt.show()


def plot_prediction_errors(y_test, predictions, ticker_name, save_path=None):
    """
    Plot prediction errors over time for each model.
    
    Shows how the error (actual - predicted) evolves over the test period.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        errors = y_test.values - pred
        
        axes[i].plot(y_test.index, errors, color=colors[i], 
                    linewidth=1.5, alpha=0.7)
        axes[i].axhline(y=0, color='black', linestyle='--', 
                       linewidth=1, alpha=0.5)
        axes[i].set_title(f'{model_name} - Prediction Errors', 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Error (CHF)', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        axes[i].text(0.02, 0.95, f'Mean Error: {mean_error:.2f} CHF\nStd: {std_error:.2f} CHF',
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Date', fontsize=13)
    fig.suptitle(f'Prediction Errors Over Time - {ticker_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved: {save_path}")
    
    plt.show()


def plot_error_distribution(y_test, predictions, save_path=None):
    """
    Plot distribution of prediction errors for all models.
    
    Creates histograms showing the distribution of errors.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        errors = y_test.values - pred
        
        axes[i].hist(errors, bins=30, color=colors[i], 
                    alpha=0.7, edgecolor='black')
        axes[i].axvline(x=0, color='black', linestyle='--', 
                       linewidth=2, label='Zero Error')
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Prediction Error (CHF)', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Distribution of Prediction Errors', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved: {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot comprehensive bar charts comparing model performances.
    
    Creates a 2x3 grid showing all metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE', 'Max_Error', 'Bias']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                       color=['blue', 'red', 'green'], legend=False)
        
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=0)
        
        # Add values on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    fig.suptitle('Comprehensive Model Performance Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved: {save_path}")
    
    plt.show()


def plot_actual_vs_predicted_scatter(y_test, predictions, save_path=None):
    """
    Create scatter plots of actual vs predicted values.
    
    Perfect predictions would lie on the diagonal line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        axes[i].scatter(y_test.values, pred, color=colors[i], 
                       alpha=0.6, s=30)
        
        # Add perfect prediction line
        min_val = min(y_test.values.min(), pred.min())
        max_val = max(y_test.values.max(), pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 
                    'k--', linewidth=2, label='Perfect Prediction')
        
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Actual Price (CHF)', fontsize=11)
        axes[i].set_ylabel('Predicted Price (CHF)', fontsize=11)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    fig.suptitle('Actual vs Predicted Values', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved: {save_path}")
    
    plt.show()


def calculate_relative_performance(results_df):
    """
    Calculate relative performance between models.
    
    Shows how much better/worse each model is compared to others.
    Returns a DataFrame with relative comparisons.
    """
    print("\n" + "="*70)
    print("RELATIVE PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Find best and worst for each metric
    metrics = ['MAE', 'RMSE', 'MAPE']
    
    for metric in metrics:
        best_value = results_df[metric].min()
        worst_value = results_df[metric].max()
        
        print(f"\n{metric}:")
        print(f"  Best:  {results_df.loc[results_df[metric].idxmin(), 'Model']} = {best_value:.2f}")
        print(f"  Worst: {results_df.loc[results_df[metric].idxmax(), 'Model']} = {worst_value:.2f}")
        
        # Calculate relative difference
        if best_value > 0:
            relative_diff = ((worst_value - best_value) / best_value) * 100
            print(f"  Relative difference: {relative_diff:.1f}% worse")
        
        # Show each model's performance relative to best
        print("  Performance relative to best:")
        for _, row in results_df.iterrows():
            value = row[metric]
            if best_value > 0:
                relative = ((value - best_value) / best_value) * 100
                status = "★ BEST" if value == best_value else f"+{relative:.1f}% worse"
                print(f"    {row['Model']:12s}: {status}")
    
    # R² comparison (higher is better)
    print("\nR² (higher is better):")
    best_r2 = results_df['R2'].max()
    worst_r2 = results_df['R2'].min()
    print(f"  Best:  {results_df.loc[results_df['R2'].idxmax(), 'Model']} = {best_r2:.4f}")
    print(f"  Worst: {results_df.loc[results_df['R2'].idxmin(), 'Model']} = {worst_r2:.4f}")
    
    print("\n  Variance explained:")
    for _, row in results_df.iterrows():
        r2_value = row['R2']
        variance_explained = r2_value * 100
        status = "★ BEST" if r2_value == best_r2 else ""
        print(f"    {row['Model']:12s}: {variance_explained:.1f}% {status}")
    
    print("\n" + "="*70)
    
    return results_df


def create_evaluation_report(results_df, y_test, ticker_name):
    """
    Generate a comprehensive evaluation report.
    
    Prints a detailed text report summarizing all findings.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)
    
    print(f"\nStock: {ticker_name}")
    print(f"Test Period: {y_test.index[0]} to {y_test.index[-1]}")
    print(f"Number of Predictions: {len(y_test)}")
    print(f"Price Range: {y_test.min():.2f} - {y_test.max():.2f} CHF")
    
    print("\n" + "-"*70)
    print("MODEL RANKINGS")
    print("-"*70)
    
    # Rank by different metrics
    metrics_to_rank = ['MAE', 'RMSE', 'MAPE']
    
    for metric in metrics_to_rank:
        ascending = True  # Lower is better for MAE, RMSE, MAPE
        ranked = results_df.sort_values(metric, ascending=ascending)
        print(f"\nBest by {metric}:")
        for rank, row in enumerate(ranked.itertuples(), 1):
            print(f"  {rank}. {row.Model}: {getattr(row, metric):.2f}")
    
    # R² ranking (higher is better)
    ranked_r2 = results_df.sort_values('R2', ascending=False)
    print("\nBest by R²:")
    for rank, row in enumerate(ranked_r2.itertuples(), 1):
        print(f"  {rank}. {row.Model}: {row.R2:.4f}")
    
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)
    
    best_model = results_df.loc[results_df['MAPE'].idxmin()]
    print(f"\nOverall Best Model: {best_model['Model']}")
    print(f"  - Average error: {best_model['MAE']:.2f} CHF ({best_model['MAPE']:.2f}%)")
    print(f"  - Variance explained: {best_model['R2']*100:.1f}%")
    
    if best_model['MAPE'] < 2:
        print("  - Performance: EXCELLENT (MAPE < 2%)")
    elif best_model['MAPE'] < 5:
        print("  - Performance: GOOD (MAPE < 5%)")
    elif best_model['MAPE'] < 10:
        print("  - Performance: ACCEPTABLE (MAPE < 10%)")
    else:
        print("  - Performance: NEEDS IMPROVEMENT (MAPE > 10%)")
    
    print("\n" + "="*70)


def save_evaluation_results(results_df, filepath="data/processed/evaluation_results.csv"):
    """
    Save detailed evaluation results to CSV.
    
    Includes all metrics for all models.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"\nEvaluation results saved to {filepath}")
