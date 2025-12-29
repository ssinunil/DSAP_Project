# Forecasting Swiss Stock Prices using ML Models

## Research Question

Which regression model performs best for predicting Swiss stock prices:
Ridge Regression, Random Forest, or XGBoost?

## Project Overview

This project implements a machine learning pipeline to predict next-day stock prices for Swiss companies listed on the SIX Swiss Exchange. The system uses percentage-based features and predicts daily returns, which are then converted back to absolute prices for evaluation.

**Key Innovation**: Instead of predicting absolute prices (which suffer from multi-collinearity), we predict percentage changes and reconstruct prices, leading to better generalization across different stocks.

## Setup

### Create Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dsap-project
```

### Download Data

The data is automatically downloaded from Yahoo Finance when you run the script. The following Swiss stocks are available:

- NESN.SW (Nestlé)
- UBSG.SW (UBS Group)
- NOVN.SW (Novartis)
- ROG.SW (Roche)
- ABBN.SW (ABB)

## Usage

### Run Full Pipeline

```bash
python main.py
```

Expected output:
- Downloaded data for 5 Swiss stocks (2018-2024)
- Feature engineering (14 percentage-based features)
- Model training with normalization
- Evaluation metrics (MAE, RMSE, MAPE, R²)
- 5 visualization plots saved in `data/plots/`

### Change Stock Selection

Edit `main.py` line ~33:

```python
SELECTED_TICKER = 'UBSG.SW'  # Change to any ticker in TICKERS list
```


## Project Structure

```
ss_project/
├── main.py                      # Main entry point
├── README.md                    # Project documentation
├── Proposal.md                  # Project proposal
├── AI_USAGE.md                  # AI usage and disclosure
├── src/                         # Source code
│   ├── data_loading.py          # Data download/loading
│   ├── feature_engineering.py   # Feature creation (% changes approach)
│   ├── models.py                # Model training with StandardScaler
│   ├── evaluation.py            # Evaluation metrics and plots
│   └── utils.py                 # Data exploration utilities
├── data/
│   ├── raw/                     # Downloaded CSV files
│   ├── processed/               # Processed features and results
│   └── plots/                   # Generated visualizations
└── environment.yml              # Dependencies
```


## Results

### Performance on Nestlé (NESN.SW)

**Test Period**: 2023-2024 (500 trading days)  
**Price Range**: 71.42 - 108.88 CHF

| Model          | MAE (CHF) | RMSE (CHF) | MAPE (%) | R²     |
|----------------|-----------|------------|----------|--------|
| Ridge          | 0.66      | 0.91       | **0.72** | -0.09  |
| Random Forest  | 0.68      | 0.93       | 0.74     | -0.07  |
| XGBoost        | 0.72      | 0.95       | 0.78     | -0.07  |

**Winner**: Ridge Regression (MAPE: 0.72%)

## Requirements

- Python 3.11+
- pandas >= 1.5.0
- numpy >= 1.24.0
- yfinance >= 0.2.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
