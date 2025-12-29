# Project Proposal

## Title
Forecasting Swiss Stock Prices using Machine Learning Models

## Category
Statistical Analysis Tools / Data Analysis & Visualization

---

## Problem Statement and Motivation

Stock price forecasting is a central problem in Finance due to the high volatility and non-linear behavior of financial markets, yet their time series often reveal meaningful trends and recurring patterns. Traditionnal linear models often lack the flexibility to capture complex relationship between past and future price movements.

Therefore, the goal of this project is to forecast the next day closing price of major Swiss stocks (e.g. Nestlé, UBS, Novartis, …) using three different machine learning models.

The objective is to evaluate whether modern ML techniques can provide accurate short-term predictions when trained on historical market data from the Swiss stock exchange.

---

## Planned Approach and Technologies

### Data Collection
Find daily closing prices and trading volumes of selected Swiss stocks (2018-2024) using yfinance.

### Data Preparation
Clean and process the data using pandas and numpy. Create supervised learning datasets by transforming time series into lag-based features (e.g. past 5 days to predict the next day)

### Exploring data
Compute features such as daily returns, moving averages and volatility. Visualizing time series, price distributions and inter-stock correlations using matplotlib.

### Modeling and Forecasting
Implement and compare three machine learning models

- Ridge Regression  
- Random Forest Regressor  
- XGBoost Regressor  

All models will be trained on 2018-2022 and evaluated on 2023-2024 to simulate realistic out-of-sample forecasting.

### Evaluation
Compare models using standard performance metrics and visualize predicted versus actual price trajectories to assess the models’s ability to follow market trends.

---

## Expected Challenges and how I will address them

- Lag feature selection : test different lag windows (3, 5, 10 days) to determine which provides best predictive signal  
- Non stationarity : stock prices are non-stationary, so I will apply differencing and log transformations to stabilize stocks’ variance.  
- Model tuning : Optimize key parameters  

---

## Success Criteria

This project will be considered successful if :

- The data is accurately collected, cleaned, visualized and transformed into a supervised ML dataset  
- All three models are correctly implemented and evaluated  
- At least one model produces reasonable forecasts on 2023-2024 data.  
- The analysis is well-documented, reproducible and visually supported by clear charts and metrics  

---

## Stretch Goals (If Time Permits)

- Extend the analysis to multiple sectors of the Swiss market.  
- Compare alternative prediction horizons (e.g. 5-day ahead return)  
- Develop an interactive dashboard  
