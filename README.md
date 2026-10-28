
# MarketLens AI

A Python-based machine-learning market research project for Indian and international equities, combining interactive market-data visualization with technical feature engineering and time-series ML experiments.

## Overview

MarketLens uses historical market data to explore price behavior, generate technical and statistical features, and evaluate machine-learning approaches to predicting next-session price direction.

The current Streamlit dashboard supports ticker discovery, historical market-data exploration, charts, and summary metrics. Machine-learning experiments are implemented separately; displaying model predictions and evaluation results in the dashboard is planned.

This project is intended for research and education only. It does not provide financial advice, guarantee returns, or execute trades.

## Implemented Features

### Market Data and Dashboard
- Company-name and ticker-symbol search using yfinance
- Direct entry of exact ticker symbols
- Historical OHLCV data retrieval
- Market-data validation and error handling
- Interactive candlestick charts using Plotly
- Trading-volume visualization and recent-price tables
- Summary metrics, including latest closing price, period return, and average volume

### Feature Engineering
- Technical and statistical feature generation using pandas and NumPy
- Ten features covering returns, moving-average relationships, volatility, momentum, volume, and intraday price behavior
- Next-session direction target generation for binary classification
- Dataset validation before machine-learning experiments

### Machine Learning and Evaluation
- Scikit-learn logistic-regression baseline pipeline
- Comparison against an always-up prediction baseline
- Chronological train/test holdout evaluation
- Walk-forward validation for time-series experiments
- Model evaluation using accuracy and balanced accuracy
- Automated tests using pytest

## Planned Features

- Displaying machine-learning predictions and model evaluation results in the Streamlit dashboard
- Explainable model signals and feature-level interpretation
- Probability-based market research signals
- Trading-strategy backtesting with realistic transaction costs
- Additional machine-learning models and comparative evaluation

## Tech Stack

- **Language:** Python
- **Data Processing:** pandas, NumPy
- **Market Data:** yfinance
- **Machine Learning:** scikit-learn
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Testing:** pytest
- **Version Control:** Git and GitHub

## Getting Started

Install the dependencies:
pip install -r requirements.txt

Launch the market-data dashboard:
streamlit run app.py

Run automated tests:
python -m pytest -q

## Project Status

**Active development.**

The market-data dashboard, technical feature pipeline, baseline ML experiments, and time-series validation are implemented.

Model-result visualization, explainable signals, and trading-strategy backtesting remain under development.
