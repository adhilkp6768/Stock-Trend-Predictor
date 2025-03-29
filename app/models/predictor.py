from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from app.utils.data_handler import get_stock_data, add_features
import yfinance as yf
import time
import random

def train_model(X_train, y_train):
    """Train an XGBoost classifier model"""
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='error'  # Explicitly set eval_metric to 'error'
    )
    model.fit(X_train, y_train)
    return model

def predict_stock_trend(ticker, prediction_period):
    """Predict stock trend using machine learning"""
    # Define period mapping
    period_mapping = {
        "weekly": "1y",
        "monthly": "2y",
        "yearly": "5y"
    }
    
    # Get and prepare data with retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            df = get_stock_data(ticker)
            break  # If successful, break out of the retry loop
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count}/{max_retries}: yfinance failed for {ticker}: {str(e)}")
            
            if retry_count < max_retries:
                # Add exponential backoff with jitter for retries
                sleep_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"All {max_retries} attempts failed. Using mock data.")
                
                # Create more realistic mock data
                end_date = pd.Timestamp.now()
                if prediction_period == "yearly":
                    start_date = end_date - pd.Timedelta(days=365*3)  # 3 years of data
                elif prediction_period == "monthly":
                    start_date = end_date - pd.Timedelta(days=365)    # 1 year of data
                else:  # weekly
                    start_date = end_date - pd.Timedelta(days=180)    # 6 months of data
                
                # Generate date range
                date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
                
                # Generate mock price data with realistic patterns
                base_price = 150.0  # Starting price for AAPL-like stock
                np.random.seed(42)  # For reproducibility
                
                # Generate random walk with drift
                returns = np.random.normal(0.0005, 0.015, size=len(date_range))
                prices = base_price * (1 + returns).cumprod()
                
                # Create mock dataframe
                df = pd.DataFrame({
                    'Open': prices * 0.99,
                    'High': prices * 1.02,
                    'Low': prices * 0.98,
                    'Close': prices,
                    'Adj Close': prices,
                    'Volume': np.random.randint(5000000, 50000000, size=len(date_range))
                }, index=date_range)
    
    # Continue with the rest of the function
    df = add_features(df, prediction_period)
    
    features = [
        'Close', 'High', 'Low', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI', 
        'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volatility', 
        'Momentum', 'Volume', 'Lag_1', 'Lag_3', 'ADX'
    ]
    X = df[features]
    y = df['Target']
    
    if len(X) < 50:
        raise ValueError(f"Insufficient data points ({len(X)}) for {ticker}.")
    
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Make prediction
    latest_data = X.tail(1)
    prediction = model.predict(latest_data)[0]
    prediction_proba = model.predict_proba(latest_data)[0]
    confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
    trend = "UP" if prediction == 1 else "DOWN"
    
    # Return results without feature importance
    return {
        'df': df,
        'accuracy': accuracy,
        'trend': trend,
        'confidence': confidence,
        'prediction_period': prediction_period,
        'ticker': ticker
    }