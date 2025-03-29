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
    # Reduce complexity for faster training
    model = XGBClassifier(
        n_estimators=50,  # Reduced from 100
        learning_rate=0.1,
        max_depth=3,      # Reduced from 5
        eval_metric='error',
        n_jobs=-1         # Use all available cores
    )
    model.fit(X_train, y_train)
    return model

def predict_stock_trend(ticker, prediction_period):
    """Predict stock trend using machine learning"""
    # Define period mapping with shorter timeframes to reduce data size
    period_mapping = {
        "weekly": "6mo",  # Reduced from 1y
        "monthly": "1y",  # Reduced from 2y
        "yearly": "2y"    # Reduced from 5y
    }
    
    # Get and prepare data with retry logic
    max_retries = 2  # Reduced from 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            df = get_stock_data(ticker, period=period_mapping[prediction_period])  # Pass period parameter
            break
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count}/{max_retries}: yfinance failed for {ticker}: {str(e)}")
            
            if retry_count < max_retries:
                sleep_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"All {max_retries} attempts failed. Using mock data.")
                
                # Create simpler mock data with fewer points
                end_date = pd.Timestamp.now()
                if prediction_period == "yearly":
                    start_date = end_date - pd.Timedelta(days=365)  # Reduced to 1 year
                elif prediction_period == "monthly":
                    start_date = end_date - pd.Timedelta(days=180)  # Reduced to 6 months
                else:  # weekly
                    start_date = end_date - pd.Timedelta(days=90)   # Reduced to 3 months
                
                # Generate date range with fewer points
                date_range = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly instead of business days
                
                # Generate mock price data
                base_price = 150.0
                np.random.seed(42)
                returns = np.random.normal(0.0005, 0.015, size=len(date_range))
                prices = base_price * (1 + returns).cumprod()
                
                df = pd.DataFrame({
                    'Open': prices * 0.99,
                    'High': prices * 1.02,
                    'Low': prices * 0.98,
                    'Close': prices,
                    'Adj Close': prices,
                    'Volume': np.random.randint(5000000, 50000000, size=len(date_range))
                }, index=date_range)
    
    # Optimize feature calculation
    df = add_features(df, prediction_period)
    
    # Reduce number of features for faster processing
    features = [
        'Close', 'High', 'Low', 'SMA_10', 'SMA_50', 'RSI', 
        'MACD', 'BB_Upper', 'BB_Lower', 'Volume'
    ]
    
    X = df[features].dropna()  # Ensure we drop NaN values
    y = df['Target'].loc[X.index]  # Align target with features
    
    # Reduce minimum data points requirement
    if len(X) < 30:  # Reduced from 50
        raise ValueError(f"Insufficient data points ({len(X)}) for {ticker}.")
    
    # Use a smaller test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
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
    
    return {
        'df': df,
        'accuracy': accuracy,
        'trend': trend,
        'confidence': confidence,
        'prediction_period': prediction_period,
        'ticker': ticker
    }