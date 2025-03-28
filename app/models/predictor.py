from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from app.utils.data_handler import get_stock_data, add_features

def train_model(X_train, y_train):
    """Train an XGBoost classifier model"""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def predict_stock_trend(ticker, prediction_period):
    """Predict stock trend using machine learning"""
    # Get and prepare data
    df = get_stock_data(ticker)
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
        'prediction_period': prediction_period
    }