from flask import Flask, render_template, request, Response
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
import warnings
import threading
import time
warnings.filterwarnings("ignore")

app = Flask(__name__)
server_running = True

# Mock data function
def get_mock_data(ticker):
    dates = pd.date_range(start="2023-03-18", end="2025-03-18", freq="B")  # Extended for yearly
    prices = np.random.normal(100, 10, len(dates)).cumsum()
    df = pd.DataFrame({
        "Close": prices,
        "High": prices + np.random.uniform(0, 5, len(dates)),
        "Low": prices - np.random.uniform(0, 5, len(dates)),
        "Volume": np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    return df

# Add features to dataframe
def add_features(df, period):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=14).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Lag_1'] = df['Returns'].shift(1)
    df['Lag_3'] = df['Returns'].shift(3)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                         np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                   abs(df['Low'] - df['Close'].shift(1))))
    df['ADX'] = df['TR'].rolling(window=14).mean()
    
    # Adjust target based on period
    if period == "yearly":
        shift_days = -252  # Approx 1 year (252 trading days)
    elif period == "monthly":
        shift_days = -21   # Approx 1 month (21 trading days)
    else:  # weekly
        shift_days = -5    # Approx 1 week (5 trading days)
    
    df['Target'] = (df['Close'].shift(shift_days) > df['Close']).astype(int)
    return df.dropna()

# Fetch stock data
def get_stock_data(ticker, period="2y", fallback_period="1y", short_period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            print(f"No data for {ticker} with period={period}. Trying {fallback_period}...")
            df = stock.history(period=fallback_period)
        if df.empty:
            print(f"No data for {ticker} with period={fallback_period}. Trying {short_period}...")
            df = stock.history(period=short_period)
        if df.empty:
            raise ValueError(f"No data available for {ticker} via yfinance. Using mock data.")
    except Exception as e:
        print(f"yfinance failed for {ticker}: {str(e)}. Using mock data.")
        return get_mock_data(ticker)
    return df

# Prediction function
def predict_trend(ticker, prediction_period):
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    latest_data = X.tail(1)
    prediction = model.predict(latest_data)[0]
    trend = "UP" if prediction == 1 else "DOWN"
    
    # In the predict_trend function, modify the plot generation section:
    
    # Generate plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Adjust the displayed date range based on prediction period
    if prediction_period == "weekly":
        # For weekly, show last 3 months of data
        start_date = df.index[-1] - pd.Timedelta(days=90)
    elif prediction_period == "monthly":
        # For monthly, show last 6 months of data
        start_date = df.index[-1] - pd.Timedelta(days=180)
    else:  # yearly
        # For yearly, show last 2 years of data
        start_date = df.index[-1] - pd.Timedelta(days=730)
    
    # Filter data for display
    display_df = df[df.index >= start_date]
    
    # Plot the filtered data
    ax.plot(display_df.index, display_df['Close'], label="Close", color='blue', alpha=0.7)
    ax.plot(display_df.index, display_df['High'], label="High", color='green', alpha=0.5)
    ax.plot(display_df.index, display_df['Low'], label="Low", color='red', alpha=0.5)
    
    last_date = df.index[-1]
    last_price = df['Close'].iloc[-1]
    if prediction_period == "yearly":
        next_date = last_date + pd.Timedelta(days=365)
        trend_factor = 0.05  # Larger for yearly
    elif prediction_period == "monthly":
        next_date = last_date + pd.Timedelta(days=30)
        trend_factor = 0.02
    else:  # weekly
        next_date = last_date + pd.Timedelta(days=7)
        trend_factor = 0.01
    
    next_price = last_price * (1 + trend_factor) if trend == "UP" else last_price * (1 - trend_factor)
    ax.plot([last_date, next_date], [last_price, next_price], 
            label=f"Predicted Trend ({prediction_period.capitalize()})", 
            color='red', linestyle='--', linewidth=4, alpha=1.0)
    
    ax.set_title(f"{ticker} Stock Price and Predicted Trend ({prediction_period.capitalize()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return accuracy, trend, plot_url

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "INTC", "JPM", "BAC", "WMT"]
    periods = ["yearly", "monthly", "weekly"]
    result = None
    plot_url = None
    
    if request.method == 'POST':
        ticker = request.form.get('ticker') or request.form.get('custom_ticker')
        prediction_period = request.form.get('prediction_period', 'weekly')  # Default to weekly
        if ticker:
            try:
                accuracy, trend, plot_url = predict_trend(ticker.upper(), prediction_period)
                result = {
                    'ticker': ticker.upper(),
                    'accuracy': f"{accuracy:.2%}",
                    'trend': trend,
                    'period': prediction_period.capitalize()
                }
            except Exception as e:
                result = {'error': str(e)}
    
    return render_template('index.html', stocks=stocks, periods=periods, result=result, plot_url=plot_url)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global server_running
    server_running = False
    threading.Thread(target=shutdown_server).start()
    return "Shutting down..."

def shutdown_server():
    time.sleep(1)
    import os
    os._exit(0)

if __name__ == '__main__':
    app.run(debug=True)