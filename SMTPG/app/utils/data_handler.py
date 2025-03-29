import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_mock_data(ticker):
    """Generate mock stock data when real data is unavailable"""
    dates = pd.date_range(start="2023-03-18", end="2025-03-18", freq="B")  # Extended for yearly
    prices = np.random.normal(100, 10, len(dates)).cumsum()
    df = pd.DataFrame({
        "Close": prices,
        "High": prices + np.random.uniform(0, 5, len(dates)),
        "Low": prices - np.random.uniform(0, 5, len(dates)),
        "Volume": np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    return df

def get_stock_data(ticker, period="2y", fallback_period="1y", short_period="6mo"):
    """Fetch stock data from Yahoo Finance with fallback options"""
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

def add_features(df, period):
    """Add technical indicators and features to the dataframe"""
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Other indicators
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=14).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Lag_1'] = df['Returns'].shift(1)
    df['Lag_3'] = df['Returns'].shift(3)
    
    # True Range and ADX
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