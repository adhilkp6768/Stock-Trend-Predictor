import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import numpy as np

def generate_trend_plot(prediction_result):
    """Generate stock trend visualization plot"""
    df = prediction_result['df']
    trend = prediction_result['trend']
    prediction_period = prediction_result['prediction_period']
    ticker = prediction_result.get('ticker', 'Stock')
    
    # Create figure and plot historical data
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use all available historical data
    display_df = df
    
    # Plot the historical data
    ax.plot(display_df.index, display_df['Close'], label="Close", color='blue', linewidth=1.5)
    ax.plot(display_df.index, display_df['High'], label="High", color='green', alpha=0.6, linewidth=1)
    ax.plot(display_df.index, display_df['Low'], label="Low", color='red', alpha=0.6, linewidth=1)
    
    # Add prediction line
    last_date = df.index[-1]
    last_price = df['Close'].iloc[-1]
    
    # Set future prediction period
    if prediction_period == "weekly":
        future_days = 30
    elif prediction_period == "monthly":
        future_days = 90
    else:  # yearly
        future_days = 365
    
    # Create future date for prediction endpoint
    future_date = last_date + pd.Timedelta(days=future_days)
    
    # Calculate predicted price based on trend
    if trend == "UP":
        future_price = last_price * 1.05 if prediction_period == "weekly" else \
                      last_price * 1.10 if prediction_period == "monthly" else \
                      last_price * 1.15
    else:
        future_price = last_price * 0.95 if prediction_period == "weekly" else \
                      last_price * 0.90 if prediction_period == "monthly" else \
                      last_price * 0.85
    
    # Plot prediction line
    ax.plot([last_date, future_date], [last_price, future_price], 
            label=f"Predicted Trend ({prediction_period.capitalize()})", 
            color='red', linestyle='--', linewidth=2)
    
    # Customize plot
    ax.set_title(f"{ticker} Stock Price and Predicted Trend ({prediction_period.capitalize()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format date axis
    plt.gcf().autofmt_xdate()
    
    # Set y-axis limits with padding
    price_range = display_df['High'].max() - display_df['Low'].min()
    y_min = display_df['Low'].min() - (price_range * 0.1)
    y_max = max(display_df['High'].max(), future_price) + (price_range * 0.1)
    ax.set_ylim(y_min, y_max)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url