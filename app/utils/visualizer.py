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
    
    # Adjust the displayed date range based on prediction period
    if prediction_period == "weekly":
        # For weekly, show last 6 months of data
        start_date = df.index[-1] - pd.Timedelta(days=180)
        future_days = 30  # Show 30 days into the future
    elif prediction_period == "monthly":
        # For monthly, show last 12 months of data
        start_date = df.index[-1] - pd.Timedelta(days=365)
        future_days = 90  # Show 90 days into the future
    else:  # yearly
        # For yearly, show last 2 years of data
        start_date = df.index[-1] - pd.Timedelta(days=730)
        future_days = 180  # Show 180 days into the future
    
    # Filter data for display - ensure we have enough data
    try:
        display_df = df[df.index >= start_date]
        if len(display_df) < 10:  # If not enough data in range, show all available
            display_df = df
    except Exception as e:
        # Fallback to all data if date filtering fails
        display_df = df
    
    # Plot the filtered data
    ax.plot(display_df.index, display_df['Close'], label="Close", color='blue', linewidth=1.5)
    ax.plot(display_df.index, display_df['High'], label="High", color='green', alpha=0.6, linewidth=1)
    ax.plot(display_df.index, display_df['Low'], label="Low", color='salmon', alpha=0.6, linewidth=1)
    
    # Add prediction line with multiple points for smoother curve
    last_date = df.index[-1]
    last_price = df['Close'].iloc[-1]
    
    # Create a series of future dates and predicted prices
    num_points = 10
    future_dates = [last_date + pd.Timedelta(days=i*future_days/num_points) for i in range(num_points+1)]
    
    # Calculate trend factors based on historical volatility
    try:
        # Calculate historical volatility
        hist_volatility = df['Close'].pct_change().std() * np.sqrt(252)  # Annualized
        
        # Adjust trend factors based on volatility and prediction period
        base_factor = max(0.02, min(0.15, hist_volatility))
        
        if prediction_period == "yearly":
            trend_factor = base_factor * 3
        elif prediction_period == "monthly":
            trend_factor = base_factor * 1.5
        else:  # weekly
            trend_factor = base_factor
    except:
        # Fallback to default values if calculation fails
        if prediction_period == "yearly":
            trend_factor = 0.12
        elif prediction_period == "monthly":
            trend_factor = 0.06
        else:  # weekly
            trend_factor = 0.03
    
    if trend == "UP":
        predicted_prices = [last_price * (1 + trend_factor * (i/num_points)) for i in range(num_points+1)]
    else:
        predicted_prices = [last_price * (1 - trend_factor * (i/num_points)) for i in range(num_points+1)]
    
    # Plot the prediction line
    ax.plot(future_dates, predicted_prices, 
            label=f"Predicted Trend ({prediction_period.capitalize()})", 
            color='red', linestyle='--', linewidth=2)
    
    # Customize plot
    ax.set_title(f"{ticker} Stock Price and Predicted Trend ({prediction_period.capitalize()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format date axis to be more readable
    plt.gcf().autofmt_xdate()
    
    # Set y-axis limits to provide some padding
    y_min = min(display_df['Low'].min(), min(predicted_prices)) * 0.95
    y_max = max(display_df['High'].max(), max(predicted_prices)) * 1.05
    ax.set_ylim(y_min, y_max)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url