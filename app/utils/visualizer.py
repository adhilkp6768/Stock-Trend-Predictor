import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

def generate_trend_plot(prediction_result):
    """Generate stock trend visualization plot"""
    df = prediction_result['df']
    trend = prediction_result['trend']
    prediction_period = prediction_result['prediction_period']
    
    # Create figure and plot historical data
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
    
    # Add prediction line
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
    
    # Customize plot
    ax.set_title(f"Stock Price and Predicted Trend ({prediction_period.capitalize()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url