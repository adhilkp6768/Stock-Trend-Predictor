# Stock Market Predictor - Stock Details
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request
import threading
import time
import os

# Stock details dictionary with company information
STOCK_DETAILS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "description": "Consumer electronics, software and online services."},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "description": "Software, cloud computing, and hardware."},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "description": "Internet services and products, including Google."},
    "TSLA": {"name": "Tesla, Inc.", "sector": "Automotive", "description": "Electric vehicles, energy storage, and solar products."},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "description": "Graphics processing units and AI computing."},
    "AMD": {"name": "Advanced Micro Devices", "sector": "Technology", "description": "Semiconductors and processors."},
    "INTC": {"name": "Intel Corporation", "sector": "Technology", "description": "Microprocessors and integrated circuits."},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial", "description": "Banking and financial services."},
    "BAC": {"name": "Bank of America Corp.", "sector": "Financial", "description": "Banking and financial services."},
    "WMT": {"name": "Walmart Inc.", "sector": "Retail", "description": "Retail and wholesale stores and e-commerce."}
}

def get_current_price(ticker):
    """Get the current price of a stock"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return "N/A"
    except:
        return "N/A"

def create_app():
    app = Flask(__name__, template_folder='templates')
    
    # Register routes
    from app.routes.main_routes import register_routes
    register_routes(app)
    
    # Add stock details to app config
    app.config['STOCK_DETAILS'] = STOCK_DETAILS
    
    # Fetch current prices
    prices = {}
    for ticker in STOCK_DETAILS.keys():
        prices[ticker] = get_current_price(ticker)
    app.config['CURRENT_PRICES'] = prices
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)