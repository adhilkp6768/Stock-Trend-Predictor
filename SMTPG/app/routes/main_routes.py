from flask import render_template, request, current_app
from app.models.predictor import predict_stock_trend
from app.utils.visualizer import generate_trend_plot

def register_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "INTC", "JPM", "BAC", "WMT"]
        periods = ["yearly", "monthly", "weekly"]
        result = None
        plot_url = None
        error = None
        
        # Get stock details from app config
        stock_details = current_app.config.get('STOCK_DETAILS', {})
        current_prices = current_app.config.get('CURRENT_PRICES', {})
        
        if request.method == 'POST':
            ticker = request.form.get('ticker') or request.form.get('custom_ticker')
            prediction_period = request.form.get('prediction_period', 'weekly')
            
            if not ticker:
                error = "Please select a stock or enter a custom ticker symbol"
            else:
                try:
                    # Get prediction results
                    prediction_result = predict_stock_trend(ticker.upper(), prediction_period)
                    
                    # Generate visualization (only trend plot, no feature importance)
                    plot_url = generate_trend_plot(prediction_result)
                    
                    # Prepare result for template
                    result = {
                        'ticker': ticker.upper(),
                        'accuracy': f"{prediction_result['accuracy']:.2%}",
                        'trend': prediction_result['trend'],
                        'period': prediction_period.capitalize(),
                        'confidence': f"{prediction_result['confidence']:.2%}"
                    }
                except Exception as e:
                    result = {'error': str(e)}
        
        return render_template('index.html', 
                              stocks=stocks, 
                              periods=periods, 
                              result=result, 
                              plot_url=plot_url,
                              stock_details=stock_details,
                              current_prices=current_prices,
                              error=error)
    
    @app.route('/about')
    def about():
        return render_template('about.html')
    
    @app.route('/contact')
    def contact():
        return render_template('contact.html')

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        import threading
        import time
        import os
        
        def shutdown_server():
            time.sleep(1)
            os._exit(0)
        
        threading.Thread(target=shutdown_server).start()
        return "Shutting down..."