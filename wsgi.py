import os
from run import create_app

# Create the Flask application instance
app = create_app()

# This section only runs when executing the file directly
if __name__ == "__main__":
    # For local development, use localhost and port 5000
    # For production, use 0.0.0.0 and the PORT environment variable
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    else:
        app.run(host='127.0.0.1', port=5000, debug=True)