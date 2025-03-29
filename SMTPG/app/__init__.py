from flask import Flask

def create_app():
    app = Flask(__name__, template_folder='../templates')
    
    # Register routes
    from app.routes.main_routes import register_routes
    register_routes(app)
    
    return app