"""
Main Flask Application for Module 3
Runs on port 5002
""" 

from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.api import api_bp, initialize_module3
from config import Config

# Create Flask app
app = Flask(__name__)
CORS(app)

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'data', 'Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx')


@app.route('/')
def index():
    return jsonify({
        'message': 'Module 3 - Relief Resource Recommendation API',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict/<division_name>',
            'divisions': '/api/divisions'
        },
        'integration': {
            'module1': 'Connected via Supabase',
            'module2': 'Ready (mock mode active, switch to real data with USE_MOCK_DATA=False)'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'module': 'Module 3', 'port': Config.MODULE3_PORT})


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        print("Please place your Excel file in the data/ folder")
    else:
        initialize_module3(DATA_PATH)
    
    app.run(host='0.0.0.0', port=Config.MODULE3_PORT, debug=Config.DEBUG)