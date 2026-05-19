from flask import Flask, render_template, jsonify, send_file
from flask_cors import CORS
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.api import api_bp, initialize_module3
from database import get_flood_results, get_latest_summary

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, 'interface', 'frontend')

app = Flask(__name__, 
            template_folder=FRONTEND_PATH,
            static_folder=FRONTEND_PATH)

CORS(app)

app.register_blueprint(api_bp, url_prefix='/api')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'data', 'Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx')


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'port': 5001})


@app.route('/geojson')
def geojson():
    try:
        flood_data = get_flood_results()
        geojson = {"type": "FeatureCollection", "features": []}
        if flood_data:
            for item in flood_data:
                feature = {
                    "type": "Feature",
                    "geometry": item.get('geometry'),
                    "properties": {
                        "ds_division": item.get('ds_division') or item.get('name', 'Unknown'),
                        "flood_area_ha": item.get('flood_area_ha', 0),
                        "priority_label": item.get('priority_label', 'No Flood')
                    }
                }
                geojson["features"].append(feature)
        return jsonify(geojson)
    except:
        return jsonify({"features": []})


@app.route('/results')
def results():
    try:
        summary = get_latest_summary()
        return jsonify({"success": True, "data": summary})
    except:
        return jsonify({"success": False, "data": {}})


if __name__ == '__main__':
    if os.path.exists(DATA_PATH):
        initialize_module3(DATA_PATH)
    else:
        print(f"❌ Data file not found: {DATA_PATH}")

    print("\n" + "=" * 50)
    print("🌐 MODULE 3 RUNNING ON PORT 5001")
    print("=" * 50)
    print(f"📍 Dashboard: http://localhost:5001")
    print(f"📍 Frontend folder: {FRONTEND_PATH}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5001, debug=True)