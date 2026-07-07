# # """
# # interface/backend/main.py
# # Flask backend for flood detection system.
# # Team Trivia · University of Moratuwa · 2026
# # """

# import os
# import sys
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from dotenv import load_dotenv

# from pathlib import Path

# # More explicit path resolution
# env_path = Path(__file__).resolve().parent.parent.parent / '.env'
# print(f"Looking for .env at: {env_path}")
# print(f".env exists: {env_path.exists()}")
# load_dotenv(env_path)

# # Add project root to path so modules can be imported
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from modules.flood_detection import FloodDetectionProcessor, FloodDetectionConfig
# from modules.flood_detection.database import FloodDetectionDatabase

# app = Flask(__name__)
# CORS(app)

# # Verify Supabase credentials are loaded
# print(f"Supabase URL: {'✓ Set' if os.getenv('SUPABASE_URL') else '✗ Missing'}")
# print(f"Supabase Key: {'✓ Set' if os.getenv('SUPABASE_KEY') else '✗ Missing'}")

# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs('outputs', exist_ok=True)

# # ── Routes ────────────────────────────────────────────────────

# @app.route('/health', methods=['GET'])
# def health():
#     """Health check endpoint."""
#     return jsonify({'status': 'ok', 'module': 'flood_detection'})


# @app.route('/process', methods=['POST'])
# def process():
#     """
#     Main processing endpoint.
#     Accepts multipart form with:
#         before_b3, before_b5 : before flood Landsat bands
#         after_b3,  after_b5  : after flood Landsat bands
#         dem                  : merged DEM GeoTIFF
#         event_date           : optional date string
#     """
#     try:
#         # Save uploaded files
#         files_needed = ['before_b3', 'before_b5', 'after_b3', 'after_b5', 'dem']
#         paths = {}

#         for key in files_needed:
#             f = request.files.get(key)
#             if not f:
#                 return jsonify({'error': f'Missing file: {key}'}), 400
#             save_path = os.path.join(UPLOAD_FOLDER, f.filename)
#             f.save(save_path)
#             paths[key] = save_path

       

#         # Run flood detection
#         config    = FloodDetectionConfig()
#         processor = FloodDetectionProcessor(config)

#         # fix
#         event_date = request.form.get('event_date', None)   # ← add this line above

#         results = processor.process(
#             before_b3=paths['before_b3'],
#             before_b5=paths['before_b5'],
#             after_b3 =paths['after_b3'],
#             after_b5 =paths['after_b5'],
#             dem_path =paths['dem'],
#             event_date=event_date,                           # ← pass it in
#         )
#         # Save to Supabase
#         db = FloodDetectionDatabase(config)
#         db.save_results(results['gdf'], results['stats'], event_date)

#         return jsonify({
#             'success':    True,
#             'stats':      results['stats'],
#             'geojson_url': '/geojson',
#             'map_url':    '/map'
#         })

#     except Exception as e:
#         import traceback
#         return jsonify({
#             'error': str(e),
#             'trace': traceback.format_exc()
#         }), 500


# @app.route('/geojson', methods=['GET'])
# def get_geojson(): 
#     """Return the latest flood results GeoJSON."""
#     path = 'outputs/flood_results.geojson'
#     if not os.path.exists(path):
#         return jsonify({'error': 'No results yet. Run /process first.'}), 404
#     return send_file(path, mimetype='application/json')


# @app.route('/results', methods=['GET'])
# def get_results():
#     """
#     Return latest flood results from Supabase.
#     Used by Module 2 (population) and Module 3 (resources).
#     """
#     try:
#         event_date = request.args.get('event_date', None)
#         config     = FloodDetectionConfig()
#         db         = FloodDetectionDatabase(config)
#         data       = db.get_latest_results(event_date)

#         if data is None:
#             return jsonify({'error': 'No results found'}), 404

#         return jsonify({'success': True, 'data': data})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.route('/map', methods=['GET'])
# def get_map():
#     """Return the latest flood map PNG."""
#     path = 'outputs/flood_map.png'
#     if not os.path.exists(path):
#         return jsonify({'error': 'No map yet'}), 404
#     return send_file(path, mimetype='image/png')


# if __name__ == '__main__':
#     print("Starting Flood Detection Backend...")
#     print("Running on http://localhost:5001")
#     app.run(debug=True, host='0.0.0.0', port=5001)

"""
interface/backend/main.py
Flask backend - Supabase only.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.flood_detection import FloodDetectionProcessor, FloodDetectionConfig
from modules.flood_detection.database import FloodDetectionDatabase
from modules.affected_population.live_prediction_endpoint import LivePopulationRiskEndpoint

app = Flask(__name__)
CORS(app)

os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Initialize your ML weights once when the server starts up (Fast runtime memory allocation)
population_risk_engine = LivePopulationRiskEndpoint()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'module': 'flood_detection'})


@app.route('/process', methods=['POST'])
def process():
    try:
        files_needed = ['before_b3', 'before_b5', 'after_b3', 'after_b5', 'dem']
        paths = {}
        
        for key in files_needed:
            f = request.files.get(key)
            if not f:
                return jsonify({'error': f'Missing file: {key}'}), 400
            save_path = os.path.join('uploads', f.filename)
            f.save(save_path)
            paths[key] = save_path
        
        event_date = request.form.get('event_date', None)

        # Capture live rainfall from dashboard form inputs (Defaulting safely to 0.0 if omitted)
        input_precip_mm = float(request.form.get('precip_mm', 0.0))
        
        config = FloodDetectionConfig()
        processor = FloodDetectionProcessor(config)
        
        results = processor.process(
            before_b3=paths['before_b3'],
            before_b5=paths['before_b5'],
            after_b3=paths['after_b3'],
            after_b5=paths['after_b5'],
            dem_path=paths['dem'],
            event_date=event_date,
        )

        # Save Module 1 results to Supabase
        db = FloodDetectionDatabase(config)
        db.save_results(results['gdf'], results['stats'], event_date)

        # ========================================================
        # ── MODULE 2 EXECUTION: COMPUTE DEMOGRAPHIC ESTIMATION ──
        # ========================================================
        # Pass Module 1's generated GeoDataFrame matrix and the rainfall variable
        live_grid_df = results['gdf']
        demographic_payload = population_risk_engine.predict_realtime_demographics(live_grid_df, input_precip_mm)
        
        return jsonify({
            'success': True,
            'stats': results['stats'],
            'geojson_url': '/geojson',
            'map_url': '/map'
            'affected_population_demographics': demographic_payload # ← Your output added to the app payload
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


@app.route('/geojson', methods=['GET'])
def get_geojson():
    path = 'outputs/flood_results.geojson'
    if not os.path.exists(path):
        return jsonify({'error': 'No results yet'}), 404
    return send_file(path, mimetype='application/json')


@app.route('/results', methods=['GET'])
def get_results():
    try:
        event_date = request.args.get('event_date', None)
        config = FloodDetectionConfig()
        db = FloodDetectionDatabase(config)
        data = db.get_latest_results(event_date)
        
        if data is None:
            return jsonify({'error': 'No results found'}), 404
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flood Detection Backend (Supabase only)...")
    print("Running on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)