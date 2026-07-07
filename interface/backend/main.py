# """
# interface/backend/main.py
# Flask backend - Supabase only.
# """

# import os
# import sys
# from pathlib import Path
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from dotenv import load_dotenv

# # Load .env
# load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

# # Add project root to path
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# from modules.flood_detection import FloodDetectionProcessor, FloodDetectionConfig
# from modules.flood_detection.database import FloodDetectionDatabase

# app = Flask(__name__)
# CORS(app)

# os.makedirs('uploads', exist_ok=True)
# os.makedirs('outputs', exist_ok=True)


# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'ok', 'module': 'flood_detection'})


# @app.route('/process', methods=['POST'])
# def process():
#     try:
#         files_needed = ['before_b3', 'before_b5', 'after_b3', 'after_b5', 'dem']
#         paths = {}
        
#         for key in files_needed:
#             f = request.files.get(key)
#             if not f:
#                 return jsonify({'error': f'Missing file: {key}'}), 400
#             save_path = os.path.join('uploads', f.filename)
#             f.save(save_path)
#             paths[key] = save_path
        
#         event_date = request.form.get('event_date', None)
        
#         config = FloodDetectionConfig()
#         processor = FloodDetectionProcessor(config)
        
#         results = processor.process(
#             before_b3=paths['before_b3'],
#             before_b5=paths['before_b5'],
#             after_b3=paths['after_b3'],
#             after_b5=paths['after_b5'],
#             dem_path=paths['dem'],
#             event_date=event_date,
#         )
        
#         db = FloodDetectionDatabase(config)
#         db.save_results(results['gdf'], results['stats'], event_date)
        
#         return jsonify({
#             'success': True,
#             'stats': results['stats'],
#             'geojson_url': '/geojson',
#             'map_url': '/map'
#         })
        
#     except Exception as e:
#         import traceback
#         return jsonify({
#             'error': str(e),
#             'trace': traceback.format_exc()
#         }), 500


# @app.route('/geojson', methods=['GET'])
# def get_geojson():
#     path = 'outputs/flood_results.geojson'
#     if not os.path.exists(path):
#         return jsonify({'error': 'No results yet'}), 404
#     return send_file(path, mimetype='application/json')


# @app.route('/results', methods=['GET'])
# def get_results():
#     try:
#         event_date = request.args.get('event_date', None)
#         config = FloodDetectionConfig()
#         db = FloodDetectionDatabase(config)
#         data = db.get_latest_results(event_date)
        
#         if data is None:
#             return jsonify({'error': 'No results found'}), 404
        
#         return jsonify({'success': True, 'data': data})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     print("Starting Flood Detection Backend (Supabase only)...")
#     print("Running on http://localhost:5001")
#     app.run(debug=True, host='0.0.0.0', port=5001)

"""
Unified Flood Relief Backend
All three modules integrated into a single Flask app.
Team Trivia · University of Moratuwa · 2026
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# ─── SETUP PATHS ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load .env from project root
load_dotenv(PROJECT_ROOT / '.env')

# ─── MODULE 1 IMPORTS ────────────────────────────────────
from modules.flood_detection import FloodDetectionProcessor, FloodDetectionConfig
from modules.flood_detection.database import FloodDetectionDatabase

# ─── MODULE 2 IMPORTS ────────────────────────────────────
from modules.affected_population.live_prediction_endpoint import LivePopulationRiskEndpoint

# ─── MODULE 3 IMPORTS ────────────────────────────────────
module3_path = PROJECT_ROOT / 'modules' / 'resource_recommendation' / 'backend'
sys.path.insert(0, str(module3_path))

from routes.api import api_bp, initialize_module3

# ─── APP INITIALIZATION ──────────────────────────────────
app = Flask(__name__)
CORS(app)

# Register Module 3's API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# ─── INITIALIZE MODULE 2 ─────────────────────────────────
print("🔄 Loading Module 2 - Population Risk Engine...")
try:
    risk_engine = LivePopulationRiskEndpoint()
    print("✅ Module 2 loaded successfully!")
except Exception as e:
    print(f"❌ Module 2 failed to load: {e}")
    risk_engine = None

# ─── INITIALIZE MODULE 3 ─────────────────────────────────
DATA_PATH = PROJECT_ROOT / 'modules' / 'resource_recommendation' / 'data' / 'Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx'
print(f"📊 Module 3 data path: {DATA_PATH}")

if DATA_PATH.exists():
    try:
        initialize_module3(str(DATA_PATH))
        print("✅ Module 3 loaded successfully!")
    except Exception as e:
        print(f"⚠️ Module 3 initialization warning: {e}")
else:
    print(f"⚠️ Module 3 data file not found at: {DATA_PATH}")

# ─── CREATE FOLDERS ──────────────────────────────────────
os.makedirs(PROJECT_ROOT / 'uploads', exist_ok=True)
os.makedirs(PROJECT_ROOT / 'outputs', exist_ok=True)
os.makedirs(PROJECT_ROOT / 'interface' / 'backend' / 'uploads', exist_ok=True)
os.makedirs(PROJECT_ROOT / 'interface' / 'backend' / 'outputs', exist_ok=True)

# ─── STORE LATEST RESULTS ────────────────────────────────
latest_results = {
    'flood': None,
    'population': None,
    'geojson': None
}

# ═══════════════════════════════════════════════════════════
#  SUPABASE HELPERS
# ═══════════════════════════════════════════════════════════

def save_population_to_supabase(population_data, event_date=None):
    """Save Module 2 population predictions to Supabase."""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            print("⚠️ Supabase credentials not found, skipping population save")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        demographic_data = population_data.get('demographic_data', [])
        
        for div_data in demographic_data:
            record = {
                'division_name': div_data['division_name'],
                'predicted_mean_affected': div_data['summary_metrics']['predicted_mean_affected'],
                'upper_risk_limit': div_data['summary_metrics']['conservative_upper_risk_limit'],
                'male_count': div_data['gender_demographics']['male_count'],
                'female_count': div_data['gender_demographics']['female_count'],
                'children_count': div_data['age_demographics']['children_count_0_14'],
                'adult_count': div_data['age_demographics']['adult_count_15_59'],
                'elderly_count': div_data['age_demographics']['elderly_count_60_plus'],
                'upper_risk_male': div_data['gender_demographics']['upper_risk_male_count'],
                'upper_risk_female': div_data['gender_demographics']['upper_risk_female_count'],
                'upper_risk_children': div_data['age_demographics']['upper_risk_children_count'],
                'upper_risk_elderly': div_data['age_demographics']['upper_risk_elderly_count'],
                'event_date': event_date,
                'spatial_scope': population_data.get('spatial_scope', 'Gampaha District'),
                'evaluation_year': population_data.get('evaluation_year', 2025)
            }
            
            # Upsert - update if exists, insert if new
            supabase.table('population_predictions').upsert(
                record, 
                on_conflict='division_name,event_date'
            ).execute()
        
        print(f"✅ Population data saved to Supabase: {len(demographic_data)} divisions")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to save population to Supabase: {e}")
        # Also save locally as backup
        save_population_to_file(population_data, event_date)
        return False


def save_population_to_file(population_data, event_date=None):
    """Backup: Save population results to JSON file."""
    try:
        output_path = PROJECT_ROOT / 'outputs' / 'population_results.json'
        data_to_save = {
            'event_date': event_date,
            'population_data': population_data,
            'saved_at': str(datetime.datetime.now())
        }
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        print(f"💾 Population data backup saved to: {output_path}")
    except Exception as e:
        print(f"⚠️ Failed to save population backup: {e}")


# ═══════════════════════════════════════════════════════════
#  MAIN ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check for all modules."""
    return jsonify({
        'status': 'ok',
        'modules': {
            'module1_flood_detection': 'online',
            'module2_population': 'online' if risk_engine else 'offline',
            'module3_relief': 'online' if DATA_PATH.exists() else 'data_missing'
        },
        'project_root': str(PROJECT_ROOT)
    })


@app.route('/process', methods=['POST'])
def process():
    """
    Main processing endpoint - runs Module 1 (flood) + Module 2 (population).
    """
    try:
        # ─── GET UPLOADED FILES ───────────────────────────
        files_needed = ['before_b3', 'before_b5', 'after_b3', 'after_b5', 'dem']
        paths = {}
        
        upload_dir = PROJECT_ROOT / 'interface' / 'backend' / 'uploads'
        
        for key in files_needed:
            f = request.files.get(key)
            if not f:
                return jsonify({'error': f'Missing file: {key}'}), 400
            save_path = str(upload_dir / f.filename)
            f.save(save_path)
            paths[key] = save_path
        
        # ─── GET FORM DATA ────────────────────────────────
        event_date = request.form.get('event_date', None)
        input_precip_mm = float(request.form.get('precip_mm', 0.0))
        
        # ─── MODULE 1: FLOOD DETECTION ────────────────────
        print("🌊 Running Module 1: Flood Detection...")
        config = FloodDetectionConfig()
        processor = FloodDetectionProcessor(config)
        
        flood_results = processor.process(
            before_b3=paths['before_b3'],
            before_b5=paths['before_b5'],
            after_b3=paths['after_b3'],
            after_b5=paths['after_b5'],
            dem_path=paths['dem'],
            event_date=event_date,
        )
        print("✅ Module 1 complete!")
        
        # Save Module 1 results to Supabase
        db = FloodDetectionDatabase(config)
        db.save_results(flood_results['gdf'], flood_results['stats'], event_date)
        
        # ─── MODULE 2: POPULATION PREDICTION ──────────────
        population_results = None
        
        if risk_engine:
            try:
                print("👥 Running Module 2: Population Prediction...")
                live_grid_df = flood_results['gdf'].copy()
                
                # Ensure Ds_Division_Name exists
                if 'Ds_Division_Name' not in live_grid_df.columns:
                    live_grid_df['Ds_Division_Name'] = live_grid_df.get(
                        'adm3_name', 
                        live_grid_df.get('ds_division', 'Unknown')
                    )
                
                # Run prediction
                population_results = risk_engine.predict_realtime_demographics(
                    live_grid_df=live_grid_df, 
                    input_precip_mm=input_precip_mm
                )
                
                if population_results.get('status') == 'SUCCESS':
                    total = sum(d['summary_metrics']['predicted_mean_affected'] 
                               for d in population_results['demographic_data'])
                    print(f"✅ Module 2 complete! Total affected: {total:,}")
                    
                    # ✅ SAVE TO SUPABASE
                    save_population_to_supabase(population_results, event_date)
                    
            except Exception as pop_error:
                import traceback
                print(f"⚠️ Module 2 error: {pop_error}")
                print(traceback.format_exc())
                population_results = {"status": "FAILED", "error": str(pop_error)}
        else:
            population_results = {"status": "UNAVAILABLE", "error": "Module 2 not loaded"}
        
        # Store in memory
        latest_results['flood'] = flood_results
        latest_results['population'] = population_results
        
        # ─── RETURN COMBINED RESULTS ──────────────────────
        return jsonify({
            'success': True,
            'module1': {
                'stats': flood_results['stats'],
                'geojson_url': '/geojson',
                'map_url': '/map'
            },
            'module2': population_results,
            'module3_ready': DATA_PATH.exists()
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


@app.route('/geojson', methods=['GET'])
def get_geojson():
    """Return flood results GeoJSON."""
    output_dir = PROJECT_ROOT / 'interface' / 'backend' / 'outputs'
    path = output_dir / 'flood_results.geojson'
    
    if not path.exists():
        path = PROJECT_ROOT / 'outputs' / 'flood_results.geojson'
    
    if not path.exists():
        return jsonify({'error': 'No results yet. Run /process first.'}), 404
    
    return send_file(str(path), mimetype='application/json')


@app.route('/map', methods=['GET'])
def get_map():
    """Return flood map PNG."""
    output_dir = PROJECT_ROOT / 'interface' / 'backend' / 'outputs'
    path = output_dir / 'flood_map.png'
    
    if not path.exists():
        path = PROJECT_ROOT / 'outputs' / 'flood_map.png'
    
    if not path.exists():
        return jsonify({'error': 'No map yet'}), 404
    
    return send_file(str(path), mimetype='image/png')


@app.route('/results', methods=['GET'])
def get_results():
    """Get combined results from database."""
    try:
        event_date = request.args.get('event_date', None)
        config = FloodDetectionConfig()
        db = FloodDetectionDatabase(config)
        flood_data = db.get_latest_results(event_date)
        
        return jsonify({
            'success': True,
            'flood_data': flood_data,
            'population_data': latest_results.get('population')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
#  MODULE 2 ROUTES (Population)
# ═══════════════════════════════════════════════════════════

@app.route('/population', methods=['POST'])
def get_population():
    """Get population predictions from GeoJSON."""
    if not risk_engine:
        return jsonify({'status': 'ERROR', 'error': 'Module 2 not loaded'}), 503
    
    try:
        import pandas as pd
        
        data = request.get_json()
        geojson_data = data.get('geojson', {})
        precipitation = float(data.get('precipitation', 150))
        
        # Convert GeoJSON to DataFrame
        features = []
        for feature in geojson_data.get('features', []):
            props = feature.get('properties', {})
            features.append(props)
        
        live_grid_df = pd.DataFrame(features)
        
        if 'Ds_Division_Name' not in live_grid_df.columns:
            live_grid_df['Ds_Division_Name'] = live_grid_df.get(
                'adm3_name', 
                live_grid_df.get('ds_division', 'Unknown')
            )
        
        population_results = risk_engine.predict_realtime_demographics(
            live_grid_df=live_grid_df, 
            input_precip_mm=precipitation
        )
        
        latest_results['population'] = population_results
        
        # Also save to Supabase if successful
        if population_results.get('status') == 'SUCCESS':
            save_population_to_supabase(population_results)
        
        return jsonify(population_results)
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


@app.route('/population/divisions', methods=['GET'])
def get_population_divisions():
    """Get population data for all divisions from Supabase."""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            # Fallback to memory
            if latest_results.get('population'):
                return jsonify(latest_results['population'])
            return jsonify({'status': 'NO_DATA', 'error': 'No Supabase connection'}), 503
        
        supabase = create_client(supabase_url, supabase_key)
        response = supabase.table('population_predictions').select('*').execute()
        
        return jsonify({
            'status': 'SUCCESS',
            'data': response.data
        })
        
    except Exception as e:
        return jsonify({'status': 'ERROR', 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
#  MODULE 3 ROUTES (Relief)
#  Already registered via api_bp at /api/*
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
#  START SERVER
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import datetime
    
    print("\n" + "=" * 60)
    print("🌊 FLOOD RELIEF MANAGEMENT SYSTEM")
    print("=" * 60)
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print("=" * 60)
    print(f"🔹 Module 1: Flood Detection      ✅")
    print(f"🔹 Module 2: Population Impact    {'✅' if risk_engine else '❌'}")
    print(f"🔹 Module 3: Relief Recommend     {'✅' if DATA_PATH.exists() else '❌'}")
    print("=" * 60)
    print(f"🌐 Server: http://localhost:5001")
    print(f"📍 Health:  http://localhost:5001/health")
    print(f"📍 Process: POST http://localhost:5001/process")
    print(f"📍 GeoJSON: http://localhost:5001/geojson")
    print(f"📍 Population: http://localhost:5001/population")
    print(f"📍 Relief:  http://localhost:5001/api/predict/Gampaha")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)