"""
Flask API Routes for Module 3
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_preprocessing import ReliefDataPreprocessor
from models.causal_network import CausalReliefNetwork
from models.relief_predictor import ReliefPredictor
from database import db

api_bp = Blueprint('api', __name__)

preprocessor = None
predictor = None
causal_network = None


def initialize_module3(data_path):
    global preprocessor, predictor, causal_network
    print("\nINITIALIZING MODULE 3")
    preprocessor = ReliefDataPreprocessor(data_path)
    X_train, X_test, y_train, y_test, full_data = preprocessor.run_pipeline(test_year=2025, scale=False)
    predictor = ReliefPredictor()
    predictor.train_models(X_train, y_train, X_test, y_test)
    causal_network = CausalReliefNetwork()
    causal_network.build_network(full_data)
    predictor.set_causal_network(causal_network)
    print("MODULE 3 READY")
    return True


@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'module': 'Module 3',
        'port': 5002,
        'timestamp': datetime.now().isoformat()
    })


@api_bp.route('/predict/<division_name>', methods=['GET'])
def predict_for_division(division_name):
    try:
        print(f"\nProcessing: {division_name}")
        
        # STEP 1: Get flood severity from Module 1
        flood_severity = db.get_flood_severity(division_name)
        print(f"   Severity: {flood_severity}")
        
        # STEP 2: Get population data
        pop_data = db.get_population_data(division_name)
        print(f"   Population: {pop_data['affected_population']}")
        
        # STEP 3: Get ML predictions
        ml_result = predictor.predict_with_analysis(
            affected_population=pop_data['affected_population'],
            children_pct=pop_data['children_pct'],
            elderly_pct=pop_data['elderly_pct'],
            female_pct=pop_data['female_pct'],
            flood_severity=flood_severity
        )
        
        # STEP 4: Get causal explanation
        causal_predictions = causal_network.predict_relief_needs(
            affected_population=pop_data['affected_population'],
            children_pct=pop_data['children_pct'],
            elderly_pct=pop_data['elderly_pct'],
            female_pct=pop_data['female_pct'],
            flood_severity=flood_severity
        )
        
        explanation = causal_network.get_explainable_recommendation(
            input_data=ml_result['input'],
            predictions=causal_predictions,
            ml_predictions=ml_result['predictions']  # Pass with priority (internal use)
        )
        
        # STEP 5: Prepare input data for database
        input_data_for_db = {
            'affected_population': pop_data['affected_population'],
            'children_percentage': round(pop_data['children_pct'] * 100, 1),
            'elderly_percentage': round(pop_data['elderly_pct'] * 100, 1),
            'female_percentage': round(pop_data['female_pct'] * 100, 1),
            'flood_severity': flood_severity
        }
        
        # STEP 6: Save to Supabase
        db.save_prediction(
            division_name=division_name,
            input_data=input_data_for_db,
            predictions=ml_result['predictions'],
            overall_priority=ml_result['overall_priority'],
            explanation=explanation['summary']
        )
        
        # STEP 7: Return response
        # This keeps priority for internal use (causal network, database)
        cleaned_predictions = {}
        for item, details in ml_result['predictions'].items():
            cleaned_predictions[item] = {
                'quantity': details['quantity']   # Only quantity, no priority
            }
        
        response = {
            'success': True,
            'division': division_name,
            'input_data': input_data_for_db,
            'relief_predictions': cleaned_predictions,
            'overall_priority': ml_result['overall_priority'],
            'explanation': explanation['summary'],
            'drivers': explanation.get('drivers', [])
        }
        
        print(f"Success for {division_name}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/divisions', methods=['GET'])
def get_divisions():
    return jsonify({'success': True, 'divisions': db.get_division_list()})