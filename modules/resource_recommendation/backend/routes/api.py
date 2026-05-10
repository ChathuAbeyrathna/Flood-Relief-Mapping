from flask import Blueprint, request, jsonify
from datetime import datetime
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_preprocessing import ReliefDataPreprocessor
from models.causal_network import CausalReliefNetwork
from models.relief_predictor import ReliefPredictor
from database import get_flood_results, get_latest_summary

api_bp = Blueprint('api', __name__)

preprocessor = None
predictor = None
causal_network = None


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def initialize_module3(data_path):
    global preprocessor, predictor, causal_network

    print("\n" + "=" * 50)
    print("🚀 INITIALIZING MODULE 3")
    print("=" * 50)

    preprocessor = ReliefDataPreprocessor(data_path)
    X_train, X_test, y_train, y_test, full_data = preprocessor.run_pipeline(test_year=2025, scale=False)

    predictor = ReliefPredictor()
    predictor.train_models(X_train, y_train, X_test, y_test)

    causal_network = CausalReliefNetwork()
    causal_network.build_network(full_data)
    predictor.set_causal_network(causal_network)

    print("✅ MODULE 3 READY")
    return True


@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'module': 'Module 3', 'timestamp': datetime.now().isoformat()})


@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        affected_population = data.get('affected_population')
        children_pct = data.get('children_percentage', 25) / 100
        elderly_pct = data.get('elderly_percentage', 15) / 100
        flood_severity = data.get('flood_severity', 'Medium')
        division_name = data.get('division_name', 'Unknown')

        if affected_population is None:
            return jsonify({'error': 'affected_population required'}), 400

        # Get ML predictions
        ml_result = predictor.predict_with_analysis(
            affected_population=affected_population,
            children_pct=children_pct,
            elderly_pct=elderly_pct,
            flood_severity=flood_severity
        )

        # Get causal explanation
        causal_predictions = causal_network.predict_relief_needs(
            affected_population=affected_population,
            children_pct=children_pct,
            elderly_pct=elderly_pct,
            flood_severity=flood_severity
        )

        causal_explanation = causal_network.get_explainable_recommendation(
            input_data={
                'affected_population': affected_population,
                'children_pct': children_pct,
                'elderly_pct': elderly_pct,
                'flood_severity': flood_severity
            },
            predictions=causal_predictions
        )

        # Build CLEAN output format
        relief_list = []
        for item, details in ml_result['predictions'].items():
            relief_list.append({
                "item": item,
                "quantity": details['quantity']
            })

        response_data = convert_to_serializable({
            "success": True,
            "division": division_name,
            "overall_priority": ml_result['overall_priority'],
            "relief_items": relief_list,
            "explanation": {
                "summary": causal_explanation.get('summary', ''),
                "reasoning": causal_explanation.get('reasoning_steps', []),
                "recommendations": causal_explanation.get('recommendations', [])
            }
        })

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/divisions', methods=['GET'])
def get_divisions():
    try:
        divisions = preprocessor.get_division_list()
        response_data = convert_to_serializable({'success': True, 'divisions': divisions})
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/division/<name>', methods=['GET'])
def get_division_stats(name):
    try:
        stats = preprocessor.get_statistics_by_division(name)
        if stats is None:
            return jsonify({'error': 'Division not found'}), 404
        response_data = convert_to_serializable({'success': True, 'statistics': stats})
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500