"""
Traditional CBN - Single Input to Single Output (Forecast-based)
"""

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')


class TraditionalCBN:
    """
    Traditional CBN: Flood Severity → Evacuation Need only
    (Forecast-based, single input, single output)
    """
    
    def __init__(self):
        self.model = None
        self.inference = None
        
        # Only ONE relationship: Severity → Evacuation Need
        self.causal_edges = [
            ('Flood_Severity', 'Evacuation_Need')  # Single input → Single output
        ]
    
    def prepare_data(self, data):
        """Prepare data for traditional CBN"""
        prepared_data = data.copy()
        
        # Discretize Flood Severity
        if 'Flood_Severity' not in prepared_data.columns and 'Severity' in prepared_data.columns:
            prepared_data['Flood_Severity'] = prepared_data['Severity']
        
        # Create Evacuation Need based on flood severity (simple mapping)
        # If Severity is High → Evacuation Need = High
        # If Severity is Medium → Evacuation Need = Medium
        # If Severity is Low → Evacuation Need = Low
        severity_to_evacuation = {
            'High': 'High',
            'Medium': 'Medium',
            'Low': 'Low'
        }
        
        if 'Severity' in prepared_data.columns:
            prepared_data['Evacuation_Need'] = prepared_data['Severity'].map(severity_to_evacuation)
        else:
            # Fallback if no severity column
            prepared_data['Evacuation_Need'] = 'Medium'
        
        # Discretize Flood Severity to numeric (for pgmpy)
        if 'Flood_Severity' in prepared_data.columns:
            severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Severe': 4}
            prepared_data['Flood_Severity'] = prepared_data['Flood_Severity'].map(severity_map)
        
        # Discretize Evacuation Need
        if 'Evacuation_Need' in prepared_data.columns:
            evac_map = {'Low': 1, 'Medium': 2, 'High': 3}
            prepared_data['Evacuation_Need'] = prepared_data['Evacuation_Need'].map(evac_map)
        
        return prepared_data
    
    def build_network(self, training_data):
        """Build traditional CBN"""
        print("\n" + "=" * 60)
        print("BUILDING TRADITIONAL CBN (Single Input → Single Output)")
        print("=" * 60)
        print("   Input:  Flood Severity")
        print("   Output: Evacuation Need")
        print("=" * 60)
        
        prepared_data = self.prepare_data(training_data)
        
        # Select only the two variables
        available_nodes = ['Flood_Severity', 'Evacuation_Need']
        network_data = prepared_data[available_nodes].copy().dropna()
        
        print(f"\nNetwork nodes: {available_nodes}")
        print(f"Training samples: {len(network_data)}")
        print(f"Relationship: Flood_Severity → Evacuation_Need")
        
        # Build model
        self.model = DiscreteBayesianNetwork(self.causal_edges)
        
        try:
            self.model.fit(network_data, estimator=MaximumLikelihoodEstimator)
            print("\nTraditional CBN trained using MLE")
        except Exception as e:
            print(f"\nTraining failed: {e}")
            return None
        
        self.inference = VariableElimination(self.model)
        
        print(f"\nNetwork Statistics:")
        print(f"  - Nodes: {self.model.number_of_nodes()}")
        print(f"  - Edges: {self.model.number_of_edges()}")
        
        return self.model
    
    def predict(self, flood_severity):
        """Predict evacuation need based on severity only"""
        if self.model is None:
            return None
        
        # Map severity to numeric
        severity_map = {'Low': 1, 'Medium': 2, 'High': 3}
        severity_value = severity_map.get(flood_severity, 2)
        
        evidence = {'Flood_Severity': severity_value}
        
        try:
            result = self.inference.query(variables=['Evacuation_Need'], evidence=evidence)
            values = result.values.flatten()
            max_idx = np.argmax(values)
            
            # Map back to labels
            state_names = ['Low', 'Medium', 'High']
            most_likely = state_names[max_idx]
            probability = float(values[max_idx])
            
            return {
                'evacuation_need': most_likely,
                'probability': probability,
                'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
            }
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None