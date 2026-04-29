import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')


class CausalReliefNetwork:
    
    def __init__(self):
        self.model = None
        self.inference = None
        
        self.causal_edges = [
            ('Flood_Severity', 'Affected_Population'),
            ('Children_Percentage', 'Baby_Formula_Need'),
            ('Children_Percentage', 'Milk_Powder_Need'),
            ('Elderly_Percentage', 'Medical_Need'),
            ('Affected_Population', 'Water_Need'),
            ('Affected_Population', 'Food_Need'),
            ('Affected_Population', 'Sanitation_Need'),
            ('Affected_Population', 'Hygiene_Need'),
            ('Affected_Population', 'Baby_Formula_Need'),
            ('Affected_Population', 'Evacuation_Need'),
            ('Water_Need', 'Sanitation_Need'),
            ('Food_Need', 'Medical_Need'),
        ]
    
    def create_target_variables(self, data):
        relief_mappings = {
            'Water_Need': ('Water Bottles', [0, 10000, 50000, 100000, 200000, float('inf')]),
            'Food_Need': ('Cooked Food Packs', [0, 5000, 20000, 50000, 100000, float('inf')]),
            'Sanitation_Need': ('Sanitary', [0, 500, 1000, 2000, 5000, float('inf')]),
            'Hygiene_Need': ('Soap', [0, 500, 1000, 2000, 5000, float('inf')]),
            'Baby_Formula_Need': ('Infant Milk Powder Packs', [0, 50, 100, 200, 500, float('inf')]),
            'Milk_Powder_Need': ('Milk Powder Packs', [0, 100, 200, 500, 1000, float('inf')]),
            'Medical_Need': ('Noodles Packs', [0, 100, 300, 600, 1000, float('inf')]),
            'Evacuation_Need': ('Affected_Population', [0, 5000, 10000, 20000, 30000, float('inf')])
        }
        
        new_data = data.copy()
        
        for new_col, (source_col, bins) in relief_mappings.items():
            if source_col in new_data.columns:
                labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
                new_data[new_col] = pd.cut(
                    new_data[source_col], bins=bins, labels=labels, include_lowest=True
                )
        
        return new_data
    
    def discretize_data(self, df):
        discretized_df = df.copy()
        
        if 'Affected_Population' in discretized_df.columns:
            bins = [0, 5000, 10000, 20000, 30000, 50000, float('inf')]
            labels = ['0-5k', '5k-10k', '10k-20k', '20k-30k', '30k-50k', '50k+']
            discretized_df['Affected_Population'] = pd.cut(
                discretized_df['Affected_Population'], bins=bins, labels=labels, include_lowest=True
            )
        
        if 'Children_%' in discretized_df.columns:
            bins = [0, 0.15, 0.25, 0.35, 1.0]
            labels = ['Low', 'Medium', 'High', 'Very_High']
            discretized_df['Children_Percentage'] = pd.cut(
                discretized_df['Children_%'], bins=bins, labels=labels, include_lowest=True
            )
        
        if 'Elderly_%' in discretized_df.columns:
            bins = [0, 0.10, 0.15, 0.20, 1.0]
            labels = ['Low', 'Medium', 'High', 'Very_High']
            discretized_df['Elderly_Percentage'] = pd.cut(
                discretized_df['Elderly_%'], bins=bins, labels=labels, include_lowest=True
            )
        
        return discretized_df
    
    def build_network(self, training_data):
        print("=" * 60)
        print("BUILDING CAUSAL BAYESIAN NETWORK")
        print("=" * 60)
        
        prepared_data = training_data.copy()
        
        if 'Flood_Severity' not in prepared_data.columns and 'Severity' in prepared_data.columns:
            prepared_data['Flood_Severity'] = prepared_data['Severity']
        
        prepared_data = self.create_target_variables(prepared_data)
        prepared_data = self.discretize_data(prepared_data)
        
        all_nodes = set()
        for edge in self.causal_edges:
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])
        
        available_nodes = [node for node in all_nodes if node in prepared_data.columns]
        network_data = prepared_data[available_nodes].copy()
        network_data = network_data.dropna()
        
        available_edges = [(u, v) for u, v in self.causal_edges if u in available_nodes and v in available_nodes]
        
        print(f"\nNetwork nodes: {available_nodes}")
        print(f"Training samples: {len(network_data)}")
        
        self.model = DiscreteBayesianNetwork(available_edges)
        
        try:
            self.model.fit(network_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)
            print("\nModel trained using Bayesian Estimator")
        except Exception as e:
            print(f"\nBayesian Estimator failed: {e}")
            self.model.fit(network_data, estimator=MaximumLikelihoodEstimator)
            print("Model trained using Maximum Likelihood Estimator")
        
        self.inference = VariableElimination(self.model)
        
        print(f"\nNetwork Statistics:")
        print(f"  - Nodes: {self.model.number_of_nodes()}")
        print(f"  - Edges: {self.model.number_of_edges()}")
        
        return self.model
    
    def predict_relief_needs(self, affected_population, children_pct, elderly_pct, flood_severity='Medium'):
        if self.model is None:
            print("Network not built yet")
            return None
        
        # Categorize
        if affected_population < 5000:
            pop_cat = '0-5k'
        elif affected_population < 10000:
            pop_cat = '5k-10k'
        elif affected_population < 20000:
            pop_cat = '10k-20k'
        elif affected_population < 30000:
            pop_cat = '20k-30k'
        elif affected_population < 50000:
            pop_cat = '30k-50k'
        else:
            pop_cat = '50k+'
        
        if children_pct <= 0.25:
            children_cat = 'Medium'
        else:
            children_cat = 'High'
        
        if elderly_pct <= 0.15:
            elderly_cat = 'Medium'
        else:
            elderly_cat = 'High'
        
        if flood_severity not in ['Low', 'Medium', 'High']:
            flood_severity = 'Medium'
        
        # Evidence
        evidence = {}
        if 'Affected_Population' in self.model.nodes():
            evidence['Affected_Population'] = pop_cat
        if 'Flood_Severity' in self.model.nodes():
            evidence['Flood_Severity'] = flood_severity
        if 'Children_Percentage' in self.model.nodes():
            evidence['Children_Percentage'] = children_cat
        if 'Elderly_Percentage' in self.model.nodes():
            evidence['Elderly_Percentage'] = elderly_cat
        
        query_vars = ['Water_Need', 'Food_Need', 'Sanitation_Need', 'Hygiene_Need', 'Baby_Formula_Need', 'Evacuation_Need']
        query_vars = [v for v in query_vars if v in self.model.nodes()]
        
        print(f"\nEvidence: {evidence}")
        print(f"Query: {query_vars}")
        
        # Query one variable at a time
        predictions = {}
        
        for var in query_vars:
            try:
                result = self.inference.query(variables=[var], evidence=evidence)
                
                # Get the factor
                if hasattr(result, 'values'):
                    factor = result
                elif isinstance(result, list) and len(result) > 0:
                    factor = result[0]
                else:
                    print(f"  Unexpected result type for {var}")
                    continue
                
                # Get values and flatten
                values = factor.values
                if values.ndim > 1:
                    values = values.flatten()
                
                max_idx = np.argmax(values)
                probability = float(values[max_idx])
                
                # Get state names
                try:
                    cpd = self.model.get_cpds(var)
                    state_names = cpd.state_names[var]
                except:
                    state_names = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
                
                most_likely = state_names[max_idx] if max_idx < len(state_names) else 'Medium'
                
                predictions[var] = {
                    'most_likely': most_likely,
                    'probability': probability,
                    'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
                }
                
                print(f"  {var}: {most_likely} ({probability:.2f})")
                
            except Exception as e:
                print(f"  Could not query {var}: {e}")
                continue
        
        return predictions if predictions else None
    
    def get_explainable_recommendation(self, input_data, predictions):
        explanation = {
            'summary': "",
            'reasoning_steps': [],
            'recommendations': [],
            'priority_level': "Medium"
        }
        
        explanation['reasoning_steps'].append(
            f"Area has {input_data['affected_population']:,} affected people, "
            f"with {input_data['children_pct']*100:.1f}% children and {input_data['elderly_pct']*100:.1f}% elderly."
        )
        
        if input_data['flood_severity'] == 'High':
            explanation['reasoning_steps'].append("HIGH severity - substantial relief required immediately!")
        elif input_data['flood_severity'] == 'Medium':
            explanation['reasoning_steps'].append("Medium severity - moderate relief needed.")
        else:
            explanation['reasoning_steps'].append("Low severity - basic relief sufficient.")
        
        if predictions:
            for var, pred in predictions.items():
                if var == 'Water_Need':
                    explanation['reasoning_steps'].append(
                        f"Water: {pred['most_likely']} priority ({pred['probability']*100:.1f}% confidence)"
                    )
                    if pred['most_likely'] in ['High', 'Very_High']:
                        explanation['recommendations'].append("Deploy clean water immediately (3L/person/day)")
                        explanation['priority_level'] = 'High'
                
                elif var == 'Food_Need':
                    explanation['reasoning_steps'].append(f"Food: {pred['most_likely']} priority")
                    if pred['most_likely'] in ['High', 'Very_High']:
                        explanation['recommendations'].append("Distribute ready-to-eat food packs")
                
                elif var == 'Sanitation_Need':
                    explanation['reasoning_steps'].append(f"Sanitary pads: {pred['most_likely']} priority")
                    if pred['most_likely'] in ['High', 'Very_High']:
                        explanation['recommendations'].append("Deploy sanitary pads immediately")
                
                elif var == 'Hygiene_Need':
                    explanation['reasoning_steps'].append(f"Hygiene kits: {pred['most_likely']} priority")
                    if pred['most_likely'] in ['High', 'Very_High']:
                        explanation['recommendations'].append("Distribute soap, toothpaste, toothbrushes")
        
        explanation['summary'] = f"Priority: {explanation['priority_level']} level relief deployment needed."
        
        return explanation


if __name__ == "__main__":
    from data_preprocessing import ReliefDataPreprocessor
    
    print("=" * 60)
    print("TESTING CAUSAL BAYESIAN NETWORK")
    print("=" * 60)
    
    preprocessor = ReliefDataPreprocessor("../../data/Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx")
    X_train, X_test, y_train, y_test, full_data = preprocessor.run_pipeline(test_year=2025, scale=False)
    
    causal_network = CausalReliefNetwork()
    causal_network.build_network(full_data)
    
    print("\n" + "=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)
    
    predictions = causal_network.predict_relief_needs(
        affected_population=42105,
        children_pct=0.28,
        elderly_pct=0.13,
        flood_severity='High'
    )
    
    if predictions:
        print("\nPrediction Results Summary:")
        for var, pred in predictions.items():
            print(f"   {var}: {pred['most_likely']} (prob: {pred['probability']:.2f}, {pred['confidence']})")
    else:
        print("\nNo predictions returned")
    
    explanation = causal_network.get_explainable_recommendation(
        input_data={'affected_population': 42105, 'children_pct': 0.28, 'elderly_pct': 0.13, 'flood_severity': 'High'},
        predictions=predictions
    )
    
    print("\nEXPLANATION:")
    print(f"   {explanation['summary']}")
    print("\n   Reasoning:")
    for step in explanation['reasoning_steps']:
        print(f"     • {step}")
    print("\n   Recommendations:")
    for rec in explanation['recommendations']:
        print(f"     → {rec}")
    
    print("\nTEST COMPLETE!")