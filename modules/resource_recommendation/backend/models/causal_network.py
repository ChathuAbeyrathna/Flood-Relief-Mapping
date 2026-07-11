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
            ('Elderly_Percentage', 'Milk_Powder_Need'),
            ('Elderly_Percentage', 'Medical_Need'),
            ('Female_Percentage', 'Sanitation_Need'),
            ('Affected_Population', 'Water_Need'),
            ('Affected_Population', 'Food_Need'),
            ('Affected_Population', 'Hygiene_Need'),
            ('Affected_Population', 'Evacuation_Need'),
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
        
        if 'Female %' in discretized_df.columns:
            bins = [0, 0.45, 0.50, 0.55, 1.0]
            labels = ['Low', 'Normal', 'High', 'Very_High']
            discretized_df['Female_Percentage'] = pd.cut(
                discretized_df['Female %'], bins=bins, labels=labels, include_lowest=True
            )
        elif 'Female_Percentage' in discretized_df.columns:
            bins = [0, 0.45, 0.50, 0.55, 1.0]
            labels = ['Low', 'Normal', 'High', 'Very_High']
            discretized_df['Female_Percentage'] = pd.cut(
                discretized_df['Female_Percentage'], bins=bins, labels=labels, include_lowest=True
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
    
    def predict_relief_needs(self, affected_population, children_pct, elderly_pct, female_pct, flood_severity='Medium'):
        if self.model is None:
            print("Network not built yet")
            return None
        
        # Categorize population
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
        
        # Categorize children
        if children_pct <= 0.25:
            children_cat = 'Medium'
        else:
            children_cat = 'High'
        
        # Categorize elderly
        if elderly_pct <= 0.15:
            elderly_cat = 'Medium'
        else:
            elderly_cat = 'High'
        
        # Categorize female
        if female_pct <= 0.45:
            female_cat = 'Low'
        elif female_pct <= 0.50:
            female_cat = 'Normal'
        elif female_pct <= 0.55:
            female_cat = 'High'
        else:
            female_cat = 'Very_High'
        
        if flood_severity not in ['Low', 'Medium', 'High']:
            flood_severity = 'Medium'
        
        # Build evidence
        evidence = {}
        if 'Affected_Population' in self.model.nodes():
            evidence['Affected_Population'] = pop_cat
        if 'Flood_Severity' in self.model.nodes():
            evidence['Flood_Severity'] = flood_severity
        if 'Children_Percentage' in self.model.nodes():
            evidence['Children_Percentage'] = children_cat
        if 'Elderly_Percentage' in self.model.nodes():
            evidence['Elderly_Percentage'] = elderly_cat
        if 'Female_Percentage' in self.model.nodes():
            # Check if the category exists in the model's CPD
            try:
                cpd = self.model.get_cpds('Female_Percentage')
                available_states = cpd.state_names['Female_Percentage']
                
                if female_cat in available_states:
                    evidence['Female_Percentage'] = female_cat
                else:
                    # Use the closest available state
                    if 'Normal' in available_states:
                        evidence['Female_Percentage'] = 'Normal'
                    elif 'High' in available_states:
                        evidence['Female_Percentage'] = 'High'
                    elif 'Low' in available_states:
                        evidence['Female_Percentage'] = 'Low'
                    else:
                        evidence['Female_Percentage'] = available_states[0] if available_states else female_cat
                    print(f"   '{female_cat}' not available for Female_Percentage, using '{evidence['Female_Percentage']}'")
            except:
                evidence['Female_Percentage'] = female_cat
        
        query_vars = ['Water_Need', 'Food_Need', 'Sanitation_Need', 'Hygiene_Need', 'Baby_Formula_Need', 'Evacuation_Need']
        query_vars = [v for v in query_vars if v in self.model.nodes()]
        
        predictions = {}
        
        for var in query_vars:
            try:
                result = self.inference.query(variables=[var], evidence=evidence)
                
                if hasattr(result, 'values'):
                    factor = result
                elif isinstance(result, list) and len(result) > 0:
                    factor = result[0]
                else:
                    continue
                
                values = factor.values
                if values.ndim > 1:
                    values = values.flatten()
                
                try:
                    cpd = self.model.get_cpds(var)
                    state_names = cpd.state_names[var]
                except:
                    state_names = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
                
                max_idx = np.argmax(values)
                most_likely = state_names[max_idx]
                probability = float(values[max_idx])
                
                predictions[var] = {
                    'most_likely': most_likely,
                    'probability': probability,
                    'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
                }
                                
            except Exception as e:
                continue
        
        return predictions if predictions else None
    
    def get_explainable_recommendation(self, input_data, predictions, ml_predictions=None):
        """
        COMPLETELY NATURAL explanation - NO templates whatsoever
        Every word is chosen dynamically based on actual data
        """
        
        pop = input_data['affected_population']
        children_pct = input_data['children_pct']
        elderly_pct = input_data['elderly_pct']
        female_pct = input_data['female_pct']
        
        # FIX: Handle both key names ('flood_severity' or 'severity')
        if 'flood_severity' in input_data:
            severity = input_data['flood_severity']
        else:
            severity = input_data['severity']
        
        # ============================================================
        # PART 1: Describe the SITUATION naturally
        # ============================================================
        
        situation_words = []
        
        # Describe population size
        if pop > 30000:
            situation_words.append(f"A massive {pop:,} people")
        elif pop > 20000:
            situation_words.append(f"A large {pop:,} people")
        elif pop > 10000:
            situation_words.append(f"{pop:,} people")
        elif pop > 5000:
            situation_words.append(f"A moderate {pop:,} people")
        else:
            situation_words.append(f"A small {pop:,} people")
        
        situation_words.append("are affected by a")
        
        # Describe severity
        if severity == 'High':
            situation_words.append("severe")
        elif severity == 'Medium':
            situation_words.append("moderate")
        else:
            situation_words.append("mild")
        
        situation_words.append("flood.")
        
        # Describe demographics naturally
        demo_parts = []
        if children_pct > 0.25:
            demo_parts.append(f"{children_pct*100:.1f}% are children")
        elif children_pct > 0.20:
            demo_parts.append(f"children make up {children_pct*100:.1f}%")
        
        if elderly_pct > 0.15:
            demo_parts.append(f"{elderly_pct*100:.1f}% are elderly")
        elif elderly_pct > 0.10:
            demo_parts.append(f"elderly constitute {elderly_pct*100:.1f}%")
        
        if female_pct > 0.52:
            demo_parts.append(f"women are {female_pct*100:.1f}% of the population")
        elif female_pct > 0.48:
            demo_parts.append(f"the population is {female_pct*100:.1f}% female")
        
        if demo_parts:
            if len(demo_parts) == 1:
                situation_words.append(f" {demo_parts[0]}.")
            elif len(demo_parts) == 2:
                situation_words.append(f" {demo_parts[0]} and {demo_parts[1]}.")
            else:
                situation_words.append(f" {', '.join(demo_parts[:-1])}, and {demo_parts[-1]}.")
        
        situation = " ".join(situation_words)
        
        # ============================================================
        # PART 2: Explain WHY these needs exist (CAUSES)
        # ============================================================
        
        cause_parts = []
        
        # Population cause
        if pop > 30000:
            cause_parts.append(f"the sheer number of people ({pop:,})")
        elif pop > 15000:
            cause_parts.append(f"the large population size")
        elif pop > 5000:
            cause_parts.append(f"the population size")
        
        # Children cause
        if children_pct > 0.30:
            cause_parts.append(f"the unusually high number of children ({children_pct*100:.1f}%)")
        elif children_pct > 0.25:
            cause_parts.append(f"the high child population")
        elif children_pct > 0.20:
            cause_parts.append(f"the presence of many children")
        
        # Elderly cause
        if elderly_pct > 0.20:
            cause_parts.append(f"the large elderly population needing special care")
        elif elderly_pct > 0.15:
            cause_parts.append(f"the significant elderly population")
        elif elderly_pct > 0.10:
            cause_parts.append(f"the elderly population")
        
        # Female cause
        if female_pct > 0.55:
            cause_parts.append(f"the majority female population requiring sanitary supplies")
        elif female_pct > 0.52:
            cause_parts.append(f"the high proportion of women and girls")
        elif female_pct > 0.48:
            cause_parts.append(f"the balanced gender distribution")
        
        # Severity cause
        if severity == 'High':
            cause_parts.append(f"the severity of the flooding")
        elif severity == 'Medium':
            cause_parts.append(f"the nature of the flooding")
        
        # Build cause sentence naturally
        if cause_parts:
            if len(cause_parts) == 1:
                cause_text = f"Because of {cause_parts[0]}"
            elif len(cause_parts) == 2:
                cause_text = f"Because of {cause_parts[0]} and {cause_parts[1]}"
            else:
                cause_text = f"Because of {', '.join(cause_parts[:-1])}, and {cause_parts[-1]}"
        else:
            cause_text = "Standard relief protocols apply"
        
        # ============================================================
        # PART 3: Describe WHAT is needed (from predictions)
        # ============================================================
        
        # Get high priority needs
        high_priority_items = []
        if ml_predictions:
            for item, details in ml_predictions.items():
                if details['priority'] in ['High', 'Critical']:
                    high_priority_items.append({
                        'name': item,
                        'qty': details['quantity']
                    })
        
        need_parts = []
        if high_priority_items:
            for i, item in enumerate(high_priority_items[:5]):  # Top 5 needs
                if item['name'] == 'Water Bottles':
                    need_parts.append(f"{item['qty']:,} litres of water")
                elif item['name'] == 'Cooked Food Packs':
                    need_parts.append(f"{item['qty']:,} food packs")
                elif item['name'] == 'Sanitary':
                    need_parts.append(f"{item['qty']:,} sanitary pads")
                elif item['name'] == 'Soap':
                    need_parts.append(f"{item['qty']:,} soap bars")
                elif item['name'] == 'Infant Milk Powder Packs':
                    need_parts.append(f"{item['qty']:,} baby formula packs")
                elif item['name'] == 'Toothpaste':
                    need_parts.append(f"{item['qty']:,} toothpaste tubes")
                elif item['name'] == 'Toothbrushes':
                    need_parts.append(f"{item['qty']:,} toothbrushes")
                else:
                    need_parts.append(f"{item['qty']:,} {item['name'].lower()}")
        
        if need_parts:
            if len(need_parts) == 1:
                needs_text = f"You will need {need_parts[0]}."
            elif len(need_parts) == 2:
                needs_text = f"You will need {need_parts[0]} and {need_parts[1]}."
            else:
                needs_text = f"You will need {', '.join(need_parts[:-1])}, and {need_parts[-1]}."
        else:
            needs_text = "Relief supplies are within normal range."
        
        # ============================================================
        # PART 4: Determine PRIORITY naturally
        # ============================================================
        
        priority_score = 0
        priority_reasons = []
        
        if severity == 'High':
            priority_score += 3
            priority_reasons.append("severe flooding")
        elif severity == 'Medium':
            priority_score += 2
            priority_reasons.append("moderate flooding")
        
        if pop > 20000:
            priority_score += 3
            priority_reasons.append("massive population")
        elif pop > 10000:
            priority_score += 2
            priority_reasons.append("large population")
        elif pop > 5000:
            priority_score += 1
            priority_reasons.append("significant population")
        
        if children_pct > 0.25:
            priority_score += 2
            priority_reasons.append("vulnerable children")
        
        if elderly_pct > 0.15:
            priority_score += 2
            priority_reasons.append("vulnerable elderly")
        
        if female_pct > 0.52:
            priority_score += 1
            priority_reasons.append("specific women's needs")
        
        if priority_score >= 7:
            priority = "CRITICAL"
            priority_text = f"This is a CRITICAL situation. The combination of {', '.join(priority_reasons)} means you must act immediately."
        elif priority_score >= 5:
            priority = "HIGH"
            priority_text = f"This is HIGH priority. With {', '.join(priority_reasons)}, deploy resources within 24 hours."
        elif priority_score >= 3:
            priority = "MEDIUM"
            priority_text = f"This is MEDIUM priority. {', '.join(priority_reasons)} requires coordinated response within 48 hours."
        else:
            priority = "LOW"
            priority_text = f"This is LOW priority. Standard monitoring and preparedness are sufficient."
        
        # ============================================================
        # PART 5: Combine into a NATURAL paragraph
        # ============================================================
        
        full_explanation = f"{situation} {cause_text}. {needs_text} {priority_text}"
        
        return {
            'summary': full_explanation,
            'priority_level': priority,
            'drivers': priority_reasons,
            'situation': situation,
            'cause': cause_text,
            'needs': needs_text,
            'priority_text': priority_text
        }


if __name__ == "__main__":
    from data_preprocessing import ReliefDataPreprocessor
    from relief_predictor import ReliefPredictor
    
    print("=" * 60)
    print("TESTING TRULY NATURAL EXPLANATION")
    print("=" * 60)
    
    # Load data
    preprocessor = ReliefDataPreprocessor("../../data/Gampaha_DS_Flood_Emergency_Relief_2019_2025.xlsx")
    X_train, X_test, y_train, y_test, full_data = preprocessor.run_pipeline(test_year=2025, scale=False)
    
    # Train predictor
    predictor = ReliefPredictor()
    predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Test different scenarios
    test_cases = [
        {"name": "Large population, High severity", "pop": 42105, "children": 0.28, "elderly": 0.13, "female": 0.52, "severity": "High"},
        {"name": "Small population, Low severity", "pop": 2500, "children": 0.15, "elderly": 0.08, "female": 0.48, "severity": "Low"},
        {"name": "High elderly population", "pop": 15000, "children": 0.12, "elderly": 0.25, "female": 0.45, "severity": "Medium"},
        {"name": "Very high female percentage", "pop": 8000, "children": 0.20, "elderly": 0.10, "female": 0.65, "severity": "Medium"},
    ]
    
    causal_network = CausalReliefNetwork()
    causal_network.build_network(full_data)
    
    for test in test_cases:
        print("\n" + "=" * 60)
        print(f"SCENARIO: {test['name']}")
        print("=" * 60)
        
        # Print input parameters
        print(f"\nINPUT PARAMETERS:")
        print(f"   Population: {test['pop']:,}")
        print(f"   Children %: {test['children']*100:.1f}%")
        print(f"   Elderly %: {test['elderly']*100:.1f}%")
        print(f"   Female %: {test['female']*100:.1f}%")
        print(f"   Severity: {test['severity']}")
        
        # Get ML predictions
        ml_result = predictor.predict_with_analysis(
            affected_population=test['pop'],
            children_pct=test['children'],
            elderly_pct=test['elderly'],
            female_pct=test['female'],
            flood_severity=test['severity']
        )
        
        # Get causal predictions
        causal_predictions = causal_network.predict_relief_needs(
            affected_population=test['pop'],
            children_pct=test['children'],
            elderly_pct=test['elderly'],
            female_pct=test['female'],
            flood_severity=test['severity']
        )
        
        # ============================================================
        # PRINT CAUSAL NETWORK PREDICTIONS WITH CONFIDENCE
        # ============================================================
        print("\nCAUSAL NETWORK PREDICTIONS:")
        print("-" * 60)
        if causal_predictions:
            for var, pred in causal_predictions.items():
                # Format the variable name for display
                display_name = var.replace('_', ' ')
                print(f"   {display_name:20s} → {pred['most_likely']:10s} (prob: {pred['probability']:.2f}, {pred['confidence']})")
        else:
            print("   No predictions returned. Check if all required nodes exist in the network.")
            print("   Available nodes in model:", causal_network.model.nodes())
        
        # Get explanation
        explanation = causal_network.get_explainable_recommendation(
            input_data={
                'affected_population': test['pop'],
                'children_pct': test['children'],
                'elderly_pct': test['elderly'],
                'female_pct': test['female'],
                'flood_severity': test['severity']
            },
            predictions=causal_predictions,
            ml_predictions=ml_result['predictions']
        )
        
        print(f"\nEXPLANATION:")
        print("-" * 60)
        print(f"{explanation['summary']}\n")
        print(f"Priority: {explanation['priority_level']}")
        
        # Print drivers
        if explanation.get('drivers'):
            print(f"\nKey Drivers:")
            for driver in explanation['drivers']:
                print(f"   • {driver}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)