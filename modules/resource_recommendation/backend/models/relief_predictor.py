import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ReliefPredictor:

    def __init__(self, model_dir='models_saved/'):
        self.model_dir = model_dir
        self.models = {}
        self.feature_columns = ['Affected_Population', 'Children_%', 'Elderly_%', 'Female %', 'Severity_Code']
        self.target_columns = None
        self.causal_network = None
        self.severity_map = {'Low': 1, 'Medium': 2, 'High': 3}

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def set_causal_network(self, causal_network):
        self.causal_network = causal_network
        print("Causal network connected")

    def train_models(self, X_train, y_train, X_test=None, y_test=None):
        print("\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)

        X_train = X_train[self.feature_columns]
        self.target_columns = y_train.columns.tolist()

        print(f"Features: {self.feature_columns}")
        print(f"Targets: {self.target_columns}")
        print(f"Training samples: {len(X_train)}")

        tscv = TimeSeriesSplit(n_splits=5)

        for target in self.target_columns:
            print(f"\nTraining → {target}")

            models = {
                "Ridge": Ridge(alpha=1.0),
                "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42)
            }

            best_model = None
            best_score = -np.inf
            best_name = ""

            for name, model in models.items():
                try:
                    cv_scores = cross_val_score(model, X_train, y_train[target], cv=tscv, scoring='r2')
                    model.fit(X_train, y_train[target])
                    score = cv_scores.mean()

                    if X_test is not None:
                        preds = model.predict(X_test[X_train.columns])
                        test_r2 = r2_score(y_test[target], preds)
                        print(f"  {name}: CV={score:.3f}, Test={test_r2:.3f}")
                        score = test_r2
                    else:
                        print(f"  {name}: CV={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_name = name
                except Exception as e:
                    print(f"  {name} failed: {str(e)[:50]}")

            self.models[target] = best_model
            joblib.dump(best_model, os.path.join(self.model_dir, f"{target.replace(' ', '_')}.pkl"))
            print(f"Best model: {best_name} ({best_score:.3f})")

        print("\nTraining completed")

    def predict(self, X_input):
        if isinstance(X_input, dict):
            X_input = pd.DataFrame([X_input])
        X_input = X_input[self.feature_columns]

        results = {}
        for target, model in self.models.items():
            pred = model.predict(X_input)[0]
            results[target] = int(max(0, round(pred)))

        return pd.DataFrame([results])

    def predict_with_analysis(self, affected_population, children_pct, elderly_pct, female_pct, flood_severity='Medium'):
        severity_code = self.severity_map.get(flood_severity, 2)

        X_input = pd.DataFrame([{
            "Affected_Population": int(affected_population),
            "Children_%": float(children_pct),
            "Elderly_%": float(elderly_pct),
            "Female %": float(female_pct),
            "Severity_Code": int(severity_code)
        }])

        preds = self.predict(X_input)

        result = {
            "input": {
                "affected_population": int(affected_population),
                "children_pct": float(children_pct),
                "elderly_pct": float(elderly_pct),
                "female_pct": float(female_pct),
                "severity": flood_severity
            },
            "predictions": {},
            "overall_priority": "Medium"
        }

        for col in preds.columns:
            value = preds[col].iloc[0]
            value = int(value)

            if col in ['Water Bottles', 'Cooked Food Packs']:
                if value > 100000:
                    priority = "Critical"
                elif value > 50000:
                    priority = "High"
                elif value > 20000:
                    priority = "Medium"
                else:
                    priority = "Low"
            elif col in ['Sanitary', 'Soap']:
                if value > 5000:
                    priority = "High"
                elif value > 1000:
                    priority = "Medium"
                else:
                    priority = "Low"
            else:
                if value > 2000:
                    priority = "High"
                elif value > 500:
                    priority = "Medium"
                else:
                    priority = "Low"

            result["predictions"][col] = {
                "quantity": int(value),
                "priority": priority
            }

        priorities = [p['priority'] for p in result['predictions'].values()]
        if 'Critical' in priorities:
            result['overall_priority'] = 'Critical'
        elif 'High' in priorities:
            result['overall_priority'] = 'High'
        elif 'Medium' in priorities:
            result['overall_priority'] = 'Medium'
        else:
            result['overall_priority'] = 'Low'

        return result

    def load_models(self):
        if not os.path.exists(self.model_dir):
            return False

        loaded_count = 0
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pkl'):
                target = filename.replace('.pkl', '').replace('_', ' ')
                model_path = os.path.join(self.model_dir, filename)
                try:
                    self.models[target] = joblib.load(model_path)
                    loaded_count += 1
                    if self.target_columns is None:
                        self.target_columns = []
                    if target not in self.target_columns:
                        self.target_columns.append(target)
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

        print(f"Loaded {loaded_count} models")
        return loaded_count > 0