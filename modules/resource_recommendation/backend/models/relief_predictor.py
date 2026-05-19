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
        self.feature_columns = ['Affected_Population', 'Children_%', 'Elderly_%', 'Severity_Code']
        self.target_columns = None
        self.causal_network = None
        self.severity_map = {'Low': 1, 'Medium': 2, 'High': 3}

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def set_causal_network(self, causal_network):
        """Set causal network for explainability"""
        self.causal_network = causal_network
        print("Causal network connected to predictor")

    def train_models(self, X_train, y_train, X_test=None, y_test=None):
        """Train multiple models for each relief item"""
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

        print("\n Training completed")

    def predict(self, X_input):
        """Predict relief needs for given input"""
        if isinstance(X_input, dict):
            X_input = pd.DataFrame([X_input])
        X_input = X_input[self.feature_columns]

        results = {}
        for target, model in self.models.items():
            pred = model.predict(X_input)[0]
            # Convert numpy int to Python int (for JSON serialization)
            results[target] = int(max(0, round(pred)))

        return pd.DataFrame([results])

    def predict_with_analysis(self, affected_population, children_pct, elderly_pct, flood_severity='Medium'):
        """Predict relief needs with causal explanation"""
        severity_code = self.severity_map.get(flood_severity, 2)

        X_input = pd.DataFrame([{
            "Affected_Population": int(affected_population),
            "Children_%": float(children_pct),
            "Elderly_%": float(elderly_pct),
            "Severity_Code": int(severity_code)
        }])

        preds = self.predict(X_input)

        result = {
            "input": {
                "affected_population": int(affected_population),
                "children_pct": float(children_pct),
                "elderly_pct": float(elderly_pct),
                "severity": flood_severity
            },
            "predictions": {},
            "overall_priority": "Medium"
        }

        # Priority logic based on quantities
        for col in preds.columns:
            value = preds[col].iloc[0]
            
            # Convert to Python int
            value = int(value)

            # Determine priority
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
                "quantity": int(value),  # Ensure Python int
                "priority": priority
            }

        # Set overall priority
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

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        X_test = X_test[self.feature_columns]
        results = {}

        for target in self.target_columns:
            model = self.models[target]
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test[target], preds)
            rmse = np.sqrt(mean_squared_error(y_test[target], preds))
            r2 = r2_score(y_test[target], preds)

            safe_mape = np.mean(np.abs((y_test[target] - preds) / np.maximum(y_test[target], 1))) * 100

            results[target] = {
                "MAE": float(round(mae, 2)),
                "RMSE": float(round(rmse, 2)),
                "R2": float(round(r2, 3)),
                "MAPE": float(round(safe_mape, 2))
            }

            print(f"\n{target}")
            print(f" MAE : {mae:.2f}")
            print(f" RMSE: {rmse:.2f}")
            print(f" R2  : {r2:.3f}")
            print(f" MAPE: {safe_mape:.1f}%")

        return results

    def load_models(self):
        """Load saved models from disk"""
        if not os.path.exists(self.model_dir):
            print(f"Model directory not found: {self.model_dir}")
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


# Test the predictor
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RELIEF PREDICTOR")
    print("=" * 60)

    # Create sample data for testing
    sample_X = pd.DataFrame({
        'Affected_Population': [10000, 5000, 20000],
        'Children_%': [0.25, 0.30, 0.20],
        'Elderly_%': [0.15, 0.10, 0.20],
        'Severity_Code': [2, 1, 3]
    })

    sample_y = pd.DataFrame({
        'Water Bottles': [30000, 15000, 60000],
        'Cooked Food Packs': [10000, 5000, 20000],
        'Soap': [300, 150, 600]
    })

    predictor = ReliefPredictor(model_dir='test_models/')
    predictor.train_models(sample_X, sample_y)

    result = predictor.predict_with_analysis(
        affected_population=10000,
        children_pct=0.25,
        elderly_pct=0.15,
        flood_severity='High'
    )

    print("\nPrediction Result:")
    print(f"   Priority Level: {result['overall_priority']}")
    print(f"\n   Relief Items:")
    for item, details in result['predictions'].items():
        print(f"     - {item}: {details['quantity']} ({details['priority']})")

    print("\nTest completed!")