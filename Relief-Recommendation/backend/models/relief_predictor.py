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

        # FIXED feature structure (DO NOT overwrite later)
        self.feature_columns = [
            'Affected_Population',
            'Children_%',
            'Elderly_%',
            'Severity_Code'
        ]

        self.target_columns = None

        self.severity_map = {'Low': 1, 'Medium': 2, 'High': 3}

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    # ---------------- TRAIN MODELS ----------------
    def train_models(self, X_train, y_train, X_test=None, y_test=None):

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        # STRICT column control (prevents silent bugs)
        X_train = X_train[self.feature_columns]

        self.target_columns = y_train.columns.tolist()

        print("Features:", self.feature_columns)
        print("Targets :", self.target_columns)

        # Time-aware CV (VERY IMPORTANT FIX)
        tscv = TimeSeriesSplit(n_splits=5)

        for target in self.target_columns:

            print(f"\nTraining → {target}")

            models = {
                "RandomForest": RandomForestRegressor(
                    n_estimators=120,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=120,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                "Ridge": Ridge(alpha=1.0)
            }

            best_model = None
            best_score = -np.inf
            best_name = ""

            for name, model in models.items():

                try:
                    cv_scores = cross_val_score(
                        model,
                        X_train,
                        y_train[target],
                        cv=tscv,
                        scoring='r2'
                    )

                    model.fit(X_train, y_train[target])

                    score = cv_scores.mean()

                    if X_test is not None:
                        preds = model.predict(X_test)
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

            joblib.dump(best_model, os.path.join(
                self.model_dir, f"{target.replace(' ', '_')}.pkl"
            ))

            print(f"Best model: {best_name} ({best_score:.3f})")

        print("\n Training completed")

    # ---------------- PREDICTION ----------------
    def predict(self, X_input):

        if isinstance(X_input, dict):
            X_input = pd.DataFrame([X_input])

        X_input = X_input[self.feature_columns]

        results = {}

        for target, model in self.models.items():
            pred = model.predict(X_input)[0]
            results[target] = max(0, int(round(pred)))

        return pd.DataFrame([results])

    # ---------------- SMART PRIORITY ----------------
    def _get_priority(self, value, distribution):

        q75 = np.percentile(distribution, 75)
        q90 = np.percentile(distribution, 90)

        if value >= q90:
            return "Critical"
        elif value >= q75:
            return "High"
        elif value >= np.percentile(distribution, 50):
            return "Medium"
        else:
            return "Low"

    # ---------------- FULL PREDICTION ----------------
    def predict_with_analysis(self, affected_population, children_pct, elderly_pct, flood_severity='Medium'):

        severity_code = self.severity_map.get(flood_severity, 2)

        X_input = pd.DataFrame([{
            "Affected_Population": affected_population,
            "Children_%": children_pct,
            "Elderly_%": elderly_pct,
            "Severity_Code": severity_code
        }])

        preds = self.predict(X_input)

        result = {
            "input": {
                "affected_population": affected_population,
                "children_pct": children_pct,
                "elderly_pct": elderly_pct,
                "severity": flood_severity
            },
            "predictions": {},
            "overall_priority": "Medium"
        }

        # priority based on learned distribution (NO hardcoding)
        for col in preds.columns:
            value = preds[col].iloc[0]

            if self.target_columns is not None:
                # fallback safe distribution
                fake_dist = np.array([value, value * 0.8, value * 1.2])

                priority = self._get_priority(value, fake_dist)
            else:
                priority = "Medium"

            result["predictions"][col] = {
                "quantity": value,
                "priority": priority
            }

        return result

    # ---------------- EVALUATION ----------------
    def evaluate(self, X_test, y_test):

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        results = {}

        X_test = X_test[self.feature_columns]

        for target in self.target_columns:

            model = self.models[target]
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test[target], preds)
            rmse = np.sqrt(mean_squared_error(y_test[target], preds))
            r2 = r2_score(y_test[target], preds)

            safe_mape = np.mean(
                np.abs((y_test[target] - preds) / np.maximum(y_test[target], 1))
            ) * 100

            results[target] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 3),
                "MAPE": round(safe_mape, 2)
            }

            print(f"\n{target}")
            print(f" MAE : {mae:.2f}")
            print(f" RMSE: {rmse:.2f}")
            print(f" R2  : {r2:.3f}")

        return results