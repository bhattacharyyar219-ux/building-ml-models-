import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


class UniversalXGBRegressor:
    def __init__(self, num_cols=None, cat_cols=None, xgb_params=None):
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1
        }
        self.pipeline = None

    def _infer_columns(self, X):
        if not self.num_cols and not self.cat_cols:
            self.num_cols = list(X.select_dtypes(include=[np.number]).columns)
            self.cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    def _build_pipeline(self):
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
        ], remainder="drop")

        model = XGBRegressor(**self.xgb_params)

        self.pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._infer_columns(X)
        self._build_pipeline()
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict(X)

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        return {
            "mae": mean_absolute_error(y_true, preds),
            "rmse": np.sqrt(mean_squared_error(y_true, preds)),
            "r2": r2_score(y_true, preds)
        }

    def save(self, path):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Nothing to save.")
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=300, n_features=6, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = UniversalXGBRegressor()
    reg.fit(X_train, y_train)
    metrics = reg.evaluate(X_test, y_test)
    print(metrics)
