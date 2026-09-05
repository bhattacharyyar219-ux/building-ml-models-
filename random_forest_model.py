import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def _infer_columns(X, num_cols, cat_cols):
    if not num_cols and not cat_cols:
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)
    return num_cols, cat_cols


def _build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="drop")


class UniversalRandomForestClassifier:
    def __init__(self, num_cols=None, cat_cols=None, model_params=None):
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.model_params = model_params or {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        self.pipeline = None

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.num_cols, self.cat_cols = _infer_columns(X, self.num_cols, self.cat_cols)
        preprocessor = _build_preprocessor(self.num_cols, self.cat_cols)
        model = RandomForestClassifier(**self.model_params)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y_true, preds),
            "f1_weighted": f1_score(y_true, preds, average="weighted")
        }

    def feature_importances(self):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.pipeline.named_steps["model"].feature_importances_

    def save(self, path):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Nothing to save.")
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class UniversalRandomForestRegressor:
    def __init__(self, num_cols=None, cat_cols=None, model_params=None):
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.model_params = model_params or {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        self.pipeline = None

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.num_cols, self.cat_cols = _infer_columns(X, self.num_cols, self.cat_cols)
        preprocessor = _build_preprocessor(self.num_cols, self.cat_cols)
        model = RandomForestRegressor(**self.model_params)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
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
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "r2": r2_score(y_true, preds)
        }

    def feature_importances(self):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.pipeline.named_steps["model"].feature_importances_

    def save(self, path):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Nothing to save.")
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    Xc, yc = make_classification(n_samples=300, n_features=6, random_state=42)
    Xc = pd.DataFrame(Xc, columns=[f"f{i}" for i in range(6)])
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    print("Random Forest Classifier:", UniversalRandomForestClassifier().fit(Xc_tr, yc_tr).evaluate(Xc_te, yc_te))

    Xr, yr = make_regression(n_samples=300, n_features=6, noise=8, random_state=42)
    Xr = pd.DataFrame(Xr, columns=[f"f{i}" for i in range(6)])
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    print("Random Forest Regressor:", UniversalRandomForestRegressor().fit(Xr_tr, yr_tr).evaluate(Xr_te, yr_te))
