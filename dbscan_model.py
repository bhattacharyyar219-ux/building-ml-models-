import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


class UniversalDBSCAN:
    def __init__(self, num_cols=None, cat_cols=None, model_params=None):
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.model_params = model_params or {"eps": 0.5, "min_samples": 5}
        self.pipeline = None
        self.labels_ = None

    def _infer_columns(self, X):
        if not self.num_cols and not self.cat_cols:
            self.num_cols = list(X.select_dtypes(include=[np.number]).columns)
            self.cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    def fit_predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._infer_columns(X)
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
        ], remainder="drop")
        model = DBSCAN(**self.model_params)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        X_transformed = self.pipeline.named_steps["preprocess"].fit_transform(X)
        self.labels_ = self.pipeline.named_steps["model"].fit_predict(X_transformed)
        return self.labels_

    def evaluate(self, X):
        if self.labels_ is None:
            raise RuntimeError("Call fit_predict() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_transformed = self.pipeline.named_steps["preprocess"].transform(X)
        mask = self.labels_ != -1
        n_clusters = len(set(self.labels_[mask]))
        result = {"n_clusters": n_clusters, "n_noise": int((self.labels_ == -1).sum())}
        if n_clusters >= 2 and mask.sum() > 1:
            result["silhouette"] = float(silhouette_score(X_transformed[mask], self.labels_[mask]))
        return result

    def save(self, path):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Nothing to save.")
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    dbscan = UniversalDBSCAN(model_params={"eps": 1.0, "min_samples": 5})
    dbscan.fit_predict(X)
    print("DBSCAN:", dbscan.evaluate(X))
