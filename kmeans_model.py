import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class UniversalKMeans:
    def __init__(self, n_clusters=3, num_cols=None, cat_cols=None, model_params=None):
        self.n_clusters = n_clusters
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.model_params = model_params or {}
        self.pipeline = None

    def _infer_columns(self, X):
        if not self.num_cols and not self.cat_cols:
            self.num_cols = list(X.select_dtypes(include=[np.number]).columns)
            self.cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    def fit(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._infer_columns(X)
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols),
        ], remainder="drop")
        params = {"n_clusters": self.n_clusters, "random_state": 42, "n_init": 10, **self.model_params}
        model = KMeans(**params)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline.fit(X)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict(X)

    def evaluate(self, X):
        labels = self.predict(X)
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_transformed = self.pipeline.named_steps["preprocess"].transform(X)
        result = {"inertia": float(self.pipeline.named_steps["model"].inertia_)}
        if len(set(labels)) >= 2:
            result["silhouette"] = float(silhouette_score(X_transformed, labels))
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

    kmeans = UniversalKMeans(n_clusters=3).fit(X)
    print("KMeans:", kmeans.evaluate(X))
