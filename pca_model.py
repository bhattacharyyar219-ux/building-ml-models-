import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


class UniversalPCA:
    def __init__(self, n_components=2, num_cols=None, cat_cols=None, model_params=None):
        self.n_components = n_components
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
        params = {"n_components": self.n_components, "random_state": 42, **self.model_params}
        model = PCA(**params)
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.pipeline.named_steps["model"].explained_variance_ratio_

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

    pca = UniversalPCA(n_components=2)
    transformed = pca.fit_transform(X)
    print("PCA explained variance ratio:", pca.explained_variance_ratio())
    print("PCA transformed shape:", transformed.shape)
