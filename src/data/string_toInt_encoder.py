from sklearn.base import BaseEstimator, TransformerMixin

class StringToIntEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        self.encoders = {}

    def fit(self, X, y=None):
        self.encoders = {}
        for col in self.columns_to_encode:
            unique_vals = X[col].dropna().unique()
            self.encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in self.columns_to_encode:
            encoder = self.encoders[col]
            X_encoded[col] = X_encoded[col].map(lambda val: encoder.get(val, -1))
        return X_encoded
