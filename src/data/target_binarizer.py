from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TargetBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=50):
        self.threshold = threshold

    def fit(self, y, X=None):
        return self

    def transform(self, y):
        y_binary = np.where(y >= self.threshold, 1, 0)
        return y_binary
