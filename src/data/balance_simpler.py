from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import pandas as pd

class BalancedResampler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, Xy):
        # Xy is expected to be a tuple (X, y)
        X, y = Xy

        # Combine for easier resampling
        df = X.copy()
        df['target'] = y

        # Split by class
        df_min = df[df['target'] == 1]
        df_maj = df[df['target'] == 0]

        # Oversample class 1 to 2x its original size
        df_min_oversampled = resample(
            df_min,
            replace=True,
            n_samples=len(df_min) * 2,
            random_state=42
        )

        # Undersample class 0 to match new size of class 1
        df_maj_undersampled = resample(
            df_maj,
            replace=False,
            n_samples=len(df_min_oversampled),
            random_state=42
        )

        # Combine and shuffle
        df_balanced = pd.concat([df_min_oversampled, df_maj_undersampled])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split back
        X_resampled = df_balanced.drop(columns='target')
        y_resampled = df_balanced['target']

        return X_resampled, y_resampled
