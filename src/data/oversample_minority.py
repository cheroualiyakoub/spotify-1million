from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import numpy as np

class OversampleMinority(BaseEstimator, TransformerMixin):
    def __init__(self, target_minority_percentage=0.5, random_state=42, **kwargs):
        self.target_minority_percentage = target_minority_percentage
        self.random_state = random_state
        self.smote_params = kwargs
        self.smote = None

    def fit(self, X, y):
        # Get original class counts
        _, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        target_minority_count = int((self.target_minority_percentage / (1 - self.target_minority_percentage)) * majority_count)
        
        # Compute actual sampling strategy for SMOTE
        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        minority_class = min(class_counts, key=class_counts.get)
        self.sampling_strategy = {minority_class: target_minority_count}
        
        self.smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            **self.smote_params
        )
        self.smote.fit_resample(X, y)
        return self

    def fit_resample(self, X, y):
        return self.smote.fit_resample(X, y)

    def transform(self, X, y=None):
        return X, y

if (__name__ == "__main__"):
    print("oversampling")
