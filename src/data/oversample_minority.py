from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import numpy as np


class OversampleMinority(BaseEstimator, TransformerMixin):
    def __init__(self, target_minority_percentage=0.4, random_state=42, **kwargs):
        self.target_minority_percentage = target_minority_percentage
        self.random_state = random_state
        self.smote_params = kwargs
        self.sampler_ = None  # Renamed from smote to sampler_

    def fit_resample(self, X, y):
        # Calculate sampling strategy
        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        minority_class = min(class_counts, key=class_counts.get)
        majority_count = max(class_counts.values())
        
        target_minority_count = int(
            (self.target_minority_percentage / (1 - self.target_minority_percentage)) 
            * majority_count
        )
        
        self.sampler_ = SMOTE(
            sampling_strategy={minority_class: target_minority_count},
            random_state=self.random_state,
            **self.smote_params
        )
        return self.sampler_.fit_resample(X, y)

    def transform(self, X, y=None):
        return X, y