from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class UndersampleMajority(BaseEstimator, TransformerMixin):
    def __init__(self, target_minority_percentage=0.3, random_state=42, **kwargs):
        self.target_minority_percentage = target_minority_percentage
        self.random_state = random_state
        self.under_params = kwargs
        self.sampler_ = None  # Changed from undersampler to sampler_

    def fit_resample(self, X, y):
        # Calculate sampling strategy
        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        minority_count = class_counts[minority_class]
        
        target_majority_count = int(
            (minority_count / self.target_minority_percentage) - minority_count
        )
        
        self.sampler_ = RandomUnderSampler(
            sampling_strategy={majority_class: target_majority_count},
            random_state=self.random_state,
            **self.under_params
        )
        return self.sampler_.fit_resample(X, y)

    def transform(self, X, y=None):
        return X, y