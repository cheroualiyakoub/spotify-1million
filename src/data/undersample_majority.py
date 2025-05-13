from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class UndersampleMajority(BaseEstimator, TransformerMixin):
    def __init__(self, target_minority_percentage=0.5, random_state=42, **kwargs):
        self.target_minority_percentage = target_minority_percentage
        self.random_state = random_state
        self.under_params = kwargs
        self.undersampler = None
        self.sampling_strategy_ = None

    def fit(self, X, y):
        # Get original class counts
        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        minority_count = class_counts[minority_class]
        
        # Calculate target majority count
        target_majority_count = int((minority_count / self.target_minority_percentage) - minority_count)
        
        # Set sampling strategy
        self.sampling_strategy_ = {majority_class: target_majority_count}
        
        # Initialize undersampler
        self.undersampler = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy_,
            random_state=self.random_state,
            **self.under_params
        )
        return self

    def fit_resample(self, X, y):
        return self.undersampler.fit_resample(X, y)  # Changed from smote to undersampler

    def transform(self, X, y=None):
        return X, y