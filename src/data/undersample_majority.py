from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class UndersampleMajority(BaseEstimator, TransformerMixin):
    def __init__(self, target_minority_percentage=0.5, random_state=42, **kwargs):
        self.target_minority_percentage = target_minority_percentage
        self.random_state = random_state
        self.under_params = kwargs
        self.undersampler = None

    def fit(self, X, y):
        # Get original class counts
        _, counts = np.unique(y, return_counts=True)
        minority_count = min(counts)
        target_majority_count = int((1 - self.target_minority_percentage) / self.target_minority_percentage * minority_count)
        
        # Compute actual sampling strategy
        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        majority_class = max(class_counts, key=class_counts.get)
        self.sampling_strategy = {majority_class: target_majority_count}
        
        self.undersampler = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            **self.under_params
        )
        self.undersampler.fit_resample(X, y)
        return self

    def transform(self, X, y):
        return self.undersampler.fit_resample(X, y)

if (__name__ == "__main__"):
    print("undersampling")
