class TrainTimeOversamplingWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessing, oversampler, classifier):
        self.preprocessing = preprocessing
        self.oversampler = oversampler
        self.classifier = classifier

    def fit(self, X, y):
        # Preprocess
        X_transformed = self.preprocessing.fit_transform(X)
        # Oversample
        X_resampled, y_resampled = self.oversampler.fit(X_transformed, y).transform(X_transformed, y)
        # Fit classifier
        self.classifier.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        X_transformed = self.preprocessing.transform(X)
        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.preprocessing.transform(X)
        return self.classifier.predict_proba(X_transformed)

    def score(self, X, y):
        return self.predict(X) == y  # or use accuracy_score if needed
