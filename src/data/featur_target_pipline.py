
class FeatureTargetPipeline:
    def __init__(self, preprocessing_pipeline, target_transformer):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.target_transformer = target_transformer

    def fit_transform(self, df):
        df_preprocessed = self.preprocessing_pipeline.fit_transform(df)
        X = df_preprocessed.drop(columns='popularity')
        y = df_preprocessed['popularity']
        y_bin = self.target_transformer.fit_transform(y)
        return X, y_bin