import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
import logging 
import mlflow

logger = logging.getLogger(__name__)

class ModelService1:
    def __init__(self, model_uri):
        self.model = self._load_model(model_uri)
        self.feature_columns = self._get_expected_features()
        
    def _load_model(self, model_uri):
        try:
            logger.info(f"Loading model from {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Model loading failed")

    def _get_expected_features(self):
        """Get features expected by the model from its signature"""
        if self.model.metadata.signature:
            return self.model.metadata.signature.inputs.input_names()
        return None

    def _validate_features(self, df):
        """Validate input features against model expectations"""
        if self.feature_columns:
            missing = set(self.feature_columns) - set(df.columns)
            extra = set(df.columns) - set(self.feature_columns)
            
            if missing:
                logger.error(f"Missing features: {missing}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required features: {list(missing)}"
                )
                
            if extra:
                logger.warning(f"Extra features detected: {extra}")
                df = df[self.feature_columns]
                
        return df

    def predict(self, input_data: pd.DataFrame):
        try:
            # Convert to proper dtypes
            input_data = input_data.convert_dtypes()
            
            # Validate and align features
            validated_data = self._validate_features(input_data)
            
            # Make prediction
            logger.info("Making predictions...")
            return self.model.predict(validated_data)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Prediction failed")