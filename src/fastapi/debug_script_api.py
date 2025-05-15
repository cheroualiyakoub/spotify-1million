# File: debug_steps.py
"""
Debug script to help isolate and fix issues with the Spotify Popularity Prediction API
"""
import mlflow
from mlflow import MlflowClient
import logging
import pandas as pd
from fastapi.testclient import TestClient
from main import app
from model_service import ModelService 



# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test client
client = TestClient(app)

def test_mlflow_connection():
    """Test connection to MLflow tracking server"""
    try:
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # List experiments
        # client = MlflowClient()
        # experiments = client.list_experiments()
        # logger.info(f"Available experiments: {[exp.name for exp in experiments]}")
        
        # Check for model in registry
        try:
            model_versions = mlflow.tracking.MlflowClient().search_model_versions("name='SpotifyPopularityClassifier'")
            logger.info(f"Model versions: {[mv.version for mv in model_versions]}")
            if not model_versions:
                logger.error("No model versions found for SpotifyPopularityClassifier")
            return True
        except Exception as e:
            logger.error(f"Error searching model registry: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"MLflow connection error: {str(e)}")
        return False

def test_model_loading(model_uri):
    """Test loading model from MLflow"""
    try:
        logger.info(f"Attempting to load model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully")
        
        # Check model metadata
        logger.info(f"Model run_id: {model.metadata.run_id if hasattr(model, 'metadata') else 'No metadata'}")
        
        # Check model signature if available
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'signature'):
            if model.metadata.signature:
                logger.info(f"Model input features: {model.metadata.signature.inputs.input_names()}")
            else:
                logger.warning("Model has no signature")
        else:
            logger.warning("Model has no metadata or signature")
            
        return True
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = client.get("/")
        logger.info(f"Root endpoint response status: {response.status_code}")
        logger.info(f"Response content: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return False

def test_predict_endpoint():
    """Test the predict endpoint with sample data"""
    try:
        # Create a sample prediction request
        sample_data = {
            "tracks": [
                {
                "artist_name": "Artist",
                "track_name": "Track",
                "track_id": "0J3Gf2MDWB2nurkJiqkwiw",
                "year": 2023,
                "genre": "pop",
                "danceability": 0.7,
                "energy": 0.8,
                "key": 5,
                "loudness": -5.3,
                "mode": 1,
                "speechiness": 0.04,
                "acousticness": 0.2,
                "instrumentalness": 0.2,
                "liveness": 0.1,
                "valence": 0.6,
                "tempo": 120.2,
                "duration_ms": 210000,
                "time_signature": 4
                }
            ]
        }


        
        response = client.post("/predict", json=sample_data)
        logger.info(f"Predict endpoint response status: {response.status_code}")
        
        if response.status_code == 200:
            logger.info(f"Prediction results: {response.json()}")
            return True
        else:
            logger.error(f"Predict endpoint error: {response.json()}")
            return False
    except Exception as e:
        logger.error(f"Predict endpoint test error: {str(e)}")
        return False

def run_all_tests():
    """Run all tests to diagnose issues"""
    results = {
        "mlflow_connection": test_mlflow_connection(),
        "model_loading_production": test_model_loading("models:/SpotifyPopularityClassifier@production"),
        "root_endpoint": test_root_endpoint(),
        "predict_endpoint": test_predict_endpoint()
    }
    
    logger.info("Test results summary:")
    for test_name, result in results.items():
        logger.info(f"{test_name}: {'PASS' if result else 'FAIL'}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
