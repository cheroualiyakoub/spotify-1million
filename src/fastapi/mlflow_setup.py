# File: mlflow_setup.py
"""
Helper script to check and setup MLflow environment for the Spotify API
"""
import mlflow
import logging
import sys
import os
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_mlflow_connection(tracking_uri="http://localhost:5000"):
    """Check connection to MLflow tracking server"""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        # experiments = client.list_experiments()
        logger.info(f"Successfully connected to MLflow at {tracking_uri}")
        # logger.info(f"Found {len(experiments)} experiments")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MLflow at {tracking_uri}: {str(e)}")
        return False

def check_model_registry(model_name="SpotifyPopularityClassifier"):
    """Check if the model exists in the registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            logger.info(f"Found {len(versions)} versions of model {model_name}")
            for v in versions:
                logger.info(f"Version {v.version}: {v.current_stage}")
            return True
        else:
            logger.warning(f"No versions found for model {model_name}")
            return False
    except Exception as e:
        logger.error(f"Error checking model registry: {str(e)}")
        return False

def main():
    """Main function to setup MLflow for testing"""
    # Check if MLflow server is running
    if not check_mlflow_connection():
        logger.error("MLflow server not available. Please start it first.")
        logger.info("You can start MLflow server with: mlflow server --host 0.0.0.0 --port 5000")
        return False
    
    # Check if model exists in registry
    if not check_model_registry():
        logger.warning("Model not found in registry")
        return False
    
    logger.info("MLflow setup completed successfully.")
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
