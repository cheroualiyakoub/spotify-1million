# File: main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
import pandas as pd
import mlflow
import logging
from datetime import datetime

# Import the necessary classes from other files
from predict_request import PredictionRequest
import model_service
from model_service import ModelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)

# Create FastAPI app
app = FastAPI()

# Initialize model service - with fallback to None if it fails
model_service = None
try:
    model_uri = "models:/SpotifyPopularityClassifier@production"  # Note: changed format
    model_service = ModelService(model_uri)
    logger.info(f"Model service initialized successfully with {model_uri}")
except Exception as e:
    logger.error(f"Failed to initialize model service: {str(e)}")
    # We'll continue with model_service as None and handle this in the endpoints

@app.get("/")
async def root():
    try:
        if model_service is None:
            return {
                "message": "Spotify Track Popularity Prediction API",
                "status": "WARNING: Model service not initialized",
                "error": "Unable to load model. See server logs for details."
            }
        
        return {
            "message": "Spotify Track Popularity Prediction API",
            "model_version": model_service.model.metadata.run_id if hasattr(model_service.model, "metadata") else "Unknown",
            "expected_features": model_service.feature_columns
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"API health check failed: {str(e)}"}
        )

@app.post("/predict", response_model=dict)
async def predict(request: PredictionRequest):
    """Predict popularity for multiple tracks"""
    try:
        logger.info(f"Received prediction request for {len(request.tracks)} tracks")
        
        df = pd.DataFrame([t.dict() for t in request.tracks]).astype({
                        'danceability': 'float64',
                        'energy': 'float64',
                        'loudness': 'float64',
                        'speechiness': 'float64',
                        'acousticness': 'float64',
                        'instrumentalness': 'float64',
                        'liveness': 'float64',
                        'valence': 'float64',
                        'tempo': 'float64',
                        'key': 'int64',
                        'mode': 'int64',
                        'time_signature': 'int64',
                        'duration_ms': 'int64',
                        'year': 'int64'
                    })
        

        # Check if model service is initialized
        if model_service is None:
            logger.error("Model service not initialized")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": "Model service not available. Please check server logs."}
            )
        
        # Convert to DataFrame with proper schema
        df = pd.DataFrame([t.dict() for t in request.tracks])
        
        # # Remove unnecessary fields not used in training
        # df = df.drop(columns=['track_name', 'track_id', 'artist_name'], errors='ignore')
        
        # Make predictions
        predictions = model_service.predict(df)
        
        # Prepare human-readable results
        results = []
        for track, pred in zip(request.tracks, predictions):
            results.append({
                "track_id": track.track_id,
                "track_name": track.track_name,
                "artist": track.artist_name,
                "popularity_score": float(pred),
                "popularity_class": "Hit" if pred > 0.5 else "Non-hit"
            })
        
        # MLflow logging
        try:
            with mlflow.start_run(run_name="API Prediction") as run:
                mlflow.log_params({
                    "num_tracks": len(predictions),
                    "request_source": "API"
                })
                mlflow.log_metrics({
                    "avg_popularity": float(predictions.mean()),
                    "hit_rate": float((predictions > 0.5).mean())
                })
                mlflow.log_dict(
                    {"predictions": results}, 
                    f"predictions/{datetime.now().isoformat()}.json"
                )
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
            # Continue with the response even if MLflow logging fails
            
        return {"predictions": results}
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Prediction failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
