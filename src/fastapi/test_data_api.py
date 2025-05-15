# File: test_data.py
"""
Sample test data for manual testing of the Spotify Popularity Prediction API
"""
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoint
API_URL = "http://localhost:8000"

# Sample track data
sample_tracks = {
    "tracks": [
        {
            "track_id": "7qiZfU4dY1lWllzX7mPBI3",
            "track_name": "Shape of You",
            "artist_name": "lotfi DK",
            "danceability": 0.825,
            "energy": 0.652,
            "key": 1,
            "loudness": -3.183,
            "mode": 0,
            "speechiness": 0.0802,
            "acousticness": 0.581,
            "instrumentalness": 0.1,
            "liveness": 0.0931,
            "valence": 0.931,
            "genre" : "pop",
            "year" : 2001,
            "tempo": 95.977,
            "duration_ms": 233713,
            "time_signature": 4
        }
    ]
}

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_URL}/")
        logger.info(f"API Health Check Status: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    try:
        response = requests.post(f"{API_URL}/predict", json=sample_tracks)
        logger.info(f"Prediction Status: {response.status_code}")
        
        if response.status_code == 200:
            predictions = response.json()
            logger.info(f"Predictions: {json.dumps(predictions, indent=2)}")
            return True
        else:
            logger.error(f"Prediction failed with status {response.status_code}")
            logger.error(f"Error message: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Prediction request failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Testing API Health...")
    test_health_endpoint()
    
    logger.info("\nTesting Prediction Endpoint...")
    test_prediction_endpoint()
