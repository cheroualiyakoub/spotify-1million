# Project Structure and Setup Guide

## Project Files


1. **main.py** - The FastAPI application entry point
2. **model_service.py** - Contains the ModelService class for loading and using the ML model
3. **predict_request.py** - Contains Pydantic models for API request validation
4. **mlflow_setup.py** - Helper script to check and setup MLflow environment
5. **debug_steps.py** - Diagnostic script for troubleshooting
6. **test_data.py** - Contains sample data for testing the API

## File Structure

```
spotify-prediction-api/
│
├── main.py                 # Main API application
├── model_service.py        # Model loading and prediction service
├── predict_request.py      # API request models
├── mlflow_setup.py         # MLflow setup helper
├── debug_steps.py          # Debugging utility
└── test_data.py            # Test data and test scripts
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pandas mlflow scikit-learn pydantic
```

### 2. Setup MLflow

Before running the API, ensure MLflow is properly set up:

```bash
# Start MLflow tracking server (in a separate terminal)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# In another terminal, run the MLflow setup helper
python mlflow_setup.py
```

### 3. Run the API

```bash
# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API

```bash
# Test the API endpoints
python test_data.py
```

## Troubleshooting

If you encounter issues:

1. Run the debugging script:
   ```bash
   python debug_steps.py
   ```

2. Check the MLflow setup:
   ```bash
   python mlflow_setup.py
   ```

3. Verify your model exists in MLflow:
   ```bash
   # Access MLflow UI
   # Open http://localhost:5000 in your browser
   ```

4. Check the logs:
   ```bash
   # Set log level to DEBUG for more information
   LOGLEVEL=DEBUG uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
   ```
