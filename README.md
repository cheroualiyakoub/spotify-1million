spotify-popularity-classifier/
│
├── data/                        # Raw data or processed subset
│   └── spotify.csv
│
├── notebooks/                   # EDA or prototyping notebooks
│   └── 01_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── load_data.py         # Read/preprocess data
│   ├── features/
│   │   └── build_features.py    # Pipelines, transformations
│   ├── models/
│   │   └── train_model.py       # Training logic + MLflow logging
│   ├── predict/
│   │   └── predict.py           # Load model and make predictions
│
├── api/
│   └── main.py                  # FastAPI or Flask server
│
├── streamlit_app/
│   └── app.py                   # Streamlit frontend
│
├── mlruns/                      # MLflow tracking folder
│
├── requirements.txt
├── Dockerfile                   # For containerized deployment
├── build_pipeline.yaml          # Optional: CI/CD with GitHub Actions
└── README.md


✅ Development Roadmap
1. EDA & Preprocessing (notebooks/, src/data)
Analyze class balance and key features.

Convert the target into binary classification (popular = 1 or 0).

2. Feature Engineering (src/features)
Build a preprocessing pipeline using Pipeline([...]) for both numerical and categorical features.

3. Model Training (src/models/train_model.py)
Train classification models and log with MLflow:

Log hyperparameters and metrics (accuracy, F1-score, AUC, etc.).

Save the fitted pipeline using mlflow.sklearn.log_model(...).

Register the trained model in the MLflow Model Registry.

4. Prediction API (api/main.py)
Create a POST endpoint using FastAPI or Flask:

Accepts track features in JSON format.

Loads the production model via mlflow.pyfunc.load_model(...).

Returns prediction as JSON: { "popular": 1 } or { "popular": 0 }.

5. Frontend UI (streamlit_app/app.py)
Use Streamlit sliders for features like tempo, energy, etc.

Call the backend API and display the prediction result in real time.

6. Model Registry (via MLflow UI or code)
Register your best-performing model under a name like "spotify_classifier".

Promote the selected model to "Staging" or "Production" stage.

7. (Optional) CI/CD & Containerization
Add Docker support.

Set up GitHub Actions or other CI/CD pipelines for automated deployment.

