import mlflow
import numpy as np
import json
import git
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlflow.models import infer_signature
from mlflow.pyfunc import log_model
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class MLflowExperimentRunner:
    """
    A reusable class for running and tracking machine learning experiments with MLflow.
    
    This class provides a structured way to:
    - Train models with preprocessing and sampling
    - Log comprehensive experiment metadata
    - Track model performance
    - Register models in the MLflow model registry
    """
    
    def __init__(
        self, 
        experiment_name="Default_Experiment",
        tracking_uri="http://localhost:5000",
        evaluator_class=None
    ):
        """
        Initialize the experiment runner.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the MLflow experiment to use
        tracking_uri : str
            URI for the MLflow tracking server
        evaluator_class : class, optional
            Default evaluator class to use if not specified in experiments
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.evaluator_class = evaluator_class
        
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
    
    def _get_git_commit(self):
        """Get the current git commit hash."""
        try:
            repo = git.Repo(Path.cwd(), search_parent_directories=True)
            return repo.head.object.hexsha
        except Exception:
            return None
    
    def _log_feature_importance(self, model, feature_names, tmp_dir):
        """Create and log feature importance plot."""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            
            # Create series with proper feature names
            importances = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values()
            
            importances.plot.barh()
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(f"{tmp_dir}/feature_importance.png")
            plt.close()
            
    def _get_feature_names(self, preprocessor, model):
        """Extract feature names from the preprocessor."""
        try:
            encoder = preprocessor.named_steps.get('encoder', None)
            if encoder and hasattr(encoder, 'get_feature_names_out'):
                return encoder.get_feature_names_out()
            else:
                return [f"feature_{i}" for i in range(len(model.feature_importances_))]
        except Exception as e:
            print(f"Could not retrieve feature names due to: {e}")
            return [f"feature_{i}" for i in range(len(model.feature_importances_))]
    
    def run_experiments(self, experiments):
        """
        Run multiple experiments from a list of experiment configurations.
        
        Parameters:
        -----------
        experiments : list
            List of dictionaries containing experiment configurations
            
        Returns:
        --------
        list
            List of experiment results
        """
        results = []
        for exp_config in experiments:
            # Create a copy of the configuration to avoid modifying the original
            exp = exp_config.copy()
            
            # Use default evaluator class if not specified
            evaluator_class = exp.pop('evaluator_class', self.evaluator_class)
            if evaluator_class is None:
                raise ValueError("No evaluator_class provided. Either set it during initialization or include it in experiment config.")
                
            # Set experiment name if provided
            custom_experiment_name = exp.pop('experiment_name', None)
            if custom_experiment_name:
                original_experiment_name = self.experiment_name
                self.experiment_name = custom_experiment_name
            
            # Run the experiment
            try:
                result = self.run_experiment(evaluator_class=evaluator_class, **exp)
                results.append(result)
            finally:
                # Restore original experiment name if it was changed
                if custom_experiment_name:
                    self.experiment_name = original_experiment_name
                    
        return results
    
    def run_experiment(
        self,
        model: BaseEstimator, 
        preprocessor: Pipeline, 
        X_train, y_train, 
        X_val, y_val,
        evaluator_class=None,
        sampler=None,
        sampler_params=None,
        model_params=None,
        dataset_version="1.0.0",
        input_example=None,
        metadata=None,
        tags=None,
        log_artifacts=True,
        registered_model_name=None
    ):
        """
        Run a machine learning experiment with comprehensive MLflow logging.
        
        Parameters:
        -----------
        model : BaseEstimator
            Scikit-learn compatible model
        preprocessor : Pipeline
            Preprocessing pipeline
        X_train : DataFrame
            Training features
        y_train : Series
            Training target
        X_val : DataFrame
            Validation features
        y_val : Series
            Validation target
        evaluator_class : class, optional
            Evaluator class with evaluate method that returns metrics
        sampler : object, optional
            Data resampling strategy (like SMOTE)
        sampler_params : dict, optional
            Parameters for the sampler
        model_params : dict, optional
            Parameters for the model
        dataset_version : str, optional
            Version identifier for the dataset
        input_example : DataFrame, optional
            Sample input data for model signature
        metadata : dict, optional
            Additional metadata to log
        tags : dict, optional
            Key-value pairs for experiment organization
        log_artifacts : bool, optional
            Whether to save feature importance and configs
        registered_model_name : str, optional
            Name for model registry
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Initialize default parameters
        sampler_params = sampler_params or {}
        model_params = model_params or {}
        tags = tags or {}
        metadata = metadata or {}
        
        # Use instance evaluator_class if none provided
        if evaluator_class is None:
            evaluator_class = self.evaluator_class
            if evaluator_class is None:
                raise ValueError("No evaluator_class provided")
        
        # Set up MLflow
        self._setup_mlflow()
        
        with mlflow.start_run():
            # Get git commit if available
            git_commit = self._get_git_commit()
            if git_commit:
                tags["git_commit"] = git_commit
            
            # =====================
            # Core Experiment Logic
            # =====================
            # Preprocess data
            X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
            
            # Apply sampling
            if sampler:
                sampler_instance = sampler(**sampler_params)
                X_resampled, y_resampled = sampler_instance.fit_resample(X_train_preprocessed, y_train)
            else:
                X_resampled, y_resampled = X_train_preprocessed, y_train

            # Create and train model
            final_model = model.set_params(**model_params)
            final_model.fit(X_resampled, y_resampled)

            # =====================
            # MLflow Logging
            # =====================
            # Create pipeline and signature
            full_pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", final_model)
            ])
            
            # Generate input example and signature
            input_example = X_train.sample(5, random_state=42) if input_example is None else input_example
            signature = infer_signature(input_example, full_pipeline.predict(input_example))

            # Log parameters
            sampler_name = sampler.__name__ if sampler else "None"
            sampler_params_prefixed = {f"sampler_{k}": v for k, v in (sampler_params or {}).items()}
            
            mlflow.log_params({
                "sampler": sampler_name,
                "model": model.__class__.__name__,
                **sampler_params_prefixed,
                **model_params,
                "dataset_version": dataset_version
            })
            
            # For full transparency, also log the sampler object
            if sampler:
                mlflow.log_dict(
                    {
                        "sampler_class": sampler_name,
                        "sampler_params": sampler_params or {},
                    },
                    "sampler_config.json"
                )

            # Tags and metadata
            project_name = metadata.get("project_name", "ml_project")
            mlflow.set_tags({
                **tags,
                "project": project_name,
                "model_type": metadata.get("model_type", "classifier"),
                **metadata
            })

            # Model logging with metadata
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                metadata={
                    "features": list(X_train.columns),
                    "target": y_train.name if hasattr(y_train, 'name') else "target",
                    **metadata
                }
            )

            # =====================
            # Additional Artifacts
            # =====================
            if log_artifacts:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Feature importance plot
                    feature_names = self._get_feature_names(preprocessor, final_model)
                    self._log_feature_importance(final_model, feature_names, tmp_dir)

                    # Preprocessing config
                    preprocessor_config = {
                        "steps": list(preprocessor.named_steps.keys()),
                        "params": preprocessor.get_params()
                    }
                    # with open(f"{tmp_dir}/preprocessor_config.json", 'w') as f:
                    #     json.dump(preprocessor_config, f)

                    mlflow.log_artifacts(tmp_dir)

            # =====================
            # Evaluation and Metrics
            # =====================
            evaluator = evaluator_class(full_pipeline, X_val, y_val)
            metrics = evaluator.evaluate(log_to_mlflow=True)
            
            return metrics
