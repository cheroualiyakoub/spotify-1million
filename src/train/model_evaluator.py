# A simple ModelEvaluator implementation for illustration
class ModelEvaluator:
    """
    Model evaluator for classification tasks.
    
    This evaluator calculates common classification metrics including
    accuracy, precision, recall, F1 score, and AUC.
    """
    
    def __init__(self, model, X_test, y_test):
        """
        Initialize the evaluator with a model and test data.
        
        Parameters:
        -----------
        model : trained model or pipeline
            The trained model to evaluate
        X_test : DataFrame
            Test features
        y_test : Series
            Test target values
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
    def evaluate(self, log_to_mlflow=True):
        """
        Evaluate the model and optionally log metrics to MLflow.
        
        Parameters:
        -----------
        log_to_mlflow : bool
            Whether to log metrics to MLflow
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        import mlflow
        
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        
        # For AUC and other metrics that need probabilities
        try:
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            has_proba = True
        except:
            has_proba = False
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_test, y_pred, average='weighted'),
            "f1": f1_score(self.y_test, y_pred, average='weighted'),
        }
        
        # Add AUC if probabilities are available
        if has_proba:
            try:
                metrics["auc"] = roc_auc_score(self.y_test, y_prob)
            except:
                # In case of single class
                pass
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Log metrics to MLflow if requested
        if log_to_mlflow:
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            
            # Log confusion matrix as a figure
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                
                # Save confusion matrix to a temporary file and log it
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                    plt.savefig(tmp.name)
                    mlflow.log_artifact(tmp.name, "confusion_matrix.png")
                plt.close()
            except Exception as e:
                print(f"Could not log confusion matrix: {e}")
        
        return metrics