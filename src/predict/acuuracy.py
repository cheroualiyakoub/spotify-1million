import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import mlflow
import tempfile
import os

class ModelEvaluator:
    def __init__(self, pipeline, X_test, y_test):
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_proba = None

    def evaluate(self, log_to_mlflow=False):
        # Generate predictions
        self.y_pred = self.pipeline.predict(self.X_test)
        
        # Generate probabilities if available
        if hasattr(self.pipeline, 'predict_proba'):
            self.y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1': f1_score(self.y_test, self.y_pred)
        }
        
        if self.y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_proba)

        # Generate visualizations
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Confusion Matrix plots
            cm_path = self._plot_confusion_matrix(
                confusion_matrix(self.y_test, self.y_pred),
                title="Confusion Matrix",
                log_to_mlflow=log_to_mlflow,
                tmp_dir=tmp_dir
            )
            
            # Normalized Confusion Matrix
            cm_norm_path = self._plot_confusion_matrix(
                confusion_matrix(self.y_test, self.y_pred, normalize='true'),
                title="Normalized Confusion Matrix",
                normalized=True,
                log_to_mlflow=log_to_mlflow,
                tmp_dir=tmp_dir
            )

            # Log to MLflow if requested
            if log_to_mlflow:
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(cm_norm_path)
                
                # Log classification report as text
                report = classification_report(self.y_test, self.y_pred)
                report_path = os.path.join(tmp_dir, "classification_report.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                mlflow.log_artifact(report_path)

        return metrics

    def _plot_confusion_matrix(self, cm, title, normalized=False, log_to_mlflow=False, tmp_dir=None):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalized else "d", cmap='Blues')
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if log_to_mlflow:
            filename = f"{title.lower().replace(' ', '_')}.png"
            path = os.path.join(tmp_dir, filename)
            plt.savefig(path)
            plt.close()
            return path
        else:
            plt.tight_layout()
            plt.show()
            return None