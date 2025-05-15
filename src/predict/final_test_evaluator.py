import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.tracking import MlflowClient


class FinalTestEvaluator:
    def __init__(self, test_data_path, target_column, model_uri, evaluator_class, expirement_name):
        self.test_data_path = test_data_path
        self.target_column = target_column
        self.model_uri = model_uri
        self.evaluator_class = evaluator_class
        self.expirement_name = expirement_name

    def run(self):
        mlflow.set_experiment(self.expirement_name)
        with mlflow.start_run(nested=True) as run:
            self.run_id = run.info.run_id
            mlflow.set_tag("phase", "final_test")
            mlflow.set_tag("validated", "true")
            mlflow.set_tag("model_uri_tested", self.model_uri)

            if self.model_uri.startswith("models:/"):
                uri_parts = self.model_uri.replace("models:/", "").split("/")
                if len(uri_parts) == 2:
                    model_name, stage_or_version = uri_parts
                    client = mlflow.tracking.MlflowClient()

                    # Try to get the version info
                    try:
                        if stage_or_version.lower() in ["staging", "production"]:
                            versions = client.get_latest_versions(model_name, [stage_or_version])
                            if versions:
                                version = versions[0].version
                                source_run_id = versions[0].run_id
                        else:
                            version = stage_or_version
                            mv = client.get_model_version(model_name, version)
                            source_run_id = mv.run_id

                        # Set experiment of source run (optional)
                        parent_experiment_id = client.get_run(source_run_id).info.experiment_id
                        mlflow.set_experiment(experiment_id=parent_experiment_id)

                        # Tag test run with model info
                        mlflow.set_tag("model_uri_tested", self.model_uri)
                        mlflow.set_tag("model_name_tested", model_name)
                        mlflow.set_tag("model_version_or_stage_tested", stage_or_version)
                        mlflow.set_tag("linked_model_version", f"{model_name}:{version}")
                        mlflow.set_tag("linked_model_run_id", source_run_id)

                    except Exception as e:
                        print(f"[WARN] Could not resolve model version info: {e}")
                else:
                    print("[WARN] Model URI must be in format models:/<name>/<version_or_stage>")



            self.load_data()
            self.load_model()
            self.predict()
            self.evaluate()

            # Log metrics and artifacts with clear test prefixes
            mlflow.log_metric("test_accuracy", self.metrics["accuracy"])
            self.log_confusion_matrix()
            self.log_classification_report()

            # Also tag model version in the registry
            self.tag_model_version()

            return self.metrics

    def load_data(self):
        self.data = pd.read_csv(self.test_data_path)
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

    def load_model(self):
        self.model = mlflow.pyfunc.load_model(self.model_uri)

    def predict(self):
        self.y_pred = self.model.predict(self.X)

    def evaluate(self):
        acc = accuracy_score(self.y, self.y_pred)
        conf_matrix = confusion_matrix(self.y, self.y_pred)
        class_report = classification_report(self.y, self.y_pred, output_dict=True)

        self.metrics = {
            "accuracy": acc
        }
        self.conf_matrix = conf_matrix
        self.class_report = class_report

    def log_confusion_matrix(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Final Test Confusion Matrix")
        plt.tight_layout()
        fig_path = "test_confusion_matrix.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)

    def log_classification_report(self):
        df = pd.DataFrame(self.class_report).transpose()
        report_path = "test_classification_report.csv"
        df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

    def tag_model_version(self):
        if self.model_uri.startswith("models:/"):
            try:
                client = MlflowClient()

                # Extract model name and version from URI
                parts = self.model_uri.replace("models:/", "").split("/")
                if len(parts) != 2:
                    print("Invalid model URI format for tagging.")
                    return

                model_name, version_or_stage = parts

                # If it's a stage (e.g., 'Staging'), resolve to version
                if version_or_stage.lower() in ["staging", "production", "none", "archived"]:
                    version = client.get_latest_versions(model_name, stages=[version_or_stage])[0].version
                else:
                    version = version_or_stage

                # Tag the model version
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="final_tested",
                    value="true"
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="final_test_run_id",
                    value=self.run_id
                )
            except Exception as e:
                print(f"Failed to tag model version: {e}")
        
    def get_prediction_probabilities(self):
        """
        Returns predicted class probabilities if supported by the model.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict(self.X)
        elif hasattr(self.model, "predict"):
            try:
                return self.model.predict(self.X)
            except TypeError:
                raise NotImplementedError("Model does not support probability predictions.")
        else:
            raise NotImplementedError("Model does not have predict_proba method.")

