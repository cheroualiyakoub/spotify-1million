import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

class ModelEvaluator:
    def __init__(self, pipeline, X_test, y_test):
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None

    def evaluate(self):
        print("Running evaluation...")
        self.y_pred = self.pipeline.predict(self.X_test)

        # Accuracy
        acc = accuracy_score(self.y_test, self.y_pred)
        print(f"\nAccuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Normalized Confusion Matrix
        cm_normalized = confusion_matrix(self.y_test, self.y_pred, normalize='true')

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

        # Plot Confusion Matrix
        self._plot_confusion_matrix(cm, title="Confusion Matrix")
        self._plot_confusion_matrix(cm_normalized, title="Normalized Confusion Matrix", normalized=True)

    def _plot_confusion_matrix(self, cm, title, normalized=False):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalized else "d", cmap='Blues')
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
