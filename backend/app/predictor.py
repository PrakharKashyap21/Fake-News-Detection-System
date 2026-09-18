import os
import math
import re
import joblib
from typing import Dict, Any

class NewsPredictor:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

        self.vectorizer_path = os.path.join(project_root, "backend", "models", "tfidf_vectorizer.joblib")
        self.model_path = os.path.join(project_root, "backend", "models", "linear_svm_model.joblib")

        if not os.path.exists(self.vectorizer_path) or not os.path.exists(self.model_path):
            # Fallback to relative path
            self.vectorizer_path = os.path.join("backend", "models", "tfidf_vectorizer.joblib")
            self.model_path = os.path.join("backend", "models", "linear_svm_model.joblib")

        if not os.path.exists(self.vectorizer_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"CRITICAL ERROR: Production model artifacts not found at {self.vectorizer_path} or {self.model_path}."
            )

        # Load models once during initialization
        print(f"Loading production TF-IDF vectorizer from: {self.vectorizer_path}...")
        self.vectorizer = joblib.load(self.vectorizer_path)

        print(f"Loading production LinearSVM model from:   {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print("Production models loaded into memory successfully.")

    def _normalize_text(self, text: str) -> str:
        """Safely normalizes input text by stripping and reducing repeated whitespace."""
        if not text:
            return ""
        text_str = str(text).strip()
        return re.sub(r"\s+", " ", text_str)

    def predict(self, title: str = "", text: str = "") -> Dict[str, Any]:
        norm_title = self._normalize_text(title)
        norm_text = self._normalize_text(text)

        # Combine title and body matching training pipeline logic
        if norm_title and norm_text:
            combined_content = f"{norm_title} {norm_text}"
        elif norm_title:
            combined_content = norm_title
        elif norm_text:
            combined_content = norm_text
        else:
            raise ValueError("Content cannot be empty after text normalization.")

        # Transform using pre-fitted TF-IDF vectorizer (without refitting)
        X_vec = self.vectorizer.transform([combined_content])

        # Model prediction and decision score
        pred_label = int(self.model.predict(X_vec)[0])
        decision_score = float(self.model.decision_function(X_vec)[0])

        # Note: This is a transformed SVM decision-margin score (|decision_margin|), not a calibrated probability.
        confidence_margin = abs(decision_score)
        confidence_percent = round(min(99.99, 100.0 * (1.0 - math.exp(-confidence_margin))), 2)

        if pred_label == 0:
            prediction_str = "LIKELY FAKE NEWS"
            message_str = "The model detected linguistic patterns associated with fake-news examples in its training data."
        else:
            prediction_str = "LIKELY REAL NEWS"
            message_str = "The model detected linguistic patterns associated with real-news examples in its training data."

        return {
            "prediction": prediction_str,
            "label": pred_label,
            "confidence": confidence_percent,
            "message": message_str
        }

# Global singleton instance loaded once at startup
predictor_instance = None

def get_predictor() -> NewsPredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = NewsPredictor()
    return predictor_instance
