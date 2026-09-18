import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    x_train_path = os.path.join(project_root, "backend", "data", "X_train_tfidf.npz")
    x_test_path = os.path.join(project_root, "backend", "data", "X_test_tfidf.npz")
    y_train_path = os.path.join(project_root, "backend", "data", "y_train.csv")
    y_test_path = os.path.join(project_root, "backend", "data", "y_test.csv")

    if not os.path.exists(x_train_path):
        x_train_path = os.path.join("backend", "data", "X_train_tfidf.npz")
        x_test_path = os.path.join("backend", "data", "X_test_tfidf.npz")
        y_train_path = os.path.join("backend", "data", "y_train.csv")
        y_test_path = os.path.join("backend", "data", "y_test.csv")

    print("=" * 65)
    print("Baseline Logistic Regression Model Training & Evaluation")
    print("=" * 65)

    # Check file existence
    missing_files = []
    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            missing_files.append(p)
    if missing_files:
        raise FileNotFoundError(f"CRITICAL ERROR: Missing required input file(s): {missing_files}")

    print("1. LOADING SPARSE TF-IDF MATRICES AND LABELS...")
    # 1 & 4. Load sparse matrices (KEEP IN SPARSE FORMAT)
    X_train = load_npz(x_train_path)
    X_test = load_npz(x_test_path)

    # 2. Load labels
    y_train = pd.read_csv(y_train_path)["label"].values
    y_test = pd.read_csv(y_test_path)["label"].values

    # 3. Verification checks
    if X_train.shape[0] != len(y_train):
        raise ValueError(f"CRITICAL ERROR: X_train rows ({X_train.shape[0]}) do not match y_train length ({len(y_train)})!")

    if X_test.shape[0] != len(y_test):
        raise ValueError(f"CRITICAL ERROR: X_test rows ({X_test.shape[0]}) do not match y_test length ({len(y_test)})!")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"CRITICAL ERROR: X_train feature count ({X_train.shape[1]}) does not match X_test ({X_test.shape[1]})!")

    unique_train_labels = set(np.unique(y_train))
    unique_test_labels = set(np.unique(y_test))

    if not unique_train_labels.issubset({0, 1}):
        raise ValueError(f"CRITICAL ERROR: y_train contains invalid labels: {unique_train_labels}")

    if not unique_test_labels.issubset({0, 1}):
        raise ValueError(f"CRITICAL ERROR: y_test contains invalid labels: {unique_test_labels}")

    print("   [✓] Data verification passed successfully.")
    print(f"   - Training set: {X_train.shape[0]:,} samples x {X_train.shape[1]:,} features")
    print(f"   - Testing set:  {X_test.shape[0]:,} samples x {X_test.shape[1]:,} features")

    # 5. Baseline LogisticRegression Classifier configuration
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    )

    # 13. Print model configuration used
    print("\n2. MODEL CONFIGURATION:")
    print(f"   - Classifier: {model.__class__.__name__}")
    print(f"   - Parameters: {model.get_params()}")

    # 6. Train model
    print("\n3. TRAINING BASELINE MODEL ON TRAIN DATA...")
    start_train_time = time.time()
    model.fit(X_train, y_train)
    end_train_time = time.time()
    train_duration = end_train_time - start_train_time
    print(f"   - Training completed in: {train_duration:.4f} seconds")

    # 7. Evaluate model
    print("\n4. EVALUATING BASELINE MODEL ON TEST DATA...")
    start_pred_time = time.time()
    y_pred = model.predict(X_test)
    end_pred_time = time.time()
    pred_duration = end_pred_time - start_pred_time
    print(f"   - Prediction completed in: {pred_duration:.4f} seconds")

    # 8. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Binary / Overall metrics (for label 1 = REAL)
    precision_overall = precision_score(y_test, y_pred, pos_label=1)
    recall_overall = recall_score(y_test, y_pred, pos_label=1)
    f1_overall = f1_score(y_test, y_pred, pos_label=1)

    # Specific FAKE metrics (for label 0 = FAKE)
    precision_fake = precision_score(y_test, y_pred, pos_label=0)
    recall_fake = recall_score(y_test, y_pred, pos_label=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=0)

    # Weighted metrics
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # 10 & 11. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Reports
    print("\n" + "=" * 65)
    print("BASELINE LOGISTIC REGRESSION EVALUATION REPORT")
    print("=" * 65)

    print(f"Timing Statistics:")
    print(f"   - Training Time:   {train_duration:.4f} seconds")
    print(f"   - Prediction Time: {pred_duration:.4f} seconds")

    print(f"\nOverall Performance Metrics:")
    print(f"   - Accuracy:           {accuracy * 100:.2f}% ({accuracy:.4f})")
    print(f"   - Weighted Precision: {precision_weighted * 100:.2f}% ({precision_weighted:.4f})")
    print(f"   - Weighted Recall:    {recall_weighted * 100:.2f}% ({recall_weighted:.4f})")
    print(f"   - Weighted F1-Score:  {f1_weighted * 100:.2f}% ({f1_weighted:.4f})")

    print(f"\nREAL Class (Label 1) Metrics:")
    print(f"   - Precision:          {precision_overall * 100:.2f}% ({precision_overall:.4f})")
    print(f"   - Recall:             {recall_overall * 100:.2f}% ({recall_overall:.4f})")
    print(f"   - F1-Score:           {f1_overall * 100:.2f}% ({f1_overall:.4f})")

    print(f"\nFAKE Class (Label 0) Metrics:")
    print(f"   - Precision:          {precision_fake * 100:.2f}% ({precision_fake:.4f})")
    print(f"   - Recall:             {recall_fake * 100:.2f}% ({recall_fake:.4f})")
    print(f"   - F1-Score:           {f1_fake * 100:.2f}% ({f1_fake:.4f})")

    # 9. Full Classification Report
    print("\n" + "-" * 65)
    print("CLASSIFICATION REPORT:")
    print("-" * 65)
    print(classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4))

    # 10. Formatted Confusion Matrix
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print("-" * 65)
    print("                  Predicted")
    print("                  FAKE      REAL")
    print(f"Actual FAKE     {tn:>6d}    {fp:>6d}")
    print(f"Actual REAL     {fn:>6d}    {tp:>6d}")
    print("-" * 65)
    print("Confusion Matrix Element Breakdown:")
    print(f"   - True Negatives  (TN - Actual FAKE, Predicted FAKE): {tn:,}")
    print(f"   - False Positives (FP - Actual FAKE, Predicted REAL): {fp:,}")
    print(f"   - False Negatives (FN - Actual REAL, Predicted FAKE): {fn:,}")
    print(f"   - True Positives  (TP - Actual REAL, Predicted REAL): {tp:,}")

    print("\n" + "=" * 65)
    print("BASELINE MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
