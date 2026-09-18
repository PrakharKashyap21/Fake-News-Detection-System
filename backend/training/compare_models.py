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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    x_train_path = os.path.join(project_root, "backend", "data", "X_train_tfidf.npz")
    y_train_path = os.path.join(project_root, "backend", "data", "y_train.csv")

    if not os.path.exists(x_train_path):
        x_train_path = os.path.join("backend", "data", "X_train_tfidf.npz")
        y_train_path = os.path.join("backend", "data", "y_train.csv")

    print("=" * 70)
    print("WELFake Model Comparison (Inner Train/Validation Split)")
    print("=" * 70)

    # 28. Fail loudly if required files are missing
    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Training artifact(s) missing: {x_train_path}, {y_train_path}")

    print("1. LOADING SPARSE TRAINING TF-IDF MATRIX AND LABELS...")
    # 1 & 3. Keep matrix in sparse format
    X_train_full = load_npz(x_train_path)
    y_train_full = pd.read_csv(y_train_path)["label"].values

    # 4. Verifications
    if X_train_full.shape[0] != len(y_train_full):
        raise ValueError(f"CRITICAL ERROR: X_train rows ({X_train_full.shape[0]}) do not match y_train length ({len(y_train_full)})!")

    unique_labels = set(np.unique(y_train_full))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"CRITICAL ERROR: Invalid label values found in y_train: {unique_labels}. Expected [0, 1].")

    print("   [✓] Input verification passed.")
    print(f"   - Full training set shape: {X_train_full.shape[0]:,} samples x {X_train_full.shape[1]:,} features")

    # 5. Create reproducible stratified validation split (80% inner train, 20% validation)
    print("\n2. CREATING STRATIFIED INNER-TRAIN & VALIDATION SPLIT (80/20)...")
    X_inner_train, X_val, y_inner_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full
    )

    # 28. Fail loudly if either class is missing
    if set(np.unique(y_inner_train)) != {0, 1}:
        raise ValueError("CRITICAL ERROR: Inner training split is missing target class(es)!")

    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError("CRITICAL ERROR: Validation split is missing target class(es)!")

    print(f"   - Inner Training Set: {X_inner_train.shape[0]:,} samples")
    print(f"   - Validation Set:     {X_val.shape[0]:,} samples")

    # 7. Define Candidate Classifiers
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear"
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0
        )
    }

    results = []

    print("\n" + "=" * 70)
    print("3. TRAINING & EVALUATING CANDIDATE MODELS")
    print("=" * 70)

    for name, clf in models.items():
        print(f"\n--- {name} ---")
        
        # 12. Train on inner-train
        start_train = time.time()
        clf.fit(X_inner_train, y_inner_train)
        train_time = time.time() - start_train

        # 13. Evaluate on validation set
        start_pred = time.time()
        y_val_pred = clf.predict(X_val)
        pred_time = time.time() - start_pred

        # 14. Metrics calculation
        acc = accuracy_score(y_val, y_val_pred)
        
        macro_p = precision_score(y_val, y_val_pred, average="macro")
        macro_r = recall_score(y_val, y_val_pred, average="macro")
        macro_f1 = f1_score(y_val, y_val_pred, average="macro")

        fake_p = precision_score(y_val, y_val_pred, pos_label=0)
        fake_r = recall_score(y_val, y_val_pred, pos_label=0)
        fake_f1 = f1_score(y_val, y_val_pred, pos_label=0)

        real_p = precision_score(y_val, y_val_pred, pos_label=1)
        real_r = recall_score(y_val, y_val_pred, pos_label=1)
        real_f1 = f1_score(y_val, y_val_pred, pos_label=1)

        # 15 & 16. Confusion Matrix (rows = actual, cols = predicted, labels = [0, 1])
        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"Training Time:   {train_time:.4f} seconds")
        print(f"Prediction Time: {pred_time:.4f} seconds")
        print(f"Accuracy:        {acc * 100:.2f}%")
        print(f"Macro F1-Score:  {macro_f1:.4f}")
        print(f"FAKE Class F1:   {fake_f1:.4f} (Precision: {fake_p:.4f}, Recall: {fake_r:.4f})")
        print(f"REAL Class F1:   {real_f1:.4f} (Precision: {real_p:.4f}, Recall: {real_r:.4f})")
        
        print("\nConfusion Matrix:")
        print("                  Predicted")
        print("                  FAKE      REAL")
        print(f"Actual FAKE     {tn:>6d}    {fp:>6d}")
        print(f"Actual REAL     {fn:>6d}    {tp:>6d}")
        print(f"  [TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}]")

        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro P": macro_p,
            "Macro R": macro_r,
            "Macro F1": macro_f1,
            "FAKE P": fake_p,
            "FAKE R": fake_r,
            "FAKE F1": fake_f1,
            "REAL P": real_p,
            "REAL R": real_r,
            "REAL F1": real_f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Train Time (s)": train_time,
            "Pred Time (s)": pred_time
        })

    # 18. Final Comparison Table sorted by Macro F1 Score
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("4. MODEL VALIDATION COMPARISON SUMMARY TABLE")
    print("=" * 70)
    summary_cols = ["Model", "Accuracy", "Macro F1", "FAKE F1", "REAL F1", "Train Time (s)", "Pred Time (s)"]
    print(results_df_sorted[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)

    print("\nNOTE: Final model selection & test evaluation will be performed in subsequent steps.")

if __name__ == "__main__":
    main()
