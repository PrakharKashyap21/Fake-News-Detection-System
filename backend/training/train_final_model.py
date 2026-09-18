import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    train_path = os.path.join(project_root, "backend", "data", "train.csv")
    test_path = os.path.join(project_root, "backend", "data", "test.csv")
    vectorizer_path = os.path.join(project_root, "backend", "models", "tfidf_vectorizer.joblib")
    model_path = os.path.join(project_root, "backend", "models", "linear_svm_model.joblib")

    if not os.path.exists(train_path):
        train_path = os.path.join("backend", "data", "train.csv")
        test_path = os.path.join("backend", "data", "test.csv")
        vectorizer_path = os.path.join("backend", "models", "tfidf_vectorizer.joblib")
        model_path = os.path.join("backend", "models", "linear_svm_model.joblib")

    print("=" * 70)
    print("Final Production Model Training & Held-Out Test Evaluation")
    print("=" * 70)

    # 1 & 2. Load and verify datasets
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Input dataset(s) missing: {train_path}, {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    expected_cols = {"content", "label"}
    if not expected_cols.issubset(set(train_df.columns)) or not expected_cols.issubset(set(test_df.columns)):
        raise ValueError("CRITICAL ERROR: Dataset missing required columns 'content' or 'label'")

    if len(train_df) != 50937:
        print(f"WARNING: train.csv row count is {len(train_df):,} (expected 50,937)")

    if len(test_df) != 12735:
        print(f"WARNING: test.csv row count is {len(test_df):,} (expected 12,735)")

    train_labels = set(train_df["label"].unique())
    test_labels = set(test_df["label"].unique())

    if not train_labels.issubset({0, 1}) or not test_labels.issubset({0, 1}):
        raise ValueError(f"CRITICAL ERROR: Invalid label values found! Train: {train_labels}, Test: {test_labels}")

    # 3. Verify no normalized content overlap
    train_content_norm = set(train_df["content"].str.lower().str.strip())
    test_content_norm = set(test_df["content"].str.lower().str.strip())
    overlap = train_content_norm.intersection(test_content_norm)

    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Data leakage detected! {len(overlap)} overlapping records found between train and test.")

    print("1. DATA INTEGRITY VERIFICATION PASSED:")
    print(f"   - Training Samples Count: {len(train_df):,}")
    print(f"   - Testing Samples Count:  {len(test_df):,}")
    print(f"   - Content Overlap:        {len(overlap)} items")

    # 4 & 5. Fit FINAL TfidfVectorizer on ALL training content
    print("\n2. FITTING FINAL TF-IDF VECTORIZER ON ALL TRAINING DATA...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )

    t0_fit = time.time()
    X_train = vectorizer.fit_transform(train_df["content"])
    t1_fit = time.time()
    tfidf_fit_time = t1_fit - t0_fit

    t0_trans = time.time()
    X_test = vectorizer.transform(test_df["content"])
    t1_trans = time.time()
    tfidf_trans_time = t1_trans - t0_trans

    vocab_size = len(vectorizer.vocabulary_)

    # 6. Train final LinearSVC
    print("3. TRAINING FINAL LINEARSVC CLASSIFIER (C=2.0)...")
    model = LinearSVC(
        C=2.0,
        max_iter=5000,
        random_state=42
    )

    t0_train = time.time()
    model.fit(X_train, train_df["label"])
    t1_train = time.time()
    model_train_time = t1_train - t0_train

    total_training_time = tfidf_fit_time + model_train_time

    # 7. Evaluate final model on locked test set (EXACTLY ONCE)
    print("4. EVALUATING FINAL MODEL ON LOCKED TEST SET...")
    t0_pred = time.time()
    y_test_pred = model.predict(X_test)
    t1_pred = time.time()
    model_pred_time = t1_pred - t0_pred

    total_prediction_time = tfidf_trans_time + model_pred_time

    y_test_true = test_df["label"].values

    # Metrics
    accuracy = accuracy_score(y_test_true, y_test_pred)
    macro_f1 = f1_score(y_test_true, y_test_pred, average="macro")
    weighted_f1 = f1_score(y_test_true, y_test_pred, average="weighted")

    fake_p = precision_score(y_test_true, y_test_pred, pos_label=0)
    fake_r = recall_score(y_test_true, y_test_pred, pos_label=0)
    fake_f1 = f1_score(y_test_true, y_test_pred, pos_label=0)

    real_p = precision_score(y_test_true, y_test_pred, pos_label=1)
    real_r = recall_score(y_test_true, y_test_pred, pos_label=1)
    real_f1 = f1_score(y_test_true, y_test_pred, pos_label=1)

    cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 9. Save Production Artifacts
    print("\n5. SAVING PRODUCTION MODEL ARTIFACTS...")
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(vectorizer, vectorizer_path)
    print(f"   - Saved TF-IDF Vectorizer: {vectorizer_path}")

    joblib.dump(model, model_path)
    print(f"   - Saved LinearSVC Model:   {model_path}")

    # 10. Print required final summary section
    print("\n" + "=" * 70)
    print("FINAL MODEL SUMMARY & EVALUATION")
    print("=" * 70)

    print("FINAL MODEL")
    print("-----------")
    print(f"TF-IDF: TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)")
    print(f"Classifier: LinearSVC(C=2.0, max_iter=5000, random_state=42)")
    print(f"Vocabulary: {vocab_size:,} n-grams")
    print(f"Training Samples: {len(train_df):,}")
    print(f"Test Samples:     {len(test_df):,}")
    print(f"Train Non-Zero Entries (nnz): {X_train.nnz:,}")
    print(f"Test Non-Zero Entries (nnz):  {X_test.nnz:,}")
    print(f"Training Time:    {total_training_time:.2f} seconds")
    print(f"Prediction Time:  {total_prediction_time:.2f} seconds")

    print("\nFINAL TEST PERFORMANCE")
    print("----------------------")
    print(f"Accuracy:    {accuracy * 100:.2f}% ({accuracy:.4f})")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nFAKE (Label 0):")
    print(f"Precision:   {fake_p * 100:.2f}% ({fake_p:.4f})")
    print(f"Recall:      {fake_r * 100:.2f}% ({fake_r:.4f})")
    print(f"F1:          {fake_f1:.4f}")

    print("\nREAL (Label 1):")
    print(f"Precision:   {real_p * 100:.2f}% ({real_p:.4f})")
    print(f"Recall:      {real_r * 100:.2f}% ({real_r:.4f})")
    print(f"F1:          {real_f1:.4f}")

    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  FAKE      REAL")
    print(f"Actual FAKE     {tn:>6d}    {fp:>6d}")
    print(f"Actual REAL     {fn:>6d}    {tp:>6d}")
    print(f"  [TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}]")

    print("\nFull Classification Report:")
    print("-" * 70)
    print(classification_report(y_test_true, y_test_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4))

    # 11. Explicit methodology statement
    print("=" * 70)
    print("Test set was used only for final evaluation and was not used for model selection.")
    print("=" * 70)

if __name__ == "__main__":
    main()
