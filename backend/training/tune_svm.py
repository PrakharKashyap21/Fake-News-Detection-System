import gc
import os
import time
import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    train_path = os.path.join(project_root, "backend", "data", "train.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("backend", "data", "train.csv")

    print("=" * 75)
    print("WELFake Controlled Hyperparameter & TF-IDF Optimization for Linear SVM")
    print("=" * 75)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Training file not found at {train_path}!")

    print(f"Loading training data from: {train_path}...\n")
    df_train = pd.read_csv(train_path)

    if "content" not in df_train.columns or "label" not in df_train.columns:
        raise ValueError("CRITICAL ERROR: train.csv must contain 'content' and 'label' columns.")

    total_train_rows = len(df_train)
    print(f"1. INITIAL DATASET VERIFICATION PASSED: {total_train_rows:,} rows")

    # 2. Create internal stratified train/validation split (80% inner train, 20% validation)
    print("\n2. CREATING STRATIFIED INNER-TRAIN & VALIDATION SPLIT (80/20)...")
    train_text, val_text, y_inner_train, y_val = train_test_split(
        df_train["content"],
        df_train["label"].values,
        test_size=0.20,
        random_state=42,
        stratify=df_train["label"].values
    )

    inner_train_size = len(train_text)
    val_size = len(val_text)

    print(f"   - Inner Training Set Size: {inner_train_size:,} samples")
    print(f"   - Validation Set Size:     {val_size:,} samples")

    # Define TF-IDF configurations
    tfidf_configs = {
        "Config A (Baseline: unigram+bigram, min_df=2)": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True
        },
        "Config B (Unigram-only: min_df=2)": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True
        },
        "Config C (Unigram+bigram, min_df=3)": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 2),
            "min_df": 3,
            "max_df": 0.95,
            "sublinear_tf": True
        },
        "Config D (Unigram+bigram, min_df=2, max_df=0.98, max_features=1M)": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
            "max_features": 1000000
        }
    }

    c_values = [0.5, 1.0, 2.0]
    experiments = []

    print("\n" + "=" * 75)
    print("3. EXECUTING 12 HYPERPARAMETER OPTIMIZATION EXPERIMENTS")
    print("=" * 75)

    exp_num = 1
    best_y_val_pred = None
    best_model_obj = None

    for config_name, tfidf_params in tfidf_configs.items():
        print(f"\nEvaluating {config_name}...")
        
        # Fit TF-IDF ONLY on inner training text
        t0_tfidf = time.time()
        vectorizer = TfidfVectorizer(**tfidf_params)
        X_inner_train_sparse = vectorizer.fit_transform(train_text)
        t1_tfidf = time.time()
        tfidf_fit_time = t1_tfidf - t0_tfidf

        # Transform validation text
        X_val_sparse = vectorizer.transform(val_text)
        t2_tfidf = time.time()
        tfidf_transform_time = t2_tfidf - t1_tfidf

        vocab_size = len(vectorizer.vocabulary_)
        print(f"   [TF-IDF] Vocabulary Size: {vocab_size:,} | Fit Time: {tfidf_fit_time:.2f}s | Transform Time: {tfidf_transform_time:.2f}s")

        for c_val in c_values:
            print(f"   --> Experiment {exp_num}/12: LinearSVC(C={c_val})...", end=" ", flush=True)

            t0_svm = time.time()
            clf = LinearSVC(C=c_val, max_iter=5000, random_state=42)
            clf.fit(X_inner_train_sparse, y_inner_train)
            t1_svm = time.time()
            svm_train_time = t1_svm - t0_svm

            t0_pred = time.time()
            y_val_pred = clf.predict(X_val_sparse)
            t1_pred = time.time()
            svm_pred_time = t1_pred - t0_pred

            total_train_time = tfidf_fit_time + svm_train_time
            total_pred_time = tfidf_transform_time + svm_pred_time

            acc = accuracy_score(y_val, y_val_pred)
            macro_f1 = f1_score(y_val, y_val_pred, average="macro")

            fake_p = precision_score(y_val, y_val_pred, pos_label=0)
            fake_r = recall_score(y_val, y_val_pred, pos_label=0)
            fake_f1 = f1_score(y_val, y_val_pred, pos_label=0)

            real_p = precision_score(y_val, y_val_pred, pos_label=1)
            real_r = recall_score(y_val, y_val_pred, pos_label=1)
            real_f1 = f1_score(y_val, y_val_pred, pos_label=1)

            print(f"Macro F1: {macro_f1:.4f} | Accuracy: {acc*100:.2f}% | Train Time: {total_train_time:.2f}s")

            exp_record = {
                "Exp #": exp_num,
                "TF-IDF Config": config_name,
                "C": c_val,
                "Vocab Size": vocab_size,
                "Accuracy": acc,
                "Macro F1": macro_f1,
                "FAKE Precision": fake_p,
                "FAKE Recall": fake_r,
                "FAKE F1": fake_f1,
                "REAL Precision": real_p,
                "REAL Recall": real_r,
                "REAL F1": real_f1,
                "Train Time (s)": total_train_time,
                "Pred Time (s)": total_pred_time,
                "_y_val_pred": y_val_pred,
                "_config_name": config_name
            }
            experiments.append(exp_record)
            exp_num += 1

        # Cleanup memory for sparse matrices
        del X_inner_train_sparse, X_val_sparse, vectorizer
        gc.collect()

    # Create Summary Table
    df_exp = pd.DataFrame(experiments)
    
    # Primary sorting: Macro F1 descending, then Accuracy descending
    df_sorted = df_exp.sort_values(by=["Macro F1", "Accuracy"], ascending=[False, False]).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("4. EXPERIMENT RESULTS SUMMARY TABLE (Sorted by Macro F1)")
    print("=" * 75)

    display_cols = ["Exp #", "TF-IDF Config", "C", "Vocab Size", "Accuracy", "Macro F1", "FAKE F1", "REAL F1", "Train Time (s)"]
    print(df_sorted[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Select Best Configuration
    best_exp = df_sorted.iloc[0]

    print("\n" + "=" * 75)
    print("5. BEST CONFIGURATION EVALUATION & CLASSIFICATION DETAILS")
    print("=" * 75)
    print(f"Best Experiment Rank #1:")
    print(f"   - TF-IDF Configuration: {best_exp['TF-IDF Config']}")
    print(f"   - LinearSVC C Parameter: {best_exp['C']}")
    print(f"   - Learned Vocabulary Size: {best_exp['Vocab Size']:,}")
    print(f"   - Validation Accuracy: {best_exp['Accuracy'] * 100:.2f}% ({best_exp['Accuracy']:.4f})")
    print(f"   - Validation Macro F1: {best_exp['Macro F1']:.4f}")
    print(f"   - FAKE Class F1:       {best_exp['FAKE F1']:.4f}")
    print(f"   - REAL Class F1:       {best_exp['REAL F1']:.4f}")

    best_pred = best_exp["_y_val_pred"]

    print("\nBest Configuration Classification Report:")
    print("-" * 75)
    print(classification_report(y_val, best_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4))

    cm = confusion_matrix(y_val, best_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("-" * 75)
    print("Best Configuration Confusion Matrix:")
    print("-" * 75)
    print("                  Predicted")
    print("                  FAKE      REAL")
    print(f"Actual FAKE     {tn:>6d}    {fp:>6d}")
    print(f"Actual REAL     {fn:>6d}    {tp:>6d}")
    print("-" * 75)
    print(f"   - True Negatives  (TN - Actual FAKE, Predicted FAKE): {tn:,}")
    print(f"   - False Positives (FP - Actual FAKE, Predicted REAL): {fp:,}")
    print(f"   - False Negatives (FN - Actual REAL, Predicted FAKE): {fn:,}")
    print(f"   - True Positives  (TP - Actual REAL, Predicted REAL): {tp:,}")

    # Methodological Summary
    print("\n" + "=" * 75)
    print("METHODOLOGICAL EXPERIMENT SUMMARY")
    print("=" * 75)
    print(f"   - Total Experiments Run:            12")
    print(f"   - Inner Training Set Size:           {inner_train_size:,} samples")
    print(f"   - Internal Validation Set Size:      {val_size:,} samples")
    print(f"   - Best TF-IDF Configuration:         {best_exp['TF-IDF Config']}")
    print(f"   - Best LinearSVC C Parameter:       {best_exp['C']}")
    print(f"   - Best Validation Macro F1 Score:   {best_exp['Macro F1']:.4f}")
    print(f"   - Best Validation Accuracy:          {best_exp['Accuracy'] * 100:.2f}% ({best_exp['Accuracy']:.4f})")
    print("=" * 75)

if __name__ == "__main__":
    main()
