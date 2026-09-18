import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

def validate_dataframe(df: pd.DataFrame, filename: str):
    expected_cols = {"content", "label"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CRITICAL ERROR: {filename} missing required columns. Expected {expected_cols}, got {list(df.columns)}")
    
    if df["content"].isnull().sum() > 0:
        raise ValueError(f"CRITICAL ERROR: {filename} contains null content values!")

    if df["content"].fillna("").astype(str).str.strip().eq("").sum() > 0:
        raise ValueError(f"CRITICAL ERROR: {filename} contains empty string content values!")

    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"CRITICAL ERROR: {filename} contains invalid label values: {unique_labels}. Expected [0, 1].")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    train_path = os.path.join(project_root, "backend", "data", "train.csv")
    test_path = os.path.join(project_root, "backend", "data", "test.csv")
    
    vectorizer_path = os.path.join(project_root, "backend", "models", "tfidf_vectorizer.joblib")
    x_train_path = os.path.join(project_root, "backend", "data", "X_train_tfidf.npz")
    x_test_path = os.path.join(project_root, "backend", "data", "X_test_tfidf.npz")
    y_train_path = os.path.join(project_root, "backend", "data", "y_train.csv")
    y_test_path = os.path.join(project_root, "backend", "data", "y_test.csv")

    if not os.path.exists(train_path):
        train_path = os.path.join("backend", "data", "train.csv")
        test_path = os.path.join("backend", "data", "test.csv")
        vectorizer_path = os.path.join("backend", "models", "tfidf_vectorizer.joblib")
        x_train_path = os.path.join("backend", "data", "X_train_tfidf.npz")
        x_test_path = os.path.join("backend", "data", "X_test_tfidf.npz")
        y_train_path = os.path.join("backend", "data", "y_train.csv")
        y_test_path = os.path.join("backend", "data", "y_test.csv")

    print("=" * 65)
    print("WELFake Dataset TF-IDF Feature Engineering Pipeline")
    print("=" * 65)
    print(f"Loading train dataset from: {train_path}")
    print(f"Loading test dataset from:  {test_path}\n")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"CRITICAL ERROR: train.csv or test.csv not found!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2 & 3. Validate DataFrames
    validate_dataframe(train_df, "train.csv")
    validate_dataframe(test_df, "test.csv")

    print("1. INPUT DATA INTEGRITY CHECKS PASSED:")
    print(f"   - Training Documents Count: {len(train_df):,}")
    print(f"   - Testing Documents Count:  {len(test_df):,}")

    # 4 & 8 & 9. Baseline TfidfVectorizer Configuration
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    print("\n2. FITTING TF-IDF VECTORIZER ON TRAINING DATA ONLY...")
    # 5. Fit ONLY on training content
    X_train = vectorizer.fit_transform(train_df["content"])

    print("3. TRANSFORMING TEST CONTENT USING TRAINED VECTORIZER...")
    # 6 & 7. Transform test content (DO NOT FIT ON TEST)
    X_test = vectorizer.transform(test_df["content"])

    # 11. Verifications
    num_train_docs, num_vocab = X_train.shape
    num_test_docs, test_vocab_cols = X_test.shape

    if num_vocab != test_vocab_cols:
        raise ValueError(f"CRITICAL ERROR: X_train columns ({num_vocab}) and X_test columns ({test_vocab_cols}) do not match!")

    # Check for NaN / Inf in sparse matrices
    if not np.all(np.isfinite(X_train.data)):
        raise ValueError("CRITICAL ERROR: X_train contains non-finite (NaN or Inf) values!")

    if not np.all(np.isfinite(X_test.data)):
        raise ValueError("CRITICAL ERROR: X_test contains non-finite (NaN or Inf) values!")

    # Compute non-zero elements and sparsity
    train_nnz = X_train.nnz
    test_nnz = X_test.nnz
    
    train_sparsity = (1.0 - (train_nnz / (num_train_docs * num_vocab))) * 100
    test_sparsity = (1.0 - (test_nnz / (num_test_docs * num_vocab))) * 100

    # 10. Print clear TF-IDF report
    print("\n" + "-" * 65)
    print("TF-IDF FEATURE ENGINEERING REPORT")
    print("-" * 65)
    print(f"   - Training Document Count:          {num_train_docs:,}")
    print(f"   - Testing Document Count:           {num_test_docs:,}")
    print(f"   - Learned Vocabulary Size (n-grams):{num_vocab:,}")
    print(f"   - X_train Shape:                    {X_train.shape}")
    print(f"   - X_test Shape:                     {X_test.shape}")
    print(f"   - X_train Non-Zero Features (nnz):  {train_nnz:,}")
    print(f"   - X_test Non-Zero Features (nnz):   {test_nnz:,}")
    print(f"   - X_train Matrix Sparsity:          {train_sparsity:.4f}%")
    print(f"   - X_test Matrix Sparsity:           {test_sparsity:.4f}%")

    # 12, 13, 14. Save vectorizer, matrices, and labels
    print("\n4. SAVING GENERATED ARTIFACTS...")
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    os.makedirs(os.path.dirname(x_train_path), exist_ok=True)

    joblib.dump(vectorizer, vectorizer_path)
    print(f"   - Saved fitted vectorizer:          {vectorizer_path}")

    save_npz(x_train_path, X_train)
    print(f"   - Saved X_train matrix:             {x_train_path}")

    save_npz(x_test_path, X_test)
    print(f"   - Saved X_test matrix:              {x_test_path}")

    train_df[["label"]].to_csv(y_train_path, index=False)
    print(f"   - Saved y_train labels:             {y_train_path}")

    test_df[["label"]].to_csv(y_test_path, index=False)
    print(f"   - Saved y_test labels:              {y_test_path}")

    print("\n" + "=" * 65)
    print("TF-IDF FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
