import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    cleaned_path = os.path.join(project_root, "backend", "data", "WELFake_cleaned.csv")
    train_path = os.path.join(project_root, "backend", "data", "train.csv")
    test_path = os.path.join(project_root, "backend", "data", "test.csv")

    if not os.path.exists(cleaned_path):
        cleaned_path = os.path.join("backend", "data", "WELFake_cleaned.csv")
        train_path = os.path.join("backend", "data", "train.csv")
        test_path = os.path.join("backend", "data", "test.csv")

    print("=" * 65)
    print("WELFake Dataset Train/Test Stratified Splitting Pipeline")
    print("=" * 65)
    print(f"Loading cleaned dataset from: {cleaned_path}...\n")

    df = pd.read_csv(cleaned_path)

    # 18. Validation: Required columns must be present
    required_cols = {"content", "label"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CRITICAL ERROR: Required column(s) missing from cleaned dataset: {missing}")

    # 4. Check missing/empty content or labels
    null_labels = df["label"].isnull().sum()
    null_content = df["content"].isnull().sum()
    empty_content = df["content"].fillna("").astype(str).str.strip().eq("").sum()

    print("1. PRE-SPLIT DATA INTEGRITY REPORT:")
    print(f"   - Total input rows:               {len(df):,}")
    print(f"   - Null labels found:              {null_labels:,}")
    print(f"   - Null content rows found:        {null_content:,}")
    print(f"   - Empty string content rows:      {empty_content:,}")

    # Drop any invalid/empty rows if found and report explicitly
    invalid_mask = df["label"].isnull() | df["content"].fillna("").astype(str).str.strip().eq("")
    if invalid_mask.sum() > 0:
        print(f"   - WARNING: Removing {invalid_mask.sum():,} invalid/empty rows before splitting.")
        df = df[~invalid_mask].copy()

    # 18. Validation: Invalid labels check
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        invalid_set = unique_labels - {0, 1}
        raise ValueError(f"CRITICAL ERROR: Invalid label values found in dataset: {invalid_set}. Allowed labels: [0, 1]")

    # 5. Perform Stratified Train/Test Split (80% Train, 20% Test, random_state=42)
    train_df, test_df = train_test_split(
        df[["content", "label"]],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 6. Verification: Check content overlap (case-insensitive normalized content)
    train_content_set = set(train_df["content"].str.lower().str.strip())
    test_content_set = set(test_df["content"].str.lower().str.strip())
    overlap_set = train_content_set.intersection(test_content_set)
    overlap_count = len(overlap_set)

    if overlap_count > 0:
        raise ValueError(f"CRITICAL ERROR: Data leakage detected! {overlap_count} overlapping content records found between train and test sets.")

    # 6 & 18. Verification: Check that both classes exist in both splits
    train_classes = set(train_df["label"].unique())
    test_classes = set(test_df["label"].unique())

    if train_classes != {0, 1}:
        raise ValueError(f"CRITICAL ERROR: Training split is missing target class(es)! Found classes: {train_classes}")

    if test_classes != {0, 1}:
        raise ValueError(f"CRITICAL ERROR: Testing split is missing target class(es)! Found classes: {test_classes}")

    # Calculate statistics
    total_count = len(df)
    train_count = len(train_df)
    test_count = len(test_df)
    train_pct = (train_count / total_count) * 100
    test_pct = (test_count / total_count) * 100

    train_fake = (train_df["label"] == 0).sum()
    train_real = (train_df["label"] == 1).sum()
    train_fake_pct = (train_fake / train_count) * 100
    train_real_pct = (train_real / train_count) * 100

    test_fake = (test_df["label"] == 0).sum()
    test_real = (test_df["label"] == 1).sum()
    test_fake_pct = (test_fake / test_count) * 100
    test_real_pct = (test_real / test_count) * 100

    # 8. Save output datasets
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 7. Print clear detailed report
    print("\n" + "-" * 65)
    print("STRATIFIED TRAIN / TEST SPLIT REPORT")
    print("-" * 65)
    print(f"   - Total Processed Dataset Rows:     {total_count:,}")
    print(f"   - Training Row Count:                {train_count:,} ({train_pct:.2f}%)")
    print(f"   - Testing Row Count:                 {test_count:,} ({test_pct:.2f}%)")
    print("\nTraining Split Class Breakdown:")
    print(f"   - FAKE (0):                         {train_fake:,} ({train_fake_pct:.2f}%)")
    print(f"   - REAL (1):                         {train_real:,} ({train_real_pct:.2f}%)")
    print("\nTesting Split Class Breakdown:")
    print(f"   - FAKE (0):                         {test_fake:,} ({test_fake_pct:.2f}%)")
    print(f"   - REAL (1):                         {test_real:,} ({test_real_pct:.2f}%)")
    print("\nData Leakage Check:")
    print(f"   - Overlapping Normalized Records:   {overlap_count}")

    print("\nSaved Output Files:")
    print(f"   - Training Set:                     {train_path}")
    print(f"   - Testing Set:                      {test_path}")

    print("\n" + "=" * 65)
    print("TRAIN/TEST SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
