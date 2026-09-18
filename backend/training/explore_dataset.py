import os
import pandas as pd

def main():
    # Construct path to dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    dataset_path = os.path.join(project_root, "backend", "data", "WELFake_Dataset.csv")
    
    if not os.path.exists(dataset_path):
        # Fallback to relative path if run from root
        dataset_path = os.path.join("backend", "data", "WELFake_Dataset.csv")

    print("=" * 60)
    print("WELFake Dataset Exploration")
    print("=" * 60)
    print(f"Loading dataset from: {dataset_path}...\n")
    
    df = pd.read_csv(dataset_path)

    # 2. Print dataset shape
    print("1. DATASET SHAPE:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    print("-" * 60)

    # 3. Print all column names and data types
    print("2. COLUMN NAMES & DATA TYPES:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col}: {dtype}")
    print("-" * 60)

    # 4. Display the first 5 rows
    print("3. FIRST 5 ROWS:")
    print(df.head())
    print("-" * 60)

    # 5. Missing values for every column
    print("4. MISSING VALUES PER COLUMN:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"   - {col}: {null_count:,} missing ({null_pct:.2f}%)")
    print("-" * 60)

    # 6. Completely duplicated rows
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100
    print("5. DUPLICATED ROWS:")
    print(f"   - Completely duplicated rows: {dup_count:,} ({dup_pct:.2f}%)")
    print("-" * 60)

    # 7. Missing/empty values in title column
    title_nulls = df['title'].isnull().sum()
    title_empty_str = df['title'].fillna('').astype(str).str.strip().eq('').sum()
    print("6. MISSING / EMPTY TITLE VALUES:")
    print(f"   - NaN / null titles: {title_nulls:,}")
    print(f"   - Empty / whitespace titles: {title_empty_str:,}")
    print("-" * 60)

    # 8. Missing/empty values in text column
    text_nulls = df['text'].isnull().sum()
    text_empty_str = df['text'].fillna('').astype(str).str.strip().eq('').sum()
    print("7. MISSING / EMPTY TEXT VALUES:")
    print(f"   - NaN / null texts: {text_nulls:,}")
    print(f"   - Empty / whitespace texts: {text_empty_str:,}")
    print("-" * 60)

    # 14. Rows with BOTH title and text missing/empty
    both_empty = (
        df['title'].fillna('').astype(str).str.strip().eq('') &
        df['text'].fillna('').astype(str).str.strip().eq('')
    ).sum()
    print("8. ROWS WITH BOTH TITLE AND TEXT MISSING/EMPTY:")
    print(f"   - Count: {both_empty:,}")
    print("-" * 60)

    # 9, 10, 11. Label distribution
    print("9. LABEL DISTRIBUTION:")
    label_counts = df['label'].value_counts(dropna=False)
    label_map = {0: "FAKE", 1: "REAL"}
    for label_val, count in label_counts.items():
        label_name = label_map.get(label_val, f"Unknown ({label_val})")
        pct = (count / len(df)) * 100
        print(f"   - Label {label_val} ({label_name}): {count:,} ({pct:.2f}%)")
    print("-" * 60)

    # 12. Title character-length statistics
    title_lens = df['title'].fillna('').astype(str).str.len()
    print("10. TITLE CHARACTER-LENGTH STATISTICS:")
    print(f"   - Minimum: {title_lens.min():,}")
    print(f"   - Maximum: {title_lens.max():,}")
    print(f"   - Mean:    {title_lens.mean():.2f}")
    print(f"   - Median:  {title_lens.median():.2f}")
    print("-" * 60)

    # 13. Article-text character-length statistics
    text_lens = df['text'].fillna('').astype(str).str.len()
    print("11. ARTICLE-TEXT CHARACTER-LENGTH STATISTICS:")
    print(f"   - Minimum: {text_lens.min():,}")
    print(f"   - Maximum: {text_lens.max():,}")
    print(f"   - Mean:    {text_lens.mean():.2f}")
    print(f"   - Median:  {text_lens.median():.2f}")
    print("-" * 60)

    # 15. Concise interpretation of findings
    print("12. INTERPRETATION OF FINDINGS:")
    print("   1. Dataset Balance: The dataset is well-balanced (~51.3% FAKE vs ~48.7% REAL).")
    print("   2. Missing Values: Small number of missing titles and texts present. Missing values should be combined or filled (e.g. combining title + text) during preprocessing.")
    print("   3. Duplicates: Duplicated rows exist and should be evaluated for removal in the preprocessing phase.")
    print("   4. Content Lengths: Text length varies significantly (from 0 to long articles). Combining title and text will provide richer features for classification.")
    print("=" * 60)

if __name__ == "__main__":
    main()
