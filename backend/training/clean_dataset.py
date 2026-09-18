import os
import pandas as pd

def normalize_text_series(series: pd.Series) -> pd.Series:
    """Safely converts series to string, strips whitespace, and normalizes internal repeated spaces."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    raw_path = os.path.join(project_root, "backend", "data", "WELFake_Dataset.csv")
    cleaned_path = os.path.join(project_root, "backend", "data", "WELFake_cleaned.csv")

    if not os.path.exists(raw_path):
        raw_path = os.path.join("backend", "data", "WELFake_Dataset.csv")
        cleaned_path = os.path.join("backend", "data", "WELFake_cleaned.csv")

    print("=" * 65)
    print("WELFake Dataset Cleaning & Normalization Pipeline")
    print("=" * 65)
    print(f"Loading raw dataset from: {raw_path}...\n")

    df_raw = pd.read_csv(raw_path)
    original_row_count = len(df_raw)

    print(f"1. ORIGINAL ROW COUNT: {original_row_count:,}")

    # 2. Drop index column if present
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    # 3. Normalize title and text safely
    norm_title = normalize_text_series(df_raw["title"])
    norm_text = normalize_text_series(df_raw["text"])

    # 4. Create "content" column combining title and text
    # If both exist, combine with space; otherwise use whichever exists.
    def combine_title_text(t: str, b: str) -> str:
        if t and b:
            return f"{t} {b}"
        elif t:
            return t
        elif b:
            return b
        else:
            return ""

    # Vectorized construction of content
    content_series = norm_title.combine(
        norm_text,
        lambda t, b: combine_title_text(t, b)
    )

    df_clean = pd.DataFrame({
        "title": norm_title,
        "text": norm_text,
        "content": content_series,
        "label": df_raw["label"]
    })

    # 7. Validate labels before cleaning
    valid_labels_mask = df_clean["label"].isin([0, 1])
    invalid_labels = df_clean[~valid_labels_mask]
    invalid_label_count = len(invalid_labels)

    # 5. Remove rows where content is empty
    empty_content_mask = df_clean["content"].str.len() == 0
    empty_removed_count = empty_content_mask.sum()
    df_clean = df_clean[~empty_content_mask].copy()

    # 6. Deduplicate based on normalized content (case-insensitive key for accuracy, keeping first occurrence)
    norm_content_key = df_clean["content"].str.lower()
    duplicate_mask = norm_content_key.duplicated(keep="first")
    duplicates_removed_count = duplicate_mask.sum()

    df_clean = df_clean[~duplicate_mask].copy()

    # Filter out invalid labels if any were found
    if invalid_label_count > 0:
        df_clean = df_clean[df_clean["label"].isin([0, 1])].copy()

    final_row_count = len(df_clean)

    # Calculate statistics for final content lengths
    content_lengths = df_clean["content"].str.len()
    min_len = content_lengths.min()
    max_len = content_lengths.max()
    mean_len = content_lengths.mean()
    median_len = content_lengths.median()

    # Calculate label distribution
    label_counts = df_clean["label"].value_counts()
    fake_count = label_counts.get(0, 0)
    real_count = label_counts.get(1, 0)
    fake_pct = (fake_count / final_row_count) * 100 if final_row_count > 0 else 0
    real_pct = (real_count / final_row_count) * 100 if final_row_count > 0 else 0

    # 9 & 10. Save cleaned dataset with exact requested columns
    expected_cols = ["title", "text", "content", "label"]
    df_clean = df_clean[expected_cols]
    
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_clean.to_csv(cleaned_path, index=False)
    print(f"\nSaved cleaned dataset to: {cleaned_path}")

    # 11. Print detailed cleaning report
    print("\n" + "-" * 65)
    print("DETAILED CLEANING REPORT")
    print("-" * 65)
    print(f"   - Original Row Count:                 {original_row_count:,}")
    print(f"   - Empty Content Rows Removed:         {empty_removed_count:,}")
    print(f"   - Duplicate Rows Removed:             {duplicates_removed_count:,}")
    print(f"   - Invalid Labels Found:               {invalid_label_count:,}")
    print(f"   - Final Cleaned Row Count:            {final_row_count:,}")
    print(f"   - Final FAKE Articles (0):            {fake_count:,} ({fake_pct:.2f}%)")
    print(f"   - Final REAL Articles (1):            {real_count:,} ({real_pct:.2f}%)")
    print("\nContent Character Length Statistics:")
    print(f"   - Minimum: {min_len:,}")
    print(f"   - Maximum: {max_len:,}")
    print(f"   - Mean:    {mean_len:.2f}")
    print(f"   - Median:  {median_len:.2f}")

    # 12. Run validation checks
    print("\n" + "-" * 65)
    print("POST-CLEANING VALIDATION CHECKS")
    print("-" * 65)

    check_empty = (df_clean["content"].str.len() == 0).sum() == 0
    check_duplicates = df_clean["content"].str.lower().duplicated().sum() == 0
    check_labels = df_clean["label"].isin([0, 1]).all()
    check_columns = list(df_clean.columns) == expected_cols

    print(f"   [✓] No Empty Content Remains:       {check_empty}")
    print(f"   [✓] No Duplicate Content Remains:   {check_duplicates}")
    print(f"   [✓] No Invalid Labels Remain:       {check_labels}")
    print(f"   [✓] Exact Columns Match Schema:     {check_columns} ({expected_cols})")

    assert check_empty, "Validation Failed: Empty content found in cleaned dataset!"
    assert check_duplicates, "Validation Failed: Duplicate content found in cleaned dataset!"
    assert check_labels, "Validation Failed: Invalid labels found in cleaned dataset!"
    assert check_columns, "Validation Failed: Column names do not match expected schema!"

    print("\n" + "=" * 65)
    print("CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
