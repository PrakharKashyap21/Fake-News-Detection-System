import os
import re
import pandas as pd

def main():
    # Construct path to dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    dataset_path = os.path.join(project_root, "backend", "data", "WELFake_Dataset.csv")

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("backend", "data", "WELFake_Dataset.csv")

    print("=" * 65)
    print("WELFake Dataset Data Quality & Duplicate Content Analysis")
    print("=" * 65)
    print(f"Loading dataset from: {dataset_path}...\n")

    df = pd.read_csv(dataset_path)

    # Remove 'Unnamed: 0' for analysis purposes if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    total_rows = len(df)
    print(f"Total Rows Loaded: {total_rows:,}\n")

    # 5 & 6. Check title and body presence/absence categories
    print("-" * 65)
    print("1. TITLE AND BODY CONTENT AVAILABILITY CATEGORIES")
    print("-" * 65)

    is_title_empty = df["title"].fillna("").astype(str).str.strip().eq("")
    is_text_empty = df["text"].fillna("").astype(str).str.strip().eq("")

    title_only = (df["title"].notnull() & ~is_title_empty) & is_text_empty
    text_only = is_title_empty & (df["text"].notnull() & ~is_text_empty)
    both_empty = is_title_empty & is_text_empty
    both_present = (~is_title_empty) & (~is_text_empty)

    cnt_title_only = title_only.sum()
    cnt_text_only = text_only.sum()
    cnt_both_empty = both_empty.sum()
    cnt_both_present = both_present.sum()

    print(f"   - Both Title and Body Present: {cnt_both_present:,} ({cnt_both_present / total_rows * 100:.2f}%)")
    print(f"   - Title Present but Body Empty: {cnt_title_only:,} ({cnt_title_only / total_rows * 100:.2f}%)")
    print(f"   - Body Present but Title Empty: {cnt_text_only:,} ({cnt_text_only / total_rows * 100:.2f}%)")
    print(f"   - Both Title and Body Empty:   {cnt_both_empty:,} ({cnt_both_empty / total_rows * 100:.2f}%)")

    # 3. Normalize title and text for duplicate detection
    print("\n" + "-" * 65)
    print("2. NORMALIZING CONTENT FOR DUPLICATE DETECTION")
    print("-" * 65)

    norm_title = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    norm_text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    df["combined_norm"] = norm_title + " " + norm_text

    # 4. Duplicate Analysis
    print("\n" + "-" * 65)
    print("3. DUPLICATE CONTENT ANALYSIS")
    print("-" * 65)

    # Exact redundant duplicate rows (excluding the first occurrence)
    num_redundant_duplicates = df.duplicated(subset=["combined_norm"]).sum()
    
    # Rows involved in duplicate groups (keep=False marks ALL instances of duplicates)
    duplicated_rows_mask = df.duplicated(subset=["combined_norm"], keep=False)
    num_rows_involved = duplicated_rows_mask.sum()

    # Number of distinct duplicate groups
    num_duplicate_groups = df[duplicated_rows_mask]["combined_norm"].nunique()

    print(f"   - Number of redundant duplicate rows: {num_redundant_duplicates:,} ({num_redundant_duplicates / total_rows * 100:.2f}%)")
    print(f"   - Total rows involved in duplicate content: {num_rows_involved:,} ({num_rows_involved / total_rows * 100:.2f}%)")
    print(f"   - Number of distinct duplicate content groups: {num_duplicate_groups:,}")

    # Label consistency check across duplicate groups
    print("\n" + "-" * 65)
    print("4. LABEL CONSISTENCY IN DUPLICATE CONTENT")
    print("-" * 65)

    group_label_counts = df.groupby("combined_norm")["label"].nunique()
    conflicting_groups = group_label_counts[group_label_counts > 1]
    num_conflicting_groups = len(conflicting_groups)

    if num_conflicting_groups > 0:
        conflicting_rows_count = df[df["combined_norm"].isin(conflicting_groups.index)].shape[0]
        print(f"   - CONFLICT DETECTED: Yes, duplicate content with conflicting labels exists.")
        print(f"   - Number of conflicting content groups (contain both FAKE & REAL labels): {num_conflicting_groups:,}")
        print(f"   - Total rows involved in conflicting duplicate groups: {conflicting_rows_count:,} ({conflicting_rows_count / total_rows * 100:.2f}%)")
    else:
        print("   - CONFLICT DETECTED: No conflicting labels found across duplicate groups.")

    print("\n" + "=" * 65)
    print("SUMMARY & PREPROCESSING RECOMMENDATIONS:")
    print("=" * 65)
    print("   1. Duplicates: A significant number of duplicate rows exist after whitespace and case normalization.")
    print("   2. Label Conflicts: Any duplicate groups with conflicting labels (same content marked both FAKE and REAL) must be carefully handled/removed during preprocessing to avoid noise.")
    print("   3. Missing Body/Title: Articles missing body text or title should be handled by merging title + text into a single content feature.")
    print("=" * 65)

if __name__ == "__main__":
    main()
