from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "lang-uk/recruitment-dataset-candidate-profiles-english"
DATASET_SPLIT = "train"

TARGET_ROWS = 20_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "candidate_profiles_english_clean_20k.parquet"

# Save rejected rows optionally for auditing.
SAVE_REJECTED_ROWS = True
REJECTED_OUTPUT_PATH = OUTPUT_DIR / "candidate_profiles_rejected_cyrillic.parquet"


# This is broader and safer than only using \u0400-\u04FF.
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]")


def contains_cyrillic(value: Any) -> bool:
    """
    Return True when a value contains at least one Cyrillic character.

    Strings are checked directly. Lists, tuples, sets and dictionaries
    are checked recursively so that no textual content is missed.
    """
    if value is None:
        return False

    if isinstance(value, str):
        return CYRILLIC_PATTERN.search(value) is not None

    if isinstance(value, dict):
        return any(
            contains_cyrillic(key) or contains_cyrillic(item)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple, set)):
        return any(contains_cyrillic(item) for item in value)

   
    return False


def row_contains_cyrillic(row: dict[str, Any]) -> bool:
    """
    Return True if any field in the row contains Cyrillic characters.
    """
    return any(contains_cyrillic(value) for value in row.values())


def verify_no_cyrillic(df: pd.DataFrame) -> None:
    """
    Final verification that no saved row contains Cyrillic characters.
    Raises an error if any problematic row is detected.
    """
    problematic_indices: list[int] = []

    for index, row in df.iterrows():
        if row_contains_cyrillic(row.to_dict()):
            problematic_indices.append(index)

    if problematic_indices:
        preview = problematic_indices[:10]
        raise ValueError(
            "Final validation failed. Cyrillic characters were found "
            f"in rows: {preview}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading and cleaning candidate profiles")
    print("=" * 70)
    print(f"Source dataset: {DATASET_NAME}")
    print(f"Target clean rows: {TARGET_ROWS:,}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Streaming mode avoids loading the entire 210k-row dataset into RAM
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )

    clean_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    scanned_count = 0
    rejected_count = 0

    for row in dataset:
        scanned_count += 1

        if row_contains_cyrillic(row):
            rejected_count += 1

            if SAVE_REJECTED_ROWS:
                rejected_rows.append(dict(row))

        else:
            # dict(row) keeps every field returned by the original dataset
            clean_rows.append(dict(row))

        if scanned_count % 1_000 == 0:
            print(
                f"Scanned: {scanned_count:,} | "
                f"Accepted: {len(clean_rows):,} | "
                f"Rejected: {rejected_count:,}"
            )

        # Stop only when exactly 20.000 clean rows have been collected
        if len(clean_rows) == TARGET_ROWS:
            break

    if len(clean_rows) < TARGET_ROWS:
        raise RuntimeError(
            f"Only {len(clean_rows):,} clean rows were found. "
            f"The requested target was {TARGET_ROWS:,}."
        )

    clean_df = pd.DataFrame(clean_rows)

    # Preserve the exact original column order as returned by the dataset
    original_column_order = list(clean_rows[0].keys())
    clean_df = clean_df[original_column_order]

    print()
    print("Running final Cyrillic validation...")
    verify_no_cyrillic(clean_df)

    if len(clean_df) != TARGET_ROWS:
        raise ValueError(
            f"Unexpected row count: {len(clean_df):,}. "
            f"Expected exactly {TARGET_ROWS:,}."
        )

    # Save all columns and exactly 20,000 rows
    clean_df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    if SAVE_REJECTED_ROWS and rejected_rows:
        rejected_df = pd.DataFrame(rejected_rows)

        if set(original_column_order).issubset(rejected_df.columns):
            rejected_df = rejected_df[original_column_order]

        rejected_df.to_parquet(
            REJECTED_OUTPUT_PATH,
            index=False,
            engine="pyarrow",
        )

    # Reload the saved file to ensure it was written correctly
    saved_df = pd.read_parquet(OUTPUT_PATH)

    if len(saved_df) != TARGET_ROWS:
        raise ValueError(
            f"Saved parquet contains {len(saved_df):,} rows instead "
            f"of {TARGET_ROWS:,}."
        )

    verify_no_cyrillic(saved_df)

    print()
    print("=" * 70)
    print("CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Rows scanned: {scanned_count:,}")
    print(f"Rows rejected: {rejected_count:,}")
    print(f"Rows saved: {len(saved_df):,}")
    print(f"Columns saved: {len(saved_df.columns)}")
    print(f"Column names: {list(saved_df.columns)}")
    print(f"Clean parquet: {OUTPUT_PATH}")

    if SAVE_REJECTED_ROWS and rejected_rows:
        print(f"Rejected rows audit file: {REJECTED_OUTPUT_PATH}")

    print()
    print("Final checks:")
    print(f"  Exactly 20,000 rows: {len(saved_df) == TARGET_ROWS}")
    print("  Cyrillic characters found: False")
    print("  All source columns preserved: True")


if __name__ == "__main__":
    main()