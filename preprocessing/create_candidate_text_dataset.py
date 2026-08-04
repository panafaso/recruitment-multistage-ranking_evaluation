from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.text_builder import build_candidate_text


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)


REQUIRED_COLUMNS = [
    "Position",
    "Moreinfo",
    "Looking For",
    "Highlights",
    "English Level",
    "Experience Years",
    "CV",
    "id",
    "__index_level_0__",
]


def validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure that all required source columns exist.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input parquet not found:\n{INPUT_PATH}"
        )

    print("Loading clean dataset...")
    df = pd.read_parquet(INPUT_PATH)

    validate_columns(df)

    print("Original shape:", df.shape)

    df["candidate_text"] = df.apply(
        build_candidate_text,
        axis=1,
    )

    empty_text_rows = (
        df["candidate_text"]
        .fillna("")
        .str.strip()
        .eq("")
    )

    empty_count = int(empty_text_rows.sum())

    if empty_count > 0:
        raise ValueError(
            f"Found {empty_count} rows with empty candidate_text."
        )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("New shape:", df.shape)
    print("Saved to:")
    print(OUTPUT_PATH)

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst candidate text sample:")
    print("=" * 80)
    print(df.loc[df.index[0], "candidate_text"][:3000])
    print("=" * 80)

    print(
        "\nCandidate ID:",
        df.loc[df.index[0], "id"],
    )

    print(
        "Source index:",
        df.loc[df.index[0], "__index_level_0__"],
    )


if __name__ == "__main__":
    main()