from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    print("Shape:", df.shape)
    print("Duplicate IDs:", df["id"].duplicated().sum())
    print("Empty candidate_text:", df["candidate_text"].fillna("").str.strip().eq("").sum())
    print("Missing IDs:", df["id"].isna().sum())
    print("Missing source indexes:", df["__index_level_0__"].isna().sum())

    print("\nCandidate text length statistics:")
    print(df["candidate_text"].str.len().describe())

    print("\nFirst 3 identifiers:")
    print(df[["id", "__index_level_0__"]].head(3))


if __name__ == "__main__":
    main()