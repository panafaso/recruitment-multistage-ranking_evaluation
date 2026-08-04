from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "annotation_pool_top50_annotated_revised_balanced.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "qrels.csv"
)


REQUIRED_COLUMNS = [
    "query_id",
    "candidate_id",
    "relevance",
]

ALLOWED_LABELS = {0, 1, 2}

EXPECTED_QUERIES = 16
EXPECTED_JUDGEMENTS = 2335


if not ANNOTATION_PATH.exists():
    raise FileNotFoundError(
        f"Annotation file not found:\n{ANNOTATION_PATH}"
    )


df = pd.read_csv(
    ANNOTATION_PATH,
    dtype={
        "query_id": str,
        "candidate_id": str,
    },
)


missing_columns = (
    set(REQUIRED_COLUMNS)
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
    )


qrels = df[REQUIRED_COLUMNS].copy()


# Check for missing relevance judgements before conversion.
missing_relevance = qrels["relevance"].isna().sum()

if missing_relevance > 0:
    raise ValueError(
        f"Found {missing_relevance} rows without relevance labels."
    )


qrels["relevance"] = pd.to_numeric(
    qrels["relevance"],
    errors="raise",
).astype(int)


observed_labels = set(
    qrels["relevance"].unique()
)

invalid_labels = observed_labels - ALLOWED_LABELS

if invalid_labels:
    raise ValueError(
        f"Invalid relevance labels found: "
        f"{sorted(invalid_labels)}"
    )


duplicate_count = qrels.duplicated(
    subset=[
        "query_id",
        "candidate_id",
    ]
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Found {duplicate_count} duplicate qrels."
    )

query_count = qrels["query_id"].nunique()
judgement_count = len(qrels)

if query_count != EXPECTED_QUERIES:
    raise ValueError(
        f"Expected {EXPECTED_QUERIES} queries, "
        f"but found {query_count}."
    )

if judgement_count != EXPECTED_JUDGEMENTS:
    raise ValueError(
        f"Expected {EXPECTED_JUDGEMENTS} judgements, "
        f"but found {judgement_count}."
    )

qrels = qrels.sort_values(
    by=[
        "query_id",
        "candidate_id",
    ],
    ascending=[
        True,
        True,
    ],
).reset_index(drop=True)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


qrels.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


label_distribution = (
    qrels["relevance"]
    .value_counts()
    .sort_index()
)


print("=" * 60)
print("QRELS CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Queries: {query_count}")
print(f"Judgements: {judgement_count}")
print(f"Duplicate qrels: {duplicate_count}")

print("\nRelevance distribution:")
print(label_distribution.to_string())

print("\nExpected distribution:")
print("0 = 602")
print("1 = 1293")
print("2 = 440")

print(f"\nSaved to:\n{OUTPUT_PATH}")