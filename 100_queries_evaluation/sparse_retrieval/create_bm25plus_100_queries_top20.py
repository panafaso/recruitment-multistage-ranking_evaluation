import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from rank_bm25 import BM25Plus
from nltk.corpus import stopwords

# Script location: evaluation_100_queries/sparse_retrieval/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from queries_100 import load_queries

# Prefer the ORIGINAL shared dataset to preserve candidate index alignment.
# If this file is absent, use the local project's data/ copy.
ORIGINAL_DATA_PATH = PROJECT_ROOT.parent / "data" / "candidate_profiles_english_clean_20k_with_text.parquet"
LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "candidate_profiles_english_clean_20k_with_text.parquet"
DATA_PATH = ORIGINAL_DATA_PATH if ORIGINAL_DATA_PATH.is_file() else LOCAL_DATA_PATH
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Cannot find corpus parquet. Checked: {ORIGINAL_DATA_PATH} and {LOCAL_DATA_PATH}")
OUTPUT_DIR = Path(__file__).resolve().parent

MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "bm25plus_100_queries_top20.md"
CSV_OUTPUT_PATH = OUTPUT_DIR / "bm25plus_100_queries_top20.csv"

TOP_K = 20

K1 = 1.5
B = 0.60
DELTA = 1.0

df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
if len(df) != 20000:
    raise ValueError(f"Expected the same 20,000 candidate profiles, found {len(df)}")

df["candidate_id"] = df.index.astype(str)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

if "candidate_text" not in df.columns:
    raise ValueError("candidate_text column not found")


texts = (
    df["candidate_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

def safe_value(value):
    return "" if pd.isna(value) else str(value)


def log(line, file):
    print(line)
    file.write(line + "\n")



stop_words = set(stopwords.words("english"))

important_words = {
    "no",
    "not",
    "more",
    "most",
}

stop_words = stop_words - important_words


def tokenize(text):
    text = str(text).lower()


    text = re.sub(r"[^a-z0-9+#. ]", " ", text)

    tokens = text.split()

    return [
        token
        for token in tokens
        if token not in stop_words
    ]

print("Using candidate data:", DATA_PATH)
print("Tokenizing corpus...")

tokenized_corpus = [
    tokenize(document)
    for document in texts
]

print("Tokenization complete!")

print("Building BM25Plus index...")

bm25 = BM25Plus(
    tokenized_corpus,
    k1=K1,
    b=B,
    delta=DELTA,
)

print("BM25Plus ready!")


def retrieve_bm25(query, top_k=TOP_K):
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices, scores[top_indices]



def write_results_md(query, top_indices, top_scores, file):
    log(f"\n### Query: {query}", file)
    log(f"\nTOP {TOP_K} BM25PLUS RESULTS\n", file)

    for rank, (idx, score) in enumerate(
        zip(top_indices, top_scores),
        start=1,
    ):
        row = df.iloc[idx]

        candidate_id = row["candidate_id"]

        log("---", file)
        log(f"### Rank {rank}", file)
        log(f"- BM25Plus Score: {score:.4f}", file)
        log(f"- Dataset Index: {idx}", file)
        log(f"- Candidate ID: {candidate_id}", file)

        log(
            f"- Position: {safe_value(row.get('Position'))}",
            file,
        )

        log(
            f"- Experience Years: "
            f"{safe_value(row.get('Experience Years'))}",
            file,
        )

        log(
            f"- English Level: "
            f"{safe_value(row.get('English Level'))}",
            file,
        )

        log("\n#### Looking For", file)
        log(
            safe_value(row.get("Looking For")),
            file,
        )

        log("\n#### Highlights", file)
        log(
            safe_value(row.get("Highlights")),
            file,
        )

        log("\n#### CV snippet", file)

        cv = safe_value(row.get("CV"))

        log(
            cv,
            file,
        )

        log("", file)


def validate_results(results_df):
    expected_queries = len(load_queries())

    expected_rows = expected_queries * TOP_K

    actual_queries = results_df["query_id"].nunique()
    actual_rows = len(results_df)

    duplicate_count = results_df.duplicated(
        subset=["query_id", "candidate_id"]
    ).sum()

    rows_per_query = results_df.groupby("query_id").size()

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("Expected queries:", expected_queries)
    print("Actual queries:", actual_queries)

    print("Expected rows:", expected_rows)
    print("Actual rows:", actual_rows)

    print(
        "Duplicate query/candidate pairs:",
        duplicate_count,
    )

    print("\nRows per query:")
    print(rows_per_query.to_string())

    if actual_queries != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} queries, "
            f"but found {actual_queries}."
        )

    expected_ids = {item["query_id"] for item in load_queries()}
    if set(results_df["query_id"]) != expected_ids:
        raise ValueError("Query ID mismatch with frozen 100-query source.")

    if actual_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"but found {actual_rows}."
        )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate candidate IDs were found "
            "inside one or more query rankings."
        )

    if not (rows_per_query == TOP_K).all():
        raise ValueError(
            f"Not every query contains exactly {TOP_K} results."
        )

    print("\nValidation successful.")


if __name__ == "__main__":

    all_results = []

    with open(
        MARKDOWN_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        log(
            "# BM25Plus + Stopwords Retrieval Results — 100 Queries",
            file,
        )

        log(f"\nTop K: {TOP_K}", file)
        log(f"k1: {K1}", file)
        log(f"b: {B}", file)
        log(f"delta: {DELTA}", file)
        log("Stopword removal: yes", file)
        log("Stemming: no", file)
        log("Retrieval text: candidate_text", file)
        log(
            "Candidate ID: stable dataframe index after reset_index",
            file,
        )

        for entry in load_queries():
            query_id = entry["query_id"]
            category = entry["category"]
            query = entry["query"]
            log("\n" + "=" * 80, file)
            log(f"## {query_id} | {category}", file)
            log("=" * 80, file)

            top_indices, top_scores = retrieve_bm25(
                query
            )

            write_results_md(
                query=query,
                top_indices=top_indices,
                top_scores=top_scores,
                file=file,
            )

            for rank, (candidate_idx, score) in enumerate(
                zip(top_indices, top_scores),
                start=1,
            ):
                row = df.iloc[candidate_idx]

                all_results.append({
                    "query_id": query_id,
                    "category": category,
                    "query": query,
                    "candidate_id": str(
                        row["candidate_id"]
                    ),
                    "dataset_index": int(
                        candidate_idx
                    ),
                    "rank": int(rank),
                    "score": float(score),
                })

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=["query_id", "rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    results_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nMarkdown results saved to: "
        f"{MARKDOWN_OUTPUT_PATH}"
    )

    print(
        f"CSV rankings saved to: "
        f"{CSV_OUTPUT_PATH}"
    )

    print(
        "Number of queries:",
        results_df["query_id"].nunique(),
    )

    print(
        "Number of rows:",
        len(results_df),
    )

    validate_results(results_df)