from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Shared immutable evaluation queries (Q001-Q100).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from queries_100 import load_queries


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "cv_embeddings_bge_m3_20k_2048.npy"
)

MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "bge_m3_100_queries_top20.md"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "bge_m3_100_queries_top20.csv"
)

MODEL_NAME = "BAAI/bge-m3"

TOP_K = 20


# The same canonical query texts/IDs are used by all systems.
QUERIES = load_queries()
if len(QUERIES) != 100:
    raise ValueError("Expected exactly 100 evaluation queries")


print("Loading dataset...")

df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

# Create a stable candidate ID.
# BM25Plus, BGE-M3, E5 and MXBAI must all create/use it in the same way.
df["candidate_id"] = df.index.astype(str)

print("Dataset rows:", len(df))
print("Dataset columns:", df.columns.tolist())

if "candidate_text" not in df.columns:
    raise ValueError("candidate_text column not found")


print("\nLoading BGE-M3 embeddings...")

document_embeddings = np.load(EMBEDDINGS_PATH)

print("Embeddings shape:", document_embeddings.shape)

if document_embeddings.ndim != 2:
    raise ValueError(
        "The document embeddings array must be two-dimensional."
    )

if len(df) != len(document_embeddings):
    raise ValueError(
        f"Dataset has {len(df)} rows, but embeddings have "
        f"{len(document_embeddings)} rows."
    )

# Check that the precomputed document vectors are L2-normalised,
# as required for dot-product scoring to equal cosine similarity.
probe_norms = np.linalg.norm(document_embeddings[::max(1, len(document_embeddings)//100)], axis=1)
if not np.allclose(probe_norms, 1.0, atol=1e-3):
    raise ValueError("Document embeddings are not L2-normalised; check original embedding generation.")

# Use float32 consistently.
document_embeddings = document_embeddings.astype(
    np.float32,
    copy=False,
)


print("\nLoading BGE-M3 model...")

model = SentenceTransformer(MODEL_NAME)

model.max_seq_length = 512

print("BGE-M3 model ready!")


def safe_value(value):
    return "" if pd.isna(value) else str(value)


def log(line, file):
    print(line)
    file.write(line + "\n")


def retrieve_bge(
    query,
    top_k=TOP_K,
):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.ndim != 1:
        query_embedding = query_embedding.reshape(-1)

    if (
        document_embeddings.shape[1]
        != query_embedding.shape[0]
    ):
        raise ValueError(
            "Embedding dimension mismatch: "
            f"documents have dimension "
            f"{document_embeddings.shape[1]}, "
            f"but the query has dimension "
            f"{query_embedding.shape[0]}."
        )

    # This is cosine similarity only if the saved
    # document embeddings were also normalized.
    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    return top_indices, scores[top_indices]


def write_results_md(
    query,
    top_indices,
    top_scores,
    file,
):
    log(f"\n### Query: {query}", file)
    log(f"\nTOP {TOP_K} BGE-M3 RESULTS\n", file)

    for rank, (idx, score) in enumerate(
        zip(top_indices, top_scores),
        start=1,
    ):
        row = df.iloc[idx]

        log("---", file)
        log(f"### Rank {rank}", file)
        log(
            f"- BGE-M3 Similarity: {score:.6f}",
            file,
        )
        log(
            f"- Dataset Index: {idx}",
            file,
        )
        log(
            f"- Candidate ID: {row['candidate_id']}",
            file,
        )

        log(
            f"- Position: "
            f"{safe_value(row.get('Position'))}",
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
        log(
            safe_value(row.get("CV")),
            file,
        )

        log("", file)

def validate_results(results_df):
    expected_queries = len(QUERIES)

    expected_rows = expected_queries * TOP_K

    actual_queries = results_df[
        "query_id"
    ].nunique()

    actual_rows = len(results_df)

    duplicate_count = results_df.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    rows_per_query = results_df.groupby(
        "query_id"
    ).size()

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        "Expected queries:",
        expected_queries,
    )
    print(
        "Actual queries:",
        actual_queries,
    )

    print(
        "Expected rows:",
        expected_rows,
    )
    print(
        "Actual rows:",
        actual_rows,
    )

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

    if not (
        rows_per_query == TOP_K
    ).all():
        raise ValueError(
            f"Not every query contains exactly "
            f"{TOP_K} results."
        )

    print("\nValidation successful.")


if __name__ == "__main__":

    MARKDOWN_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    total_queries = len(QUERIES)

    query_counter = 0

    with open(
        MARKDOWN_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        log(
            "# BGE-M3 Dense Retrieval Results",
            file,
        )

        log(
            f"\nModel: {MODEL_NAME}",
            file,
        )

        log(
            f"Dataset rows: {len(df)}",
            file,
        )

        log(
            f"Top K per query: {TOP_K}",
            file,
        )

        log(
            "Similarity: dot product between "
            "normalized embeddings",
            file,
        )

        log(
            "Document embeddings normalized: yes",
            file,
        )

        log(
            "Query embeddings normalized: yes",
            file,
        )

        log(
            "Retrieval text: candidate_text",
            file,
        )

        log(
            "Candidate ID: stable dataframe index "
            "after reset_index",
            file,
        )

        for query_counter, item in enumerate(QUERIES, start=1):
            query_id = item["query_id"]
            category = item["category"]
            query = item["query"]

            print(f"\nQuery {query_counter}/{total_queries}: {query_id} ({category})")
            log("\n" + "=" * 80, file)
            log(f"## {query_id} — {category}", file)
            log("=" * 80, file)

            top_indices, top_scores = retrieve_bge(
                query=query,
                top_k=TOP_K,
            )

            write_results_md(
                query=query,
                top_indices=top_indices,
                top_scores=top_scores,
                file=file,
            )

            for rank, (candidate_idx, score) in enumerate(
                zip(top_indices, top_scores), start=1
            ):
                row = df.iloc[candidate_idx]
                all_results.append({
                    "query_id": query_id,
                    "category": category,
                    "query": query,
                    "candidate_id": str(row["candidate_id"]),
                    "dataset_index": int(candidate_idx),
                    "rank": int(rank),
                    "score": float(score),
                })

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        by=[
            "query_id",
            "rank",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    validate_results(results_df)

    results_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nBGE-M3 retrieval completed and validated!")

    print(
        "Markdown results saved to:",
        MARKDOWN_OUTPUT_PATH,
    )

    print(
        "CSV rankings saved to:",
        CSV_OUTPUT_PATH,
    )

    print(
        "Number of queries:",
        results_df["query_id"].nunique(),
    )

    print(
        "Number of rows:",
        len(results_df),
    )

    validate_results(
        results_df
    )