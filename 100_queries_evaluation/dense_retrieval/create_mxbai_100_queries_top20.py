from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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
    / "cv_embeddings_mxbai_embed_large_20k.npy"
)

MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "mxbai_100_queries_top20.md"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "mxbai_100_queries_top20.csv"
)

MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

TOP_K = 20


# A single, fixed query source shared by all 100-query experiments.
QUERIES = load_queries()
assert len(QUERIES) == 100, "Expected exactly 100 queries"
assert [x["query_id"] for x in QUERIES] == [f"Q{i:03d}" for i in range(1, 101)]
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
if not EMBEDDINGS_PATH.is_file():
    raise FileNotFoundError(f"Missing MXBAI embeddings: {EMBEDDINGS_PATH}")

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

# Stable candidate ID shared across every retriever.
df["candidate_id"] = df.index.astype(str)

print("Dataset rows:", len(df))
if len(df) != 20000:
    raise ValueError(f"Expected 20,000 candidate profiles, found {len(df)}")
print("Dataset columns:", df.columns.tolist())

if "candidate_text" not in df.columns:
    raise ValueError("candidate_text column not found")


print("\nLoading MXBAI embeddings...")

document_embeddings = np.load(EMBEDDINGS_PATH)

if document_embeddings.ndim != 2:
    raise ValueError(
        "The document embeddings array must be two-dimensional."
    )

document_embeddings = document_embeddings.astype(
    np.float32,
    copy=False,
)

print("Embeddings shape:", document_embeddings.shape)

# The original script computes a dot product and assumes normalized CV vectors.
# Check that assumption rather than silently changing scoring.
document_norms = np.linalg.norm(document_embeddings, axis=1)
if not np.allclose(document_norms, 1.0, atol=1e-3):
    raise ValueError(
        "Stored MXBAI document embeddings are not L2-normalized; "
        "verify the original embedding-generation pipeline before proceeding."
    )


if len(df) != len(document_embeddings):
    raise ValueError(
        f"Dataset has {len(df)} rows, but embeddings have "
        f"{len(document_embeddings)} rows."
    )


print("\nLoading MXBAI model...")

model = SentenceTransformer(MODEL_NAME)

model.max_seq_length = 512

print("MXBAI model ready!")



def safe_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def log(line, file):
    print(line)
    file.write(line + "\n")


def retrieve_mxbai(
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

    # Dot product equals cosine similarity only when
    # both document and query embeddings are normalized.
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
    log(f"\nTOP {TOP_K} MXBAI RESULTS\n", file)

    for rank, (idx, score) in enumerate(
        zip(top_indices, top_scores),
        start=1,
    ):
        row = df.iloc[idx]

        log("---", file)
        log(f"### Rank {rank}", file)
        log(
            f"- MXBAI Similarity: {score:.6f}",
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

    if set(results_df["query_id"]) != {q["query_id"] for q in QUERIES}:
        raise ValueError("Query IDs do not match queries_100.py")

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
            "# MXBAI Dense Retrieval Results",
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

        for item in QUERIES:
            query_counter += 1
            query_id = item["query_id"]
            category = item["category"]
            query = item["query"]

            print(f"\nQuery {query_counter}/{total_queries}: {query_id} - {category}")

            top_indices, top_scores = retrieve_mxbai(
                query=query, top_k=TOP_K
            )

            log("\n" + "=" * 80, file)
            log(f"## {query_id} | Category: {category}", file)
            log("=" * 80, file)
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

    print("\nMXBAI retrieval completed!")

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

