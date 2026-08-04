from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_PATH = (
    PROJECT_ROOT
    / "sparse_retrieval"
    / "bm25plus_top50.csv"
)

MXBAI_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "mxbai_top50.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "hybrid_retrieval"
    / "hybrid_bm25_mxbai_rrf_top50.csv"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "hybrid_retrieval"
    / "hybrid_bm25_mxbai_rrf_top50.md"
)

TOP_K = 50
RRF_K = 60


bm25 = pd.read_csv(
    BM25_PATH,
    dtype={
        "query_id": str,
        "candidate_id": str,
    },
)

mxbai = pd.read_csv(
    MXBAI_PATH,
    dtype={
        "query_id": str,
        "candidate_id": str,
    },
)


required_columns = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
    "score",
}

for name, df in [
    ("BM25Plus", bm25),
    ("MXBAI", mxbai),
]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )

    duplicates = df.duplicated(
        subset=["query_id", "candidate_id"]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"{name} contains {duplicates} duplicate "
            "query/candidate pairs."
        )


if set(bm25["query_id"]) != set(mxbai["query_id"]):
    raise ValueError(
        "BM25Plus and MXBAI do not contain the same queries."
    )

bm25 = bm25.rename(
    columns={
        "rank": "bm25_rank",
        "score": "bm25_score",
    }
)

bm25["bm25_rrf"] = (
    1 / (RRF_K + bm25["bm25_rank"])
)

bm25 = bm25[
    [
        "query_id",
        "category",
        "query",
        "candidate_id",
        "dataset_index",
        "bm25_rank",
        "bm25_score",
        "bm25_rrf",
    ]
]


mxbai = mxbai.rename(
    columns={
        "rank": "mxbai_rank",
        "score": "mxbai_score",
    }
)

mxbai["mxbai_rrf"] = (
    1 / (RRF_K + mxbai["mxbai_rank"])
)

mxbai = mxbai[
    [
        "query_id",
        "category",
        "query",
        "candidate_id",
        "dataset_index",
        "mxbai_rank",
        "mxbai_score",
        "mxbai_rrf",
    ]
]


hybrid = bm25.merge(
    mxbai,
    on=[
        "query_id",
        "candidate_id",
    ],
    how="outer",
    suffixes=(
        "_bm25",
        "_mxbai",
    ),
)


hybrid["category"] = (
    hybrid["category_bm25"]
    .combine_first(hybrid["category_mxbai"])
)

hybrid["query"] = (
    hybrid["query_bm25"]
    .combine_first(hybrid["query_mxbai"])
)

hybrid["dataset_index"] = (
    hybrid["dataset_index_bm25"]
    .combine_first(hybrid["dataset_index_mxbai"])
)


hybrid["bm25_rrf"] = hybrid["bm25_rrf"].fillna(0)
hybrid["mxbai_rrf"] = hybrid["mxbai_rrf"].fillna(0)

hybrid["score"] = (
    hybrid["bm25_rrf"]
    + hybrid["mxbai_rrf"]
)


hybrid["retrieved_by"] = "BM25Plus;MXBAI"

hybrid.loc[
    hybrid["bm25_rank"].notna()
    & hybrid["mxbai_rank"].isna(),
    "retrieved_by",
] = "BM25Plus"

hybrid.loc[
    hybrid["bm25_rank"].isna()
    & hybrid["mxbai_rank"].notna(),
    "retrieved_by",
] = "MXBAI"


hybrid["best_rank"] = hybrid[
    [
        "bm25_rank",
        "mxbai_rank",
    ]
].min(axis=1, skipna=True)

hybrid = hybrid.sort_values(
    by=[
        "query_id",
        "score",
        "best_rank",
        "candidate_id",
    ],
    ascending=[
        True,
        False,
        True,
        True,
    ],
)

hybrid["rank"] = (
    hybrid
    .groupby("query_id")
    .cumcount()
    + 1
)

hybrid = hybrid[
    hybrid["rank"] <= TOP_K
].copy()


hybrid = hybrid[
    [
        "query_id",
        "category",
        "query",
        "candidate_id",
        "dataset_index",
        "rank",
        "score",
        "retrieved_by",
        "bm25_rank",
        "bm25_score",
        "mxbai_rank",
        "mxbai_score",
    ]
]

hybrid = hybrid.sort_values(
    by=[
        "query_id",
        "rank",
    ]
).reset_index(drop=True)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

hybrid.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

# ============================================================
# Save readable Markdown results
# ============================================================

candidate_df = pd.read_parquet(
    DATA_PATH
).reset_index(drop=True)

candidate_df["candidate_id"] = (
    candidate_df.index.astype(str)
)


def safe_value(value):
    return "" if pd.isna(value) else str(value)


with open(
    MARKDOWN_OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "# Hybrid BM25Plus + MXBAI RRF Retrieval Results\n"
    )

    file.write(f"\nTop K: {TOP_K}\n")
    file.write(f"RRF constant: {RRF_K}\n")
    file.write("Sparse retriever: BM25Plus\n")
    file.write("Dense retriever: MXBAI\n")
    file.write("Fusion method: Reciprocal Rank Fusion\n")
    file.write("Retrieval text: candidate_text\n")

    for category, category_group in hybrid.groupby(
        "category",
        sort=False,
    ):

        file.write("\n" + "=" * 80 + "\n")
        file.write(f"## Category: {category}\n")
        file.write("=" * 80 + "\n")

        for query_id, query_group in category_group.groupby(
            "query_id",
            sort=False,
        ):

            query_group = query_group.sort_values("rank")
            query = query_group["query"].iloc[0]

            file.write(f"\n### Query: {query}\n")
            file.write(
                f"\nTOP {TOP_K} HYBRID RRF RESULTS\n\n"
            )

            for _, result in query_group.iterrows():

                dataset_index = int(result["dataset_index"])
                candidate_row = candidate_df.iloc[dataset_index]

                file.write("---\n")
                file.write(
                    f"### Rank {int(result['rank'])}\n"
                )

                file.write(
                    f"- Hybrid RRF Score: "
                    f"{float(result['score']):.6f}\n"
                )

                file.write(
                    f"- Retrieved By: "
                    f"{safe_value(result['retrieved_by'])}\n"
                )

                bm25_rank = (
                    ""
                    if pd.isna(result["bm25_rank"])
                    else str(int(result["bm25_rank"]))
                )

                mxbai_rank = (
                    ""
                    if pd.isna(result["mxbai_rank"])
                    else str(int(result["mxbai_rank"]))
                )

                bm25_score = (
                    ""
                    if pd.isna(result["bm25_score"])
                    else f"{float(result['bm25_score']):.6f}"
                )

                mxbai_score = (
                    ""
                    if pd.isna(result["mxbai_score"])
                    else f"{float(result['mxbai_score']):.6f}"
                )

                file.write(
                    f"- BM25Plus Rank: {bm25_rank}\n"
                )

                file.write(
                    f"- BM25Plus Score: {bm25_score}\n"
                )

                file.write(
                    f"- MXBAI Rank: {mxbai_rank}\n"
                )

                file.write(
                    f"- MXBAI Score: {mxbai_score}\n"
                )

                file.write(
                    f"- Dataset Index: {dataset_index}\n"
                )

                file.write(
                    f"- Candidate ID: "
                    f"{safe_value(result['candidate_id'])}\n"
                )

                file.write(
                    f"- Position: "
                    f"{safe_value(candidate_row.get('Position'))}\n"
                )

                file.write(
                    f"- Experience Years: "
                    f"{safe_value(candidate_row.get('Experience Years'))}\n"
                )

                file.write(
                    f"- English Level: "
                    f"{safe_value(candidate_row.get('English Level'))}\n"
                )

                file.write("\n#### Looking For\n")

                file.write(
                    safe_value(
                        candidate_row.get("Looking For")
                    )
                    + "\n"
                )

                file.write("\n#### Highlights\n")

                file.write(
                    safe_value(
                        candidate_row.get("Highlights")
                    )
                    + "\n"
                )

                file.write("\n#### CV snippet\n")

                file.write(
                    safe_value(
                        candidate_row.get("CV")
                    )
                    + "\n\n"
                )

rows_per_query = (
    hybrid
    .groupby("query_id")
    .size()
)

duplicate_pairs = hybrid.duplicated(
    subset=[
        "query_id",
        "candidate_id",
    ]
).sum()

if duplicate_pairs > 0:
    raise ValueError(
        "Duplicate query/candidate pairs found "
        "in hybrid results."
    )

if not (rows_per_query == TOP_K).all():
    raise ValueError(
        f"Not every query contains exactly {TOP_K} results."
    )


print("=" * 60)
print("RRF HYBRID BM25PLUS + MXBAI COMPLETED")
print("=" * 60)
print(f"Queries: {hybrid['query_id'].nunique()}")
print(f"Rows: {len(hybrid)}")
print(f"Top K per query: {TOP_K}")
print(f"RRF constant: {RRF_K}")
print(f"CSV saved to: {OUTPUT_PATH}")
print(f"Markdown saved to: {MARKDOWN_OUTPUT_PATH}")