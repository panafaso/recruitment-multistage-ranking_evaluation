from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_PATH = (
    PROJECT_ROOT
    / "sparse_retrieval"
    / "bm25plus_100_queries_top20.csv"
)

JINA_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "jina_v5_100_queries_top20.csv"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "hybrid_retrieval"
    / "hybrid_bm25_jina_rrf_100_queries_top20.csv"
)

MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "hybrid_retrieval"
    / "hybrid_bm25_jina_rrf_100_queries_top20.md"
)

TOP_K = 20
RRF_K = 60

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
    "score",
}


def validate_file_exists(
    path: Path,
    description: str,
) -> None:
    """Raise an informative error when a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def load_ranking(
    path: Path,
    system_name: str,
) -> pd.DataFrame:
    """Load and validate one Top-20 ranking file."""

    ranking = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    missing_columns = (
        REQUIRED_RANKING_COLUMNS
        - set(ranking.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{system_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    ranking = ranking[
        [
            "query_id",
            "category",
            "query",
            "candidate_id",
            "dataset_index",
            "rank",
            "score",
        ]
    ].copy()

    ranking["rank"] = pd.to_numeric(
        ranking["rank"],
        errors="raise",
    ).astype(int)

    ranking["score"] = pd.to_numeric(
        ranking["score"],
        errors="raise",
    ).astype(float)

    ranking["dataset_index"] = pd.to_numeric(
        ranking["dataset_index"],
        errors="raise",
    ).astype(int)

    duplicate_count = ranking.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{system_name} contains {duplicate_count} duplicate "
            "query/candidate pairs."
        )

    rows_per_query = (
        ranking
        .groupby("query_id")
        .size()
    )

    if not (rows_per_query == TOP_K).all():
        problematic_queries = rows_per_query[
            rows_per_query != TOP_K
        ]

        raise ValueError(
            f"{system_name} does not contain exactly "
            f"{TOP_K} candidates for every query:\n"
            f"{problematic_queries}"
        )

    for query_id, query_rows in ranking.groupby(
        "query_id"
    ):
        observed_ranks = sorted(
            query_rows["rank"].tolist()
        )

        expected_ranks = list(
            range(1, TOP_K + 1)
        )

        if observed_ranks != expected_ranks:
            raise ValueError(
                f"{system_name}, query '{query_id}', "
                f"does not contain ranks exactly 1 to {TOP_K}."
            )

    ranking = ranking.sort_values(
        by=[
            "query_id",
            "rank",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(drop=True)

    return ranking


def validate_query_compatibility(
    bm25: pd.DataFrame,
    jina: pd.DataFrame,
) -> None:
    """Verify that both ranking files contain the same queries."""

    bm25_queries = set(
        bm25["query_id"].unique()
    )

    jina_queries = set(
        jina["query_id"].unique()
    )

    if bm25_queries != jina_queries:
        raise ValueError(
            "BM25Plus and Jina-v5 do not contain the same queries.\n"
            f"Missing from Jina-v5: "
            f"{sorted(bm25_queries - jina_queries)}\n"
            f"Extra in Jina-v5: "
            f"{sorted(jina_queries - bm25_queries)}"
        )

    bm25_metadata = (
        bm25[
            [
                "query_id",
                "category",
                "query",
            ]
        ]
        .drop_duplicates()
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    jina_metadata = (
        jina[
            [
                "query_id",
                "category",
                "query",
            ]
        ]
        .drop_duplicates()
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    if not bm25_metadata.equals(jina_metadata):
        raise ValueError(
            "BM25Plus and Jina-v5 contain different "
            "query text or category metadata."
        )


def prepare_bm25(
    bm25: pd.DataFrame,
) -> pd.DataFrame:
    """Rename BM25Plus fields and calculate its RRF contribution."""

    bm25 = bm25.rename(
        columns={
            "category": "category_bm25",
            "query": "query_bm25",
            "dataset_index": "dataset_index_bm25",
            "rank": "bm25_rank",
            "score": "bm25_score",
        }
    ).copy()

    bm25["bm25_rrf"] = (
        1.0
        / (
            RRF_K
            + bm25["bm25_rank"]
        )
    )

    return bm25[
        [
            "query_id",
            "candidate_id",
            "category_bm25",
            "query_bm25",
            "dataset_index_bm25",
            "bm25_rank",
            "bm25_score",
            "bm25_rrf",
        ]
    ].copy()


def prepare_jina(
    jina: pd.DataFrame,
) -> pd.DataFrame:
    """Rename Jina fields and calculate its RRF contribution."""

    jina = jina.rename(
        columns={
            "category": "category_jina",
            "query": "query_jina",
            "dataset_index": "dataset_index_jina",
            "rank": "jina_rank",
            "score": "jina_score",
        }
    ).copy()

    jina["jina_rrf"] = (
        1.0
        / (
            RRF_K
            + jina["jina_rank"]
        )
    )

    return jina[
        [
            "query_id",
            "candidate_id",
            "category_jina",
            "query_jina",
            "dataset_index_jina",
            "jina_rank",
            "jina_score",
            "jina_rrf",
        ]
    ].copy()


def create_hybrid(
    bm25: pd.DataFrame,
    jina: pd.DataFrame,
) -> pd.DataFrame:
    """Create the BM25Plus + Jina-v5 RRF Top-20 ranking."""

    hybrid = bm25.merge(
        jina,
        on=[
            "query_id",
            "candidate_id",
        ],
        how="outer",
        validate="one_to_one",
    )

    both_present = (
        hybrid["dataset_index_bm25"].notna()
        & hybrid["dataset_index_jina"].notna()
    )

    inconsistent_indices = hybrid.loc[
        both_present
        & (
            hybrid["dataset_index_bm25"]
            != hybrid["dataset_index_jina"]
        )
    ]

    if not inconsistent_indices.empty:
        raise ValueError(
            "The same candidate_id has different dataset_index "
            "values in BM25Plus and Jina-v5."
        )

    hybrid["category"] = (
        hybrid["category_bm25"]
        .combine_first(
            hybrid["category_jina"]
        )
    )

    hybrid["query"] = (
        hybrid["query_bm25"]
        .combine_first(
            hybrid["query_jina"]
        )
    )

    hybrid["dataset_index"] = (
        hybrid["dataset_index_bm25"]
        .combine_first(
            hybrid["dataset_index_jina"]
        )
        .astype(int)
    )

    hybrid["bm25_rrf"] = (
        hybrid["bm25_rrf"]
        .fillna(0.0)
    )

    hybrid["jina_rrf"] = (
        hybrid["jina_rrf"]
        .fillna(0.0)
    )

    hybrid["score"] = (
        hybrid["bm25_rrf"]
        + hybrid["jina_rrf"]
    )

    hybrid["retrieved_by"] = "BM25Plus;Jina-v5"

    hybrid.loc[
        hybrid["bm25_rank"].notna()
        & hybrid["jina_rank"].isna(),
        "retrieved_by",
    ] = "BM25Plus"

    hybrid.loc[
        hybrid["bm25_rank"].isna()
        & hybrid["jina_rank"].notna(),
        "retrieved_by",
    ] = "Jina-v5"

    hybrid["retrieval_count"] = (
        hybrid[
            [
                "bm25_rank",
                "jina_rank",
            ]
        ]
        .notna()
        .sum(axis=1)
        .astype(int)
    )

    hybrid["best_rank"] = (
        hybrid[
            [
                "bm25_rank",
                "jina_rank",
            ]
        ]
        .min(
            axis=1,
            skipna=True,
        )
    )

    hybrid = hybrid.sort_values(
        by=[
            "query_id",
            "score",
            "retrieval_count",
            "best_rank",
            "candidate_id",
        ],
        ascending=[
            True,
            False,
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

    hybrid["bm25_rank"] = (
        hybrid["bm25_rank"]
        .astype("Int64")
    )

    hybrid["jina_rank"] = (
        hybrid["jina_rank"]
        .astype("Int64")
    )

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
            "retrieval_count",
            "best_rank",
            "bm25_rank",
            "bm25_score",
            "bm25_rrf",
            "jina_rank",
            "jina_score",
            "jina_rrf",
        ]
    ].sort_values(
        by=[
            "query_id",
            "rank",
        ]
    ).reset_index(drop=True)

    duplicate_pairs = hybrid.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    if duplicate_pairs > 0:
        raise ValueError(
            f"Found {duplicate_pairs} duplicate "
            "query/candidate pairs in the hybrid."
        )

    rows_per_query = (
        hybrid
        .groupby("query_id")
        .size()
    )

    if not (rows_per_query == TOP_K).all():
        raise ValueError(
            f"Not every query contains exactly {TOP_K} "
            "hybrid results."
        )

    return hybrid


def safe_value(value) -> str:
    """Convert missing dataframe values to an empty string."""

    if pd.isna(value):
        return ""

    return str(value)


def optional_integer(value) -> str:
    """Format optional integer rank values for Markdown."""

    if pd.isna(value):
        return ""

    return str(int(value))


def optional_score(value) -> str:
    """Format optional original retrieval scores for Markdown."""

    if pd.isna(value):
        return ""

    return f"{float(value):.6f}"


def write_markdown(
    hybrid: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> None:
    """Write a readable Markdown version of the hybrid Top-20 results."""

    with MARKDOWN_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Hybrid BM25Plus + Jina-v5 RRF Retrieval Results\n"
        )

        file.write(f"\nTop K: {TOP_K}\n")
        file.write(f"RRF constant: {RRF_K}\n")
        file.write("Sparse retriever: BM25Plus\n")
        file.write("Dense retriever: Jina-v5\n")
        file.write(
            "Fusion method: Reciprocal Rank Fusion\n"
        )
        file.write(
            "RRF formula: "
            "1/(RRF_K + BM25 rank) + "
            "1/(RRF_K + Jina rank)\n"
        )
        file.write(
            "Retrieval text: candidate_text\n"
        )

        for category, category_group in hybrid.groupby(
            "category",
            sort=False,
        ):

            file.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            file.write(
                f"## Category: {category}\n"
            )

            file.write(
                "=" * 80
                + "\n"
            )

            for query_id, query_group in category_group.groupby(
                "query_id",
                sort=False,
            ):

                query_group = query_group.sort_values(
                    "rank"
                )

                query = query_group["query"].iloc[0]

                file.write(
                    f"\n### Query ID: {query_id}\n"
                )

                file.write(
                    f"### Query: {query}\n"
                )

                file.write(
                    f"\nTOP {TOP_K} HYBRID RRF RESULTS\n\n"
                )

                for _, result in query_group.iterrows():

                    dataset_index = int(
                        result["dataset_index"]
                    )

                    candidate_row = candidate_df.iloc[
                        dataset_index
                    ]

                    file.write("---\n")

                    file.write(
                        f"### Rank {int(result['rank'])}\n"
                    )

                    file.write(
                        f"- Hybrid RRF Score: "
                        f"{float(result['score']):.10f}\n"
                    )

                    file.write(
                        f"- Retrieved By: "
                        f"{safe_value(result['retrieved_by'])}\n"
                    )

                    file.write(
                        f"- BM25Plus Rank: "
                        f"{optional_integer(result['bm25_rank'])}\n"
                    )

                    file.write(
                        f"- BM25Plus Original Score: "
                        f"{optional_score(result['bm25_score'])}\n"
                    )

                    file.write(
                        f"- BM25Plus RRF Contribution: "
                        f"{float(result['bm25_rrf']):.10f}\n"
                    )

                    file.write(
                        f"- Jina-v5 Rank: "
                        f"{optional_integer(result['jina_rank'])}\n"
                    )

                    file.write(
                        f"- Jina-v5 Original Score: "
                        f"{optional_score(result['jina_score'])}\n"
                    )

                    file.write(
                        f"- Jina-v5 RRF Contribution: "
                        f"{float(result['jina_rrf']):.10f}\n"
                    )

                    file.write(
                        f"- Dataset Index: "
                        f"{dataset_index}\n"
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

                    file.write(
                        "\n#### Looking For\n"
                    )

                    file.write(
                        safe_value(
                            candidate_row.get(
                                "Looking For"
                            )
                        )
                        + "\n"
                    )

                    file.write(
                        "\n#### Highlights\n"
                    )

                    file.write(
                        safe_value(
                            candidate_row.get(
                                "Highlights"
                            )
                        )
                        + "\n"
                    )

                    file.write(
                        "\n#### CV\n"
                    )

                    file.write(
                        safe_value(
                            candidate_row.get("CV")
                        )
                        + "\n\n"
                    )


def main() -> None:
    """Build and save the BM25Plus + Jina-v5 RRF hybrid."""

    validate_file_exists(
        BM25_PATH,
        "BM25Plus ranking file",
    )

    validate_file_exists(
        JINA_PATH,
        "Jina-v5 ranking file",
    )

    validate_file_exists(
        DATA_PATH,
        "Candidate dataset",
    )

    bm25 = load_ranking(
        BM25_PATH,
        "BM25Plus",
    )

    jina = load_ranking(
        JINA_PATH,
        "Jina-v5",
    )

    validate_query_compatibility(
        bm25,
        jina,
    )

    prepared_bm25 = prepare_bm25(
        bm25
    )

    prepared_jina = prepare_jina(
        jina
    )

    hybrid = create_hybrid(
        prepared_bm25,
        prepared_jina,
    )

    candidate_df = pd.read_parquet(
        DATA_PATH
    ).reset_index(drop=True)

    candidate_df["candidate_id"] = (
        candidate_df.index.astype(str)
    )

    invalid_dataset_indices = hybrid.loc[
        (
            hybrid["dataset_index"] < 0
        )
        | (
            hybrid["dataset_index"]
            >= len(candidate_df)
        )
    ]

    if not invalid_dataset_indices.empty:
        raise ValueError(
            "The hybrid contains invalid dataset_index values."
        )

    if hybrid["query_id"].nunique() != 100 or len(hybrid) != 2000:
        raise ValueError(
            f"Expected 100 queries and 2000 hybrid rows, got "
            f"{hybrid['query_id'].nunique()} queries and {len(hybrid)} rows."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hybrid.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_markdown(
        hybrid,
        candidate_df,
    )

    both_systems_count = int(
        (
            hybrid["retrieval_count"] == 2
        ).sum()
    )

    bm25_only_count = int(
        (
            hybrid["retrieved_by"]
            == "BM25Plus"
        ).sum()
    )

    jina_only_count = int(
        (
            hybrid["retrieved_by"]
            == "Jina-v5"
        ).sum()
    )

    print("=" * 70)
    print("RRF HYBRID BM25PLUS + JINA-V5 COMPLETED")
    print("=" * 70)

    print(
        f"Queries: "
        f"{hybrid['query_id'].nunique()}"
    )

    print(
        f"Rows: {len(hybrid)}"
    )

    print(
        f"Top K per query: {TOP_K}"
    )

    print(
        f"RRF constant: {RRF_K}"
    )

    print(
        f"Rows retrieved by both systems: "
        f"{both_systems_count}"
    )

    print(
        f"Rows retrieved only by BM25Plus: "
        f"{bm25_only_count}"
    )

    print(
        f"Rows retrieved only by Jina-v5: "
        f"{jina_only_count}"
    )

    print(
        f"CSV saved to:\n{OUTPUT_PATH}"
    )

    print(
        f"Markdown saved to:\n"
        f"{MARKDOWN_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()