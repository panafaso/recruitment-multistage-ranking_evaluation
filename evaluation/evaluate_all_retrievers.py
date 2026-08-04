from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QRELS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "qrels.csv"
)

RANKING_FILES = {
    "BM25Plus": (
        PROJECT_ROOT
        / "sparse_retrieval"
        / "bm25plus_top50.csv"
    ),
    "BGE-M3": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "bge_m3_top50.csv"
    ),
    "MXBAI": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "mxbai_top50.csv"
    ),
    "Jina-v5": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "jina_v5_text_small_top50.csv"
    ),
    "Nomic-v1.5": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "nomic_embed_text_v1_5_top50.csv"
    ),
    # HYBRID RRF SYSTEMS

    "Hybrid BM25+Jina": (
        PROJECT_ROOT
        / "hybrid_retrieval"
        / "hybrid_bm25_jina_rrf_top50.csv"
    ),
    "Hybrid BM25+Nomic": (
        PROJECT_ROOT
        / "hybrid_retrieval"
        / "hybrid_bm25_nomic_rrf_top50.csv"
    ),
    "Hybrid BM25+BGE": (
        PROJECT_ROOT
        / "hybrid_retrieval"
        / "hybrid_bm25_bge_rrf_top50.csv"
    ),

    # BGE CROSS-ENCODER RERANKING

    "BM25+Nomic → BGE-Reranker": (
        PROJECT_ROOT
        / "reranking"
        / "outputs"
        / "bge_reranker_v2_m3"
        / "hybrid_bm25_nomic__bge_reranker_v2_m3_rankings.csv"
    ),

    "BM25+Jina → BGE-Reranker": (
        PROJECT_ROOT
        / "reranking"
        / "outputs"
        / "bge_reranker_v2_m3"
        / "hybrid_bm25_jina__bge_reranker_v2_m3_rankings.csv"
    ),
    
    "BM25+Nomic → GTE-ModernColBERT": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "gte_moderncolbert_v1"
    / "hybrid_bm25_nomic__gte_moderncolbert_v1_rankings.csv"
),

"BM25+Jina → GTE-ModernColBERT": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "gte_moderncolbert_v1"
    / "hybrid_bm25_jina__gte_moderncolbert_v1_rankings.csv"
),
"BM25+Nomic → Jina-v3.5 Top-10": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "jina_reranker_v3_5"
    / "hybrid_bm25_nomic__jina_reranker_v3_5__top10_rankings.csv"
),

"BM25+Jina → Jina-v3.5 Top-10": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "jina_reranker_v3_5"
    / "hybrid_bm25_jina__jina_reranker_v3_5__top10_rankings.csv"
),
"BM25+Nomic → Ettin-150M": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "ettin_reranker_150m"
    / "hybrid_bm25_nomic__ettin_reranker_150m_rankings.csv"
),

"BM25+Jina → Ettin-150M": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "ettin_reranker_150m"
    / "hybrid_bm25_jina__ettin_reranker_150m_rankings.csv"
),
"BM25+Nomic → Llama-Nemotron-1B": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "llama_nemotron_rerank_1b_v2"
    / "hybrid_bm25_nomic__llama_nemotron_rerank_1b_v2_rankings.csv"
),

"BM25+Jina → Llama-Nemotron-1B": (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / "llama_nemotron_rerank_1b_v2"
    / "hybrid_bm25_jina__llama_nemotron_rerank_1b_v2_rankings.csv"
),
}


RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

PER_QUERY_OUTPUT = (
    RESULTS_DIR
    / "metrics_per_query.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "metrics_summary_updated.csv"
)

BROAD_OUTPUT = (
    RESULTS_DIR
    / "metrics_broad_relevance.csv"
)

STRICT_OUTPUT = (
    RESULTS_DIR
    / "metrics_strict_relevance.csv"
)


K_VALUES = [5, 10, 20, 30, 40, 50]

# Broad relevance:
# relevance 1 and 2 are treated as relevant.
BROAD_THRESHOLD = 1

# Strict relevance:
# only relevance 2 is treated as relevant.
STRICT_THRESHOLD = 2

REQUIRED_QRELS_COLUMNS = {
    "query_id",
    "candidate_id",
    "relevance",
}

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "candidate_id",
    "rank",
}

def validate_file_exists(path: Path, description: str) -> None:
    """Raise an informative error if an input file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def load_qrels(path: Path) -> pd.DataFrame:
    """Load and validate qrels."""

    qrels = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    missing_columns = (
        REQUIRED_QRELS_COLUMNS
        - set(qrels.columns)
    )

    if missing_columns:
        raise ValueError(
            "qrels.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    qrels = qrels[
        [
            "query_id",
            "candidate_id",
            "relevance",
        ]
    ].copy()

    if qrels["relevance"].isna().any():
        missing_count = qrels["relevance"].isna().sum()

        raise ValueError(
            f"qrels.csv contains {missing_count} missing "
            "relevance judgments."
        )

    qrels["relevance"] = pd.to_numeric(
        qrels["relevance"],
        errors="raise",
    ).astype(int)

    invalid_labels = sorted(
        set(qrels["relevance"].unique())
        - {0, 1, 2}
    )

    if invalid_labels:
        raise ValueError(
            "qrels.csv contains invalid relevance labels: "
            f"{invalid_labels}. Expected only 0, 1, or 2."
        )

    duplicate_count = qrels.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"qrels.csv contains {duplicate_count} duplicate "
            "query/candidate pairs."
        )

    return qrels


def load_ranking(
    path: Path,
    system_name: str,
) -> pd.DataFrame:
    """Load and validate one retriever ranking."""

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
            "candidate_id",
            "rank",
        ]
    ].copy()

    ranking["rank"] = pd.to_numeric(
        ranking["rank"],
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

    invalid_ranks = ranking.loc[
        ranking["rank"] < 1
    ]

    if not invalid_ranks.empty:
        raise ValueError(
            f"{system_name} contains ranks smaller than 1."
        )

    ranking = ranking.sort_values(
        by=[
            "query_id",
            "rank",
        ]
    ).reset_index(drop=True)

    return ranking


def precision_at_k(
    binary_relevance: np.ndarray,
    k: int,
) -> float:
    """Precision@k."""

    top_k = binary_relevance[:k]

    if k == 0:
        return 0.0

    return float(top_k.sum() / k)


def recall_at_k(
    binary_relevance: np.ndarray,
    total_relevant: int,
    k: int,
) -> float:
    """Recall@k relative to the pooled qrels."""

    if total_relevant == 0:
        return 0.0

    top_k = binary_relevance[:k]

    return float(
        top_k.sum()
        / total_relevant
    )


def f1_at_k(
    precision: float,
    recall: float,
) -> float:
    """Harmonic mean of Precision@k and Recall@k."""

    if precision + recall == 0:
        return 0.0

    return float(
        2 * precision * recall
        / (precision + recall)
    )


def reciprocal_rank_at_k(
    binary_relevance: np.ndarray,
    k: int,
) -> float:
    """Reciprocal rank of the first relevant result up to k."""

    top_k = binary_relevance[:k]

    relevant_positions = np.flatnonzero(
        top_k == 1
    )

    if len(relevant_positions) == 0:
        return 0.0

    first_rank = relevant_positions[0] + 1

    return float(1.0 / first_rank)


def average_precision_at_k(
    binary_relevance: np.ndarray,
    total_relevant: int,
    k: int,
) -> float:
    """
    Average Precision@k.

    The denominator is min(total relevant documents, k), which is a common
    truncated AP@k definition. Recall remains measured against all pooled
    relevant documents.
    """

    if total_relevant == 0:
        return 0.0

    top_k = binary_relevance[:k]

    precision_sum = 0.0
    relevant_seen = 0

    for rank_position, relevance in enumerate(
        top_k,
        start=1,
    ):
        if relevance == 1:
            relevant_seen += 1

            precision_sum += (
                relevant_seen
                / rank_position
            )

    denominator = min(
        total_relevant,
        k,
    )

    return float(
        precision_sum
        / denominator
    )


def dcg_at_k(
    graded_relevance: np.ndarray,
    k: int,
) -> float:
    """
    Discounted Cumulative Gain@k.

    Uses the standard exponential gain:
    gain = 2^relevance - 1
    """

    top_k = graded_relevance[:k]

    if len(top_k) == 0:
        return 0.0

    gains = (
        np.power(2.0, top_k)
        - 1.0
    )

    discounts = np.log2(
        np.arange(
            2,
            len(top_k) + 2,
        )
    )

    return float(
        np.sum(
            gains
            / discounts
        )
    )


def ndcg_at_k(
    graded_relevance: np.ndarray,
    ideal_relevance: np.ndarray,
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain@k."""

    dcg = dcg_at_k(
        graded_relevance,
        k,
    )

    ideal_dcg = dcg_at_k(
        ideal_relevance,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        dcg
        / ideal_dcg
    )


def evaluate_query(
    system_name: str,
    query_id: str,
    ranking_query: pd.DataFrame,
    qrels_query: pd.DataFrame,
    threshold_name: str,
    relevance_threshold: int,
    k: int,
) -> dict:
    """Evaluate one retrieval system for one query and one cutoff."""

    relevance_lookup = dict(
        zip(
            qrels_query["candidate_id"],
            qrels_query["relevance"],
        )
    )

    ranked_candidate_ids = (
        ranking_query
        .sort_values("rank")
        ["candidate_id"]
        .tolist()
    )

    # All retrieved candidates should normally be judged because the
    # annotation pool was created from the union of all Top-50 rankings.
    graded_relevance = np.array(
        [
            relevance_lookup.get(
                candidate_id,
                0,
            )
            for candidate_id
            in ranked_candidate_ids
        ],
        dtype=float,
    )

    binary_relevance = (
        graded_relevance
        >= relevance_threshold
    ).astype(int)

    total_relevant = int(
        (
            qrels_query["relevance"]
            >= relevance_threshold
        ).sum()
    )

    ideal_relevance = np.array(
        sorted(
            qrels_query["relevance"].tolist(),
            reverse=True,
        ),
        dtype=float,
    )

    precision = precision_at_k(
        binary_relevance,
        k,
    )

    recall = recall_at_k(
        binary_relevance,
        total_relevant,
        k,
    )

    f1 = f1_at_k(
        precision,
        recall,
    )

    average_precision = average_precision_at_k(
        binary_relevance,
        total_relevant,
        k,
    )

    reciprocal_rank = reciprocal_rank_at_k(
        binary_relevance,
        k,
    )

    ndcg = ndcg_at_k(
        graded_relevance,
        ideal_relevance,
        k,
    )

    judged_retrieved_at_k = sum(
        candidate_id in relevance_lookup
        for candidate_id
        in ranked_candidate_ids[:k]
    )

    return {
        "system": system_name,
        "query_id": query_id,
        "threshold": threshold_name,
        "relevance_threshold": relevance_threshold,
        "k": k,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": average_precision,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "total_relevant_in_pool": total_relevant,
        "retrieved_at_k": min(
            k,
            len(ranked_candidate_ids),
        ),
        "judged_retrieved_at_k": judged_retrieved_at_k,
    }


def main() -> None:
    """Evaluate all retrieval systems at all requested cutoffs."""

    validate_file_exists(
        QRELS_PATH,
        "qrels.csv",
    )

    for system_name, ranking_path in RANKING_FILES.items():
        validate_file_exists(
            ranking_path,
            f"{system_name} ranking file",
        )

    qrels = load_qrels(
        QRELS_PATH
    )

    qrels_query_ids = set(
        qrels["query_id"].unique()
    )

    evaluation_rows = []

    threshold_settings = {
        "broad": BROAD_THRESHOLD,
        "strict": STRICT_THRESHOLD,
    }

    for system_name, ranking_path in RANKING_FILES.items():
        ranking = load_ranking(
            ranking_path,
            system_name,
        )

        ranking_query_ids = set(
            ranking["query_id"].unique()
        )

        missing_queries = (
            qrels_query_ids
            - ranking_query_ids
        )

        extra_queries = (
            ranking_query_ids
            - qrels_query_ids
        )

        if missing_queries or extra_queries:
            raise ValueError(
                f"Query mismatch for {system_name}.\n"
                f"Missing queries: {sorted(missing_queries)}\n"
                f"Extra queries: {sorted(extra_queries)}"
            )

        for query_id in sorted(
            qrels_query_ids
        ):
            ranking_query = ranking[
                ranking["query_id"]
                == query_id
            ].copy()

            qrels_query = qrels[
                qrels["query_id"]
                == query_id
            ].copy()

            ranking_size = len(
                ranking_query
            )

            if ranking_size < max(K_VALUES):
                raise ValueError(
                    f"{system_name} has only {ranking_size} results "
                    f"for query '{query_id}', but evaluation requires "
                    f"at least {max(K_VALUES)}."
                )

            retrieved_ids = set(
                ranking_query["candidate_id"]
            )

            judged_ids = set(
                qrels_query["candidate_id"]
            )

            unjudged_ids = (
                retrieved_ids
                - judged_ids
            )

            if unjudged_ids:
                raise ValueError(
                    f"{system_name} contains {len(unjudged_ids)} "
                    f"unjudged candidates for query '{query_id}'. "
                    "Rebuild the annotation pool or verify candidate IDs."
                )

            for threshold_name, threshold in (
                threshold_settings.items()
            ):
                for k in K_VALUES:
                    evaluation_rows.append(
                        evaluate_query(
                            system_name=system_name,
                            query_id=query_id,
                            ranking_query=ranking_query,
                            qrels_query=qrels_query,
                            threshold_name=threshold_name,
                            relevance_threshold=threshold,
                            k=k,
                        )
                    )

    per_query_metrics = pd.DataFrame(
        evaluation_rows
    )

    summary_metrics = (
        per_query_metrics
        .groupby(
            [
                "system",
                "threshold",
                "relevance_threshold",
                "k",
            ],
            as_index=False,
        )
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            map=("map", "mean"),
            mrr=("mrr", "mean"),
            ndcg=("ndcg", "mean"),
            queries=("query_id", "nunique"),
            mean_relevant_in_pool=(
                "total_relevant_in_pool",
                "mean",
            ),
            mean_judged_retrieved_at_k=(
                "judged_retrieved_at_k",
                "mean",
            ),
        )
    )

    summary_metrics = summary_metrics.sort_values(
        by=[
            "threshold",
            "k",
            "ndcg",
            "map",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    broad_metrics = summary_metrics[
        summary_metrics["threshold"]
        == "broad"
    ].reset_index(drop=True)

    strict_metrics = summary_metrics[
        summary_metrics["threshold"]
        == "strict"
    ].reset_index(drop=True)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_query_metrics.to_csv(
        PER_QUERY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_metrics.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    broad_metrics.to_csv(
        BROAD_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    strict_metrics.to_csv(
        STRICT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 75)
    print("ALL RETRIEVERS EVALUATED SUCCESSFULLY")
    print("=" * 75)

    print(
        f"Retrieval systems: "
        f"{len(RANKING_FILES)}"
    )

    print(
        f"Queries: "
        f"{qrels['query_id'].nunique()}"
    )

    print(
        f"Cutoffs: "
        f"{K_VALUES}"
    )

    print(
        "Thresholds: "
        f"broad >= {BROAD_THRESHOLD}, "
        f"strict >= {STRICT_THRESHOLD}"
    )

    print(
        f"Per-query rows: "
        f"{len(per_query_metrics)}"
    )

    print(
        "\nSaved files:"
    )

    print(
        f"- {PER_QUERY_OUTPUT}"
    )

    print(
        f"- {SUMMARY_OUTPUT}"
    )

    print(
        f"- {BROAD_OUTPUT}"
    )

    print(
        f"- {STRICT_OUTPUT}"
    )

    display_columns = [
        "system",
        "threshold",
        "k",
        "precision",
        "recall",
        "f1",
        "map",
        "mrr",
        "ndcg",
    ]

    print(
        "\nSummary preview:"
    )

    print(
        summary_metrics[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()