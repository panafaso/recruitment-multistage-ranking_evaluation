from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm


# ============================================================
# Project paths
# ============================================================

def find_project_root() -> Path:
    """
    Find the project root reliably.

    The script is expected to be stored under:
        project/reranking/scripts/

    The fallback search also allows it to be launched from the
    project root or moved one directory without breaking paths.
    """

    script_path = Path(__file__).resolve()
    candidates = [
        *script_path.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]

    checked: set[Path] = set()

    for candidate in candidates:
        if candidate in checked:
            continue

        checked.add(candidate)

        data_dir = candidate / "data"
        hybrid_dir = candidate / "hybrid_retrieval"

        if data_dir.is_dir() and hybrid_dir.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Expected one directory "
        "containing both 'data' and 'hybrid_retrieval'."
    )


PROJECT_ROOT = find_project_root()

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

INPUT_FILES = {
    'jina_v5': PROJECT_ROOT / 'dense_retrieval' / 'jina_v5_100_queries_top20.csv',
    'hybrid_bm25_jina': PROJECT_ROOT / 'hybrid_retrieval' / 'hybrid_bm25_jina_rrf_100_queries_top20.csv',
}

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_NAME = "bge_reranker_v2_m3"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reranking"
    / "outputs_100q_jina_vs_hybrid"
)

LOG_DIR = (
    PROJECT_ROOT
    / "reranking"
    / "logs_100q_jina_vs_hybrid"
)

TOP_K = 20

BATCH_SIZE = 2

MAX_LENGTH = 2048

FAIL_ON_TRUNCATION = False

SAVE_TOKEN_AUDIT = True


# Set True to run this model on CPU when GPU memory is insufficient.
FORCE_CPU = False

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
}

def create_logger(log_path: Path) -> logging.Logger:
    """Create a console and file logger for one experiment."""

    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def validate_file_exists(
    path: Path,
    description: str,
) -> None:
    """Raise an informative error for a missing input file."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_candidate_dataset(
    path: Path,
) -> pd.DataFrame:
    """
    Load candidate_text exactly as stored in the parquet file.

    No shortening, slicing, summarisation or character truncation
    is performed here.
    """

    validate_file_exists(
        path,
        "Candidate dataset",
    )

    candidates = pd.read_parquet(
        path
    ).reset_index(drop=True)

    if "candidate_text" not in candidates.columns:
        raise ValueError(
            "The candidate dataset does not contain "
            "'candidate_text'."
        )

    candidates["candidate_id"] = (
        candidates.index.astype(str)
    )

    candidates["dataset_index"] = (
        candidates.index.astype(int)
    )

    descriptive_columns = [
        "Position",
        "Moreinfo",
        "Experience Years",
        "English Level",
        "Looking For",
        "Highlights",
        "CV",
    ]

    available_columns = [
        column
        for column in descriptive_columns
        if column in candidates.columns
    ]

    result = candidates[
        [
            "candidate_id",
            "dataset_index",
            "candidate_text",
            *available_columns,
        ]
    ].copy()

    result["candidate_text"] = (
        result["candidate_text"]
        .fillna("")
        .astype(str)
    )

    empty_texts = int(
        result["candidate_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_texts:
        raise ValueError(
            f"The dataset contains {empty_texts} empty "
            "candidate_text values."
        )

    return result


def load_hybrid_ranking(
    path: Path,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Load one source Top-20 and attach full candidate text."""

    validate_file_exists(
        path,
        "Hybrid ranking file",
    )

    ranking = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    missing = (
        REQUIRED_RANKING_COLUMNS
        - set(ranking.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    ranking["dataset_index"] = pd.to_numeric(
        ranking["dataset_index"],
        errors="raise",
    ).astype(int)

    ranking["rank"] = pd.to_numeric(
        ranking["rank"],
        errors="raise",
    ).astype(int)

    ranking = (
        ranking
        .sort_values(
            [
                "query_id",
                "rank",
            ],
            kind="stable",
        )
        .groupby(
            "query_id",
            group_keys=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    duplicate_pairs = int(
        ranking.duplicated(
            [
                "query_id",
                "candidate_id",
            ]
        ).sum()
    )

    if duplicate_pairs:
        raise ValueError(
            f"{path.name} contains {duplicate_pairs} duplicate "
            "query-candidate pairs."
        )

    query_text_counts = (
        ranking
        .groupby("query_id")["query"]
        .nunique()
    )

    conflicting_queries = query_text_counts[
        query_text_counts != 1
    ]

    if not conflicting_queries.empty:
        raise ValueError(
            "Some query IDs have more than one query text: "
            f"{conflicting_queries.index.tolist()}"
        )

    # Use the full canonical CV from parquet, even if the source CSV embeds CV fields.
    ranking = ranking.drop(columns=[
        'candidate_text', 'Position', 'Moreinfo', 'Experience Years',
        'English Level', 'Looking For', 'Highlights', 'CV'
    ], errors='ignore')
    ranking = ranking.merge(
        candidates,
        on=[
            "candidate_id",
            "dataset_index",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_texts = int(
        ranking["candidate_text"].isna().sum()
    )

    if missing_texts:
        raise ValueError(
            f"{missing_texts} ranking rows could not be matched "
            "to candidate_text in the parquet dataset."
        )

    counts = (
        ranking
        .groupby("query_id")
        .size()
    )

    incomplete = counts[
        counts != top_k
    ]

    if not incomplete.empty:
        raise ValueError(
            f"{path.name} does not contain exactly {top_k} "
            "candidates for every query:\n"
            f"{incomplete.to_string()}"
        )

    expected_ranks = list(
        range(1, top_k + 1)
    )

    for query_id, group in ranking.groupby(
        "query_id",
        sort=False,
    ):
        observed_ranks = sorted(
            group["rank"].tolist()
        )

        if observed_ranks != expected_ranks:
            raise ValueError(
                f"{path.name}, query '{query_id}', does not "
                f"contain ranks exactly 1..{top_k}."
            )

    return ranking

def select_device() -> str:
    """Use CUDA when available, otherwise fall back to CPU."""

    return (
        "cuda"
        if torch.cuda.is_available() and not FORCE_CPU
        else "cpu"
    )


def load_cross_encoder(
    model_name: str,
    device: str,
) -> CrossEncoder:
    """
    Load BGE as a pairwise CrossEncoder.

    The model receives:
        (query, full candidate_text)

    max_length controls the total token count of the pair.
    """

    model = CrossEncoder(
        model_name,
        max_length=MAX_LENGTH,
        device=device,
        activation_fn=torch.nn.Identity(),
    )

    # The official BGE example supports FP16 inference.
    # This materially reduces GPU memory use.
    if device == "cuda":
        model.model.half()

    model.model.eval()

    return model


def count_pair_tokens(
    tokenizer,
    query: str,
    document: str,
) -> int:
    """
    Count the complete query-document pair without truncation.

    This mirrors the pair input used by the CrossEncoder:
        tokenizer(query, document, add_special_tokens=True)
    """

    encoded = tokenizer(
        query,
        document,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    return len(
        encoded["input_ids"]
    )


def create_token_audit(
    ranking: pd.DataFrame,
    tokenizer,
    retriever_name: str,
) -> pd.DataFrame:
    """Measure every full query-CV pair before inference."""

    rows: list[dict[str, object]] = []

    for row in tqdm(
        ranking.itertuples(index=False),
        total=len(ranking),
        desc=f"Token audit: {retriever_name}",
    ):
        query_text = str(
            row.query
        )

        candidate_text = str(
            row.candidate_text
        )

        pair_tokens = count_pair_tokens(
            tokenizer=tokenizer,
            query=query_text,
            document=candidate_text,
        )

        rows.append({
            "retriever": retriever_name,
            "query_id": row.query_id,
            "candidate_id": row.candidate_id,
            "dataset_index": int(row.dataset_index),
            "original_rank": int(row.rank),
            "query_characters": len(query_text),
            "candidate_characters": len(candidate_text),
            "pair_tokens": pair_tokens,
            "max_length": MAX_LENGTH,
            "fits_without_truncation": (
                pair_tokens <= MAX_LENGTH
            ),
            "tokens_over_limit": max(
                0,
                pair_tokens - MAX_LENGTH,
            ),
        })

    return pd.DataFrame(
        rows
    )


def save_and_validate_token_audit(
    audit: pd.DataFrame,
    retriever_name: str,
    experiment_dir: Path,
) -> None:
    """
    Save token statistics and enforce the no-truncation policy.
    """

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        experiment_dir
        / f"{retriever_name}__token_length_audit.csv"
    )

    if SAVE_TOKEN_AUDIT:
        audit.to_csv(
            audit_path,
            index=False,
            encoding="utf-8-sig",
        )

    over_limit = audit[
        ~audit["fits_without_truncation"]
    ]

    maximum = int(
        audit["pair_tokens"].max()
    )

    mean = float(
        audit["pair_tokens"].mean()
    )

    p95 = float(
        audit["pair_tokens"].quantile(0.95)
    )

    print(
        f"{retriever_name} token audit | "
        f"max={maximum} | "
        f"mean={mean:.2f} | "
        f"p95={p95:.2f} | "
        f"over_limit={len(over_limit)}"
    )

    print(
        f"Audit file: {audit_path}"
    )

    if (
        FAIL_ON_TRUNCATION
        and not over_limit.empty
    ):
        examples = over_limit[
            [
                "query_id",
                "candidate_id",
                "pair_tokens",
                "tokens_over_limit",
            ]
        ].sort_values(
            "pair_tokens",
            ascending=False,
        ).head(10)

        raise ValueError(
            f"\n{retriever_name} contains {len(over_limit)} "
            f"query-CV pairs above the {MAX_LENGTH}-token "
            "model limit.\n"
            "The reranking run was stopped because "
            "FAIL_ON_TRUNCATION=True.\n"
            "Therefore no CV was silently cut.\n\n"
            "Longest examples:\n"
            f"{examples.to_string(index=False)}\n\n"
            f"Full audit:\n{audit_path}"
        )


# Reranking

def rerank_one_system(
    ranking: pd.DataFrame,
    model: CrossEncoder,
    retriever_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Rerank the Top-20 candidates for every query."""

    output_groups: list[pd.DataFrame] = []

    grouped = ranking.groupby(
        "query_id",
        sort=False,
    )

    progress = tqdm(
        grouped,
        total=ranking["query_id"].nunique(),
        desc=(
            f"{retriever_name} "
            f"-> {RERANKER_NAME}"
        ),
    )

    for query_id, group in progress:
        group = group.copy()

        query_text = str(
            group["query"].iloc[0]
        )

        documents = (
            group["candidate_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        pairs = [
            (
                query_text,
                document,
            )
            for document in documents
        ]

        
        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = np.asarray(
            scores
        ).reshape(-1)

        if len(scores) != len(group):
            raise RuntimeError(
                f"Score mismatch for query {query_id}: "
                f"{len(scores)} scores for {len(group)} rows."
            )

        group["retriever"] = retriever_name
        group["reranker"] = RERANKER_NAME
        group["original_rank"] = group["rank"]

        if "score" in group.columns:
            group["original_score"] = (
                group["score"]
            )
        elif "rrf_score" in group.columns:
            group["original_score"] = (
                group["rrf_score"]
            )
        else:
            group["original_score"] = np.nan

        # Raw BGE logits; higher means more relevant.
        group["reranker_score"] = (
            scores.astype(float)
        )

        clipped = np.clip(
            scores.astype(float),
            -50.0,
            50.0,
        )

        group["reranker_sigmoid"] = (
            1.0
            / (
                1.0
                + np.exp(-clipped)
            )
        )

        group = group.sort_values(
            [
                "reranker_score",
                "original_rank",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        ).reset_index(drop=True)

        group["reranked_rank"] = np.arange(
            1,
            len(group) + 1,
        )

        group["rank_change"] = (
            group["original_rank"]
            - group["reranked_rank"]
        )

        output_groups.append(
            group
        )

    reranked = pd.concat(
        output_groups,
        ignore_index=True,
    )

    reranked = reranked.sort_values(
        [
            "query_id",
            "reranked_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)

    logger.info(
        "Reranked rows: %d | Queries: %d",
        len(reranked),
        reranked["query_id"].nunique(),
    )

    return reranked


# Evaluation-compatible output

def create_evaluation_ranking(
    reranked: pd.DataFrame,
    system_name: str,
) -> pd.DataFrame:
    """Create the compact ranking CSV used by the metrics script."""

    evaluation = reranked.copy()

    evaluation["system"] = system_name
    evaluation["rank"] = (
        evaluation["reranked_rank"]
    )
    evaluation["score"] = (
        evaluation["reranker_score"]
    )

    preferred_columns = [
        "system",
        "query_id",
        "category",
        "query",
        "candidate_id",
        "dataset_index",
        "rank",
        "score",
    ]

    return evaluation[
        preferred_columns
    ].copy()


# Markdown reporting


def markdown_value(
    value: object,
) -> str:
    """Return clean text for Markdown output."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def format_score(
    value: object,
) -> str:
    """Format a numeric score consistently."""

    if pd.isna(value):
        return ""

    return f"{float(value):.6f}"


def save_markdown_report(
    reranked: pd.DataFrame,
    experiment_dir: Path,
    experiment_name: str,
    retriever_name: str,
    logger: logging.Logger,
) -> None:
    """Save one combined Markdown report for an experiment."""

    report_path = (
        experiment_dir
        / f"{experiment_name}_results.md"
    )

    lines: list[str] = [
        (
            "# BGE Reranker v2-m3 Results: "
            f"{experiment_name}"
        ),
        "",
        f"Model: {MODEL_NAME}",
        f"Top K reranked: {TOP_K}",
        f"Initial retriever: {retriever_name}",
        f"Maximum pair length: {MAX_LENGTH} tokens",
        f"Batch size: {BATCH_SIZE}",
        "Input: query + complete candidate_text",
        (
            "Truncation policy: token audit first; "
            "run stops if a pair exceeds the model limit"
        ),
        "",
    ]

    ordered = reranked.sort_values(
        [
            "category",
            "query_id",
            "reranked_rank",
        ],
        kind="stable",
    )

    for (
        category,
        query_id,
    ), group in ordered.groupby(
        [
            "category",
            "query_id",
        ],
        sort=False,
    ):
        group = group.sort_values(
            "reranked_rank",
            kind="stable",
        )

        query_text = markdown_value(
            group["query"].iloc[0]
        )

        lines.extend([
            "=" * 80,
            (
                f"## Category: "
                f"{markdown_value(category)}"
            ),
            "=" * 80,
            "",
            f"### Query ID: {query_id}",
            f"### Query: {query_text}",
            "",
            (
                f"TOP {len(group)} "
                "BGE CROSS-ENCODER RERANKED RESULTS"
            ),
            "",
        ])

        for row in group.itertuples(
            index=False
        ):
            row_data = row._asdict()

            lines.extend([
                "---",
                (
                    f"### Rank "
                    f"{int(row.reranked_rank)}"
                ),
                (
                    "- BGE Raw Logit: "
                    f"{format_score(row.reranker_score)}"
                ),
                (
                    "- BGE Sigmoid: "
                    f"{format_score(row.reranker_sigmoid)}"
                ),
                (
                    "- Original Hybrid Rank: "
                    f"{int(row.original_rank)}"
                ),
                (
                    "- Rank Change: "
                    f"{int(row.rank_change):+d}"
                ),
                (
                    "- Original Hybrid Score: "
                    f"{format_score(row.original_score)}"
                ),
                (
                    "- Dataset Index: "
                    f"{int(row.dataset_index)}"
                ),
                (
                    "- Candidate ID: "
                    f"{markdown_value(row.candidate_id)}"
                ),
            ])

            if "Position" in row_data:
                lines.append(
                    "- Position: "
                    f"{markdown_value(row_data['Position'])}"
                )

            if "Experience_Years" in row_data:
                lines.append(
                    "- Experience Years: "
                    f"{markdown_value(row_data['Experience_Years'])}"
                )

            if "English_Level" in row_data:
                lines.append(
                    "- English Level: "
                    f"{markdown_value(row_data['English_Level'])}"
                )

            lines.append("")

            if "Looking_For" in row_data:
                lines.extend([
                    "#### Looking For",
                    markdown_value(
                        row_data["Looking_For"]
                    ),
                    "",
                ])

            if "Highlights" in row_data:
                lines.extend([
                    "#### Highlights",
                    markdown_value(
                        row_data["Highlights"]
                    ),
                    "",
                ])

            if "CV" in row_data:
                lines.extend([
                    "#### Full CV",
                    markdown_value(
                        row_data["CV"]
                    ),
                    "",
                ])
            else:
                lines.extend([
                    "#### Complete candidate text",
                    markdown_value(
                        row_data["candidate_text"]
                    ),
                    "",
                ])

        lines.append("")

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "Markdown report: %s",
        report_path,
    )

# Saving

def save_outputs(
    reranked: pd.DataFrame,
    retriever_name: str,
    logger: logging.Logger,
) -> None:
    """Save detailed, evaluation-ready and Markdown outputs."""

    experiment_name = (
        f"{retriever_name}"
        f"__{RERANKER_NAME}"
    )

    experiment_dir = (
        OUTPUT_DIR
        / RERANKER_NAME
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        experiment_dir
        / f"{experiment_name}_detailed.csv"
    )

    evaluation_path = (
        experiment_dir
        / f"{experiment_name}_rankings.csv"
    )

    reranked.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig",
    )

    evaluation = create_evaluation_ranking(
        reranked=reranked,
        system_name=experiment_name,
    )

    evaluation.to_csv(
        evaluation_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_markdown_report(
        reranked=reranked,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        retriever_name=retriever_name,
        logger=logger,
    )

    logger.info(
        "Detailed output: %s",
        detailed_path,
    )

    logger.info(
        "Evaluation output: %s",
        evaluation_path,
    )


def main() -> None:
    """Run BGE reranking for Jina v5 and BM25+Jina."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = select_device()

    print("=" * 78)
    print(
        "BGE RERANKER V2-M3 "
        "— JINA-V5 AND BM25+JINA TOP-20"
    )
    print("=" * 78)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Top K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum pair length: {MAX_LENGTH}")
    print(
        "Silent truncation allowed: "
        f"{not FAIL_ON_TRUNCATION}"
    )

    candidates = load_candidate_dataset(
        DATASET_PATH
    )

    hybrid_rankings: dict[
        str,
        pd.DataFrame,
    ] = {}

    for (
        retriever_name,
        ranking_path,
    ) in INPUT_FILES.items():
        hybrid_rankings[retriever_name] = (
            load_hybrid_ranking(
                path=ranking_path,
                candidates=candidates,
                top_k=TOP_K,
            )
        )

        print(
            f"Loaded {retriever_name}: "
            f"{len(hybrid_rankings[retriever_name])} rows"
        )

    print("\nLoading BGE reranker...")

    model = load_cross_encoder(
        model_name=MODEL_NAME,
        device=device,
    )

    print(
        "CrossEncoder configured max length: "
        f"{model.max_length}"
    )

    tokenizer_max_length = int(
        model.tokenizer.model_max_length
    )

    print(
        "Tokenizer model max length: "
        f"{tokenizer_max_length}"
    )

    if MAX_LENGTH > tokenizer_max_length:
        raise ValueError(
            f"Configured MAX_LENGTH={MAX_LENGTH} exceeds the "
            f"tokenizer limit {tokenizer_max_length}."
        )

    experiment_dir = (
        OUTPUT_DIR
        / RERANKER_NAME
    )

    for (
        retriever_name,
        ranking,
    ) in hybrid_rankings.items():
        audit = create_token_audit(
            ranking=ranking,
            tokenizer=model.tokenizer,
            retriever_name=retriever_name,
        )

        save_and_validate_token_audit(
            audit=audit,
            retriever_name=retriever_name,
            experiment_dir=experiment_dir,
        )

    total_experiments = len(
        hybrid_rankings
    )

    for experiment_number, (
        retriever_name,
        ranking,
    ) in enumerate(
        hybrid_rankings.items(),
        start=1,
    ):
        log_path = (
            LOG_DIR
            / (
                f"{retriever_name}"
                f"__{RERANKER_NAME}.log"
            )
        )

        logger = create_logger(
            log_path
        )

        logger.info(
            "Experiment %d/%d",
            experiment_number,
            total_experiments,
        )

        logger.info(
            "Retriever: %s",
            retriever_name,
        )

        logger.info(
            "Reranker: %s",
            MODEL_NAME,
        )

        logger.info(
            "Device: %s",
            device,
        )

        logger.info(
            (
                "Top K: %d | Batch size: %d | "
                "Maximum pair length: %d"
            ),
            TOP_K,
            BATCH_SIZE,
            MAX_LENGTH,
        )

        start_time = time.perf_counter()

        try:
            reranked = rerank_one_system(
                ranking=ranking,
                model=model,
                retriever_name=retriever_name,
                logger=logger,
            )

            save_outputs(
                reranked=reranked,
                retriever_name=retriever_name,
                logger=logger,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            logger.info(
                "Completed successfully in %.2f seconds.",
                elapsed,
            )

        except torch.cuda.OutOfMemoryError as error:
            logger.exception(
                (
                    "CUDA out of memory with BATCH_SIZE=%d and "
                    "MAX_LENGTH=%d. BATCH_SIZE is already 1. "
                    "Do not reduce MAX_LENGTH if the goal is to "
                    "preserve every full CV; use CPU inference, "
                    "chunking, or a GPU with more memory."
                ),
                BATCH_SIZE,
                MAX_LENGTH,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise error

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(
        "ALL BGE RERANKER V2-M3 RUNS COMPLETED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()