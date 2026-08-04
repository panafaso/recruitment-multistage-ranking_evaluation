from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

HYBRID_FILES = {
    "hybrid_bm25_nomic": (
        PROJECT_ROOT
        / "hybrid_retrieval"
        / "hybrid_bm25_nomic_rrf_top50.csv"
    ),
    "hybrid_bm25_jina": (
        PROJECT_ROOT
        / "hybrid_retrieval"
        / "hybrid_bm25_jina_rrf_top50.csv"
    ),
}

MODEL_NAME = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_NAME = "llama_nemotron_rerank_1b_v2"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
)

LOG_DIR = (
    PROJECT_ROOT
    / "reranking"
    / "logs"
)


TOP_K = 50

BATCH_SIZE = 1


MAX_LENGTH = 8192

FAIL_ON_TRUNCATION = True

SAVE_TOKEN_AUDIT = True

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
}

def create_logger(log_path: Path) -> logging.Logger:
    """Create one console + file logger for an experiment."""

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
    """Raise an informative error if a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def load_candidate_dataset(
    path: Path,
) -> pd.DataFrame:
    """
    Load candidate metadata and the complete candidate_text.

    No truncation or shortening is applied here.
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

    return candidates[
        [
            "candidate_id",
            "dataset_index",
            "candidate_text",
            *available_columns,
        ]
    ].copy()


def load_hybrid_ranking(
    path: Path,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Load one hybrid Top-50 ranking and attach candidate text."""

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

    missing_columns = (
        REQUIRED_RANKING_COLUMNS
        - set(ranking.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    ranking["dataset_index"] = pd.to_numeric(
        ranking["dataset_index"],
        errors="raise",
    ).astype(int)

    ranking["rank"] = pd.to_numeric(
        ranking["rank"],
        errors="raise",
    ).astype(int)

    duplicate_pairs = int(
        ranking.duplicated(
            subset=[
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

    ranking = (
        ranking
        .sort_values(
            by=[
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

    rows_per_query = (
        ranking
        .groupby("query_id")
        .size()
    )

    incomplete_queries = rows_per_query[
        rows_per_query != top_k
    ]

    if not incomplete_queries.empty:
        raise ValueError(
            f"{path.name} does not contain exactly "
            f"{top_k} candidates for every query:\n"
            f"{incomplete_queries}"
        )

    expected_ranks = list(
        range(1, top_k + 1)
    )

    for query_id, group in ranking.groupby(
        "query_id"
    ):
        observed_ranks = sorted(
            group["rank"].tolist()
        )

        if observed_ranks != expected_ranks:
            raise ValueError(
                f"{path.name}, query '{query_id}', "
                f"does not contain ranks exactly 1..{top_k}."
            )

    return ranking


def prompt_template(
    query: str,
    passage: str,
) -> str:
    """
    Format the pair exactly as shown in NVIDIA's model card.

    The reranker receives one combined sequence rather than a
    SentenceTransformers pair object.
    """

    return (
        f"question:{query}"
        " \n \n "
        f"passage:{passage}"
    )


def select_device_and_dtype() -> tuple[str, torch.dtype]:
    """Select a practical inference device and numerical dtype."""

    if torch.cuda.is_available():
        device = "cuda:0"

        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    return device, dtype


def load_tokenizer(
    model_name: str,
):
    """Load the tokenizer using NVIDIA's required settings."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    tokenizer,
):
    """Load the custom Nemotron sequence-classification model."""

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        .eval()
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = (
            tokenizer.eos_token_id
        )

    # KV caching is not needed for sequence-classification scoring
    # and disabling it reduces unnecessary memory use.
    model.config.use_cache = False

    model = model.to(device)

    return model

def count_prompt_tokens(
    tokenizer,
    prompts: list[str],
) -> list[int]:
    """
    Count full token lengths without truncation.

    This is performed before inference so that no CV is silently cut.
    """

    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_length=True,
    )

    lengths = encoded.get("length")

    if lengths is None:
        lengths = [
            len(input_ids)
            for input_ids in encoded["input_ids"]
        ]

    return [
        int(length)
        for length in lengths
    ]


def create_token_audit(
    ranking: pd.DataFrame,
    tokenizer,
    retriever_name: str,
) -> pd.DataFrame:
    """Create one token-length audit row per query-CV pair."""

    audit_rows: list[dict] = []

    for row in tqdm(
        ranking.itertuples(index=False),
        total=len(ranking),
        desc=f"Token audit: {retriever_name}",
    ):
        query_text = str(row.query)

        document_text = (
            ""
            if pd.isna(row.candidate_text)
            else str(row.candidate_text)
        )

        prompt = prompt_template(
            query_text,
            document_text,
        )

        token_length = count_prompt_tokens(
            tokenizer,
            [prompt],
        )[0]

        audit_rows.append({
            "retriever": retriever_name,
            "query_id": row.query_id,
            "candidate_id": row.candidate_id,
            "dataset_index": int(row.dataset_index),
            "original_rank": int(row.rank),
            "prompt_tokens": token_length,
            "max_length": MAX_LENGTH,
            "exceeds_max_length": (
                token_length > MAX_LENGTH
            ),
        })

    return pd.DataFrame(audit_rows)


def validate_full_cv_lengths(
    audit: pd.DataFrame,
    retriever_name: str,
    experiment_dir: Path,
) -> None:
    """
    Save and validate token lengths.

    When FAIL_ON_TRUNCATION=True, any over-limit pair stops the run.
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
        audit["exceeds_max_length"]
    ]

    print(
        f"{retriever_name} token audit: "
        f"max={int(audit['prompt_tokens'].max())}, "
        f"mean={audit['prompt_tokens'].mean():.2f}, "
        f"over_limit={len(over_limit)}"
    )

    if (
        FAIL_ON_TRUNCATION
        and not over_limit.empty
    ):
        examples = over_limit[
            [
                "query_id",
                "candidate_id",
                "prompt_tokens",
            ]
        ].head(10)

        raise ValueError(
            f"{retriever_name} contains {len(over_limit)} "
            f"query-CV pairs longer than {MAX_LENGTH} tokens.\n"
            "The run was stopped because FAIL_ON_TRUNCATION=True.\n"
            "This guarantees that no CV is silently truncated.\n"
            "Examples:\n"
            f"{examples.to_string(index=False)}\n"
            f"Full audit saved to:\n{audit_path}"
        )


def score_prompts(
    prompts: list[str],
    tokenizer,
    model,
    device: str,
) -> np.ndarray:
    """
    Score prompt-formatted query-CV pairs.

    The returned values are raw logits. Applying sigmoid is optional
    and unnecessary for ranking because sigmoid preserves order.
    """

    all_scores: list[float] = []

    for start in range(
        0,
        len(prompts),
        BATCH_SIZE,
    ):
        batch_prompts = prompts[
            start:start + BATCH_SIZE
        ]

        
        batch = tokenizer(
            batch_prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        with torch.inference_mode():
            logits = model(
                **batch
            ).logits

        batch_scores = (
            logits
            .view(-1)
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        all_scores.extend(
            batch_scores
        )

        del batch
        del logits

    return np.asarray(
        all_scores,
        dtype=float,
    )


def rerank_one_system(
    ranking: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    retriever_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Rerank all Top-50 candidate lists for one hybrid retriever."""

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

        prompts = [
            prompt_template(
                query_text,
                document,
            )
            for document in documents
        ]

        scores = score_prompts(
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        if len(scores) != len(group):
            raise RuntimeError(
                f"Score mismatch for query {query_id}: "
                f"{len(scores)} scores for {len(group)} rows."
            )

        group["retriever"] = retriever_name
        group["reranker"] = RERANKER_NAME
        group["original_rank"] = group["rank"]

        if "score" in group.columns:
            group["original_score"] = group["score"]
        elif "rrf_score" in group.columns:
            group["original_score"] = (
                group["rrf_score"]
            )
        else:
            group["original_score"] = np.nan

        group["reranker_score"] = (
            scores.astype(float)
        )

        # Optional probability-like representation.
        # The raw logit remains the authoritative ranking score.
        clipped_scores = np.clip(
            scores,
            -50.0,
            50.0,
        )

        group["reranker_sigmoid"] = (
            1.0
            / (
                1.0
                + np.exp(-clipped_scores)
            )
        )

        group = group.sort_values(
            by=[
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
        by=[
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


def create_evaluation_ranking(
    reranked: pd.DataFrame,
    system_name: str,
) -> pd.DataFrame:
    """Create the minimal ranking file used by evaluation metrics."""

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
    """Format numeric values consistently."""

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
    """Save one readable Markdown report for the whole experiment."""

    report_path = (
        experiment_dir
        / f"{experiment_name}_results.md"
    )

    lines: list[str] = [
        (
            "# Llama Nemotron Reranking Results: "
            f"{experiment_name}"
        ),
        "",
        f"Model: {MODEL_NAME}",
        f"Top K: {TOP_K}",
        f"Initial retriever: {retriever_name}",
        f"Maximum input length: {MAX_LENGTH}",
        f"Batch size: {BATCH_SIZE}",
        (
            "Input template: "
            "question:{query} / passage:{candidate_text}"
        ),
        (
            "Truncation policy: "
            "disabled after full token-length validation"
        ),
        "",
    ]

    ordered = reranked.sort_values(
        by=[
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
                "LLAMA NEMOTRON RERANKED RESULTS"
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
                    "- Nemotron Raw Logit: "
                    f"{format_score(row.reranker_score)}"
                ),
                (
                    "- Nemotron Sigmoid: "
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
                    "#### CV",
                    markdown_value(
                        row_data["CV"]
                    ),
                    "",
                ])
            else:
                lines.extend([
                    "#### Candidate text",
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


def save_outputs(
    reranked: pd.DataFrame,
    retriever_name: str,
    logger: logging.Logger,
) -> None:
    """Save detailed, evaluation and Markdown outputs."""

    experiment_name = (
        f"{retriever_name}__{RERANKER_NAME}"
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
        reranked,
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
    """Run Nemotron reranking on BM25+Nomic and BM25+Jina."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device, dtype = (
        select_device_and_dtype()
    )

    print("=" * 78)
    print(
        "LLAMA NEMOTRON RERANK 1B V2 "
        "— TWO HYBRID TOP-50 RUNS"
    )
    print("=" * 78)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Top K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum input length: {MAX_LENGTH}")
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
    ) in HYBRID_FILES.items():
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

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer(
        MODEL_NAME
    )

    # Token-length audits happen before the 1B model is loaded.
    for (
        retriever_name,
        ranking,
    ) in hybrid_rankings.items():
        experiment_dir = (
            OUTPUT_DIR
            / RERANKER_NAME
        )

        audit = create_token_audit(
            ranking=ranking,
            tokenizer=tokenizer,
            retriever_name=retriever_name,
        )

        validate_full_cv_lengths(
            audit=audit,
            retriever_name=retriever_name,
            experiment_dir=experiment_dir,
        )

    print("\nLoading Nemotron reranker...")

    model = load_model(
        model_name=MODEL_NAME,
        device=device,
        dtype=dtype,
        tokenizer=tokenizer,
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
            "Device: %s | Dtype: %s",
            device,
            dtype,
        )

        logger.info(
            (
                "Top K: %d | Batch size: %d | "
                "Maximum input length: %d"
            ),
            TOP_K,
            BATCH_SIZE,
            MAX_LENGTH,
        )

        start_time = time.perf_counter()

        try:
            reranked = rerank_one_system(
                ranking=ranking,
                tokenizer=tokenizer,
                model=model,
                device=device,
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
                    "CUDA out of memory with BATCH_SIZE=%d "
                    "and MAX_LENGTH=%d. BATCH_SIZE is already "
                    "configured conservatively. To preserve the "
                    "full CV, use CPU inference or a GPU with more "
                    "memory instead of silently reducing MAX_LENGTH."
                ),
                BATCH_SIZE,
                MAX_LENGTH,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise error

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(
        "ALL LLAMA NEMOTRON RERANKING RUNS COMPLETED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()