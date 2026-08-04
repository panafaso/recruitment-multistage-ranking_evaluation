from __future__ import annotations

import gc
import logging
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import numpy as np
import pandas as pd
import torch
from packaging.version import Version
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

MIN_SENTENCE_TRANSFORMERS = "5.4.0"
MIN_TRANSFORMERS = "5.4.0"


def require_package_version(
    package_name: str,
    minimum_version: str,
) -> None:
    try:
        installed_version = version(package_name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Required package '{package_name}' is not installed."
        ) from exc

    if Version(installed_version) < Version(minimum_version):
        raise RuntimeError(
            f"{package_name}>={minimum_version} is required, but "
            f"version {installed_version} is installed.\n\n"
            "Install the required versions with:\n"
            'python -m pip install -U "transformers>=5.4.0" '
            '"sentence-transformers[image]>=5.4.0"'
        )


require_package_version(
    "sentence-transformers",
    MIN_SENTENCE_TRANSFORMERS,
)
require_package_version(
    "transformers",
    MIN_TRANSFORMERS,
)

MODEL_NAME = "lightonai/LightOn-rerank-PW-2B"
RERANKER_NAME = "lighton_rerank_pw_2b"
DISPLAY_NAME = "LightOn-rerank-PW-2B"

TOP_K = 50

MAX_LENGTH = 4096

BATCH_SIZE = 1

FAIL_ON_TRUNCATION = True

SAVE_TOKEN_AUDIT = True
SAVE_MARKDOWN_REPORT = True
REQUIRE_CUDA = True

CUDA_CLEAR_INTERVAL = 10

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
}


def find_project_root() -> Path:
    """Find the directory containing both data/ and hybrid_retrieval/."""

    script_path = Path(__file__).resolve()
    current_path = Path.cwd().resolve()

    candidates = [
        *script_path.parents,
        current_path,
        *current_path.parents,
    ]

    checked: set[Path] = set()

    for candidate in candidates:
        if candidate in checked:
            continue

        checked.add(candidate)

        if (
            (candidate / "data").is_dir()
            and (candidate / "hybrid_retrieval").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root containing both "
        "'data' and 'hybrid_retrieval'."
    )


PROJECT_ROOT = find_project_root()

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reranking"
    / "outputs"
    / RERANKER_NAME
)

LOG_DIR = (
    PROJECT_ROOT
    / "reranking"
    / "logs"
)


def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return (
        f"allocated={allocated:.2f} GiB | "
        f"reserved={reserved:.2f} GiB | "
        f"total={total:.2f} GiB"
    )

def validate_file_exists(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_candidate_dataset(path: Path) -> pd.DataFrame:
    """Load complete candidate_text values without character truncation."""

    validate_file_exists(
        path,
        "Candidate dataset",
    )

    candidates = pd.read_parquet(
        path
    ).reset_index(drop=True)

    if "candidate_text" not in candidates.columns:
        raise ValueError(
            "The candidate dataset does not contain 'candidate_text'."
        )

    candidates["candidate_id"] = (
        candidates.index.astype(str)
    )
    candidates["dataset_index"] = (
        candidates.index.astype(int)
    )

    descriptive_columns = [
        "Position",
        "Primary Keyword",
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

    empty_count = int(
        result["candidate_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_count:
        raise ValueError(
            f"The dataset contains {empty_count} empty candidate_text values."
        )

    return result


def load_hybrid_ranking(
    path: Path,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Load one hybrid Top-50 and attach each candidate's full text."""

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

    ranking = (
        ranking
        .sort_values(
            ["query_id", "rank"],
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
            ["query_id", "candidate_id"]
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
            "Some query IDs have multiple query texts: "
            f"{conflicting_queries.index.tolist()}"
        )

    ranking = ranking.merge(
        candidates,
        on=["candidate_id", "dataset_index"],
        how="left",
        validate="many_to_one",
    )

    missing_texts = int(
        ranking["candidate_text"].isna().sum()
    )

    if missing_texts:
        raise ValueError(
            f"{missing_texts} ranking rows could not be matched to "
            "candidate_text in the parquet dataset."
        )

    counts = ranking.groupby("query_id").size()
    incomplete = counts[counts != top_k]

    if not incomplete.empty:
        raise ValueError(
            f"{path.name} does not contain exactly {top_k} candidates "
            "for every query:\n"
            f"{incomplete.to_string()}"
        )

    expected_ranks = list(range(1, top_k + 1))

    for query_id, group in ranking.groupby(
        "query_id",
        sort=False,
    ):
        observed_ranks = sorted(
            group["rank"].tolist()
        )

        if observed_ranks != expected_ranks:
            raise ValueError(
                f"{path.name}, query '{query_id}', does not contain "
                f"ranks exactly 1..{top_k}."
            )

    return ranking


# =============================================================================
# LightOn model loading
# =============================================================================


def select_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device = "cuda"

        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        return device, compute_dtype

    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA was not detected. This 2.2B reranker is intended to run "
            "on the CUDA environment used by the other reranking scripts."
        )

    return "cpu", torch.float32


def load_lighton_cross_encoder(
    device: str,
    compute_dtype: torch.dtype,
) -> tuple[CrossEncoder, str]:
    """
    Load the official Sentence Transformers CrossEncoder integration.

    The bundled 'reranker' chat template automatically applies the fixed
    LightOn system/user prompts and returns logit(Yes) - logit(No).
    """

    attempts: list[tuple[str, dict[str, Any]]] = []

    if device == "cuda":
        attempts.extend([
            (
                "sdpa",
                {
                    "torch_dtype": compute_dtype,
                    "attn_implementation": "sdpa",
                },
            ),
            (
                "default",
                {
                    "torch_dtype": compute_dtype,
                },
            ),
        ])
    else:
        attempts.append(
            (
                "default",
                {
                    "torch_dtype": compute_dtype,
                },
            )
        )

    errors: list[str] = []

    for attention_name, model_kwargs in attempts:
        try:
            model = CrossEncoder(
                MODEL_NAME,
                max_length=MAX_LENGTH,
                device=device,
                model_kwargs=model_kwargs,
            )

            model.eval()

            return model, attention_name

        except Exception as error:
            errors.append(
                f"{attention_name}: {type(error).__name__}: {error}"
            )
            clear_memory()

    raise RuntimeError(
        "Could not load the LightOn reranker.\n"
        + "\n".join(errors)
    )


def count_formatted_pair_tokens(
    model: CrossEncoder,
    query: str,
    document: str,
) -> int:
    """
    Count the tokens after the model's bundled reranker chat template.

    This is more accurate than tokenizer(query, document), because the model
    internally adds its trained system prompt, user template and generation
    prompt before scoring Yes versus No.
    """

    features = model.tokenize(
        [(query, document)],
        processing_kwargs={
            "text": {
                "truncation": False,
                "padding": False,
            }
        },
    )

    input_ids = features.get("input_ids")

    if input_ids is None:
        raise RuntimeError(
            "The model tokenizer did not return input_ids during the audit."
        )

    attention_mask = features.get("attention_mask")

    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask[0].sum().item())

    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])

    if isinstance(input_ids, np.ndarray):
        return int(input_ids.shape[-1])

    if isinstance(input_ids, list):
        first = input_ids[0]

        if isinstance(first, list):
            return len(first)

        return len(input_ids)

    raise TypeError(
        f"Unsupported input_ids type: {type(input_ids)}"
    )


def create_token_audit(
    ranking: pd.DataFrame,
    model: CrossEncoder,
    retriever_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in tqdm(
        ranking.itertuples(index=False),
        total=len(ranking),
        desc=f"Token audit: {retriever_name}",
    ):
        query_text = str(row.query)
        candidate_text = str(row.candidate_text)

        pair_tokens = count_formatted_pair_tokens(
            model=model,
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
            "formatted_pair_tokens": pair_tokens,
            "max_length": MAX_LENGTH,
            "fits_without_truncation": pair_tokens <= MAX_LENGTH,
            "tokens_over_limit": max(0, pair_tokens - MAX_LENGTH),
        })

    return pd.DataFrame(rows)


def save_and_validate_token_audit(
    audit: pd.DataFrame,
    retriever_name: str,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        OUTPUT_DIR
        / f"{retriever_name}__token_length_audit_4096.csv"
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
        audit["formatted_pair_tokens"].max()
    )
    mean = float(
        audit["formatted_pair_tokens"].mean()
    )
    p95 = float(
        audit["formatted_pair_tokens"].quantile(0.95)
    )

    print(
        f"{retriever_name} token audit | "
        f"max={maximum} | mean={mean:.2f} | p95={p95:.2f} | "
        f"over_limit={len(over_limit)}"
    )
    print(f"Audit file: {audit_path}")

    if not over_limit.empty:
        examples = (
            over_limit[
                [
                    "query_id",
                    "candidate_id",
                    "formatted_pair_tokens",
                    "tokens_over_limit",
                ]
            ]
            .sort_values(
                "formatted_pair_tokens",
                ascending=False,
            )
            .head(15)
        )

        print(
            "\nLongest inputs above 4096 tokens:\n"
            f"{examples.to_string(index=False)}\n"
        )

    if FAIL_ON_TRUNCATION and not over_limit.empty:
        raise ValueError(
            f"{retriever_name} contains {len(over_limit)} fully formatted "
            f"query-CV pairs above MAX_LENGTH={MAX_LENGTH}.\n"
            "The experiment stopped before inference, so no CV was silently "
            "truncated. Inspect the token audit and decide whether to use a "
            "larger limit or a documented long-document strategy.\n"
            f"Audit: {audit_path}"
        )


def score_pairs_safely(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
    query_id: str,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Score one query-CV pair at a time to minimise peak VRAM use.

    The returned value is the official raw LightOn score:
        logit("Yes") - logit("No")
    """

    scores: list[float] = []

    for pair_index, pair in enumerate(pairs, start=1):
        try:
            with torch.inference_mode():
                prediction = model.predict(
                    [pair],
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    processing_kwargs={
                        "text": {
                            "max_length": MAX_LENGTH,
                            "truncation": True,
                            "padding": True,
                        }
                    },
                )

        except torch.OutOfMemoryError as exc:
            clear_memory()

            raise torch.OutOfMemoryError(
                f"CUDA OOM for query '{query_id}', candidate pair "
                f"{pair_index}/{len(pairs)}, MAX_LENGTH={MAX_LENGTH}, "
                f"BATCH_SIZE={BATCH_SIZE}. Current memory: "
                f"{cuda_memory_snapshot()}"
            ) from exc

        prediction_array = np.asarray(
            prediction,
            dtype=np.float32,
        ).reshape(-1)

        if len(prediction_array) != 1:
            raise RuntimeError(
                "LightOn returned an unexpected number of scores for one pair: "
                f"{len(prediction_array)}"
            )

        scores.append(
            float(prediction_array[0])
        )

        if (
            torch.cuda.is_available()
            and pair_index % CUDA_CLEAR_INTERVAL == 0
        ):
            torch.cuda.empty_cache()

    return np.asarray(
        scores,
        dtype=np.float32,
    )


def rerank_one_system(
    ranking: pd.DataFrame,
    model: CrossEncoder,
    retriever_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    output_groups: list[pd.DataFrame] = []

    grouped = ranking.groupby(
        "query_id",
        sort=False,
    )

    progress = tqdm(
        grouped,
        total=ranking["query_id"].nunique(),
        desc=f"{retriever_name} -> {RERANKER_NAME}",
    )

    for query_id, group in progress:
        group = (
            group
            .sort_values(
                "rank",
                kind="stable",
            )
            .reset_index(drop=True)
            .copy()
        )

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
            (query_text, document)
            for document in documents
        ]

        scores = score_pairs_safely(
            model=model,
            pairs=pairs,
            query_id=str(query_id),
            logger=logger,
        )

        if len(scores) != len(group):
            raise RuntimeError(
                f"Score mismatch for query {query_id}: "
                f"{len(scores)} scores for {len(group)} candidates."
            )

        group["retriever"] = retriever_name
        group["reranker"] = RERANKER_NAME
        group["original_rank"] = group["rank"]

        if "score" in group.columns:
            group["original_score"] = group["score"]
        elif "rrf_score" in group.columns:
            group["original_score"] = group["rrf_score"]
        else:
            group["original_score"] = np.nan

        group["reranker_score"] = scores.astype(float)

        clipped_scores = np.clip(
            scores.astype(float),
            -50.0,
            50.0,
        )
        group["reranker_sigmoid"] = (
            1.0
            / (1.0 + np.exp(-clipped_scores))
        )

        group["batch_size_used"] = BATCH_SIZE
        group["maximum_input_tokens"] = MAX_LENGTH

        group = group.sort_values(
            ["reranker_score", "original_rank"],
            ascending=[False, True],
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

        output_groups.append(group)

        del pairs
        del scores
        clear_memory()

    if not output_groups:
        raise RuntimeError(
            "No query groups were available for reranking."
        )

    reranked = pd.concat(
        output_groups,
        ignore_index=True,
    )

    reranked = reranked.sort_values(
        ["query_id", "reranked_rank"],
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
    evaluation = reranked.copy()

    evaluation["system"] = system_name
    evaluation["rank"] = evaluation["reranked_rank"]
    evaluation["score"] = evaluation["reranker_score"]

    return evaluation[
        [
            "system",
            "query_id",
            "category",
            "query",
            "candidate_id",
            "dataset_index",
            "rank",
            "score",
        ]
    ].copy()


def markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def format_score(value: object) -> str:
    if pd.isna(value):
        return ""

    return f"{float(value):.6f}"


def save_markdown_report(
    reranked: pd.DataFrame,
    experiment_name: str,
    retriever_name: str,
    attention_name: str,
    logger: logging.Logger,
) -> None:
    if not SAVE_MARKDOWN_REPORT:
        return

    report_path = (
        OUTPUT_DIR
        / f"{experiment_name}_results.md"
    )

    lines: list[str] = [
        f"# {DISPLAY_NAME} Results: {experiment_name}",
        "",
        f"- Model: `{MODEL_NAME}`",
        "- Type: pointwise multimodal cross-encoder reranker",
        "- Text scoring: logit(Yes) - logit(No)",
        f"- Initial retriever: {retriever_name}",
        f"- Candidate pool reranked: Top-{TOP_K}",
        f"- Maximum formatted input length: {MAX_LENGTH} tokens",
        f"- Batch size: {BATCH_SIZE}",
        f"- Attention implementation: {attention_name}",
        "- Input: job query + complete candidate_text",
        "- Truncation policy: disabled methodologically; token audit must pass",
        "",
    ]

    ordered = reranked.sort_values(
        ["category", "query_id", "reranked_rank"],
        kind="stable",
    )

    for (category, query_id), group in ordered.groupby(
        ["category", "query_id"],
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
            f"## Category: {markdown_value(category)}",
            "=" * 80,
            "",
            f"### Query ID: {query_id}",
            f"### Query: {query_text}",
            "",
            f"TOP {len(group)} LIGHTON POINTWISE RERANKED RESULTS",
            "",
        ])

        for row in group.itertuples(index=False):
            row_data = row._asdict()

            lines.extend([
                "---",
                f"### Rank {int(row.reranked_rank)}",
                f"- LightOn Raw Score: {format_score(row.reranker_score)}",
                f"- LightOn Sigmoid: {format_score(row.reranker_sigmoid)}",
                f"- Original Hybrid Rank: {int(row.original_rank)}",
                f"- Rank Change: {int(row.rank_change):+d}",
                f"- Original Hybrid Score: {format_score(row.original_score)}",
                f"- Dataset Index: {int(row.dataset_index)}",
                f"- Candidate ID: {markdown_value(row.candidate_id)}",
            ])

            if "Position" in row_data:
                lines.append(
                    f"- Position: {markdown_value(row_data['Position'])}"
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
                    markdown_value(row_data["Looking_For"]),
                    "",
                ])

            if "Highlights" in row_data:
                lines.extend([
                    "#### Highlights",
                    markdown_value(row_data["Highlights"]),
                    "",
                ])

            if "CV" in row_data:
                lines.extend([
                    "#### Full CV",
                    markdown_value(row_data["CV"]),
                    "",
                ])
            else:
                lines.extend([
                    "#### Complete candidate text",
                    markdown_value(row_data["candidate_text"]),
                    "",
                ])

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
    attention_name: str,
    logger: logging.Logger,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_name = (
        f"{retriever_name}__{RERANKER_NAME}"
    )

    detailed_path = (
        OUTPUT_DIR
        / f"{experiment_name}_detailed.csv"
    )

    evaluation_path = (
        OUTPUT_DIR
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
        experiment_name=experiment_name,
        retriever_name=retriever_name,
        attention_name=attention_name,
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
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device, compute_dtype = select_device_and_dtype()

    print("=" * 79)
    print(
        "LIGHTON-RERANK-PW-2B — POINTWISE TOP-50 — 4096 TOKENS"
    )
    print("=" * 79)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Compute dtype: {compute_dtype}")
    print(f"Top K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum formatted input length: {MAX_LENGTH}")
    print(f"Fail on truncation: {FAIL_ON_TRUNCATION}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM: {cuda_memory_snapshot()}")

    candidates = load_candidate_dataset(
        DATASET_PATH
    )

    hybrid_rankings: dict[str, pd.DataFrame] = {}

    for retriever_name, ranking_path in HYBRID_FILES.items():
        hybrid_rankings[retriever_name] = load_hybrid_ranking(
            path=ranking_path,
            candidates=candidates,
            top_k=TOP_K,
        )

        print(
            f"Loaded {retriever_name}: "
            f"{len(hybrid_rankings[retriever_name])} rows"
        )

    print("\nLoading LightOn pointwise reranker...")

    model, attention_name = load_lighton_cross_encoder(
        device=device,
        compute_dtype=compute_dtype,
    )

    print(f"Attention implementation: {attention_name}")
    print(
        "CrossEncoder maximum sequence length: "
        f"{model.max_seq_length}"
    )
    print(f"VRAM after model load: {cuda_memory_snapshot()}")

    # Audit all fully formatted inputs before any score is calculated.
    for retriever_name, ranking in hybrid_rankings.items():
        audit = create_token_audit(
            ranking=ranking,
            model=model,
            retriever_name=retriever_name,
        )

        save_and_validate_token_audit(
            audit=audit,
            retriever_name=retriever_name,
        )

    total_experiments = len(hybrid_rankings)

    for experiment_number, (
        retriever_name,
        ranking,
    ) in enumerate(
        hybrid_rankings.items(),
        start=1,
    ):
        log_path = (
            LOG_DIR
            / f"{retriever_name}__{RERANKER_NAME}.log"
        )

        logger = create_logger(log_path)

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
            "Device: %s | dtype: %s | attention: %s",
            device,
            compute_dtype,
            attention_name,
        )
        logger.info(
            "Top K: %d | batch size: %d | max length: %d",
            TOP_K,
            BATCH_SIZE,
            MAX_LENGTH,
        )
        logger.info(
            "Memory before run: %s",
            cuda_memory_snapshot(),
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
                attention_name=attention_name,
                logger=logger,
            )

            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            logger.info(
                "Completed successfully in %.2f minutes.",
                elapsed_seconds / 60.0,
            )
            logger.info(
                "Memory after run: %s",
                cuda_memory_snapshot(),
            )

        except torch.OutOfMemoryError:
            logger.exception(
                "CUDA OOM at MAX_LENGTH=%d and BATCH_SIZE=%d. "
                "Close other GPU programs before retrying. The script does "
                "not silently lower the input length because the experiment "
                "must remain methodologically consistent.",
                MAX_LENGTH,
                BATCH_SIZE,
            )
            clear_memory()
            raise

        clear_memory()

    del model
    clear_memory()

    print("\n" + "=" * 79)
    print("ALL LIGHTON-RERANK-PW-2B RUNS COMPLETED")
    print("=" * 79)


if __name__ == "__main__":
    main()