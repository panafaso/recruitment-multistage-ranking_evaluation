from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel


def find_project_root() -> Path:
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

        if (
            (candidate / "data").is_dir()
            and (candidate / "hybrid_retrieval").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root containing "
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

MODEL_NAME = "jinaai/jina-reranker-v3.5"
RERANKER_NAME = "jina_reranker_v3_5"

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
RERANK_DEPTH = 10

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
}


def create_logger(log_path: Path) -> logging.Logger:
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
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_candidate_dataset(path: Path) -> pd.DataFrame:
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

    empty_count = int(
        result["candidate_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_count:
        raise ValueError(
            f"The dataset contains {empty_count} empty "
            "candidate_text values."
        )

    return result


def load_hybrid_ranking(
    path: Path,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
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


def select_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device = "cuda"

        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    return device, dtype


def load_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    return model


def normalize_results(
    results: list[dict],
    expected_count: int,
) -> dict[int, float]:
    if len(results) != expected_count:
        raise RuntimeError(
            f"Jina returned {len(results)} results for "
            f"{expected_count} documents."
        )

    score_by_index: dict[int, float] = {}

    for result in results:
        if "index" not in result:
            raise KeyError(
                "A Jina result does not contain 'index'."
            )

        if "relevance_score" not in result:
            raise KeyError(
                "A Jina result does not contain "
                "'relevance_score'."
            )

        index = int(
            result["index"]
        )

        score = float(
            result["relevance_score"]
        )

        if index in score_by_index:
            raise RuntimeError(
                f"Duplicate Jina result index: {index}"
            )

        score_by_index[index] = score

    expected_indexes = set(
        range(expected_count)
    )

    observed_indexes = set(
        score_by_index
    )

    if observed_indexes != expected_indexes:
        raise RuntimeError(
            "Jina result indexes do not match input documents. "
            f"Missing={sorted(expected_indexes - observed_indexes)}; "
            f"Unexpected={sorted(observed_indexes - expected_indexes)}"
        )

    return score_by_index


def clear_cuda_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def rerank_head(
    group: pd.DataFrame,
    query_text: str,
    model,
    rerank_depth: int,
) -> pd.DataFrame:
    head = (
        group
        .head(rerank_depth)
        .copy()
        .reset_index(drop=True)
    )

    documents = (
        head["candidate_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    with torch.inference_mode():
        results = model.rerank(
            query=query_text,
            documents=documents,
            top_n=None,
            return_embeddings=False,
        )

    score_by_index = normalize_results(
        results=results,
        expected_count=len(documents),
    )

    head["input_list_index"] = np.arange(
        len(head)
    )

    head["reranker_score"] = (
        head["input_list_index"]
        .map(score_by_index)
        .astype(float)
    )

    if head["reranker_score"].isna().any():
        raise RuntimeError(
            "Some Jina reranker scores are missing."
        )

    head = head.sort_values(
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

    del results
    del score_by_index

    return head


def rerank_one_query(
    group: pd.DataFrame,
    model,
    query_id: str,
    logger: logging.Logger,
) -> pd.DataFrame:
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

    group["original_rank"] = (
        group["rank"]
    )

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

    logger.info(
        "Query '%s': reranking Top-%d",
        query_id,
        RERANK_DEPTH,
    )

    reranked_head = rerank_head(
        group=group,
        query_text=query_text,
        model=model,
        rerank_depth=RERANK_DEPTH,
    )

    tail = (
        group
        .iloc[RERANK_DEPTH:]
        .copy()
        .reset_index(drop=True)
    )

    tail["input_list_index"] = pd.NA
    tail["reranker_score"] = np.nan

    reranked_head["was_reranked"] = True
    tail["was_reranked"] = False

    reranked_head["rerank_depth_used"] = RERANK_DEPTH
    tail["rerank_depth_used"] = RERANK_DEPTH

    combined = pd.concat(
        [
            reranked_head,
            tail,
        ],
        ignore_index=True,
    )

    combined["reranked_rank"] = np.arange(
        1,
        len(combined) + 1,
    )

    combined["rank_change"] = (
        combined["original_rank"]
        - combined["reranked_rank"]
    )

    return combined

def rerank_one_system(
    ranking: pd.DataFrame,
    model,
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
        desc=(
            f"{retriever_name} "
            f"-> {RERANKER_NAME}"
        ),
    )

    for query_id, group in progress:
        reranked_group = rerank_one_query(
            group=group,
            model=model,
            query_id=query_id,
            logger=logger,
        )

        reranked_group["retriever"] = retriever_name
        reranked_group["reranker"] = RERANKER_NAME

        output_groups.append(
            reranked_group
        )

        clear_cuda_memory()

    if not output_groups:
        raise RuntimeError(
            "No query groups were available for reranking."
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

    logger.info(
        "All %d queries were reranked at depth %d.",
        reranked["query_id"].nunique(),
        RERANK_DEPTH,
    )

    return reranked


def create_evaluation_ranking(
    reranked: pd.DataFrame,
    system_name: str,
) -> pd.DataFrame:
    evaluation = reranked.copy()

    evaluation["system"] = system_name
    evaluation["rank"] = (
        evaluation["reranked_rank"]
    )

    evaluation["score"] = np.where(
        evaluation["was_reranked"],
        evaluation["reranker_score"],
        -evaluation["reranked_rank"].astype(float),
    )

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
    experiment_dir: Path,
    experiment_name: str,
    retriever_name: str,
    logger: logging.Logger,
) -> None:
    report_path = (
        experiment_dir
        / f"{experiment_name}_results.md"
    )

    lines: list[str] = [
        (
            "# Jina Reranker v3.5 Results: "
            f"{experiment_name}"
        ),
        "",
        f"Model: {MODEL_NAME}",
        f"Initial retriever: {retriever_name}",
        f"Candidate pool: Top-{TOP_K}",
        f"Listwise rerank depth: {RERANK_DEPTH}",
        (
            "Candidates below the selected rerank depth preserve "
            "their original hybrid order."
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

        used_depth = int(
            group["rerank_depth_used"].iloc[0]
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
            f"### Rerank depth used: {used_depth}",
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
                    "- Was reranked: "
                    f"{bool(row.was_reranked)}"
                ),
                (
                    "- Jina Relevance Score: "
                    f"{format_score(row.reranker_score)}"
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
    experiment_name = (
        f"{retriever_name}"
        f"__{RERANKER_NAME}"
        f"__top{RERANK_DEPTH}"
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
        "JINA RERANKER V3.5 — TOP-10 ONLY"
    )
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Candidate pool: Top-{TOP_K}")
    print(f"Rerank depth: {RERANK_DEPTH}")

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

    print("\nLoading Jina reranker v3.5...")

    model = load_model(
        model_name=MODEL_NAME,
        device=device,
        dtype=dtype,
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
                f"__{RERANKER_NAME}"
                f"__top{RERANK_DEPTH}.log"
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
            "Candidate pool: %d | Rerank depth: %d",
            TOP_K,
            RERANK_DEPTH,
        )

        start_time = time.perf_counter()

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

        clear_cuda_memory()

    del model
    clear_cuda_memory()

    print("\n" + "=" * 78)
    print(
        "ALL JINA RERANKER V3.5 RUNS COMPLETED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()