from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm.auto import tqdm

try:
    from sentence_transformers import SparseEncoder
except ImportError as exc:
    raise ImportError(
        "SparseEncoder is unavailable. Install/update dependencies with:\n"
        "python -m pip install -U sentence-transformers scipy pandas pyarrow tqdm"
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "naver/splade-v3"
SYSTEM_NAME = "SPLADE-v3"
TOP_K = 50

# Official SPLADE-v3 maximum input length.
MODEL_MAX_LENGTH = 512

# 510 content tokens + BERT special tokens = at most 512 tokens.
DOC_CHUNK_TOKENS = 510
DOC_CHUNK_OVERLAP = 64

# Number of original CVs prepared in one logical block.
DOC_BLOCK_SIZE = 96

# Automatic CUDA memory fallback.
DOC_BATCH_SIZES = (8, 4, 2, 1)
QUERY_BATCH_SIZE = 8

# Save resumable indexing progress every N original CVs.
CHECKPOINT_EVERY_DOCS = 960

FORCE_REBUILD_INDEX = False
SAVE_MARKDOWN = True
SAVE_QUERY_EXPANSIONS = True
QUERY_EXPANSION_TOP_K = 30


# =============================================================================
# Paths
# =============================================================================

def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates = [*script_path.parents, cwd, *cwd.parents]

    checked: set[Path] = set()
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)

        if not (candidate / "data").is_dir():
            continue

        if any(
            (candidate / folder).is_dir()
            for folder in (
                "sparse_retrieval",
                "dense_retrieval",
                "hybrid_retrieval",
                "evaluation",
            )
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find the project root containing data/ and retrieval folders."
    )


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "sparse_retrieval" / "outputs" / "splade_v3"
CACHE_DIR = PROJECT_ROOT / "sparse_retrieval" / "cache" / "splade_v3"
LOG_DIR = PROJECT_ROOT / "sparse_retrieval" / "logs"

CORPUS_MATRIX_PATH = CACHE_DIR / "splade_v3_chunked_maxpool_corpus.npz"
CORPUS_METADATA_PATH = CACHE_DIR / "splade_v3_corpus_metadata.csv"
TOKEN_AUDIT_PATH = CACHE_DIR / "splade_v3_token_audit.csv"
PARTIAL_MATRIX_PATH = CACHE_DIR / "splade_v3_partial_corpus.npz"
PARTIAL_AUDIT_PATH = CACHE_DIR / "splade_v3_partial_token_audit.csv"
PROGRESS_PATH = CACHE_DIR / "splade_v3_progress.json"

RANKINGS_PATH = OUTPUT_DIR / "splade_v3_top50_rankings.csv"
DETAILED_PATH = OUTPUT_DIR / "splade_v3_top50_detailed.csv"
EXPANSIONS_PATH = OUTPUT_DIR / "splade_v3_query_expansions.csv"
MARKDOWN_PATH = OUTPUT_DIR / "splade_v3_top50_results.md"
LOG_PATH = LOG_DIR / "splade_v3_chunked_retrieval.log"


# =============================================================================
# Logging and helpers
# =============================================================================

def create_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("splade_v3")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


LOGGER = create_logger()


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Input discovery
# =============================================================================

def discover_dataset_path() -> Path:
    preferred = [
        PROJECT_ROOT / "data" / "candidate_profiles_english_clean_20k_with_text.parquet",
        PROJECT_ROOT / "data" / "candidate_profiles_english_clean_with_text.parquet",
    ]
    for path in preferred:
        if path.exists():
            return path

    matches = sorted(PROJECT_ROOT.rglob("*candidate*with_text*.parquet"))
    if not matches:
        matches = sorted(PROJECT_ROOT.rglob("*.parquet"))
    if not matches:
        raise FileNotFoundError("No candidate parquet dataset was found.")

    LOGGER.warning("Using discovered parquet file: %s", matches[0])
    return matches[0]


def resolve_column(
    columns: list[str],
    alternatives: tuple[str, ...],
    required: bool = True,
) -> str | None:
    mapping = {column.lower().strip(): column for column in columns}
    for name in alternatives:
        if name.lower() in mapping:
            return mapping[name.lower()]

    if required:
        raise ValueError(
            f"Missing one of columns {alternatives}. Available columns: {columns}"
        )
    return None


def discover_query_source() -> Path:
    preferred = [
        PROJECT_ROOT / "queries" / "queries.csv",
        PROJECT_ROOT / "data" / "queries.csv",
        PROJECT_ROOT / "evaluation" / "queries.csv",
        PROJECT_ROOT / "hybrid_retrieval" / "hybrid_bm25_nomic_rrf_top50.csv",
        PROJECT_ROOT / "hybrid_retrieval" / "outputs" / "hybrid_bm25_nomic_rrf_top50.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(PROJECT_ROOT.rglob("*.csv"))
    scored: list[tuple[int, Path]] = []

    for path in candidates:
        try:
            columns = [
                column.lower().strip()
                for column in pd.read_csv(path, nrows=2).columns
            ]
        except Exception:
            continue

        has_id = bool(set(columns) & {"query_id", "qid", "id"})
        has_query = bool(
            set(columns) & {"query", "query_text", "job_query", "text"}
        )
        if not (has_id and has_query):
            continue

        name = path.name.lower()
        score = 0
        if "quer" in name:
            score += 100
        if "hybrid_bm25_nomic" in name:
            score += 80
        elif "hybrid" in name:
            score += 40
        if "top50" in name:
            score += 10
        scored.append((score, path))

    if not scored:
        raise FileNotFoundError("No CSV containing query IDs and query text was found.")

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    LOGGER.warning("Using discovered query source: %s", scored[0][1])
    return scored[0][1]


# =============================================================================
# Data loading
# =============================================================================

def load_candidates(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading candidate dataset: %s", path)
    candidates = pd.read_parquet(path).reset_index(drop=True)

    if "candidate_text" not in candidates.columns:
        raise ValueError("The parquet dataset must contain candidate_text.")

    candidates["dataset_index"] = np.arange(len(candidates), dtype=np.int64)

    if "candidate_id" in candidates.columns:
        candidates["candidate_id"] = candidates["candidate_id"].map(clean_id)
    else:
        candidates["candidate_id"] = candidates["dataset_index"].astype(str)

    candidates["candidate_text"] = (
        candidates["candidate_text"].fillna("").astype(str)
    )

    empty = candidates["candidate_text"].str.strip().eq("")
    if empty.any():
        raise ValueError(f"Found {int(empty.sum())} empty candidate_text values.")

    LOGGER.info("Loaded %d candidates.", len(candidates))
    return candidates


def load_queries(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading queries from: %s", path)
    raw = pd.read_csv(path)

    qid_col = resolve_column(raw.columns.tolist(), ("query_id", "qid", "id"))
    query_col = resolve_column(
        raw.columns.tolist(),
        ("query", "query_text", "job_query", "text"),
    )
    category_col = resolve_column(
        raw.columns.tolist(),
        ("category", "query_category", "group"),
        required=False,
    )

    selected = [qid_col, query_col]
    if category_col:
        selected.append(category_col)

    queries = raw[selected].copy()
    rename = {qid_col: "query_id", query_col: "query"}
    if category_col:
        rename[category_col] = "category"
    queries = queries.rename(columns=rename)

    if "category" not in queries.columns:
        queries["category"] = "Uncategorised"

    queries["query_id"] = queries["query_id"].map(clean_id)
    queries["query"] = queries["query"].fillna("").astype(str).str.strip()
    queries["category"] = (
        queries["category"].fillna("Uncategorised").astype(str).str.strip()
    )

    queries = queries.drop_duplicates(["query_id", "query"]).reset_index(drop=True)

    conflicts = queries.groupby("query_id")["query"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Some query IDs map to multiple texts: "
            f"{conflicts.index.tolist()}"
        )

    queries = queries.drop_duplicates("query_id").reset_index(drop=True)
    if queries["query"].eq("").any():
        raise ValueError("At least one query is empty.")

    LOGGER.info("Loaded %d unique queries.", len(queries))
    return queries


# =============================================================================
# Model and token chunking
# =============================================================================

def load_model(device: str) -> SparseEncoder:
    LOGGER.info("Loading %s on %s...", MODEL_NAME, device)
    try:
        model = SparseEncoder(MODEL_NAME, device=device)
    except Exception as exc:
        message = str(exc).lower()
        if any(term in message for term in ("gated", "401", "403", "access", "authorized")):
            raise RuntimeError(
                "SPLADE-v3 is gated. Accept the model conditions on Hugging Face "
                "and run: huggingface-cli login"
            ) from exc
        raise

    try:
        model.max_seq_length = MODEL_MAX_LENGTH
    except Exception:
        LOGGER.warning("Could not set model.max_seq_length explicitly.")

    LOGGER.info(
        "Loaded SPLADE-v3 | dimension=%s | max_length=%s",
        model.get_embedding_dimension(),
        getattr(model, "max_seq_length", "unknown"),
    )
    return model


def ids_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def split_token_ids(
    token_ids: list[int],
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")
    if len(token_ids) <= chunk_size:
        return [token_ids]

    step = chunk_size - overlap
    chunks: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunks.append(token_ids[start:end])
        if end == len(token_ids):
            break
        start += step
    return chunks


def prepare_chunks(tokenizer: Any, text: str) -> tuple[list[str], int]:
    token_ids = ids_without_special_tokens(tokenizer, text)
    id_chunks = split_token_ids(
        token_ids,
        chunk_size=DOC_CHUNK_TOKENS,
        overlap=DOC_CHUNK_OVERLAP,
    )

    text_chunks = [
        tokenizer.decode(
            chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for chunk in id_chunks
    ]
    text_chunks = [chunk if chunk.strip() else text for chunk in text_chunks]
    return text_chunks, len(token_ids)


# =============================================================================
# Sparse conversion and encoding
# =============================================================================

def torch_to_csr(tensor: torch.Tensor) -> csr_matrix:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.layout != torch.sparse_coo:
        tensor = tensor.to_sparse_coo()
    tensor = tensor.coalesce()

    indices = tensor.indices().numpy()
    values = tensor.values().float().numpy()
    shape = tuple(int(value) for value in tensor.shape)

    matrix = csr_matrix(
        (values, (indices[0], indices[1])),
        shape=shape,
        dtype=np.float32,
    )
    matrix.eliminate_zeros()
    return matrix


def sparse_row_max(matrix: csr_matrix) -> csr_matrix:
    if matrix.shape[0] < 1:
        raise ValueError("Cannot aggregate an empty chunk matrix.")

    result = matrix.getrow(0).copy()
    for row_index in range(1, matrix.shape[0]):
        result = result.maximum(matrix.getrow(row_index))
    result = result.tocsr()
    result.eliminate_zeros()
    return result


def encode_documents_once(
    model: SparseEncoder,
    texts: list[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    method = getattr(model, "encode_document", model.encode)
    return method(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        convert_to_sparse_tensor=True,
        save_to_cpu=True,
        device=device,
    )


def encode_documents(
    model: SparseEncoder,
    texts: list[str],
    device: str,
) -> tuple[torch.Tensor, int]:
    last_error: Exception | None = None
    for batch_size in DOC_BATCH_SIZES:
        try:
            return (
                encode_documents_once(model, texts, batch_size, device),
                batch_size,
            )
        except torch.OutOfMemoryError as exc:
            last_error = exc
            LOGGER.warning(
                "CUDA OOM with batch_size=%d; retrying smaller batch.",
                batch_size,
            )
            clear_memory()

    raise RuntimeError("Document encoding failed even with batch_size=1.") from last_error


def encode_queries(
    model: SparseEncoder,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    method = getattr(model, "encode_query", model.encode)
    try:
        return method(
            texts,
            batch_size=QUERY_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            device=device,
        )
    except torch.OutOfMemoryError:
        LOGGER.warning("Query encoding OOM; retrying with batch_size=1.")
        clear_memory()
        return method(
            texts,
            batch_size=1,
            show_progress_bar=True,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            device=device,
        )


# =============================================================================
# Cache and resumable indexing
# =============================================================================

def cache_config(dataset_rows: int) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "dataset_rows": int(dataset_rows),
        "model_max_length": MODEL_MAX_LENGTH,
        "doc_chunk_tokens": DOC_CHUNK_TOKENS,
        "doc_chunk_overlap": DOC_CHUNK_OVERLAP,
        "aggregation": "elementwise_sparse_max",
    }


def complete_cache_is_valid(candidates: pd.DataFrame) -> bool:
    paths = [
        CORPUS_MATRIX_PATH,
        CORPUS_METADATA_PATH,
        TOKEN_AUDIT_PATH,
        PROGRESS_PATH,
    ]
    if not all(path.exists() for path in paths):
        return False

    try:
        progress = load_json(PROGRESS_PATH)
        for key, value in cache_config(len(candidates)).items():
            if progress.get(key) != value:
                return False
        if not progress.get("complete", False):
            return False

        matrix = load_npz(CORPUS_MATRIX_PATH)
        metadata = pd.read_csv(CORPUS_METADATA_PATH)
        valid = matrix.shape[0] == len(candidates) == len(metadata)
        del matrix, metadata
        return valid
    except Exception:
        return False


def load_partial(
    candidates: pd.DataFrame,
) -> tuple[list[csr_matrix], list[dict[str, Any]], int]:
    paths = [PARTIAL_MATRIX_PATH, PARTIAL_AUDIT_PATH, PROGRESS_PATH]
    if not all(path.exists() for path in paths):
        return [], [], 0

    try:
        progress = load_json(PROGRESS_PATH)
        for key, value in cache_config(len(candidates)).items():
            if progress.get(key) != value:
                return [], [], 0
        if progress.get("complete", False):
            return [], [], 0

        processed = int(progress.get("processed_documents", 0))
        matrix = load_npz(PARTIAL_MATRIX_PATH).tocsr()
        audit = pd.read_csv(PARTIAL_AUDIT_PATH).to_dict("records")

        if matrix.shape[0] != processed or len(audit) != processed:
            LOGGER.warning("Partial cache is inconsistent; starting from zero.")
            return [], [], 0

        LOGGER.info("Resuming from %d/%d indexed CVs.", processed, len(candidates))
        return [matrix], audit, processed
    except Exception as exc:
        LOGGER.warning("Could not restore partial cache: %s", exc)
        return [], [], 0


def save_partial(
    parts: list[csr_matrix],
    audit: list[dict[str, Any]],
    processed: int,
    total: int,
) -> list[csr_matrix]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = sparse.vstack(parts, format="csr", dtype=np.float32)
    save_npz(PARTIAL_MATRIX_PATH, matrix, compressed=True)
    pd.DataFrame(audit).to_csv(
        PARTIAL_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    progress = cache_config(total)
    progress.update(
        {
            "complete": False,
            "processed_documents": int(processed),
            "matrix_rows": int(matrix.shape[0]),
            "matrix_columns": int(matrix.shape[1]),
            "matrix_nnz": int(matrix.nnz),
        }
    )
    save_json(PROGRESS_PATH, progress)
    LOGGER.info("Saved checkpoint: %d/%d CVs.", processed, total)
    return [matrix]


def candidate_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = ["candidate_id", "dataset_index"]
    optional = [
        "Position",
        "Primary Keyword",
        "Moreinfo",
        "Experience Years",
        "English Level",
        "Looking For",
        "Highlights",
    ]
    columns.extend(column for column in optional if column in candidates.columns)
    return candidates[columns].copy()


def build_index(
    model: SparseEncoder,
    candidates: pd.DataFrame,
    device: str,
) -> tuple[csr_matrix, pd.DataFrame]:
    parts, audit, start_index = load_partial(candidates)
    total = len(candidates)
    docs_since_checkpoint = 0

    progress_bar = tqdm(
        total=total,
        initial=start_index,
        desc="SPLADE indexing",
        unit="CV",
    )

    for block_start in range(start_index, total, DOC_BLOCK_SIZE):
        block_end = min(block_start + DOC_BLOCK_SIZE, total)
        block = candidates.iloc[block_start:block_end]

        chunk_texts: list[str] = []
        chunk_ranges: list[tuple[int, int]] = []
        token_counts: list[int] = []
        chunk_counts: list[int] = []

        for row in block.itertuples(index=False):
            chunks, token_count = prepare_chunks(model.tokenizer, row.candidate_text)
            start = len(chunk_texts)
            chunk_texts.extend(chunks)
            end = len(chunk_texts)
            chunk_ranges.append((start, end))
            token_counts.append(token_count)
            chunk_counts.append(len(chunks))

        chunk_embeddings, batch_size_used = encode_documents(
            model,
            chunk_texts,
            device,
        )
        chunk_matrix = torch_to_csr(chunk_embeddings)

        block_rows: list[csr_matrix] = []
        for local_index, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            document_vector = sparse_row_max(chunk_matrix[chunk_start:chunk_end])
            block_rows.append(document_vector)

            candidate = block.iloc[local_index]
            audit.append(
                {
                    "candidate_id": clean_id(candidate["candidate_id"]),
                    "dataset_index": int(candidate["dataset_index"]),
                    "original_tokens": int(token_counts[local_index]),
                    "chunk_count": int(chunk_counts[local_index]),
                    "chunk_size_tokens": DOC_CHUNK_TOKENS,
                    "chunk_overlap_tokens": DOC_CHUNK_OVERLAP,
                    "model_max_length": MODEL_MAX_LENGTH,
                    "full_document_covered": True,
                    "aggregation": "elementwise_sparse_max",
                    "active_dimensions": int(document_vector.nnz),
                    "encoding_batch_size": int(batch_size_used),
                }
            )

        parts.append(sparse.vstack(block_rows, format="csr", dtype=np.float32))
        processed = block_end
        processed_now = block_end - block_start
        docs_since_checkpoint += processed_now
        progress_bar.update(processed_now)

        del chunk_embeddings, chunk_matrix, block_rows, chunk_texts
        clear_memory()

        if docs_since_checkpoint >= CHECKPOINT_EVERY_DOCS or processed == total:
            parts = save_partial(parts, audit, processed, total)
            docs_since_checkpoint = 0

    progress_bar.close()
    corpus_matrix = sparse.vstack(parts, format="csr", dtype=np.float32)

    if corpus_matrix.shape[0] != total:
        raise RuntimeError(
            f"Index rows do not match dataset: {corpus_matrix.shape[0]} != {total}"
        )

    metadata = candidate_metadata(candidates)
    audit_frame = pd.DataFrame(audit)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(CORPUS_MATRIX_PATH, corpus_matrix, compressed=True)
    metadata.to_csv(CORPUS_METADATA_PATH, index=False, encoding="utf-8-sig")
    audit_frame.to_csv(TOKEN_AUDIT_PATH, index=False, encoding="utf-8-sig")

    progress = cache_config(total)
    progress.update(
        {
            "complete": True,
            "processed_documents": total,
            "matrix_rows": int(corpus_matrix.shape[0]),
            "matrix_columns": int(corpus_matrix.shape[1]),
            "matrix_nnz": int(corpus_matrix.nnz),
            "average_chunks_per_document": float(audit_frame["chunk_count"].mean()),
            "maximum_chunks_per_document": int(audit_frame["chunk_count"].max()),
            "average_active_dimensions": float(
                audit_frame["active_dimensions"].mean()
            ),
        }
    )
    save_json(PROGRESS_PATH, progress)

    for partial_path in (PARTIAL_MATRIX_PATH, PARTIAL_AUDIT_PATH):
        if partial_path.exists():
            partial_path.unlink()

    LOGGER.info(
        "Completed index | shape=%s | nnz=%d | avg chunks/CV=%.2f",
        corpus_matrix.shape,
        corpus_matrix.nnz,
        audit_frame["chunk_count"].mean(),
    )
    return corpus_matrix, metadata


def load_or_build_index(
    model: SparseEncoder,
    candidates: pd.DataFrame,
    device: str,
) -> tuple[csr_matrix, pd.DataFrame]:
    if FORCE_REBUILD_INDEX:
        for path in (
            CORPUS_MATRIX_PATH,
            CORPUS_METADATA_PATH,
            TOKEN_AUDIT_PATH,
            PARTIAL_MATRIX_PATH,
            PARTIAL_AUDIT_PATH,
            PROGRESS_PATH,
        ):
            if path.exists():
                path.unlink()

    if not FORCE_REBUILD_INDEX and complete_cache_is_valid(candidates):
        LOGGER.info("Loading cached SPLADE corpus index.")
        matrix = load_npz(CORPUS_MATRIX_PATH).tocsr()
        metadata = pd.read_csv(
            CORPUS_METADATA_PATH,
            dtype={"candidate_id": str},
        )
        metadata["candidate_id"] = metadata["candidate_id"].map(clean_id)
        return matrix, metadata

    return build_index(model, candidates, device)


# =============================================================================
# Query expansions and retrieval
# =============================================================================

def save_query_expansions(
    model: SparseEncoder,
    queries: pd.DataFrame,
    embeddings: torch.Tensor,
) -> None:
    if not SAVE_QUERY_EXPANSIONS:
        return

    decoded = model.decode(embeddings, top_k=QUERY_EXPANSION_TOP_K)
    rows: list[dict[str, Any]] = []

    for query_row, terms in zip(queries.itertuples(index=False), decoded, strict=True):
        for rank, (token, weight) in enumerate(terms, start=1):
            rows.append(
                {
                    "query_id": query_row.query_id,
                    "category": query_row.category,
                    "query": query_row.query,
                    "term_rank": rank,
                    "expanded_token": token,
                    "weight": float(weight),
                }
            )

    pd.DataFrame(rows).to_csv(
        EXPANSIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    LOGGER.info("Saved query expansions: %s", EXPANSIONS_PATH)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    indices = np.arange(len(scores), dtype=np.int64)
    # Descending score, then ascending corpus index for deterministic ties.
    order = np.lexsort((indices, -scores))
    return order[: min(k, len(order))]


def retrieve(
    corpus_matrix: csr_matrix,
    metadata: pd.DataFrame,
    queries: pd.DataFrame,
    query_matrix: csr_matrix,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for query_position, query_row in tqdm(
        queries.iterrows(),
        total=len(queries),
        desc="SPLADE retrieval",
        unit="query",
    ):
        query_vector = query_matrix.getrow(query_position)
        scores = np.asarray(
            corpus_matrix.dot(query_vector.transpose()).toarray()
        ).reshape(-1)

        selected = top_k_indices(scores, TOP_K)

        for rank, corpus_index in enumerate(selected, start=1):
            candidate = metadata.iloc[int(corpus_index)]
            rows.append(
                {
                    "system": SYSTEM_NAME,
                    "query_id": clean_id(query_row["query_id"]),
                    "category": clean_text(query_row["category"]),
                    "query": clean_text(query_row["query"]),
                    "candidate_id": clean_id(candidate["candidate_id"]),
                    "dataset_index": int(candidate["dataset_index"]),
                    "rank": rank,
                    "score": float(scores[corpus_index]),
                }
            )

    rankings = pd.DataFrame(rows)
    expected = len(queries) * min(TOP_K, corpus_matrix.shape[0])
    if len(rankings) != expected:
        raise RuntimeError(f"Unexpected result count: {len(rankings)} != {expected}")
    return rankings


# =============================================================================
# Outputs
# =============================================================================

def create_detailed(
    rankings: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["candidate_id", "dataset_index", "candidate_text"]
    optional = [
        "Position",
        "Primary Keyword",
        "Moreinfo",
        "Experience Years",
        "English Level",
        "Looking For",
        "Highlights",
        "CV",
    ]
    columns.extend(column for column in optional if column in candidates.columns)

    return rankings.merge(
        candidates[columns],
        on=["candidate_id", "dataset_index"],
        how="left",
        validate="many_to_one",
    )


def markdown_safe(value: Any) -> str:
    return clean_text(value).replace("|", r"\|")


def save_markdown(detailed: pd.DataFrame) -> None:
    if not SAVE_MARKDOWN:
        return

    lines = [
        "# SPLADE-v3 Chunked Sparse Retrieval",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Top K: {TOP_K}",
        f"- Model context: {MODEL_MAX_LENGTH} tokens",
        f"- CV chunk size: {DOC_CHUNK_TOKENS} tokens",
        f"- Chunk overlap: {DOC_CHUNK_OVERLAP} tokens",
        "- Chunk aggregation: element-wise max pooling",
        "- Similarity: sparse dot product",
        "",
    ]

    grouped = detailed.groupby(
        ["query_id", "category", "query"],
        sort=False,
    )

    for (query_id, category, query), group in grouped:
        lines.extend(
            [
                "---",
                "",
                f"## Query {markdown_safe(query_id)}",
                "",
                f"**Category:** {markdown_safe(category)}",
                "",
                f"**Query:** {markdown_safe(query)}",
                "",
                "| Rank | Candidate ID | Score | Position |",
                "|---:|---|---:|---|",
            ]
        )

        for row in group.sort_values("rank").itertuples(index=False):
            row_dict = row._asdict()
            position = markdown_safe(row_dict.get("Position", ""))
            lines.append(
                f"| {int(row.rank)} | {markdown_safe(row.candidate_id)} | "
                f"{float(row.score):.6f} | {position} |"
            )
        lines.append("")

    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved Markdown report: %s", MARKDOWN_PATH)


def save_outputs(rankings: pd.DataFrame, detailed: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(RANKINGS_PATH, index=False, encoding="utf-8-sig")
    detailed.to_csv(DETAILED_PATH, index=False, encoding="utf-8-sig")
    save_markdown(detailed)
    LOGGER.info("Evaluation rankings: %s", RANKINGS_PATH)
    LOGGER.info("Detailed rankings: %s", DETAILED_PATH)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = discover_dataset_path()
    query_path = discover_query_source()

    LOGGER.info("=" * 79)
    LOGGER.info("SPLADE-v3 CHUNKED LEARNED-SPARSE RETRIEVAL")
    LOGGER.info("=" * 79)
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Dataset: %s", dataset_path)
    LOGGER.info("Query source: %s", query_path)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Model: %s", MODEL_NAME)
    LOGGER.info("Top K: %d", TOP_K)
    LOGGER.info(
        "Full CV handling: %d-token chunks with %d-token overlap and max pooling.",
        DOC_CHUNK_TOKENS,
        DOC_CHUNK_OVERLAP,
    )

    start_time = time.perf_counter()

    candidates = load_candidates(dataset_path)
    queries = load_queries(query_path)
    model = load_model(device)

    corpus_matrix, metadata = load_or_build_index(
        model,
        candidates,
        device,
    )

    query_embeddings = encode_queries(
        model,
        queries["query"].tolist(),
        device,
    )
    query_matrix = torch_to_csr(query_embeddings)

    if query_matrix.shape[1] != corpus_matrix.shape[1]:
        raise RuntimeError(
            "Query/corpus dimensions differ: "
            f"{query_matrix.shape[1]} != {corpus_matrix.shape[1]}"
        )

    save_query_expansions(model, queries, query_embeddings)
    rankings = retrieve(corpus_matrix, metadata, queries, query_matrix)
    detailed = create_detailed(rankings, candidates)
    save_outputs(rankings, detailed)

    elapsed = (time.perf_counter() - start_time) / 60.0
    LOGGER.info("=" * 79)
    LOGGER.info(
        "Completed | queries=%d | candidates=%d | results=%d | %.2f minutes",
        len(queries),
        len(candidates),
        len(rankings),
        elapsed,
    )
    LOGGER.info("=" * 79)

    del model, query_embeddings, query_matrix, corpus_matrix
    clear_memory()


if __name__ == "__main__":
    main()