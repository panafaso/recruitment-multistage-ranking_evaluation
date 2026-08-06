from __future__ import annotations

"""
Professor/server IBM Granite 4.1 8B full-precision PRP-Graph reranking over a raw Hybrid RRF Top-20 candidate pool.

Primary experiment:
    BM25Plus + dense retriever -> RRF Top-50 -> keep Top-20 -> IBM Granite 4.1 8B PRP-Graph

This script does NOT perform first-stage retrieval. It:
1. loads an already-created raw Hybrid RRF ranking CSV;
2. attaches candidate_text from the Parquet dataset via candidate_id + dataset_index;
3. performs bidirectional pairwise ranking prompting with IBM Granite 4.1 8B;
4. uses label probabilities to build a directed weighted ranking graph;
5. applies weighted PageRank-style aggregation;
6. saves detailed, evaluation-ready, pairwise-comparison, and round-trace files.

Install additional Python packages inside the LUMI ROCm/PyTorch environment:
    pip install -U "transformers>=4.55.4,<6" accelerate huggingface_hub \
        safetensors sentencepiece pandas pyarrow numpy tqdm

PyTorch must be supplied by the ROCm-compatible LUMI container/environment;
do not replace it with a generic CUDA torch wheel.

If Hugging Face requests authentication:
    hf auth login
"""

import gc
import hashlib
import json
import logging
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


# =============================================================================
# Project paths and experiment configuration
# =============================================================================


def find_project_root() -> Path:
    """Find a directory containing both data/ and hybrid_retrieval/."""

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
        "Could not locate the project root. Expected one directory "
        "containing both 'data' and 'hybrid_retrieval'."
    )


PROJECT_ROOT = find_project_root()

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

# Use a RAW Hybrid RRF ranking here, not an Ettin/BGE/Jina-reranked output.
HYBRID_FILES: dict[str, Path] = {
    "hybrid_bm25_jina_rrf": (
        PROJECT_ROOT
        / "hybrid_retrieval"
        / "hybrid_bm25_jina_rrf_top50.csv"
    ),
    # Add another raw hybrid experiment if desired:
    # "hybrid_bm25_nomic_rrf": (
    #     PROJECT_ROOT
    #     / "hybrid_retrieval"
    #     / "hybrid_bm25_nomic_rrf_top50.csv"
    # ),
    # "hybrid_bm25_bge_rrf": (
    #     PROJECT_ROOT
    #     / "hybrid_retrieval"
    #     / "hybrid_bm25_bge_rrf_top50.csv"
    # ),
}

# Official IBM Granite 4.1 8B BF16 checkpoint; no quantized derivative is used.
MODEL_NAME = "ibm-granite/granite-4.1-8b"
MODEL_REVISION: str | None = None  # Set a commit hash here to pin the checkpoint.
TRUST_REMOTE_CODE = False
MODEL_SHORT_NAME = "granite_4_1_8b"
PRECISION_MODE = "full_bf16_or_fp16"
PROMPT_VERSION = "recruitment_pairwise_v1"
RERANKER_NAME = "granite_4_1_8b_full_prp_graph_r10_ctx15360"

# Keep output paths short and portable across Linux and Windows.
OUTPUT_DIR = PROJECT_ROOT / "reranking" / "outputs" / RERANKER_NAME
LOG_DIR = PROJECT_ROOT / "reranking" / "logs"

# Candidate-pool depth passed from raw Hybrid RRF to PRP-Graph.
TOP_K = 20

# PRP-Graph comparison rounds. For N=20, R=10 is a sensible primary run.
COMPARISON_ROUNDS = 10

# None = all queries. Set to 1 only for a short server smoke test.
MAX_QUERIES: int | None = None

# Operational context budget for one complete pairwise prompt:
# system prompt + query + Candidate A + Candidate B + chat-template tokens.
# This operational limit is deliberately below the model-native limit.
# It is the documented experimental budget for one query and two CVs.
MAX_CONTEXT_TOKENS = 15360
CONTEXT_SAFETY_MARGIN = 128

# Keep the beginning of candidate_text if deterministic truncation is required.
# This assumes candidate_text begins with the most informative structured fields.
FAIL_ON_TRUNCATION = False

# Graph aggregation.
PAGERANK_DAMPING = 0.85
PAGERANK_TOLERANCE = 1e-8
PAGERANK_MAX_ITERATIONS = 500

# The paper also studies interpolation with the initial lexical ranking.
# Both pure graph and fixed Hybrid-RRF-interpolated scores are saved.
# The primary ranking below uses graph_only to isolate the LLM reranker.
INTERPOLATION_ALPHA = 0.50  # alpha * graph + (1-alpha) * initial RRF
FINAL_SCORE_MODE = "graph_only"  # "graph_only" or "interpolated"

# Professor/server experiment: use the original BF16 weights when supported,
# otherwise FP16. No 4-bit quantization is used in this file.
LOAD_IN_4BIT = False
USE_DOUBLE_QUANT = False
ENABLE_4BIT_CPU_OFFLOAD = False

SEED = 42
RESOLVED_COMPUTE_DTYPE = "not_loaded"

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",
    "rank",
    "score",
}

SYSTEM_PROMPT = """You are an expert recruitment relevance evaluator.

Your task is to compare two candidate profiles against one recruitment query and determine which candidate is more relevant and suitable for that query.

Use only information explicitly stated in the recruitment query and candidate profiles. Do not invent, assume, or infer unstated skills, qualifications, experience, education, certifications, language proficiency, achievements, availability, or personal characteristics.

Apply the following job-related criteria only when they are relevant to the query:
1. Direct match with the requested role, occupation, and professional domain.
2. Coverage of explicitly requested technical, legal, business, or professional skills.
3. Relevance, recency, and depth of demonstrated professional experience.
4. Match with requested seniority and years of experience.
5. Evidence that the requested skills were applied in actual work or projects.
6. Relevant education, certifications, languages, sector knowledge, and responsibilities.
7. Mandatory requirements must be prioritized over desirable requirements.
8. Direct evidence must be preferred over vague semantic similarity or keyword repetition.

Do not use or infer protected or sensitive characteristics, including age, sex, gender, nationality, ethnicity, religion, disability, health, marital or family status, photographs, names, or addresses. Do not prefer a candidate because the profile appears first, is longer, contains more repeated keywords, or uses a more polished writing style.

You must make a strict pairwise choice. Return exactly one label: A or B. Return no explanation and no other text."""


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class PairwiseResult:
    query_id: str
    candidate_first: str
    candidate_second: str
    dataset_index_first: int
    dataset_index_second: int
    p_first_when_first: float
    p_second_when_first: float
    forward_p_a: float
    forward_p_b: float
    reverse_p_a: float
    reverse_p_b: float
    forward_prompt_tokens: int
    reverse_prompt_tokens: int
    first_original_tokens: int
    second_original_tokens: int
    first_used_tokens_forward: int
    second_used_tokens_forward: int
    second_used_tokens_reverse: int
    first_used_tokens_reverse: int
    forward_truncated: bool
    reverse_truncated: bool
    forward_winner: str
    reverse_winner: str
    order_consistent: bool


@dataclass(frozen=True)
class PreparedPrompt:
    input_ids: torch.Tensor
    prompt_tokens: int
    candidate_a_original_tokens: int
    candidate_b_original_tokens: int
    candidate_a_used_tokens: int
    candidate_b_used_tokens: int
    truncated: bool


# =============================================================================
# General utilities
# =============================================================================


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


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

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def safe_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def experiment_signature(compute_dtype: torch.dtype) -> str:
    """Create a stable signature so incompatible runs never share a cache."""

    payload = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "precision_mode": PRECISION_MODE,
        "compute_dtype": str(compute_dtype),
        "top_k": TOP_K,
        "comparison_rounds": COMPARISON_ROUNDS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "context_safety_margin": CONTEXT_SAFETY_MARGIN,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "final_score_mode": FINAL_SCORE_MODE,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def get_single_token_label_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, int]:
    """Return exact next-token IDs for A and B, failing on ambiguity."""

    encoded = {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in ("A", "B")
    }

    invalid = {label: ids for label, ids in encoded.items() if len(ids) != 1}
    if invalid:
        raise RuntimeError(
            "The selected model must tokenize each output label as exactly one token. "
            f"Observed encodings: {encoded}. Choose different one-token labels "
            "before running the experiment."
        )

    if encoded["A"][0] == encoded["B"][0]:
        raise RuntimeError("Tokenizer maps labels A and B to the same token ID.")

    return {label: ids[0] for label, ids in encoded.items()}


def model_input_device(model: torch.nn.Module) -> torch.device:
    """Return the device holding the text input embedding layer."""

    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        return embeddings.weight.device

    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        embeddings = language_model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device

    raise RuntimeError("Could not locate the model input embedding device.")


# =============================================================================
# Candidate dataset and Hybrid RRF loading
# =============================================================================


def load_candidate_dataset(path: Path) -> pd.DataFrame:
    """Load candidate_text exactly as stored in the Parquet file."""

    validate_file_exists(path, "Candidate dataset")

    candidates = pd.read_parquet(path).reset_index(drop=True)

    if "candidate_text" not in candidates.columns:
        raise ValueError(
            "The candidate dataset does not contain 'candidate_text'."
        )

    if "candidate_id" in candidates.columns:
        candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    else:
        candidates["candidate_id"] = candidates.index.astype(str)

    if "dataset_index" in candidates.columns:
        candidates["dataset_index"] = pd.to_numeric(
            candidates["dataset_index"],
            errors="raise",
        ).astype(int)
    else:
        candidates["dataset_index"] = candidates.index.astype(int)

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
        result["candidate_text"].fillna("").astype(str)
    )

    empty_texts = int(result["candidate_text"].str.strip().eq("").sum())
    if empty_texts:
        raise ValueError(
            f"The Parquet dataset contains {empty_texts} empty "
            "candidate_text values."
        )

    return result


def load_hybrid_ranking(
    path: Path,
    candidates: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Load a raw Hybrid RRF ranking and attach candidate_text."""

    validate_file_exists(path, "Raw Hybrid RRF ranking file")

    ranking = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    missing = REQUIRED_RANKING_COLUMNS - set(ranking.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    ranking["dataset_index"] = pd.to_numeric(
        ranking["dataset_index"], errors="raise"
    ).astype(int)
    ranking["rank"] = pd.to_numeric(
        ranking["rank"], errors="raise"
    ).astype(int)
    ranking["score"] = pd.to_numeric(
        ranking["score"], errors="raise"
    ).astype(float)

    ranking = (
        ranking
        .sort_values(["query_id", "rank"], kind="stable")
        .groupby("query_id", group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    duplicate_pairs = int(
        ranking.duplicated(["query_id", "candidate_id"]).sum()
    )
    if duplicate_pairs:
        raise ValueError(
            f"{path.name} contains {duplicate_pairs} duplicate "
            "query-candidate pairs."
        )

    query_text_counts = ranking.groupby("query_id")["query"].nunique()
    conflicting_queries = query_text_counts[query_text_counts != 1]
    if not conflicting_queries.empty:
        raise ValueError(
            "Some query IDs have more than one query text: "
            f"{conflicting_queries.index.tolist()}"
        )

    ranking = ranking.merge(
        candidates,
        on=["candidate_id", "dataset_index"],
        how="left",
        validate="many_to_one",
    )

    missing_texts = int(ranking["candidate_text"].isna().sum())
    if missing_texts:
        raise ValueError(
            f"{missing_texts} ranking rows could not be matched to "
            "candidate_text in the Parquet dataset."
        )

    counts = ranking.groupby("query_id").size()
    incomplete = counts[counts != top_k]
    if not incomplete.empty:
        raise ValueError(
            f"{path.name} does not contain exactly {top_k} candidates "
            f"for every query:\n{incomplete.to_string()}"
        )

    expected_ranks = list(range(1, top_k + 1))
    for query_id, group in ranking.groupby("query_id", sort=False):
        observed_ranks = sorted(group["rank"].tolist())
        if observed_ranks != expected_ranks:
            raise ValueError(
                f"{path.name}, query '{query_id}', does not contain "
                f"ranks exactly 1..{top_k}."
            )

    return ranking


# =============================================================================
# IBM Granite 4.1 8B loading
# =============================================================================


def load_model() -> tuple[Any, PreTrainedTokenizerBase, torch.dtype]:
    """Load IBM Granite 4.1 8B in BF16 or FP16 without quantization."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This full-precision IBM Granite 4.1 8B run requires a CUDA/ROCm-capable GPU."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        use_fast=True,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_token_ids = get_single_token_label_ids(tokenizer)
    print(f"Output label token IDs: {label_token_ids}")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False
    model.eval()

    configured_limit = int(
        getattr(model.config, "max_position_embeddings", 0) or 0
    )
    if configured_limit and MAX_CONTEXT_TOKENS > configured_limit:
        raise ValueError(
            f"MAX_CONTEXT_TOKENS={MAX_CONTEXT_TOKENS} exceeds the "
            f"model limit {configured_limit}."
        )

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        if any(str(device).lower() == "disk" for device in device_map.values()):
            raise RuntimeError(
                "The model was dispatched partly to disk. Use a GPU or multi-GPU "
                "server with more memory, or configure an explicit CPU-offload run "
                "as a separate experiment."
            )

    print(f"Resolved device map: {device_map}")
    print(f"Resolved compute dtype: {compute_dtype}")
    return model, tokenizer, compute_dtype


# =============================================================================
# Pairwise prompt construction and deterministic context fitting
# =============================================================================


def build_user_prompt(
    query: str,
    candidate_a: str,
    candidate_b: str,
) -> str:
    return f"""Recruitment query:
<BEGIN_QUERY>
{query.strip()}
<END_QUERY>

Candidate A:
<BEGIN_CANDIDATE_A>
{candidate_a.strip()}
<END_CANDIDATE_A>

Candidate B:
<BEGIN_CANDIDATE_B>
{candidate_b.strip()}
<END_CANDIDATE_B>

Which candidate is more relevant and suitable for the recruitment query?
Return exactly one label: A or B."""


def render_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    candidate_a: str,
    candidate_b: str,
) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                query=query,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            ),
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(rendered, torch.Tensor):
        input_ids = rendered
    elif isinstance(rendered, dict) and "input_ids" in rendered:
        input_ids = rendered["input_ids"]
    else:
        input_ids = rendered.input_ids

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    return input_ids.cpu()


def allocate_candidate_budgets(
    length_a: int,
    length_b: int,
    available: int,
) -> tuple[int, int]:
    """Split an available token budget fairly, reusing unused capacity."""

    if available <= 0:
        raise ValueError(
            "The fixed prompt is already larger than the context budget."
        )

    if length_a + length_b <= available:
        return length_a, length_b

    budget_a = min(length_a, available // 2)
    budget_b = min(length_b, available - budget_a)

    remaining = available - budget_a - budget_b

    if remaining > 0:
        extra_a = min(length_a - budget_a, remaining)
        budget_a += extra_a
        remaining -= extra_a

    if remaining > 0:
        extra_b = min(length_b - budget_b, remaining)
        budget_b += extra_b

    return max(1, budget_a), max(1, budget_b)


def prepare_pair_prompt(
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    candidate_a: str,
    candidate_b: str,
) -> PreparedPrompt:
    """Fit a query and two profiles into a deterministic token budget."""

    candidate_a_ids = tokenizer.encode(
        candidate_a,
        add_special_tokens=False,
    )
    candidate_b_ids = tokenizer.encode(
        candidate_b,
        add_special_tokens=False,
    )

    original_a = len(candidate_a_ids)
    original_b = len(candidate_b_ids)

    full_prompt_ids = render_prompt_ids(
        tokenizer=tokenizer,
        query=query,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )

    usable_limit = MAX_CONTEXT_TOKENS - CONTEXT_SAFETY_MARGIN

    if full_prompt_ids.shape[1] <= usable_limit:
        return PreparedPrompt(
            input_ids=full_prompt_ids,
            prompt_tokens=int(full_prompt_ids.shape[1]),
            candidate_a_original_tokens=original_a,
            candidate_b_original_tokens=original_b,
            candidate_a_used_tokens=original_a,
            candidate_b_used_tokens=original_b,
            truncated=False,
        )

    if FAIL_ON_TRUNCATION:
        raise ValueError(
            "A pairwise prompt exceeds the configured context budget: "
            f"{full_prompt_ids.shape[1]} > {usable_limit}."
        )

    empty_prompt_ids = render_prompt_ids(
        tokenizer=tokenizer,
        query=query,
        candidate_a="",
        candidate_b="",
    )

    fixed_tokens = int(empty_prompt_ids.shape[1])
    candidate_budget = usable_limit - fixed_tokens

    budget_a, budget_b = allocate_candidate_budgets(
        length_a=original_a,
        length_b=original_b,
        available=candidate_budget,
    )

    # Keep the start of each structured candidate_text.
    used_a_ids = candidate_a_ids[:budget_a]
    used_b_ids = candidate_b_ids[:budget_b]

    # Decoding and re-tokenization can shift length slightly. Tighten until fit.
    while True:
        fitted_a = tokenizer.decode(
            used_a_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        fitted_b = tokenizer.decode(
            used_b_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        fitted_prompt_ids = render_prompt_ids(
            tokenizer=tokenizer,
            query=query,
            candidate_a=fitted_a,
            candidate_b=fitted_b,
        )

        prompt_length = int(fitted_prompt_ids.shape[1])
        if prompt_length <= usable_limit:
            break

        excess = prompt_length - usable_limit
        reduction = max(8, math.ceil(excess / 2))

        if len(used_a_ids) >= len(used_b_ids) and len(used_a_ids) > 1:
            used_a_ids = used_a_ids[: max(1, len(used_a_ids) - reduction)]
        elif len(used_b_ids) > 1:
            used_b_ids = used_b_ids[: max(1, len(used_b_ids) - reduction)]
        else:
            raise RuntimeError(
                "Could not fit the pairwise prompt into the context budget."
            )

    return PreparedPrompt(
        input_ids=fitted_prompt_ids,
        prompt_tokens=int(fitted_prompt_ids.shape[1]),
        candidate_a_original_tokens=original_a,
        candidate_b_original_tokens=original_b,
        candidate_a_used_tokens=len(used_a_ids),
        candidate_b_used_tokens=len(used_b_ids),
        truncated=True,
    )


# =============================================================================
# Exact A/B continuation probability scoring
# =============================================================================


@torch.inference_mode()
def score_binary_labels(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
) -> dict[str, float]:
    """
    Return the exact full-vocabulary next-token probabilities of labels A and B.

    The softmax is applied once over the model's complete output vocabulary.
    The extracted A and B probabilities are therefore not renormalized to sum
    to one. This is the scoring unit used to weight PRP-Graph edges.
    """

    label_token_ids = get_single_token_label_ids(tokenizer)
    device = model_input_device(model)

    input_ids = prompt_ids.to(device, non_blocking=True)
    attention_mask = torch.ones_like(input_ids)

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
    except TypeError as error:
        raise RuntimeError(
            "The installed Transformers/model implementation does not support "
            "logits_to_keep=1. Install transformers>=4.55.4; computing logits "
            "for every token of a long prompt is intentionally disabled."
        ) from error

    # logits_to_keep=1 avoids materialising vocabulary logits for every prompt
    # position; only the distribution for the next generated token is needed.
    next_token_logits = outputs.logits[0, -1].float()
    full_vocabulary_probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    return {
        "A": float(
            full_vocabulary_probabilities[label_token_ids["A"]]
            .detach()
            .cpu()
        ),
        "B": float(
            full_vocabulary_probabilities[label_token_ids["B"]]
            .detach()
            .cpu()
        ),
    }


# =============================================================================
# Incremental pairwise cache
# =============================================================================


class PairwiseCache:
    """JSONL cache that permits safe resumption after interruption."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[tuple[str, str, str], PairwiseResult] = {}
        self._load()

    @staticmethod
    def _key(
        query_id: str,
        candidate_a: str,
        candidate_b: str,
    ) -> tuple[str, str, str]:
        first, second = sorted([str(candidate_a), str(candidate_b)])
        return str(query_id), first, second

    def _load(self) -> None:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    result = PairwiseResult(**payload)
                except Exception as error:
                    raise ValueError(
                        f"Invalid cache entry at line {line_number} in "
                        f"{self.path}: {error}"
                    ) from error

                key = self._key(
                    result.query_id,
                    result.candidate_first,
                    result.candidate_second,
                )
                self._items[key] = result

    def get(
        self,
        query_id: str,
        candidate_a: str,
        candidate_b: str,
    ) -> PairwiseResult | None:
        return self._items.get(
            self._key(query_id, candidate_a, candidate_b)
        )

    def append(self, result: PairwiseResult) -> None:
        key = self._key(
            result.query_id,
            result.candidate_first,
            result.candidate_second,
        )
        self._items[key] = result

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            file.flush()


# =============================================================================
# Bidirectional scoring PRP unit
# =============================================================================


def compare_pair_bidirectionally(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    cache: PairwiseCache,
    query_id: str,
    query: str,
    candidate_i: pd.Series,
    candidate_j: pd.Series,
) -> PairwiseResult:
    """Run u(q,di,dj) and u(q,dj,di), or reuse the cached result."""

    candidate_i_id = str(candidate_i["candidate_id"])
    candidate_j_id = str(candidate_j["candidate_id"])

    cached = cache.get(query_id, candidate_i_id, candidate_j_id)
    if cached is not None:
        return cached

    text_i = str(candidate_i["candidate_text"])
    text_j = str(candidate_j["candidate_text"])

    forward = prepare_pair_prompt(
        tokenizer=tokenizer,
        query=query,
        candidate_a=text_i,
        candidate_b=text_j,
    )
    forward_probabilities = score_binary_labels(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=forward.input_ids,
    )

    reverse = prepare_pair_prompt(
        tokenizer=tokenizer,
        query=query,
        candidate_a=text_j,
        candidate_b=text_i,
    )
    reverse_probabilities = score_binary_labels(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=reverse.input_ids,
    )

    # Forward A is candidate_i. Reverse A is candidate_j.
    p_i_when_first = float(forward_probabilities["A"])
    p_j_when_first = float(reverse_probabilities["A"])

    forward_winner = (
        candidate_i_id
        if forward_probabilities["A"] >= forward_probabilities["B"]
        else candidate_j_id
    )
    reverse_winner = (
        candidate_j_id
        if reverse_probabilities["A"] >= reverse_probabilities["B"]
        else candidate_i_id
    )

    result = PairwiseResult(
        query_id=str(query_id),
        candidate_first=candidate_i_id,
        candidate_second=candidate_j_id,
        dataset_index_first=int(candidate_i["dataset_index"]),
        dataset_index_second=int(candidate_j["dataset_index"]),
        p_first_when_first=p_i_when_first,
        p_second_when_first=p_j_when_first,
        forward_p_a=float(forward_probabilities["A"]),
        forward_p_b=float(forward_probabilities["B"]),
        reverse_p_a=float(reverse_probabilities["A"]),
        reverse_p_b=float(reverse_probabilities["B"]),
        forward_prompt_tokens=forward.prompt_tokens,
        reverse_prompt_tokens=reverse.prompt_tokens,
        first_original_tokens=forward.candidate_a_original_tokens,
        second_original_tokens=forward.candidate_b_original_tokens,
        first_used_tokens_forward=forward.candidate_a_used_tokens,
        second_used_tokens_forward=forward.candidate_b_used_tokens,
        second_used_tokens_reverse=reverse.candidate_a_used_tokens,
        first_used_tokens_reverse=reverse.candidate_b_used_tokens,
        forward_truncated=forward.truncated,
        reverse_truncated=reverse.truncated,
        forward_winner=forward_winner,
        reverse_winner=reverse_winner,
        order_consistent=(forward_winner == reverse_winner),
    )

    cache.append(result)
    return result


def probabilities_for_requested_order(
    result: PairwiseResult,
    candidate_i_id: str,
    candidate_j_id: str,
) -> tuple[float, float]:
    """Map a cached bidirectional result to the currently requested i,j order."""

    candidate_i_id = str(candidate_i_id)
    candidate_j_id = str(candidate_j_id)

    if (
        result.candidate_first == candidate_i_id
        and result.candidate_second == candidate_j_id
    ):
        return result.p_first_when_first, result.p_second_when_first

    if (
        result.candidate_first == candidate_j_id
        and result.candidate_second == candidate_i_id
    ):
        return result.p_second_when_first, result.p_first_when_first

    raise ValueError("Cached pair does not match the requested candidates.")


# =============================================================================
# PRP-Graph construction
# =============================================================================


def build_prp_graph_for_query(
    *,
    group: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    cache: PairwiseCache,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Construct the pairwise ranking graph using selective comparison rounds.

    Edge convention follows the PRP-Graph paper:
        W[j, i] = probability that document i is preferred when i is first;
        W[i, j] = probability that document j is preferred when j is first.
    """

    group = (
        group.sort_values("rank", kind="stable")
        .reset_index(drop=True)
        .copy()
    )

    query_id = str(group["query_id"].iloc[0])
    query = str(group["query"].iloc[0])
    candidate_ids = group["candidate_id"].astype(str).tolist()
    number_of_candidates = len(candidate_ids)

    if number_of_candidates != TOP_K:
        raise ValueError(
            f"Query {query_id} has {number_of_candidates} candidates, "
            f"expected {TOP_K}."
        )

    rows_by_candidate = {
        str(row["candidate_id"]): row
        for _, row in group.iterrows()
    }
    initial_rank = {
        str(row["candidate_id"]): int(row["rank"])
        for _, row in group.iterrows()
    }

    # S0 = [1, 1-1/N, 1-2/N, ..., 1/N]
    construction_scores = {
        candidate_id: 1.0 - (position / number_of_candidates)
        for position, candidate_id in enumerate(candidate_ids)
    }

    current_order = candidate_ids.copy()
    compared_pairs: set[frozenset[str]] = set()

    id_to_matrix_index = {
        candidate_id: index
        for index, candidate_id in enumerate(candidate_ids)
    }
    edge_weights = np.zeros(
        (number_of_candidates, number_of_candidates),
        dtype=np.float64,
    )

    pair_records: list[dict[str, Any]] = []
    round_records: list[dict[str, Any]] = []

    rounds_completed = 0

    for round_number in range(1, COMPARISON_ROUNDS + 1):
        used_this_round: set[str] = set()
        previous_scores = construction_scores.copy()
        updated_scores = previous_scores.copy()
        comparisons_this_round = 0

        for position_i, candidate_i_id in enumerate(current_order):
            if candidate_i_id in used_this_round:
                continue

            candidate_j_id: str | None = None

            for candidate_j_id_attempt in current_order[position_i + 1 :]:
                pair_key = frozenset(
                    [candidate_i_id, candidate_j_id_attempt]
                )

                if candidate_j_id_attempt in used_this_round:
                    continue
                if pair_key in compared_pairs:
                    continue

                candidate_j_id = candidate_j_id_attempt
                break

            if candidate_j_id is None:
                continue

            candidate_i = rows_by_candidate[candidate_i_id]
            candidate_j = rows_by_candidate[candidate_j_id]

            result = compare_pair_bidirectionally(
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                query_id=query_id,
                query=query,
                candidate_i=candidate_i,
                candidate_j=candidate_j,
            )

            p_i, p_j = probabilities_for_requested_order(
                result=result,
                candidate_i_id=candidate_i_id,
                candidate_j_id=candidate_j_id,
            )

            matrix_i = id_to_matrix_index[candidate_i_id]
            matrix_j = id_to_matrix_index[candidate_j_id]

            # Directed edge j -> i expresses evidence supporting i over j.
            edge_weights[matrix_j, matrix_i] = p_i
            # Directed edge i -> j expresses evidence supporting j over i.
            edge_weights[matrix_i, matrix_j] = p_j

            updated_scores[candidate_i_id] = (
                previous_scores[candidate_i_id]
                + p_i
                * previous_scores[candidate_j_id]
                / round_number
            )
            updated_scores[candidate_j_id] = (
                previous_scores[candidate_j_id]
                + p_j
                * previous_scores[candidate_i_id]
                / round_number
            )

            used_this_round.update([candidate_i_id, candidate_j_id])
            compared_pairs.add(frozenset([candidate_i_id, candidate_j_id]))
            comparisons_this_round += 1

            pair_records.append(
                {
                    **asdict(result),
                    "round": round_number,
                    "candidate_i": candidate_i_id,
                    "candidate_j": candidate_j_id,
                    "p_i_graph_edge": p_i,
                    "p_j_graph_edge": p_j,
                    "i_rank_before_round": (
                        current_order.index(candidate_i_id) + 1
                    ),
                    "j_rank_before_round": (
                        current_order.index(candidate_j_id) + 1
                    ),
                    "i_construction_score_before": previous_scores[
                        candidate_i_id
                    ],
                    "j_construction_score_before": previous_scores[
                        candidate_j_id
                    ],
                }
            )

        if comparisons_this_round == 0:
            logger.warning(
                "Query %s stopped after %d rounds because no unseen "
                "candidate pairs remained.",
                query_id,
                rounds_completed,
            )
            break

        construction_scores = updated_scores
        current_order = sorted(
            current_order,
            key=lambda candidate_id: (
                -construction_scores[candidate_id],
                initial_rank[candidate_id],
            ),
        )
        rounds_completed = round_number

        for temporary_rank, candidate_id in enumerate(
            current_order,
            start=1,
        ):
            round_records.append(
                {
                    "query_id": query_id,
                    "round": round_number,
                    "candidate_id": candidate_id,
                    "temporary_rank": temporary_rank,
                    "construction_score": construction_scores[candidate_id],
                    "comparisons_this_round": comparisons_this_round,
                    "cumulative_unique_pairs": len(compared_pairs),
                }
            )

        logger.info(
            "Query %s | round %d/%d | pairs=%d | cumulative=%d",
            query_id,
            round_number,
            COMPARISON_ROUNDS,
            comparisons_this_round,
            len(compared_pairs),
        )

    construction_df = group.copy()
    construction_df["construction_score"] = construction_df[
        "candidate_id"
    ].astype(str).map(construction_scores)
    construction_df["rounds_completed"] = rounds_completed
    construction_df["unique_pairs_compared"] = len(compared_pairs)

    pair_df = pd.DataFrame(pair_records)
    round_df = pd.DataFrame(round_records)

    return construction_df, pair_df, round_df, edge_weights


# =============================================================================
# Weighted graph aggregation
# =============================================================================


def weighted_pagerank(
    edge_weights: np.ndarray,
    initial_distribution: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    """Compute weighted PageRank over the PRP comparison graph."""

    if edge_weights.ndim != 2 or edge_weights.shape[0] != edge_weights.shape[1]:
        raise ValueError("edge_weights must be a square matrix.")

    number_of_nodes = edge_weights.shape[0]

    initial = np.asarray(initial_distribution, dtype=np.float64)
    initial = np.clip(initial, a_min=0.0, a_max=None)
    if initial.sum() <= 0:
        initial = np.full(number_of_nodes, 1.0 / number_of_nodes)
    else:
        initial = initial / initial.sum()

    scores = initial.copy()
    out_weight_sums = edge_weights.sum(axis=1)
    uniform_teleport = np.full(number_of_nodes, 1.0 / number_of_nodes)

    converged = False

    for iteration in range(1, PAGERANK_MAX_ITERATIONS + 1):
        new_scores = (1.0 - PAGERANK_DAMPING) * uniform_teleport

        dangling_mass = float(scores[out_weight_sums == 0.0].sum())
        if dangling_mass > 0:
            new_scores += (
                PAGERANK_DAMPING
                * dangling_mass
                * uniform_teleport
            )

        for source in range(number_of_nodes):
            out_sum = out_weight_sums[source]
            if out_sum <= 0:
                continue

            new_scores += (
                PAGERANK_DAMPING
                * scores[source]
                * edge_weights[source]
                / out_sum
            )

        new_scores_sum = float(new_scores.sum())
        if new_scores_sum <= 0:
            raise RuntimeError("PageRank produced a non-positive score sum.")
        new_scores /= new_scores_sum

        if np.linalg.norm(new_scores - scores, ord=1) < PAGERANK_TOLERANCE:
            scores = new_scores
            converged = True
            return scores, iteration, converged

        scores = new_scores

    return scores, PAGERANK_MAX_ITERATIONS, converged


def aggregate_and_rank_query(
    construction_df: pd.DataFrame,
    edge_weights: np.ndarray,
) -> pd.DataFrame:
    """Create pure graph and Hybrid-RRF-interpolated final scores."""

    group = (
        construction_df.sort_values("rank", kind="stable")
        .reset_index(drop=True)
        .copy()
    )

    initial_scores = group["score"].to_numpy(dtype=np.float64)
    initial_scores = np.clip(initial_scores, a_min=0.0, a_max=None)

    if initial_scores.sum() <= 0:
        initial_normalized = np.full(
            len(group),
            1.0 / len(group),
            dtype=np.float64,
        )
    else:
        initial_normalized = initial_scores / initial_scores.sum()

    graph_scores, iterations, converged = weighted_pagerank(
        edge_weights=edge_weights,
        initial_distribution=initial_normalized,
    )

    interpolated_scores = (
        INTERPOLATION_ALPHA * graph_scores
        + (1.0 - INTERPOLATION_ALPHA) * initial_normalized
    )

    group["initial_rank"] = group["rank"].astype(int)
    group["initial_rrf_score"] = group["score"].astype(float)
    group["initial_rrf_normalized"] = initial_normalized
    group["prp_graph_score"] = graph_scores
    group["prp_graph_rrf_interpolated_score"] = interpolated_scores
    group["pagerank_iterations"] = iterations
    group["pagerank_converged"] = converged

    if FINAL_SCORE_MODE == "graph_only":
        group["final_score"] = group["prp_graph_score"]
    elif FINAL_SCORE_MODE == "interpolated":
        group["final_score"] = group[
            "prp_graph_rrf_interpolated_score"
        ]
    else:
        raise ValueError(
            "FINAL_SCORE_MODE must be 'graph_only' or 'interpolated'."
        )

    group = group.sort_values(
        ["final_score", "initial_rank"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    group["reranked_rank"] = np.arange(1, len(group) + 1)
    group["rank_change"] = (
        group["initial_rank"] - group["reranked_rank"]
    )
    group["model"] = MODEL_NAME
    group["reranker"] = "PRP-Graph"
    group["final_score_mode"] = FINAL_SCORE_MODE

    return group


# =============================================================================
# Full-system reranking and outputs
# =============================================================================


def rerank_one_hybrid_system(
    *,
    ranking: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    retriever_name: str,
    cache: PairwiseCache,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reranked_groups: list[pd.DataFrame] = []
    all_pair_records: list[pd.DataFrame] = []
    all_round_records: list[pd.DataFrame] = []

    grouped = list(ranking.groupby("query_id", sort=False))

    for query_id, group in tqdm(
        grouped,
        total=len(grouped),
        desc=f"{retriever_name} -> IBM Granite 4.1 8B PRP-Graph",
    ):
        logger.info(
            "Starting query %s with %d candidates.",
            query_id,
            len(group),
        )

        construction_df, pair_df, round_df, edge_weights = (
            build_prp_graph_for_query(
                group=group,
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                logger=logger,
            )
        )

        reranked = aggregate_and_rank_query(
            construction_df=construction_df,
            edge_weights=edge_weights,
        )
        reranked["retriever"] = retriever_name

        reranked_groups.append(reranked)

        if not pair_df.empty:
            pair_df["retriever"] = retriever_name
            all_pair_records.append(pair_df)

        if not round_df.empty:
            round_df["retriever"] = retriever_name
            all_round_records.append(round_df)

        logger.info(
            "Completed query %s | pairs=%d | top candidate=%s",
            query_id,
            int(reranked["unique_pairs_compared"].iloc[0]),
            str(reranked["candidate_id"].iloc[0]),
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not reranked_groups:
        raise RuntimeError("No query groups were available for reranking.")

    reranked_all = pd.concat(reranked_groups, ignore_index=True)
    reranked_all = reranked_all.sort_values(
        ["query_id", "reranked_rank"],
        kind="stable",
    ).reset_index(drop=True)

    pairwise_all = (
        pd.concat(all_pair_records, ignore_index=True)
        if all_pair_records
        else pd.DataFrame()
    )
    rounds_all = (
        pd.concat(all_round_records, ignore_index=True)
        if all_round_records
        else pd.DataFrame()
    )

    return reranked_all, pairwise_all, rounds_all


def create_evaluation_ranking(
    reranked: pd.DataFrame,
    system_name: str,
) -> pd.DataFrame:
    evaluation = reranked.copy()
    evaluation["system"] = system_name
    evaluation["rank"] = evaluation["reranked_rank"]
    evaluation["score"] = evaluation["final_score"]

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


def save_run_configuration(
    output_path: Path,
    retriever_name: str,
) -> None:
    payload = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "precision_mode": PRECISION_MODE,
        "resolved_compute_dtype": RESOLVED_COMPUTE_DTYPE,
        "prompt_version": PROMPT_VERSION,
        "method": "PRP-Graph",
        "initial_retriever": retriever_name,
        "candidate_pool": "raw Hybrid RRF",
        "top_k": TOP_K,
        "comparison_rounds": COMPARISON_ROUNDS,
        "maximum_queries": MAX_QUERIES,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "context_safety_margin": CONTEXT_SAFETY_MARGIN,
        "fail_on_truncation": FAIL_ON_TRUNCATION,
        "load_in_4bit": LOAD_IN_4BIT,
        "double_quantization": USE_DOUBLE_QUANT,
        "enable_4bit_cpu_offload": ENABLE_4BIT_CPU_OFFLOAD,
        "pagerank_damping": PAGERANK_DAMPING,
        "pagerank_tolerance": PAGERANK_TOLERANCE,
        "pagerank_max_iterations": PAGERANK_MAX_ITERATIONS,
        "interpolation_alpha": INTERPOLATION_ALPHA,
        "final_score_mode": FINAL_SCORE_MODE,
        "seed": SEED,
        "probability_mode": "full_vocabulary_next_token_softmax",
        "pairwise_directionality": "bidirectional",
        "pair_selection": "selective_swiss_style_prp_graph",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "system_prompt": SYSTEM_PROMPT,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_outputs(
    *,
    reranked: pd.DataFrame,
    pairwise: pd.DataFrame,
    rounds: pd.DataFrame,
    retriever_name: str,
    logger: logging.Logger,
) -> None:
    experiment_name = f"{retriever_name}__{RERANKER_NAME}"
    experiment_dir = OUTPUT_DIR / retriever_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = experiment_dir / "detailed.csv"
    evaluation_path = experiment_dir / "rankings.csv"
    pairwise_path = experiment_dir / "pairwise.csv"
    rounds_path = experiment_dir / "round_trace.csv"
    configuration_path = experiment_dir / "config.json"

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

    if not pairwise.empty:
        pairwise.to_csv(
            pairwise_path,
            index=False,
            encoding="utf-8-sig",
        )

    if not rounds.empty:
        rounds.to_csv(
            rounds_path,
            index=False,
            encoding="utf-8-sig",
        )

    save_run_configuration(
        output_path=configuration_path,
        retriever_name=retriever_name,
    )

    logger.info("Detailed output: %s", detailed_path)
    logger.info("Evaluation output: %s", evaluation_path)
    logger.info("Pairwise output: %s", pairwise_path)
    logger.info("Round trace: %s", rounds_path)
    logger.info("Configuration: %s", configuration_path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    global RESOLVED_COMPUTE_DTYPE

    set_reproducibility(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("GRANITE-4.1-8B FULL-PRECISION PRP-GRAPH — RAW HYBRID RRF TOP-20")
    print("=" * 88)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model revision: {MODEL_REVISION}")
    print(f"Precision mode: {PRECISION_MODE}")
    print(f"Top K: {TOP_K}")
    print(f"Comparison rounds: {COMPARISON_ROUNDS}")
    print(f"Maximum operational context: {MAX_CONTEXT_TOKENS}")
    print(
        f"Usable prompt-token limit: "
        f"{MAX_CONTEXT_TOKENS - CONTEXT_SAFETY_MARGIN}"
    )
    print("4-bit loading: False")
    print(f"Maximum queries: {MAX_QUERIES}")
    print(f"Final score mode: {FINAL_SCORE_MODE}")

    candidates = load_candidate_dataset(DATASET_PATH)

    hybrid_rankings: dict[str, pd.DataFrame] = {}
    for retriever_name, ranking_path in HYBRID_FILES.items():
        hybrid_rankings[retriever_name] = load_hybrid_ranking(
            path=ranking_path,
            candidates=candidates,
            top_k=TOP_K,
        )

        if MAX_QUERIES is not None:
            current_ranking = hybrid_rankings[retriever_name]
            selected_query_ids = (
                current_ranking["query_id"]
                .drop_duplicates()
                .head(MAX_QUERIES)
            )
            hybrid_rankings[retriever_name] = current_ranking.loc[
                current_ranking["query_id"].isin(selected_query_ids)
            ].copy()

        print(
            f"Loaded {retriever_name}: "
            f"{len(hybrid_rankings[retriever_name])} rows, "
            f"{hybrid_rankings[retriever_name]['query_id'].nunique()} queries"
        )

    print("\nLoading IBM Granite 4.1 8B...")
    model, tokenizer, compute_dtype = load_model()
    RESOLVED_COMPUTE_DTYPE = str(compute_dtype)
    run_signature = experiment_signature(compute_dtype)
    print(f"Run signature: {run_signature}")

    model_limit = int(
        getattr(model.config, "max_position_embeddings", 0) or 0
    )
    print(f"Model max position embeddings: {model_limit}")
    print(f"Model input device: {model_input_device(model)}")

    for experiment_number, (
        retriever_name,
        ranking,
    ) in enumerate(hybrid_rankings.items(), start=1):
        print(
            f"\nExperiment {experiment_number}/{len(hybrid_rankings)}: "
            f"{retriever_name}"
        )

        log_path = LOG_DIR / f"{RERANKER_NAME}__{retriever_name}.log"
        logger = create_logger(log_path)

        cache_path = (
            OUTPUT_DIR
            / retriever_name
            / f"cache_{run_signature}.jsonl"
        )
        cache = PairwiseCache(cache_path)

        logger.info("Retriever: %s", retriever_name)
        logger.info("Model: %s", MODEL_NAME)
        logger.info("Top K: %d", TOP_K)
        logger.info("Comparison rounds: %d", COMPARISON_ROUNDS)
        logger.info("Cache: %s", cache_path)

        start_time = time.perf_counter()

        try:
            reranked, pairwise, rounds = rerank_one_hybrid_system(
                ranking=ranking,
                model=model,
                tokenizer=tokenizer,
                retriever_name=retriever_name,
                cache=cache,
                logger=logger,
            )

            save_outputs(
                reranked=reranked,
                pairwise=pairwise,
                rounds=rounds,
                retriever_name=retriever_name,
                logger=logger,
            )

            elapsed = time.perf_counter() - start_time
            logger.info("Completed successfully in %.2f seconds.", elapsed)

        except torch.cuda.OutOfMemoryError as error:
            logger.exception(
                "CUDA out of memory. Reduce MAX_CONTEXT_TOKENS, keep "
                "TOP_K=20, or use a larger GPU. Current context=%d.",
                MAX_CONTEXT_TOKENS,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise error

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("ALL GRANITE-4.1-8B PRP-GRAPH RUNS COMPLETED")
    print("=" * 88)


if __name__ == "__main__":
    main()
