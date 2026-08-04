from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "cv_embeddings_jina_v5_text_small_20k_2048.npy"
)

MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"

EXPECTED_ROWS = 20_000

MAX_SEQ_LENGTH = 2048

BATCH_SIZE = 2

TOKEN_AUDIT_BATCH_SIZE = 256


def load_candidate_texts() -> list[str]:
    """Load and validate the common candidate representation."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print("Dataset:", DATA_PATH)
    print("Dataset shape:", df.shape)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"but found {len(df):,}."
        )

    if "candidate_text" not in df.columns:
        raise ValueError(
            "Column 'candidate_text' is missing from the dataset."
        )

    texts = (
        df["candidate_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    empty_indices = [
        index
        for index, text in enumerate(texts)
        if not text
    ]

    if empty_indices:
        raise ValueError(
            f"Found {len(empty_indices)} empty candidate_text values. "
            f"First affected indices: {empty_indices[:10]}"
        )

    return texts


def audit_token_lengths(
    model: SentenceTransformer,
    texts: list[str],
) -> list[int]:
    """
    Measure input lengths without truncation.

    This verifies how many candidate_text values exceed the selected
    MAX_SEQ_LENGTH before embedding generation begins.
    """

    tokenizer = model.tokenizer
    token_lengths: list[int] = []

    print("\nMeasuring Jina v5 small token lengths...")

    for start in range(
        0,
        len(texts),
        TOKEN_AUDIT_BATCH_SIZE,
    ):
        batch = texts[
            start : start + TOKEN_AUDIT_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        token_lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

        completed = min(
            start + TOKEN_AUDIT_BATCH_SIZE,
            len(texts),
        )

        print(
            f"\rAudited {completed:,}/{len(texts):,} CVs",
            end="",
            flush=True,
        )

    print()

    if len(token_lengths) != len(texts):
        raise ValueError(
            "Token-length count does not match the number of CVs."
        )

    return token_lengths


def print_token_statistics(
    token_lengths: list[int],
) -> None:
    """Print truncation statistics for the selected context length."""

    lengths = np.asarray(
        token_lengths,
        dtype=np.int32,
    )

    over_limit = lengths > MAX_SEQ_LENGTH
    over_limit_count = int(over_limit.sum())

    print("\n" + "=" * 80)
    print("TOKEN-LENGTH AUDIT")
    print("=" * 80)
    print("Minimum tokens:", int(lengths.min()))
    print("Mean tokens:", f"{lengths.mean():.2f}")
    print("Median tokens:", int(np.median(lengths)))
    print("95th percentile:", int(np.percentile(lengths, 95)))
    print("99th percentile:", int(np.percentile(lengths, 99)))
    print("Maximum tokens:", int(lengths.max()))

    print(
        f"CVs over {MAX_SEQ_LENGTH} tokens:",
        f"{over_limit_count:,}/{len(lengths):,}",
        f"({100 * over_limit_count / len(lengths):.4f}%)",
    )

    if over_limit_count == 0:
        print(
            f"All CVs fit within the selected "
            f"{MAX_SEQ_LENGTH}-token limit."
        )
    else:
        longest_indices = np.argsort(lengths)[::-1][:10]

        print("\nLongest CV indices:")
        for index in longest_indices:
            print(
                f"  Dataset index {int(index)}: "
                f"{int(lengths[index])} tokens"
            )

        print(
            "\nWarning: CVs above the selected limit will be "
            "truncated during embedding generation."
        )

    print("=" * 80)


def main() -> None:
    start_time = time.perf_counter()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "The target embedding file already exists:\n"
            f"{OUTPUT_PATH}\n\n"
            "Rename or remove it before rerunning the script. "
            "This safeguard prevents accidental overwriting."
        )

    texts = load_candidate_texts()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 80)
    print("CREATING JINA V5 SMALL RETRIEVAL EMBEDDINGS")
    print("=" * 80)
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Documents:", len(texts))
    print("Embedding batch size:", BATCH_SIZE)
    print("Requested maximum sequence length:", MAX_SEQ_LENGTH)
    print("Output:", OUTPUT_PATH)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

        total_vram = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )
        print(f"Total GPU VRAM: {total_vram:.2f} GB")

    print("=" * 80)

    model_kwargs = {}
    if device == "cuda":
        # The official model card recommends bfloat16 on supported GPUs.
        # RTX 50-series GPUs support bfloat16 and it substantially reduces VRAM use.
        model_kwargs["dtype"] = torch.bfloat16

    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=device,
        model_kwargs=model_kwargs,
    )

    original_max_seq_length = model.max_seq_length

    print(
        "\nOriginal SentenceTransformer max sequence length:",
        original_max_seq_length,
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    print(
        "Effective max sequence length:",
        model.max_seq_length,
    )

    if model.max_seq_length != MAX_SEQ_LENGTH:
        raise ValueError(
            "The effective model sequence length does not match "
            f"the requested value. Requested: {MAX_SEQ_LENGTH}, "
            f"effective: {model.max_seq_length}."
        )

    token_lengths = audit_token_lengths(
        model=model,
        texts=texts,
    )

    print_token_statistics(token_lengths)

    print("\nStarting embedding generation...")

    try:
        with torch.inference_mode():
            embeddings = model.encode(
                inputs=texts,
                task="retrieval",
                prompt_name="document",
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

    except torch.cuda.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise RuntimeError(
            "\nCUDA ran out of memory while creating embeddings.\n"
            "Current configuration:\n"
            f"  MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}\n"
            f"  BATCH_SIZE = {BATCH_SIZE}\n\n"
            "Try one of:\n"
            "  1. Set BATCH_SIZE = 1\n"
            "  2. Set MAX_SEQ_LENGTH = 1536\n"
            "  3. Close other GPU applications and rerun.\n"
        ) from error

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional embeddings array, "
            f"but received shape {embeddings.shape}."
        )

    if embeddings.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            "Embedding count does not match dataset row count. "
            f"Embeddings: {embeddings.shape[0]:,}, "
            f"dataset rows: {EXPECTED_ROWS:,}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "The embeddings contain NaN or infinite values."
        )

    # normalize_embeddings=True is applied by the model. With BF16 inference,
    # tiny numerical deviations from exactly 1.0 are expected after conversion
    # to NumPy float32, so use a BF16-appropriate validation tolerance.
    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        vector_norms,
        1.0,
        atol=5e-3,
        rtol=0.0,
    ):
        raise ValueError(
            "Some embeddings are outside the expected normalized range. "
            f"Minimum norm: {vector_norms.min():.6f}, "
            f"maximum norm: {vector_norms.max():.6f}."
        )

    np.save(
        OUTPUT_PATH,
        embeddings,
    )

    # Reload the saved file to verify that it was written correctly.
    saved_embeddings = np.load(
        OUTPUT_PATH,
        mmap_mode="r",
    )

    if saved_embeddings.shape != embeddings.shape:
        raise ValueError(
            "Saved embedding shape does not match the generated "
            f"embedding shape. Saved: {saved_embeddings.shape}, "
            f"generated: {embeddings.shape}."
        )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("JINA V5 SMALL RETRIEVAL EMBEDDINGS COMPLETED")
    print("=" * 80)
    print("Shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print(
        "Minimum vector norm:",
        f"{vector_norms.min():.6f}",
    )
    print(
        "Maximum vector norm:",
        f"{vector_norms.max():.6f}",
    )
    print("Saved to:", OUTPUT_PATH)
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()