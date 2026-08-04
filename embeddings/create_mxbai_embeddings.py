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
    / "cv_embeddings_mxbai_embed_large_20k.npy"
)

MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

EXPECTED_ROWS = 20_000
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 8


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

    empty_count = sum(not text for text in texts)

    if empty_count > 0:
        raise ValueError(
            f"Found {empty_count} empty candidate_text values."
        )

    return texts


def main() -> None:
    start_time = time.perf_counter()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    texts = load_candidate_texts()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 80)
    print("Creating MXBAI candidate embeddings")
    print("=" * 80)
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Documents:", len(texts))
    print("Batch size:", BATCH_SIZE)
    print("Maximum sequence length:", MAX_SEQ_LENGTH)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    print("Effective model max sequence length:", model.max_seq_length)

    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    embeddings = embeddings.astype(
        np.float32,
        copy=False,
    )

    if embeddings.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            "Embedding count does not match dataset row count. "
            f"Embeddings: {embeddings.shape[0]}, "
            f"rows: {EXPECTED_ROWS}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "The embeddings contain NaN or infinite values."
        )

    np.save(
        OUTPUT_PATH,
        embeddings,
    )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("MXBAI embeddings completed")
    print("=" * 80)
    print("Shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Saved to:", OUTPUT_PATH)
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()