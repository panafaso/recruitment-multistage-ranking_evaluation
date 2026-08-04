from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "embeddings" / "token_length_audits"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "e5": {
        "model_name": "intfloat/multilingual-e5-large",
        "default_limit": 512,
        "prefix": "passage: ",
    },
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "default_limit": 1024,
        "prefix": "",
    },
    "mxbai": {
        "model_name": "mixedbread-ai/mxbai-embed-large-v1",
        "default_limit": 512,
        "prefix": "",
    },
}

REFERENCE_LIMITS = (512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure candidate_text token lengths without recreating embeddings."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="e5",
        help="Tokenizer/model family to audit. Default: e5.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Input-token limit to test. Otherwise uses the model default.",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=20,
        help="Number of longest examples to print. Default: 20.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tokenizer batch size. Default: 256.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 8:
        parser.error("--limit must be at least 8.")
    if args.top_examples < 1:
        parser.error("--top-examples must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    return args


def load_candidate_texts() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Candidate parquet was not found:\n{DATA_PATH}"
        )

    data = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    if "candidate_text" not in data.columns:
        raise ValueError(
            f"'candidate_text' is missing. Available columns: "
            f"{list(data.columns)}"
        )

    audit = pd.DataFrame(
        {
            "dataset_index": data.index.astype(int),
            "candidate_text": (
                data["candidate_text"]
                .fillna("")
                .astype(str)
                .str.strip()
            ),
        }
    )

    audit["position"] = (
        data["Position"].fillna("").astype(str).str.strip()
        if "Position" in data.columns
        else ""
    )

    empty = audit["candidate_text"].eq("")
    if empty.any():
        examples = audit.loc[empty, "dataset_index"].head(10).tolist()
        raise ValueError(
            f"Found {int(empty.sum())} empty candidate_text values. "
            f"Examples: {examples}"
        )

    return audit


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text, flags=re.UNICODE))


def batched_token_lengths(
    tokenizer,
    texts: list[str],
    batch_size: int,
) -> list[int]:
    lengths: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

        completed = min(start + batch_size, len(texts))
        print(
            f"\rTokenized {completed:,}/{len(texts):,} candidates",
            end="",
            flush=True,
        )

    print()
    return lengths


def descriptive_statistics(series: pd.Series) -> dict[str, float | int]:
    q = series.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    return {
        "count": int(series.size),
        "minimum": int(series.min()),
        "mean": round(float(series.mean()), 2),
        "median": int(q.loc[0.50]),
        "p25": int(q.loc[0.25]),
        "p75": int(q.loc[0.75]),
        "p90": int(q.loc[0.90]),
        "p95": int(q.loc[0.95]),
        "p99": int(q.loc[0.99]),
        "maximum": int(series.max()),
    }


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model]

    model_name = str(config["model_name"])
    tested_limit = (
        args.limit
        if args.limit is not None
        else int(config["default_limit"])
    )
    document_prefix = str(config["prefix"])

    print("=" * 80)
    print("CANDIDATE_TEXT TOKEN-LENGTH AUDIT")
    print("=" * 80)
    print("Dataset:", DATA_PATH)
    print("Tokenizer:", model_name)
    print("Tested input limit:", tested_limit)
    print(
        "Document prefix:",
        repr(document_prefix) if document_prefix else "none",
    )
    print("Existing embeddings are not modified.")
    print("=" * 80)

    audit = load_candidate_texts()
    print(f"Loaded candidates: {len(audit):,}")
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    raw_texts = audit["candidate_text"].tolist()
    model_inputs = [
        document_prefix + text
        for text in raw_texts
    ]

    audit["word_count"] = [
        count_words(text)
        for text in raw_texts
    ]

    audit["token_count"] = batched_token_lengths(
        tokenizer=tokenizer,
        texts=model_inputs,
        batch_size=args.batch_size,
    )

    audit["tested_limit"] = tested_limit
    audit["would_be_truncated"] = (
        audit["token_count"] > tested_limit
    )
    audit["tokens_removed"] = (
        audit["token_count"] - tested_limit
    ).clip(lower=0)
    audit["retained_token_ratio"] = (
        tested_limit / audit["token_count"].clip(lower=1)
    ).clip(upper=1.0).round(4)

    for limit in REFERENCE_LIMITS:
        audit[f"over_{limit}_tokens"] = (
            audit["token_count"] > limit
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_model_name = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        model_name,
    )

    detail_path = (
        OUTPUT_DIR
        / f"{safe_model_name}_limit_{tested_limit}_details.csv"
    )
    summary_path = (
        OUTPUT_DIR
        / f"{safe_model_name}_limit_{tested_limit}_summary.json"
    )
    longest_path = (
        OUTPUT_DIR
        / f"{safe_model_name}_limit_{tested_limit}_longest.csv"
    )

    audit.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    longest = (
        audit.sort_values(
            ["token_count", "word_count"],
            ascending=[False, False],
            kind="stable",
        )
        .head(max(args.top_examples, 100))
        .copy()
    )
    longest.to_csv(
        longest_path,
        index=False,
        encoding="utf-8-sig",
    )

    token_stats = descriptive_statistics(audit["token_count"])
    word_stats = descriptive_statistics(audit["word_count"])

    limit_counts: dict[str, dict[str, float | int]] = {}
    for limit in sorted(set((*REFERENCE_LIMITS, tested_limit))):
        count = int((audit["token_count"] > limit).sum())
        limit_counts[str(limit)] = {
            "over_limit_count": count,
            "over_limit_percentage": round(
                100.0 * count / len(audit),
                2,
            ),
        }

    truncated_count = int(audit["would_be_truncated"].sum())

    summary = {
        "dataset": str(DATA_PATH),
        "model_name": model_name,
        "document_prefix": document_prefix,
        "tested_limit": tested_limit,
        "candidate_count": int(len(audit)),
        "truncated_count": truncated_count,
        "truncated_percentage": round(
            100.0 * truncated_count / len(audit),
            2,
        ),
        "complete_count": int(len(audit) - truncated_count),
        "complete_percentage": round(
            100.0 * (len(audit) - truncated_count) / len(audit),
            2,
        ),
        "token_statistics": token_stats,
        "word_statistics": word_stats,
        "reference_limit_counts": limit_counts,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("Token statistics:")
    for key, value in token_stats.items():
        print(f"  {key}: {value}")

    print("\nWord statistics:")
    for key, value in word_stats.items():
        print(f"  {key}: {value}")

    complete_count = len(audit) - truncated_count
    print(f"\nAt {tested_limit} tokens:")
    print(
        f"  Complete candidate_text: "
        f"{complete_count:,}/{len(audit):,} "
        f"({100.0 * complete_count / len(audit):.2f}%)"
    )
    print(
        f"  Would be truncated: "
        f"{truncated_count:,}/{len(audit):,} "
        f"({100.0 * truncated_count / len(audit):.2f}%)"
    )

    print("\nReference limits:")
    for limit, values in limit_counts.items():
        print(
            f"  Over {limit} tokens: "
            f"{values['over_limit_count']:,} "
            f"({values['over_limit_percentage']:.2f}%)"
        )

    print(
        f"\nTop {args.top_examples} longest candidate_text examples:"
    )
    print(
        longest[
            [
                "dataset_index",
                "position",
                "word_count",
                "token_count",
                "would_be_truncated",
                "tokens_removed",
            ]
        ]
        .head(args.top_examples)
        .to_string(index=False)
    )

    print("\nSaved:")
    print("  Full details:", detail_path)
    print("  Summary:", summary_path)
    print("  Longest candidates:", longest_path)
    print("=" * 80)


if __name__ == "__main__":
    main()