from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Four selected standalone Top-20 rankings (MXBAI excluded).
RANKING_FILES = {
    "bm25plus_stopwords": PROJECT_ROOT / "sparse_retrieval" / "bm25plus_100_queries_top20.csv",
    "bge": PROJECT_ROOT / "dense_retrieval" / "bge_m3_100_queries_top20.csv",
    "jina": PROJECT_ROOT / "dense_retrieval" / "jina_v5_100_queries_top20.csv",
    "nomic": PROJECT_ROOT / "dense_retrieval" / "nomic_100_queries_top20.csv",
}
DATASET_PATH = PROJECT_ROOT / "data" / "candidate_profiles_english_clean_20k_with_text.parquet"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "annotation_pool_100_queries_top20_4systems_for_annotation.csv"
SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "annotation_pool_100_queries_top20_4systems_summary.csv"
# Check both selected hybrid outputs are covered by the standalone pool.
# Since hybrids fuse standalone Top-20 inputs, they cannot introduce new candidates.
HYBRID_FILES = {
    "bm25_nomic": PROJECT_ROOT / "hybrid_retrieval" / "hybrid_bm25_nomic_rrf_100_queries_top20.csv",
    "bm25_jina": PROJECT_ROOT / "hybrid_retrieval" / "hybrid_bm25_jina_rrf_100_queries_top20.csv",
}
EXPECTED_QUERY_IDS = [f"Q{i:03d}" for i in range(1, 101)]
EXPECTED_TOP_K = 20
CANDIDATE_COLUMNS = [
    "Position", "Moreinfo", "Looking For", "Highlights", "English Level",
    "Experience Years", "CV", "candidate_text",
]


def clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def candidate_profile(row: pd.Series) -> str:
    sections = []
    for label in CANDIDATE_COLUMNS[:-1]:  # candidate_text is retained separately
        value = clean_value(row.get(label))
        if value:
            sections.append(f"{label}:\n{value}")
    return "\n\n".join(sections)


def load_ranking(system: str, path: Path, expected_queries: dict[str, tuple[str, str]]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {system} ranking: {path}")
    ranking = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str})
    required = {"query_id", "category", "query", "candidate_id", "dataset_index", "rank"}
    if required - set(ranking.columns):
        raise ValueError(f"{system}: missing columns {sorted(required - set(ranking.columns))}")
    if ranking[list(required)].isna().any().any():
        raise ValueError(f"{system}: required columns contain missing values")
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="raise").astype(int)
    ranking["dataset_index"] = pd.to_numeric(ranking["dataset_index"], errors="raise").astype(int)
    ranking["candidate_id"] = ranking["candidate_id"].astype(str)
    if ranking.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError(f"{system}: duplicate query/candidate pairs")
    if ranking.duplicated(["query_id", "rank"]).any():
        raise ValueError(f"{system}: duplicate rank values")
    if len(ranking) != len(EXPECTED_QUERY_IDS) * EXPECTED_TOP_K:
        raise ValueError(f"{system}: expected {len(EXPECTED_QUERY_IDS) * EXPECTED_TOP_K} rows; got {len(ranking)}")
    if set(ranking["query_id"]) != set(EXPECTED_QUERY_IDS):
        raise ValueError(f"{system}: query IDs do not match Q001 through Q100")
    for qid, group in ranking.groupby("query_id", sort=False):
        if sorted(group["rank"].tolist()) != list(range(1, EXPECTED_TOP_K + 1)):
            raise ValueError(f"{system}: {qid} does not contain ranks 1..{EXPECTED_TOP_K}")
        metadata = group[["category", "query"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"{system}: inconsistent query text/category for {qid}")
        row = metadata.iloc[0]
        if (str(row["category"]), str(row["query"])) != expected_queries[qid]:
            raise ValueError(f"{system}: mismatched query text/category for {qid}")
    if (ranking["candidate_id"] != ranking["dataset_index"].astype(str)).any():
        raise ValueError(f"{system}: candidate_id and dataset_index differ; do not pool incompatible corpora")
    if not ranking["dataset_index"].between(0, 19999).all():
        raise ValueError(f"{system}: out-of-range dataset index")
    ranking = ranking[["query_id", "category", "query", "candidate_id", "dataset_index", "rank"]].copy()
    ranking["system"] = system
    print(f"  {system:<20}: {len(ranking):,} rows / {ranking['query_id'].nunique()} queries")
    return ranking


def verify_hybrid_coverage(pool: pd.DataFrame) -> None:
    """Ensure the two hybrid Top-20 rankings use only candidates in this pool."""
    pool_pairs = set(zip(pool["query_id"], pool["candidate_id"]))
    for label, path in HYBRID_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Hybrid file missing ({label}): {path}\n"
                "Run the two hybrid Top-20 scripts before creating the pool."
            )
        hybrid = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str})
        required = {"query_id", "candidate_id", "rank"}
        if not required.issubset(hybrid.columns):
            raise ValueError(f"{label}: missing hybrid columns {required - set(hybrid.columns)}")
        if (len(hybrid) != 100 * EXPECTED_TOP_K
                or set(hybrid["query_id"]) != set(EXPECTED_QUERY_IDS)):
            raise ValueError(f"{label}: expected 2,000 hybrid results across Q001-Q100")
        if hybrid.duplicated(["query_id", "candidate_id"]).any():
            raise ValueError(f"{label}: duplicate query/candidate pairs")
        for qid, group in hybrid.groupby("query_id"):
            if sorted(group["rank"].astype(int).tolist()) != list(range(1, EXPECTED_TOP_K + 1)):
                raise ValueError(f"{label}: invalid Top-20 ranks for {qid}")
        hybrid_pairs = set(zip(hybrid["query_id"], hybrid["candidate_id"]))
        missing = hybrid_pairs - pool_pairs
        if missing:
            example = sorted(missing)[:5]
            raise ValueError(
                f"{label}: {len(missing)} hybrid query/candidate pairs are absent from "
                f"the standalone pool (examples: {example}). Check input rankings "
                "and whether the hybrid used Top-50 instead of Top-20."
            )
        print(f"  Hybrid {label}: all {len(hybrid_pairs):,} pairs covered by pool")


def main() -> None:
    from queries_100 import load_queries

    queries = load_queries()
    expected = {item["query_id"]: (item["category"], item["query"]) for item in queries}
    if len(queries) != 100 or list(expected) != EXPECTED_QUERY_IDS:
        raise ValueError("queries_100.py must contain exactly Q001..Q100 in order")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing corpus: {DATASET_PATH}")
    candidates = pd.read_parquet(DATASET_PATH).reset_index(drop=True)
    if len(candidates) != 20000:
        raise ValueError(f"Expected 20000 profiles; got {len(candidates)}")
    missing_cols = set(CANDIDATE_COLUMNS) - set(candidates.columns)
    if missing_cols:
        raise ValueError(f"Missing corpus columns: {sorted(missing_cols)}")
    candidates["dataset_index"] = candidates.index
    candidates["candidate_id"] = candidates.index.astype(str)
    candidates["candidate_profile"] = candidates.apply(candidate_profile, axis=1)
    if candidates["candidate_profile"].str.strip().eq("").any():
        raise ValueError("Some candidate profiles are empty")

    print("Loading standalone Top-20 rankings:")
    frames = [load_ranking(name, path, expected) for name, path in RANKING_FILES.items()]
    all_rankings = pd.concat(frames, ignore_index=True)

    # Deduplicate only by (query_id, candidate_id): a candidate may be relevant to several queries.
    rank_table = all_rankings.pivot(
        index=["query_id", "candidate_id"], columns="system", values="rank"
    ).rename(columns={name: f"{name}_rank" for name in RANKING_FILES}).reset_index()
    rank_cols = [f"{name}_rank" for name in RANKING_FILES]

    provenance = all_rankings.groupby(["query_id", "candidate_id"], sort=False)["system"].agg(
        lambda names: ";".join(sorted(set(names)))
    ).reset_index(name="retrieved_by_systems")
    provenance["retrieval_count"] = provenance["retrieved_by_systems"].str.count(";") + 1

    query_info = pd.DataFrame(queries).rename(columns={"query": "query"})
    pool = rank_table.merge(query_info, on="query_id", validate="many_to_one")
    pool = pool.merge(provenance, on=["query_id", "candidate_id"], validate="one_to_one")
    pool = pool.merge(
        candidates[["candidate_id", "dataset_index", "candidate_profile", *CANDIDATE_COLUMNS]],
        on="candidate_id", validate="many_to_one", how="left"
    )
    if pool["dataset_index"].isna().any():
        raise ValueError("Unknown candidate_id in pool")
    pool["best_retrieval_rank"] = pool[rank_cols].min(axis=1, skipna=True)
    pool = pool.sort_values(["query_id", "best_retrieval_rank", "dataset_index"]).reset_index(drop=True)
    pool["relevance"] = ""  # Leave blank until an actual relevance assessment is completed.
    pool["annotation_notes"] = ""
    pool = pool[[
        "query_id", "category", "query", "candidate_id", "dataset_index", "candidate_profile",
        *CANDIDATE_COLUMNS, "retrieved_by_systems", "retrieval_count", *rank_cols,
        "relevance", "annotation_notes",
    ]]
    if pool.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate pairs in final pool")
    if pool["query_id"].nunique() != len(EXPECTED_QUERY_IDS):
        raise ValueError("Not all 100 queries occur in the annotation pool")
    max_possible = len(EXPECTED_QUERY_IDS) * EXPECTED_TOP_K * len(RANKING_FILES)
    if not (len(EXPECTED_QUERY_IDS) * EXPECTED_TOP_K <= len(pool) <= max_possible):
        raise ValueError(f"Unexpected pool size: {len(pool)}")
    verify_hybrid_coverage(pool)
    summary = pool.groupby(["query_id", "category"], as_index=False).agg(
        unique_candidates=("candidate_id", "size"),
        appearing_in_all_four=("retrieval_count", lambda s: int((s == 4).sum()))
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"\nUnique query/candidate pairs: {len(pool):,}")
    print(f"Pool per query: min={summary.unique_candidates.min()}, max={summary.unique_candidates.max()}")
    print(f"Annotation file: {OUTPUT_PATH}")
    print(f"Summary file:    {SUMMARY_PATH}")
    print("Relevance labels are BLANK. Do not calculate nDCG/MAP/MRR until judgments are completed.")


if __name__ == "__main__":
    main()
