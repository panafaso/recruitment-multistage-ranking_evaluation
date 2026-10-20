"""ARIADNE — evaluate existing 100-query Top-20 rankings at @5, @10, @20.

Matches the evaluation conventions of the earlier 16-query project:
  broad:  relevance >= 1; strict: relevance >= 2
  P, Recall, F1, MAP, MRR, graded nDCG at every cutoff.

Additional explicitly defined BestMatch@K = Hit@K(label == 2):
  1 if at least one highly relevant CV appears within Top-K, else 0.

No retrieval, RRF, or reranking is performed. Unjudged candidates are NEVER
silently treated as irrelevant. Evaluation of a query/system/cutoff is skipped
unless all ranks 1..K exist and all K candidates are judged. Coverage is reported.

Installation: copy to <PROJECT_ROOT>/evaluation/evaluate_existing_100q.py
Inputs: <PROJECT_ROOT>/evaluation/qrels_100_queries.csv (FINAL reviewed qrels)
Run at project root: python evaluation/evaluate_existing_100q.py
Requirements: pandas, numpy
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
QRELS_PATH = EVAL / "qrels_100_queries.csv"
OUT = EVAL / "results_100q_all_cutoffs"
CUTOFFS = (5, 10, 20)

# CSV paths match the actual project structure shown in your VS Code screenshots.
SYSTEM_FILES = {
    "BM25Plus": ROOT / "sparse_retrieval" / "bm25plus_100_queries_top20.csv",
    "BGE-M3": ROOT / "dense_retrieval" / "bge_m3_100_queries_top20.csv",
    "Jina-v5": ROOT / "dense_retrieval" / "jina_v5_100_queries_top20.csv",
    "Nomic-v1.5": ROOT / "dense_retrieval" / "nomic_100_queries_top20.csv",
    "mxbai": ROOT / "dense_retrieval" / "mxbai_100_queries_top20.csv",
    "Hybrid-BM25-BGE": ROOT / "hybrid_retrieval" / "hybrid_bm25_bge_rrf_100_queries_top20.csv",
    "Hybrid-BM25-Jina": ROOT / "hybrid_retrieval" / "hybrid_bm25_jina_rrf_100_queries_top20.csv",
    "Hybrid-BM25-Nomic": ROOT / "hybrid_retrieval" / "hybrid_bm25_nomic_rrf_100_queries_top20.csv",
    "Hybrid-BM25-mxbai": ROOT / "hybrid_retrieval" / "hybrid_bm25_mxbai_rrf_100_queries_top20.csv",
}

# Common-query comparison is calculated across the available core systems.
# mxbai was NOT in the original 4-system annotation pool.
CORE_SYSTEMS = [name for name in SYSTEM_FILES if "mxbai" not in name.lower()]
THRESHOLDS = {"broad": 1, "strict": 2}
PER_METRICS = ["precision", "recall", "f1", "map", "mrr", "ndcg", "best_match_hit",
               "best_match_count", "top1_is_best_match", "best_label_at_k"]


def normalized_query(value: object) -> str:
    s = str(value).strip().upper()
    match = re.fullmatch(r"Q[_\-\s]*0*(\d+)", s)
    return f"Q{int(match.group(1)):03d}" if match else s


def normalized_candidate(value: object) -> str:
    s = str(value).strip()
    return s[:-2] if re.fullmatch(r"\d+\.0", s) else s


def read_qrels() -> pd.DataFrame:
    if not QRELS_PATH.is_file():
        raise FileNotFoundError(f"Missing qrels: {QRELS_PATH}. Use FINAL reviewed 5,949-label file.")
    q = pd.read_csv(QRELS_PATH, dtype={"query_id": str, "candidate_id": str}, encoding="utf-8-sig")
    required = {"query_id", "candidate_id", "relevance"}
    if required - set(q.columns):
        raise ValueError(f"Qrels missing columns: {sorted(required - set(q.columns))}")
    q = q[["query_id", "candidate_id", "relevance"]].copy()
    if q[["query_id", "candidate_id"]].isna().any().any():
        raise ValueError("Missing query_id or candidate_id in qrels")
    q["query_id"] = q["query_id"].map(normalized_query)
    q["candidate_id"] = q["candidate_id"].map(normalized_candidate)
    q["relevance"] = pd.to_numeric(q["relevance"], errors="coerce")
    if q["relevance"].isna().any() or not q["relevance"].isin([0, 1, 2]).all():
        raise ValueError("Qrels must have only completed labels 0, 1, or 2")
    if q.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query–candidate pairs in qrels")
    q["relevance"] = q["relevance"].astype(int)
    print(f"Qrels: {len(q)} judged pairs for {q.query_id.nunique()} queries")
    if len(q) != 5949 or q.query_id.nunique() != 100:
        print("WARNING: expected 5,949 pairs and 100 queries. Verify the qrels file.")
    return q


def read_ranking(name: str, path: Path, qrels: pd.DataFrame) -> pd.DataFrame | None:
    if not path.exists():
        print(f"SKIP {name}: missing {path.name}")
        return None
    d = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str}, encoding="utf-8-sig")
    required = {"query_id", "candidate_id", "rank"}
    if required - set(d.columns):
        raise ValueError(f"{name}: missing columns {sorted(required - set(d.columns))}")
    d = d[["query_id", "candidate_id", "rank"]].copy()
    if d[["query_id", "candidate_id", "rank"]].isna().any().any():
        raise ValueError(f"{name}: missing query ID, candidate ID or rank")
    d["query_id"] = d["query_id"].map(normalized_query)
    d["candidate_id"] = d["candidate_id"].map(normalized_candidate)
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    if d["rank"].isna().any() or (d["rank"] < 1).any() or not (d["rank"] % 1 == 0).all():
        raise ValueError(f"{name}: ranks must be positive integers")
    d["rank"] = d["rank"].astype(int)
    d = d[d["rank"] <= max(CUTOFFS)].copy()
    if d.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError(f"{name}: duplicate candidate in a query")
    if d.duplicated(["query_id", "rank"]).any():
        raise ValueError(f"{name}: duplicate rank in a query")
    d = d.merge(qrels, on=["query_id", "candidate_id"], how="left", validate="many_to_one")
    d.insert(0, "system", name)
    return d.sort_values(["query_id", "rank"]).reset_index(drop=True)


def dcg(labels: np.ndarray) -> float:
    discounts = np.log2(np.arange(2, len(labels) + 2))
    return float(np.sum((np.power(2.0, labels) - 1.0) / discounts))


def calc_one(labels: np.ndarray, all_qrels: np.ndarray, k: int, threshold: int) -> dict:
    """Exactly reproduces old 16-query broad/strict P,Recall,F1,MAP,MRR,nDCG convention.

    A zero-positive query contributes 0 for AP/Recall/nDCG like the original code.
    nDCG is GRADED (0,1,2) and therefore identical under both binary thresholds.
    """
    assert len(labels) == k
    hits = (labels >= threshold).astype(int)
    count_relevant = int((all_qrels >= threshold).sum())
    precision = float(hits.mean())
    recall = float(hits.sum() / count_relevant) if count_relevant else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    hit_positions = np.flatnonzero(hits)
    rr = float(1.0 / (hit_positions[0] + 1)) if len(hit_positions) else 0.0
    ap = (float((np.cumsum(hits) / np.arange(1, k + 1) * hits).sum()
                / min(k, count_relevant)) if count_relevant else 0.0)
    ideal = np.sort(all_qrels)[::-1][:k]
    idcg = dcg(ideal)
    ndcg = dcg(labels) / idcg if idcg else 0.0
    best_positions = np.flatnonzero(labels == 2)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": ap,  # per-query AP@K; becomes MAP@K after mean across queries
        "mrr": rr,  # per-query RR@K; becomes MRR@K after mean across queries
        "ndcg": float(ndcg),
        "best_match_hit": int(len(best_positions) > 0),  # BestMatch@K = Hit@K(label=2)
        "best_match_count": int(len(best_positions)),
        "first_best_match_rank": int(best_positions[0] + 1) if len(best_positions) else np.nan,
        "top1_is_best_match": int(labels[0] == 2),
        "best_label_at_k": int(labels.max()),
        "total_relevant_in_pool": count_relevant,
        "eligible_best_match_in_pool": int(np.any(all_qrels == 2)),
    }


def make_wide(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cols = ["ndcg", "precision", "recall", "f1", "map", "mrr", "best_match_hit",
            "best_match_hit_among_eligible", "queries_evaluated"]
    base = summary.pivot(index=["system", "threshold"], columns="k", values=cols)
    base.columns = [f"{'BestMatch' if m == 'best_match_hit' else m.upper() if m in {'map','mrr'} else m}@{k}"
                    for m, k in base.columns.to_flat_index()]
    return base.reset_index()


def create_summary(per: pd.DataFrame) -> pd.DataFrame:
    if per.empty:
        return pd.DataFrame()
    summary = (per.groupby(["system", "threshold", "relevance_threshold", "k"], as_index=False)
               .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                    f1=("f1", "mean"), map=("map", "mean"), mrr=("mrr", "mean"),
                    ndcg=("ndcg", "mean"), best_match_hit=("best_match_hit", "mean"),
                    top1_best_match_rate=("top1_is_best_match", "mean"),
                    queries_evaluated=("query_id", "nunique"),
                    queries_with_pool_best_match=("eligible_best_match_in_pool", "sum")))
    elig = (per[per["eligible_best_match_in_pool"] == 1]
            .groupby(["system", "threshold", "relevance_threshold", "k"])["best_match_hit"].mean()
            .rename("best_match_hit_among_eligible"))
    summary = summary.merge(elig.reset_index(), on=["system", "threshold", "relevance_threshold", "k"], how="left")
    return summary.sort_values(["threshold", "k", "ndcg", "map"], ascending=[True, True, False, False])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qrels = read_qrels()
    by_q = {query: group["relevance"].to_numpy(dtype=int)
            for query, group in qrels.groupby("query_id")}
    all_queries = set(by_q)
    per_query_rows: list[dict] = []
    coverage_rows: list[dict] = []
    missing_parts: list[pd.DataFrame] = []
    labeled_parts: list[pd.DataFrame] = []
    present = []

    for system, path in SYSTEM_FILES.items():
        d = read_ranking(system, path, qrels)
        if d is None:
            continue
        present.append(system)
        labeled_parts.append(d)
        extra = set(d["query_id"]) - all_queries
        if extra:
            print(f"WARNING {system}: {len(extra)} query IDs absent from qrels")
        unjudged = d[d["relevance"].isna()]
        if not unjudged.empty:
            missing_parts.append(unjudged[["system", "query_id", "candidate_id", "rank"]])
        for k in CUTOFFS:
            valid_queries = 0
            unjudged_queries = 0
            incomplete_queries = 0
            judged_rows = 0
            for qid in sorted(all_queries):
                block = d[d["query_id"] == qid]
                top = block[block["rank"] <= k]
                judged_rows += int(top["relevance"].notna().sum())
                if top["relevance"].isna().any():
                    unjudged_queries += 1
                # Evaluate only a complete contiguous ranking 1, 2, ..., K.
                if sorted(top["rank"].tolist()) != list(range(1, k + 1)):
                    incomplete_queries += 1
                    continue
                if top["relevance"].isna().any():
                    continue
                valid_queries += 1
                labels = top.sort_values("rank")["relevance"].to_numpy(dtype=int)
                for threshold_name, threshold in THRESHOLDS.items():
                    scores = calc_one(labels, by_q[qid], k, threshold)
                    per_query_rows.append({
                        "system": system, "query_id": qid, "threshold": threshold_name,
                        "relevance_threshold": threshold, "k": k,
                        **scores,
                        "best_match_candidate_id": (
                            top.sort_values("rank").iloc[int(np.flatnonzero(labels == 2)[0])]["candidate_id"]
                            if np.any(labels == 2) else ""
                        ),
                    })
            coverage_rows.append({
                "system": system, "k": k, "queries_in_qrels": len(all_queries),
                "queries_fully_judged_and_complete": valid_queries,
                "queries_with_unjudged": unjudged_queries,
                "queries_with_missing_ranks": incomplete_queries,
                "judged_ranked_rows_within_k": judged_rows,
                "total_ranked_rows_within_k": int((d["rank"] <= k).sum()),
            })
        print(f"Read {system}: {len(d)} Top-20 rows, {len(unjudged)} unjudged pairs")

    if not present:
        raise FileNotFoundError("No ranking CSV found. Verify SYSTEM_FILES and your project folders.")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "coverage_by_system_and_cutoff.csv", index=False, encoding="utf-8-sig")
    missing = pd.concat(missing_parts, ignore_index=True) if missing_parts else pd.DataFrame(
        columns=["system", "query_id", "candidate_id", "rank"])
    missing.to_csv(OUT / "unjudged_ranked_pairs.csv", index=False, encoding="utf-8-sig")
    missing[["query_id", "candidate_id"]].drop_duplicates().to_csv(
        OUT / "unique_pairs_to_annotate.csv", index=False, encoding="utf-8-sig")
    pd.concat(labeled_parts, ignore_index=True).to_csv(
        OUT / "rankings_with_qrels_labels.csv", index=False, encoding="utf-8-sig")

    per = pd.DataFrame(per_query_rows)
    if per.empty:
        print("No fully judged query/cutoff combinations: check missing labels and rankings.")
        return
    per.to_csv(OUT / "metrics_per_query_all_cutoffs.csv", index=False, encoding="utf-8-sig")
    summary = create_summary(per)
    summary.to_csv(OUT / "metrics_summary_all_cutoffs.csv", index=False, encoding="utf-8-sig")
    summary[summary["threshold"] == "broad"].to_csv(
        OUT / "metrics_broad_relevance.csv", index=False, encoding="utf-8-sig")
    summary[summary["threshold"] == "strict"].to_csv(
        OUT / "metrics_strict_relevance.csv", index=False, encoding="utf-8-sig")
    make_wide(summary).to_csv(OUT / "metrics_wide_5_10_20.csv", index=False, encoding="utf-8-sig")
    per[per["threshold"] == "strict"][
        ["system", "query_id", "k", "best_match_hit", "best_match_count",
         "first_best_match_rank", "best_match_candidate_id", "top1_is_best_match",
         "eligible_best_match_in_pool"]].to_csv(
        OUT / "best_match_per_query.csv", index=False, encoding="utf-8-sig")

    # Apples-to-apples comparison: for each K, retain only queries fully judged
    # in all AVAILABLE core systems. Reports may differ in query count across K.
    comparable = []
    core = [s for s in CORE_SYSTEMS if s in present]
    for k in CUTOFFS:
        per_k = per[per["k"] == k]
        by_system = {s: set(per_k[per_k["system"] == s]["query_id"]) for s in core}
        common = set.intersection(*by_system.values()) if by_system else set()
        if common:
            comparable.append(per_k[per_k["query_id"].isin(common) &
                                    per_k["system"].isin(core)])
        print(f"@{k}: same fully judged queries across {len(core)} core systems: {len(common)}/100")
    comparable_per = pd.concat(comparable, ignore_index=True) if comparable else pd.DataFrame()
    comparable_per.to_csv(OUT / "metrics_comparable_per_query.csv", index=False, encoding="utf-8-sig")
    compare = create_summary(comparable_per)
    compare.to_csv(OUT / "metrics_comparable_same_queries.csv", index=False, encoding="utf-8-sig")
    make_wide(compare).to_csv(OUT / "metrics_comparable_wide_5_10_20.csv", index=False,
                              encoding="utf-8-sig")

    print("\n=== TOP-5, TOP-10, TOP-20 — ALL SYSTEMS (per-system judged queries) ===")
    print(summary[["system", "threshold", "k", "queries_evaluated", "ndcg", "map", "mrr",
                   "precision", "best_match_hit"]].round(4).to_string(index=False))
    print("\nBestMatch@K = fraction of evaluated queries with at least one label-2 CV in Top-K.")
    print("'strict' uses labels == 2 for P/Recall/F1/MAP/MRR; graded nDCG uses 0/1/2 in both.")
    print(f"\nSaved all files to: {OUT.resolve()}")
    if not missing.empty:
        print(f"WARNING: {len(missing[['query_id','candidate_id']].drop_duplicates())} unique unjudged pairs."
              " Those query/cutoffs were EXCLUDED, not set to relevance 0.")
    print("NOTE: Judgments are AI-assisted; independent validation is recommended.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
