"""ARIADNE: common 100-query evaluation for BGE, Llama-Nemotron, GTE.

Inputs (already computed): Jina-v5, Hybrid-BM25-Jina and the six Top-20 reranked
rankings. No models, embeddings or GPUs needed.

Run from the project root:
    python evaluation/evaluate_3_rerankers_100q.py
or:
    python evaluation/evaluate_3_rerankers_100q.py --qrels evaluation/qrels_100_queries.csv

Metrics at K=5,10,20: graded nDCG; broad (rel>=1) and strict (rel==2)
Precision, pooled Recall, F1, AP/MAP, RR/MRR; BestMatch/Hit@K (rel==2).
Judgment-pool R and ideal gains are based on the complete qrels for each query;
pooled Recall is NOT corpus-level recall over all 20,000 CVs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K_VALUES = (5, 10, 20)
THRESHOLDS = {"broad": 1, "strict": 2}

# Each reranker gets EXACTLY the same 20 candidates as the source retriever.
SOURCES = {
    "Jina-v5": ("jina_v5", Path("dense_retrieval/jina_v5_100_queries_top20.csv")),
    "Hybrid-BM25-Jina": ("hybrid_bm25_jina", Path("hybrid_retrieval/hybrid_bm25_jina_rrf_100_queries_top20.csv")),
}
MODELS = {
    "BGE-reranker-v2-m3": "bge_reranker_v2_m3",
    "Llama-Nemotron-1B-v2": "llama_nemotron_rerank_1b_v2",
    "GTE-ModernColBERT-v1": "gte_moderncolbert_v1",
}
RERANKED_FOLDER = Path("reranking/outputs_100q_jina_vs_hybrid")


def system_paths(root: Path) -> dict[str, Path]:
    paths = {name: root / relative for name, (_, relative) in SOURCES.items()}
    for model_display, model_id in MODELS.items():
        for source_display, (source_id, _) in SOURCES.items():
            fname = f"{source_id}__{model_id}_rankings.csv"
            paths[f"{source_display} + {model_display}"] = (
                root / RERANKED_FOLDER / model_id / fname
            )
    return paths


def parent_of(system: str) -> str | None:
    for source in SOURCES:
        if system.startswith(f"{source} + "):
            return source
    return None


def normalize_query(value: object) -> str:
    value = str(value).strip().upper()
    match = re.fullmatch(r"Q[_\-\s]*0*(\d+)", value)
    return f"Q{int(match.group(1)):03d}" if match else value


def normalize_candidate(value: object) -> str:
    value = str(value).strip()
    return value[:-2] if re.fullmatch(r"\d+\.0", value) else value


def pick_qrels(root: Path, specified: str | None) -> Path:
    if specified is not None:
        path = Path(specified)
        return path if path.is_absolute() else root / path
    options = [
        root / "evaluation/qrels_100_queries.csv",
        root / "evaluation/qrels_100_queries_AI_ASSISTED_5949.csv",
        root / "qrels_100_queries_AI_ASSISTED_5949.csv",
    ]
    found = [x for x in options if x.is_file()]
    if not found:
        raise FileNotFoundError("No qrels found. Put final 5,949-label CSV in evaluation/qrels_100_queries.csv or supply --qrels.")
    if len(found) > 1:
        print(f"NOTE: multiple qrels found; using {found[0]} (verify it is the final version).")
    return found[0]


def load_qrels(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str}, encoding="utf-8-sig")
    required = {"query_id", "candidate_id", "relevance"}
    if missing := (required - set(q.columns)):
        raise ValueError(f"Qrels missing required columns: {sorted(missing)}")
    q = q[["query_id", "candidate_id", "relevance"]].copy()
    if q[["query_id", "candidate_id", "relevance"]].isna().any().any():
        raise ValueError("Qrels contain blank query_id / candidate_id / relevance")
    q["query_id"] = q["query_id"].map(normalize_query)
    q["candidate_id"] = q["candidate_id"].map(normalize_candidate)
    q["relevance"] = pd.to_numeric(q["relevance"], errors="coerce")
    if q["relevance"].isna().any() or not q["relevance"].isin([0, 1, 2]).all():
        raise ValueError("Qrels must use ONLY 0/1/2 relevance labels")
    if q.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query–candidate pair in qrels")
    q["relevance"] = q["relevance"].astype(int)
    if len(q) != 5949 or q["query_id"].nunique() != 100:
        raise ValueError(f"Expected 5,949 pairs and 100 queries; got {len(q)} pairs and {q['query_id'].nunique()} queries")
    return q


def read_ranking(name: str, path: Path, qrels: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str}, encoding="utf-8-sig")
    required = {"query_id", "candidate_id", "rank"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"{name}: missing columns {sorted(missing)}: {path}")
    df = df[["query_id", "candidate_id", "rank"]].copy()
    if df.isna().any().any():
        raise ValueError(f"{name}: blank query_id/candidate_id/rank")
    df["query_id"] = df["query_id"].map(normalize_query)
    df["candidate_id"] = df["candidate_id"].map(normalize_candidate)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if df["rank"].isna().any() or not df["rank"].isin(range(1, 21)).all():
        raise ValueError(f"{name}: rank must be an integer from 1 to 20")
    df["rank"] = df["rank"].astype(int)
    if df.duplicated(["query_id", "rank"]).any():
        raise ValueError(f"{name}: duplicate query–rank pairs")
    if df.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError(f"{name}: duplicate query–candidate pairs")
    expected_queries = set(qrels["query_id"])
    observed_queries = set(df["query_id"])
    if observed_queries != expected_queries:
        raise ValueError(f"{name}: unexpected query IDs. Missing={sorted(expected_queries-observed_queries)}, extra={sorted(observed_queries-expected_queries)}")
    sizes = df.groupby("query_id").size()
    if len(df) != 2000 or (sizes != 20).any():
        problem = sizes[sizes != 20].head(10).to_dict()
        raise ValueError(f"{name}: expected 100 queries x 20 = 2,000 rows; got {len(df)}. Per-query issues: {problem}")
    # Since all ranks are 1..20, have 20 rows, and duplicates are forbidden,
    # the rank sequence in each query is guaranteed to be exactly 1,...,20.
    df = df.merge(qrels, on=["query_id", "candidate_id"], how="left", validate="many_to_one")
    df.insert(0, "system", name)
    return df.sort_values(["query_id", "rank"]).reset_index(drop=True)


def dcg(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.sum((np.power(2.0, labels)-1.0) / np.log2(np.arange(2, len(labels)+2))))


def calculate(labels: np.ndarray, all_judgments: np.ndarray, k: int, threshold: int) -> dict[str, float | int]:
    hits = (labels >= threshold).astype(int)
    relevant_in_pool = int((all_judgments >= threshold).sum())
    precision = float(hits.sum()/k)
    # Pool-restricted recall; may under-estimate absolute corpus recall.
    recall = float(hits.sum()/relevant_in_pool) if relevant_in_pool else 0.0
    f1 = float(2*precision*recall/(precision+recall)) if precision+recall else 0.0
    hit_idx = np.flatnonzero(hits)
    rr = float(1/(hit_idx[0]+1)) if len(hit_idx) else 0.0
    ranks = np.arange(1, k+1)
    ap = float((((np.cumsum(hits)/ranks)*hits).sum())/min(k, relevant_in_pool)) if relevant_in_pool else 0.0
    ideal = np.sort(all_judgments)[::-1][:k]
    idcg = dcg(ideal)
    ndcg = float(dcg(labels)/idcg) if idcg else 0.0
    best = np.flatnonzero(labels == 2)
    return {
        "precision": precision, "recall_pooled": recall, "f1_pooled": f1,
        "ap": ap, "rr": rr, "ndcg": ndcg,
        "best_match_hit": int(bool(len(best))),
        "best_match_count": int(len(best)),
        "first_best_match_rank": int(best[0]+1) if len(best) else np.nan,
        "top1_best_match": int(labels[0] == 2),
        "eligible_for_best_match": int(np.any(all_judgments == 2)),
        "pool_relevant_count": relevant_in_pool,
    }


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    summary = per.groupby(["system", "threshold", "k"], as_index=False).agg(
        queries_evaluated=("query_id", "nunique"),
        ndcg=("ndcg", "mean"), precision=("precision", "mean"),
        recall_pooled=("recall_pooled", "mean"), f1_pooled=("f1_pooled", "mean"),
        map=("ap", "mean"), mrr=("rr", "mean"),
        best_match_hit=("best_match_hit", "mean"),
        top1_best_match_rate=("top1_best_match", "mean"),
        eligible_queries=("eligible_for_best_match", "sum"),
    )
    eligible = (per[per["eligible_for_best_match"] == 1]
                .groupby(["system", "threshold", "k"], as_index=False)
                .agg(best_match_hit_eligible_only=("best_match_hit", "mean")))
    summary = summary.merge(eligible, on=["system", "threshold", "k"], how="left")
    return summary


def make_wide(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["ndcg", "precision", "recall_pooled", "f1_pooled", "map", "mrr", "best_match_hit", "best_match_hit_eligible_only"]
    wide = summary.pivot(index=["system", "threshold"], columns="k", values=cols)
    wide.columns = [f"{metric.upper() if metric in ('map','mrr') else metric}@{k}" for metric,k in wide.columns.to_flat_index()]
    return wide.reset_index()


def deltas(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.set_index(["system", "threshold", "k"])
    metrics = ["ndcg", "precision", "recall_pooled", "f1_pooled", "map", "mrr", "best_match_hit"]
    rows = []
    for sys_name in summary["system"].unique():
        source = parent_of(sys_name)
        if source is None:
            continue
        for threshold in THRESHOLDS:
            for k in K_VALUES:
                mine = idx.loc[(sys_name, threshold, k)]
                base = idx.loc[(source, threshold, k)]
                row = {"source": source, "reranked_system": sys_name,
                       "threshold": threshold, "k": k}
                for metric in metrics:
                    row[f"baseline_{metric}"] = float(base[metric])
                    row[f"reranked_{metric}"] = float(mine[metric])
                    row[f"delta_{metric}"] = float(mine[metric]-base[metric])
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--qrels", type=str, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dest = root / "evaluation" / "results_100q_3_rerankers"
    qrels_path = pick_qrels(root, args.qrels)
    paths = system_paths(root)
    print(f"Project: {root}\nQrels: {qrels_path}\nOutput: {dest}")
    missing_files = {name: path for name,path in paths.items() if not path.is_file()}
    if missing_files:
        lines = "\n".join(f"  {name}: {p}" for name,p in missing_files.items())
        raise FileNotFoundError(f"Missing {len(missing_files)} of the 8 mandatory ranking CSVs:\n{lines}\nComplete the reranking runs first.")
    qrels = load_qrels(qrels_path)
    print(f"Qrels loaded: {len(qrels)} judged pairs, {qrels['query_id'].nunique()} queries")
    rankings: dict[str, pd.DataFrame] = {}
    for name,path in paths.items():
        rankings[name] = read_ranking(name,path,qrels)
        unjudged = int(rankings[name]["relevance"].isna().sum())
        print(f"  {name}: {len(rankings[name])} rows; unjudged={unjudged}")
    # Validate that reranking reorders rather than changes candidates (critical for fair comparison).
    for name,df in rankings.items():
        parent = parent_of(name)
        if parent is None:
            continue
        grouped = df.groupby("query_id")["candidate_id"].apply(lambda s: set(s))
        baseline = rankings[parent].groupby("query_id")["candidate_id"].apply(lambda s: set(s))
        mismatch = [qid for qid in grouped.index if grouped.loc[qid] != baseline.loc[qid]]
        if mismatch:
            raise ValueError(f"{name}: candidate pool differs from its {parent} input in {len(mismatch)} queries; examples: {mismatch[:8]}")
    dest.mkdir(parents=True, exist_ok=True)
    all_rankings = pd.concat(rankings.values(), ignore_index=True)
    unjudged = all_rankings[all_rankings["relevance"].isna()].copy()
    coverage = (all_rankings.groupby("system", as_index=False)
                .agg(queries=("query_id", "nunique"), rows=("candidate_id", "size"),
                     judged_rows=("relevance", "count")))
    coverage["unjudged_rows"] = coverage["rows"] - coverage["judged_rows"]
    coverage["coverage_percent"] = coverage["judged_rows"] / coverage["rows"] * 100
    coverage.to_csv(dest/"coverage_by_system.csv", index=False, encoding="utf-8-sig")
    unjudged[["system", "query_id", "candidate_id", "rank"]].to_csv(dest/"unjudged_pairs.csv", index=False, encoding="utf-8-sig")
    if not unjudged.empty:
        raise ValueError(f"There are {len(unjudged)} unjudged rankings. Review {dest/'unjudged_pairs.csv'}. No metrics produced; missing labels were NOT treated as zero.")

    relevance_by_query = {qid: g["relevance"].to_numpy(dtype=int) for qid,g in qrels.groupby("query_id")}
    rows = []
    for name, df in rankings.items():
        for qid, g in df.groupby("query_id", sort=False):
            g = g.sort_values("rank")
            scores = g["relevance"].to_numpy(dtype=int)
            ids = g["candidate_id"].to_numpy(dtype=str)
            for k in K_VALUES:
                subset = scores[:k]
                best = np.flatnonzero(subset == 2)
                for th,minimum in THRESHOLDS.items():
                    metric = calculate(subset, relevance_by_query[qid], k, minimum)
                    rows.append({"system": name, "query_id": qid, "threshold": th,
                                 "relevance_threshold": minimum, "k": k,
                                 "best_match_candidate_id": ids[best[0]] if len(best) else "",
                                 **metric})
    per = pd.DataFrame(rows)
    summary = summarize(per)
    system_order = list(paths)
    summary["system"] = pd.Categorical(summary["system"], categories=system_order, ordered=True)
    summary = summary.sort_values(["threshold", "k", "system"]).copy()
    summary["system"] = summary["system"].astype(str)
    wide = make_wide(summary)
    wide["system"] = pd.Categorical(wide["system"], categories=system_order, ordered=True)
    wide = wide.sort_values(["threshold", "system"])
    wide["system"] = wide["system"].astype(str)
    delta = deltas(summary)
    per.to_csv(dest/"metrics_per_query_5_10_20.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(dest/"metrics_all_systems_long.csv",index=False,encoding="utf-8-sig")
    wide.to_csv(dest/"metrics_all_8_systems_wide.csv",index=False,encoding="utf-8-sig")
    summary[summary["threshold"] == "broad"].to_csv(dest/"metrics_broad_relevance.csv",index=False,encoding="utf-8-sig")
    summary[summary["threshold"] == "strict"].to_csv(dest/"metrics_strict_relevance.csv",index=False,encoding="utf-8-sig")
    per[per["threshold"] == "strict"][["system","query_id","k","best_match_hit","best_match_count","first_best_match_rank","best_match_candidate_id","eligible_for_best_match","top1_best_match"]].to_csv(
        dest/"best_match_per_query.csv",index=False,encoding="utf-8-sig")
    delta.to_csv(dest/"delta_vs_own_baseline.csv",index=False,encoding="utf-8-sig")
    print("\n=== STRICT (label=2): nDCG, MAP, MRR, BestMatch for all 8 systems ===")
    print(summary[summary["threshold"] == "strict"][["system", "k", "ndcg", "map", "mrr", "precision", "best_match_hit"]].round(4).to_string(index=False))
    print(f"\nSUCCESS: 8 systems, 100 queries, Top-20. Results saved: {dest}")
    print("BestMatch@K = Hit@K(label 2), reported both over 100 queries and only queries containing label 2 in the judged pool.")
    print("Recall@K and nDCG ideal use the annotated pool, not the unjudged full corpus; judgments are AI-assisted.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
