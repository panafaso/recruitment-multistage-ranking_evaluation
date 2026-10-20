#!/usr/bin/env python3
"""ARIADNE — reproducible evaluation of four SLM PRP-Graph ranking runs.

No GPU or external Python packages required (Python >=3.9).

Run from project root:
    python evaluation/slm_100q/evaluate_slm_100q.py
or directly:
    python evaluate_slm_100q.py

Metrics exactly follow the earlier 100-query evaluator conventions:
- broad relevance: grade >= 1; strict: grade == 2;
- macro Precision, pooled Recall, per-query F1, MAP@K, MRR@K;
- graded nDCG using gain 2**grade - 1 and ideal over full judged pool;
- BestMatch@K: proportion of queries with a grade-2 result among first K;
- zero-relevant queries contribute zero where appropriate;
- MAP@K denominator is min(K, count relevant in full judged pool).

Recall/ideal computed on *annotated pool*, NOT on 20k full corpus.
Rankings with missing judgments abort; missing labels are NEVER treated as zero.
"""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CUTOFFS = (5, 10, 20)
THRESHOLDS = {"broad": 1, "strict": 2}
EXPECTED_Q = 100
EXPECTED_K = 20
EXPECTED_QRELS = 5949

RANKING_FILES = {
    "Hybrid + Ministral": "hybrid_bm25_jina__ministral_3_3b_instruct_2512_rankings.csv",
    "Jina + Ministral": "jina_v5__ministral_3_3b_instruct_2512_rankings.csv",
    "Hybrid + Phi-4-mini": "hybrid_bm25_jina__phi_4_mini_instruct_rankings.csv",
    "Jina + Phi-4-mini": "jina_v5__phi_4_mini_instruct_rankings.csv",
}

METRICS = ("precision", "recall", "f1", "map", "mrr", "ndcg",
           "best_match_hit", "top1_best_match_rate")


def normalized_query(value):
    t = str(value).strip().upper()
    m = re.fullmatch(r"Q[_\-\s]*0*(\d+)", t)
    return f"Q{int(m.group(1)):03d}" if m else t


def normalized_candidate(value):
    t = str(value).strip()
    return t[:-2] if re.fullmatch(r"\d+\.0", t) else t


def csv_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def read_qrels(path):
    judgments = {}
    grouped = defaultdict(list)
    for row in csv_rows(path):
        if not {"query_id", "candidate_id", "relevance"} <= row.keys():
            raise ValueError(f"Qrels missing required columns: {path}")
        qid = normalized_query(row["query_id"])
        cid = normalized_candidate(row["candidate_id"])
        if not qid or not cid or row["relevance"] not in {"0", "1", "2"}:
            raise ValueError(f"Bad qrel: {row}")
        key = (qid, cid)
        if key in judgments:
            raise ValueError(f"Repeated qrel: {key}")
        val = int(row["relevance"])
        judgments[key] = val
        grouped[qid].append(val)
    if len(judgments) != EXPECTED_QRELS or len(grouped) != EXPECTED_Q:
        raise ValueError(f"Expected {EXPECTED_QRELS} judgments for {EXPECTED_Q} queries; "
                         f"got {len(judgments)} judgments and {len(grouped)} queries")
    return judgments, grouped


def read_rankings(name, path, expected_queries, judgments):
    groups = defaultdict(dict)
    candidate_sets = defaultdict(set)
    unjudged = []
    count = 0
    for row in csv_rows(path):
        if not {"query_id", "candidate_id", "rank"} <= row.keys():
            raise ValueError(f"{path.name}: missing required ranking columns")
        qid = normalized_query(row["query_id"])
        cid = normalized_candidate(row["candidate_id"])
        try:
            rank = int(row["rank"])
        except (TypeError, ValueError):
            raise ValueError(f"{name}: invalid rank for {qid}/{cid}")
        if rank < 1 or rank > EXPECTED_K:
            raise ValueError(f"{name}: invalid rank {rank} for {qid}/{cid}")
        if rank in groups[qid]:
            raise ValueError(f"{name}: repeated rank {rank} for {qid}")
        if cid in candidate_sets[qid]:
            raise ValueError(f"{name}: duplicate candidate {cid} for {qid}")
        groups[qid][rank] = cid
        candidate_sets[qid].add(cid)
        if (qid, cid) not in judgments:
            unjudged.append((qid, cid))
        count += 1
    if set(groups) != expected_queries:
        missing = sorted(expected_queries - set(groups))
        extra = sorted(set(groups) - expected_queries)
        raise ValueError(f"{name}: query mismatch. missing={missing[:8]} extra={extra[:8]}")
    if count != EXPECTED_Q * EXPECTED_K:
        raise ValueError(f"{name}: expected {EXPECTED_Q*EXPECTED_K} ranking rows, got {count}")
    for qid, positions in groups.items():
        if set(positions) != set(range(1, EXPECTED_K + 1)):
            raise ValueError(f"{name}: ranks for {qid} are not exactly 1..20")
    if unjudged:
        raise ValueError(f"{name}: {len(unjudged)} unjudged query–CV pairs, e.g. {unjudged[:5]}")
    return {qid: [positions[r] for r in range(1, EXPECTED_K + 1)]
            for qid, positions in groups.items()}, candidate_sets


def dcg(labels):
    return sum((2**rel - 1) / math.log2(rank + 1)
               for rank, rel in enumerate(labels, 1))


def one_metrics(labels, pool_labels, k, threshold):
    top = labels[:k]
    hits = [1 if r >= threshold else 0 for r in top]
    num_rel = sum(1 for r in pool_labels if r >= threshold)
    p = sum(hits) / k
    recall = (sum(hits) / num_rel) if num_rel else 0.0
    f1 = (2 * p * recall / (p + recall)) if (p + recall) else 0.0
    first = next((i for i, is_hit in enumerate(hits, 1) if is_hit), None)
    rr = 1 / first if first is not None else 0.0
    num_seen = 0
    ap_hits = 0.0
    for rank, hit in enumerate(hits, 1):
        num_seen += hit
        if hit:
            ap_hits += num_seen / rank
    ap = ap_hits / min(k, num_rel) if num_rel else 0.0
    ideal = sorted(pool_labels, reverse=True)[:k]
    idcg = dcg(ideal)
    ndcg = dcg(top) / idcg if idcg else 0.0
    best = next((i for i, r in enumerate(top, 1) if r == 2), None)
    return {
        "precision": p, "recall": recall, "f1": f1, "map": ap, "mrr": rr,
        "ndcg": ndcg, "best_match_hit": float(best is not None),
        "top1_best_match_rate": float(top[0] == 2),
        "best_match_count": sum(r == 2 for r in top),
        "first_best_match_rank": best or "",
        "queries_with_pool_best_match": int(2 in pool_labels),
    }


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(root, qrels_path=None):
    if qrels_path is None:
        qrels_path = root / "qrels_100_queries_AI_ASSISTED_5949.csv"
    judgments, by_qid = read_qrels(qrels_path)
    ranking_dir = root / "slm_rankings"
    by_system = {}
    pools = {}
    for name, filename in RANKING_FILES.items():
        by_system[name], pools[name] = read_rankings(
            name, ranking_dir / filename, set(by_qid), judgments)
        print(f"[OK] {name}: 100 queries × 20; 0 unjudged")
    for prefix in ("Hybrid", "Jina"):
        a = pools[f"{prefix} + Ministral"]
        b = pools[f"{prefix} + Phi-4-mini"]
        different = [qid for qid in by_qid if a[qid] != b[qid]]
        if different:
            raise ValueError(f"{prefix}: candidate pools differ between SLMs, queries={different[:8]}")
        print(f"[OK] {prefix} retrieval pool identical for both SLMs")

    rows = []
    best_rows = []
    for name, groups in by_system.items():
        for qid in sorted(groups):
            cids = groups[qid]
            labels = [judgments[(qid, cid)] for cid in cids]
            pool = by_qid[qid]
            for k in CUTOFFS:
                for threshold_name, threshold in THRESHOLDS.items():
                    item = one_metrics(labels, pool, k, threshold)
                    first_pos = item["first_best_match_rank"]
                    rows.append({"system": name, "query_id": qid, "threshold": threshold_name,
                                 "relevance_threshold": threshold, "k": k, **item})
                    if threshold_name == "strict":
                        best_rows.append({"system": name, "query_id": qid, "k": k,
                                         "best_match_hit": item["best_match_hit"],
                                         "top1_best_match": item["top1_best_match_rate"],
                                         "first_best_match_rank": first_pos,
                                         "best_match_candidate_id": cids[first_pos-1] if first_pos else "",
                                         "eligible_for_best_match": item["queries_with_pool_best_match"]})
    summarized = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["threshold"], row["k"])].append(row)
    for (name, th, k), per in grouped.items():
        summary = {"system": name, "threshold": th,
                   "relevance_threshold": THRESHOLDS[th], "k": k}
        for m in METRICS:
            summary[m] = sum(r[m] for r in per) / len(per)
        summary["queries_evaluated"] = len(per)
        eligible = [r for r in per if r["queries_with_pool_best_match"]]
        summary["queries_with_pool_best_match"] = len(eligible)
        summary["best_match_hit_among_eligible"] = (
            sum(r["best_match_hit"] for r in eligible) / len(eligible)
            if eligible else 0.0)
        summarized.append(summary)
    summarized.sort(key=lambda r: (r["threshold"] != "strict", r["k"], -r["ndcg"]))

    wide = []
    for name in RANKING_FILES:
        for threshold in THRESHOLDS:
            item = {"system": name, "threshold": threshold}
            for k in CUTOFFS:
                row = next(r for r in summarized if r["system"] == name and
                           r["threshold"] == threshold and r["k"] == k)
                for m in METRICS:
                    item[f"{m}@{k}"] = row[m]
            wide.append(item)

    pairs = []
    for prefix in ("Hybrid", "Jina"):
        for th in THRESHOLDS:
            for k in CUTOFFS:
                mini = next(r for r in summarized if r["system"] == f"{prefix} + Ministral"
                            and r["threshold"] == th and r["k"] == k)
                phi = next(r for r in summarized if r["system"] == f"{prefix} + Phi-4-mini"
                           and r["threshold"] == th and r["k"] == k)
                pairs.append({"retrieval_input": prefix, "threshold": th, "k": k,
                              **{f"delta_ministral_minus_phi4_{m}": mini[m]-phi[m]
                                 for m in METRICS}})

    out = root / "results"
    per_cols = ("system", "query_id", "threshold", "relevance_threshold", "k", *METRICS,
                "best_match_count", "first_best_match_rank", "queries_with_pool_best_match")
    summary_cols = ("system", "threshold", "relevance_threshold", "k", *METRICS,
                    "queries_evaluated", "queries_with_pool_best_match", "best_match_hit_among_eligible")
    save_csv(out / "metrics_per_query.csv", rows, per_cols)
    save_csv(out / "metrics_all_4_slm_systems.csv", summarized, summary_cols)
    save_csv(out / "metrics_strict_relevance.csv", [r for r in summarized if r["threshold"] == "strict"], summary_cols)
    save_csv(out / "metrics_broad_relevance.csv", [r for r in summarized if r["threshold"] == "broad"], summary_cols)
    save_csv(out / "metrics_wide_5_10_20.csv", wide,
             ("system", "threshold", *(f"{m}@{k}" for k in CUTOFFS for m in METRICS)))
    save_csv(out / "best_match_per_query.csv", best_rows,
             ("system", "query_id", "k", "best_match_hit", "top1_best_match",
              "first_best_match_rank", "best_match_candidate_id", "eligible_for_best_match"))
    save_csv(out / "delta_ministral_minus_phi4.csv", pairs,
             ("retrieval_input", "threshold", "k",
              *(f"delta_ministral_minus_phi4_{m}" for m in METRICS)))
    save_csv(out / "coverage_by_system.csv", [{"system": n, "queries": 100, "rows": 2000,
                                               "judged_rows": 2000, "unjudged_rows": 0}
                                              for n in RANKING_FILES],
             ("system", "queries", "rows", "judged_rows", "unjudged_rows"))

    print("\n=== STRICT LABEL 2 — nDCG / MAP / MRR / BestMatch ===")
    print(f"{'System':26s} {'K':>3s} {'nDCG':>8s} {'MAP':>8s} {'MRR':>8s} {'BestMatch':>10s}")
    for row in summarized:
        if row["threshold"] == "strict":
            print(f"{row['system']:26s} {row['k']:3d} {row['ndcg']:8.5f} {row['map']:8.5f} "
                  f"{row['mrr']:8.5f} {row['best_match_hit']:10.3f}")
    print("\nSUCCESS — outputs in", out.resolve())
    print("Method: pooled Recall; graded nDCG (0/1/2); MAP denominator min(K, pooled positives).")
    return summarized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing slm_rankings and qrels CSV")
    parser.add_argument("--qrels", type=Path, default=None, help="Override qrels path")
    args = parser.parse_args()
    evaluate(args.folder.resolve(), args.qrels.resolve() if args.qrels else None)


if __name__ == "__main__":
    main()
