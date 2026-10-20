"""Validate the 100-query Top-20 Jina v5 & BM25+Jina pools and qrels BEFORE GPU inference.
Run at the project root: python reranking/scripts/00_check_100q_reranking_inputs.py
"""
from pathlib import Path
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / 'hybrid_retrieval').is_dir() and (p / 'data').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Run this script inside ARIADNE, which must contain data/ and hybrid_retrieval/.')

QRELS = ROOT / 'evaluation' / 'qrels_100_queries.csv'
POOLS = {
    'hybrid_bm25_jina': ROOT / 'hybrid_retrieval' / 'hybrid_bm25_jina_rrf_100_queries_top20.csv',
    'jina_v5': ROOT / 'dense_retrieval' / 'jina_v5_100_queries_top20.csv',
}

def qid(s):
    s = str(s).strip().upper()
    if s.startswith('Q') and s[1:].isdigit():
        return f'Q{int(s[1:]):03d}'
    return s

def cid(s):
    s = str(s).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        return s[:-2]
    return s

if not QRELS.is_file():
    raise FileNotFoundError(f'Missing FINAL annotated qrels: {QRELS}')
qrels = pd.read_csv(QRELS, dtype={'query_id':str,'candidate_id':str}, encoding='utf-8-sig')
for col in ('query_id', 'candidate_id', 'relevance'):
    if col not in qrels.columns:
        raise ValueError(f'Missing column in qrels: {col}')
qrels['query_id']=qrels['query_id'].map(qid)
qrels['candidate_id']=qrels['candidate_id'].map(cid)
qrels['relevance']=pd.to_numeric(qrels['relevance'],errors='coerce')
if not qrels['relevance'].isin([0,1,2]).all() or qrels.duplicated(['query_id','candidate_id']).any():
    raise ValueError('Qrels contain missing/invalid labels or duplicate query-candidate pairs.')
if len(qrels)!=5949 or qrels.query_id.nunique()!=100:
    raise ValueError(f'Expected 5,949 labeled pairs / 100 queries; received {len(qrels)} / {qrels.query_id.nunique()}')
print('Qrels OK: 5,949 graded judgments across 100 queries')

for name, path in POOLS.items():
    if not path.is_file():
        raise FileNotFoundError(f'Missing {name}: {path}')
    d=pd.read_csv(path,dtype={'query_id':str,'candidate_id':str},encoding='utf-8-sig')
    missing={'query_id','candidate_id','rank','query','dataset_index'}-set(d.columns)
    if missing:
        raise ValueError(f'{name}: missing {sorted(missing)}')
    d['query_id']=d['query_id'].map(qid)
    d['candidate_id']=d['candidate_id'].map(cid)
    d['rank']=pd.to_numeric(d['rank'],errors='raise').astype(int)
    if len(d)!=2000 or d.query_id.nunique()!=100:
        raise ValueError(f'{name}: expected 2,000 rows / 100 queries; got {len(d)} / {d.query_id.nunique()}')
    if d.duplicated(['query_id','candidate_id']).any() or d.duplicated(['query_id','rank']).any():
        raise ValueError(f'{name}: duplicated query-candidate pairs or ranks')
    if any(sorted(g['rank'].tolist()) != list(range(1,21)) for _,g in d.groupby('query_id')):
        raise ValueError(f'{name}: not every query has ranks 1..20')
    if (d.groupby('query_id')['query'].nunique(dropna=False)!=1).any():
        raise ValueError(f'{name}: conflicting query text for an ID')
    missing_q=set(d.query_id)-set(qrels.query_id)
    if missing_q:
        raise ValueError(f'{name}: unknown queries in the qrels: {sorted(missing_q)}')
    matches=d[['query_id','candidate_id','rank']].merge(qrels[['query_id','candidate_id','relevance']],how='left',on=['query_id','candidate_id'],validate='many_to_one')
    unknown=matches[matches.relevance.isna()]
    if not unknown.empty:
        out=ROOT/'reranking'/f'{name}_UNJUDGED.csv'
        out.parent.mkdir(parents=True,exist_ok=True)
        unknown.to_csv(out,index=False)
        raise ValueError(f'{name}: {len(unknown)} ranked pairs NOT present in the qrels. See {out}')
    print(f'{name}: 100 queries × Top-20 = 2,000 ranked pairs, ALL JUDGED')

DATASET=ROOT/'data'/'candidate_profiles_english_clean_20k_with_text.parquet'
if not DATASET.exists():
    raise FileNotFoundError(f'Missing original candidate parquet: {DATASET}')
print('Dataset found:',DATASET)
print('PASS — Run the 4 rerankers. No additional labeling needed for these pools.')
