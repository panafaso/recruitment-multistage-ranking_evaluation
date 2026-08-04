from __future__ import annotations
import gc
import logging
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from pylate import models, rank
from tqdm.auto import tqdm
from transformers import AutoTokenizer

def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [*script_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    checked: set[Path] = set()
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)
        if (candidate / 'data').is_dir() and (candidate / 'hybrid_retrieval').is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root. Expected one directory containing both 'data' and 'hybrid_retrieval'.")
PROJECT_ROOT = find_project_root()
DATASET_PATH = PROJECT_ROOT / 'data' / 'candidate_profiles_english_clean_20k_with_text.parquet'
HYBRID_FILES = {'hybrid_bm25_nomic': PROJECT_ROOT / 'hybrid_retrieval' / 'hybrid_bm25_nomic_rrf_top50.csv', 'hybrid_bm25_jina': PROJECT_ROOT / 'hybrid_retrieval' / 'hybrid_bm25_jina_rrf_top50.csv'}
MODEL_NAME = 'lightonai/GTE-ModernColBERT-v1'
RERANKER_NAME = 'gte_moderncolbert_v1'
OUTPUT_DIR = PROJECT_ROOT / 'reranking' / 'outputs'
LOG_DIR = PROJECT_ROOT / 'reranking' / 'logs'
TOP_K = 50
QUERY_LENGTH = 128
DOCUMENT_LENGTH = 2048
QUERY_BATCH_SIZE = 8
DOCUMENT_BATCH_SIZE = 1
FAIL_ON_TRUNCATION = False
SAVE_TOKEN_AUDIT = True
QUERY_PREFIX = '[Q] '
DOCUMENT_PREFIX = '[D] '
REQUIRED_RANKING_COLUMNS = {'query_id', 'category', 'query', 'candidate_id', 'dataset_index', 'rank'}

def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def validate_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'{description} not found:\n{path}')

def load_candidate_dataset(path: Path) -> pd.DataFrame:
    validate_file_exists(path, 'Candidate dataset')
    candidates = pd.read_parquet(path).reset_index(drop=True)
    if 'candidate_text' not in candidates.columns:
        raise ValueError("The candidate dataset does not contain 'candidate_text'.")
    candidates['candidate_id'] = candidates.index.astype(str)
    candidates['dataset_index'] = candidates.index.astype(int)
    descriptive_columns = ['Position', 'Moreinfo', 'Experience Years', 'English Level', 'Looking For', 'Highlights', 'CV']
    available_columns = [column for column in descriptive_columns if column in candidates.columns]
    result = candidates[['candidate_id', 'dataset_index', 'candidate_text', *available_columns]].copy()
    result['candidate_text'] = result['candidate_text'].fillna('').astype(str)
    empty_texts = int(result['candidate_text'].str.strip().eq('').sum())
    if empty_texts:
        raise ValueError(f'The dataset contains {empty_texts} empty candidate_text values.')
    return result

def load_hybrid_ranking(path: Path, candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    validate_file_exists(path, 'Hybrid ranking file')
    ranking = pd.read_csv(path, dtype={'query_id': str, 'candidate_id': str})
    missing = REQUIRED_RANKING_COLUMNS - set(ranking.columns)
    if missing:
        raise ValueError(f'{path.name} is missing columns: {sorted(missing)}')
    ranking['dataset_index'] = pd.to_numeric(ranking['dataset_index'], errors='raise').astype(int)
    ranking['rank'] = pd.to_numeric(ranking['rank'], errors='raise').astype(int)
    ranking = ranking.sort_values(['query_id', 'rank'], kind='stable').groupby('query_id', group_keys=False).head(top_k).reset_index(drop=True)
    duplicate_pairs = int(ranking.duplicated(['query_id', 'candidate_id']).sum())
    if duplicate_pairs:
        raise ValueError(f'{path.name} contains {duplicate_pairs} duplicate query-candidate pairs.')
    query_text_counts = ranking.groupby('query_id')['query'].nunique()
    conflicting_queries = query_text_counts[query_text_counts != 1]
    if not conflicting_queries.empty:
        raise ValueError(f'Some query IDs have more than one query text: {conflicting_queries.index.tolist()}')
    ranking = ranking.merge(candidates, on=['candidate_id', 'dataset_index'], how='left', validate='many_to_one')
    missing_texts = int(ranking['candidate_text'].isna().sum())
    if missing_texts:
        raise ValueError(f'{missing_texts} ranking rows could not be matched to candidate_text in the parquet dataset.')
    counts = ranking.groupby('query_id').size()
    incomplete = counts[counts != top_k]
    if not incomplete.empty:
        raise ValueError(f'{path.name} does not contain exactly {top_k} candidates for every query:\n{incomplete.to_string()}')
    expected_ranks = list(range(1, top_k + 1))
    for query_id, group in ranking.groupby('query_id', sort=False):
        observed_ranks = sorted(group['rank'].tolist())
        if observed_ranks != expected_ranks:
            raise ValueError(f"{path.name}, query '{query_id}', does not contain ranks exactly 1..{top_k}.")
    return ranking

def select_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_audit_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1000000
    return tokenizer

def load_gte_moderncolbert(model_name: str, device: str) -> models.ColBERT:
    model = models.ColBERT(model_name_or_path=model_name, device=device, query_length=QUERY_LENGTH, document_length=DOCUMENT_LENGTH)
    if device == 'cuda':
        model.half()
    model.eval()
    return model

def count_tokens(tokenizer, text: str, prefix: str) -> int:
    encoded = tokenizer(prefix + text, add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False, return_token_type_ids=False)
    return len(encoded['input_ids'])

def create_token_audit(ranking: pd.DataFrame, tokenizer, retriever_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in tqdm(ranking.itertuples(index=False), total=len(ranking), desc=f'Token audit: {retriever_name}'):
        query_text = str(row.query)
        document_text = str(row.candidate_text)
        query_tokens = count_tokens(tokenizer=tokenizer, text=query_text, prefix=QUERY_PREFIX)
        document_tokens = count_tokens(tokenizer=tokenizer, text=document_text, prefix=DOCUMENT_PREFIX)
        rows.append({'retriever': retriever_name, 'query_id': row.query_id, 'candidate_id': row.candidate_id, 'dataset_index': int(row.dataset_index), 'original_rank': int(row.rank), 'query_characters': len(query_text), 'candidate_characters': len(document_text), 'query_tokens': query_tokens, 'query_limit': QUERY_LENGTH, 'query_fits_without_truncation': query_tokens <= QUERY_LENGTH, 'document_tokens': document_tokens, 'document_limit': DOCUMENT_LENGTH, 'document_fits_without_truncation': document_tokens <= DOCUMENT_LENGTH, 'document_tokens_over_limit': max(0, document_tokens - DOCUMENT_LENGTH)})
    return pd.DataFrame(rows)

def save_and_validate_token_audit(audit: pd.DataFrame, retriever_name: str, experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    audit_path = experiment_dir / f'{retriever_name}__token_length_audit.csv'
    if SAVE_TOKEN_AUDIT:
        audit.to_csv(audit_path, index=False, encoding='utf-8-sig')
    query_over_limit = audit[~audit['query_fits_without_truncation']]
    document_over_limit = audit[~audit['document_fits_without_truncation']]
    document_max = int(audit['document_tokens'].max())
    document_mean = float(audit['document_tokens'].mean())
    document_p95 = float(audit['document_tokens'].quantile(0.95))
    query_max = int(audit['query_tokens'].max())
    print(f'{retriever_name} token audit | document_max={document_max} | document_mean={document_mean:.2f} | document_p95={document_p95:.2f} | documents_over_limit={len(document_over_limit)} | query_max={query_max} | queries_over_limit={len(query_over_limit)}')
    print(f'Audit file: {audit_path}')
    if FAIL_ON_TRUNCATION and (not query_over_limit.empty or not document_over_limit.empty):
        sections: list[str] = []
        if not query_over_limit.empty:
            query_examples = query_over_limit[['query_id', 'query_tokens', 'query_limit']].drop_duplicates().head(10)
            sections.append(f'Queries above limit:\n{query_examples.to_string(index=False)}')
        if not document_over_limit.empty:
            document_examples = document_over_limit[['query_id', 'candidate_id', 'document_tokens', 'document_tokens_over_limit']].sort_values('document_tokens', ascending=False).head(10)
            sections.append(f'CVs above limit:\n{document_examples.to_string(index=False)}')
        raise ValueError(f'\n{retriever_name} contains inputs above the configured ColBERT limits.\nThe run stopped because FAIL_ON_TRUNCATION=True.\nNo query or CV was silently truncated.\n\n' + '\n\n'.join(sections) + f'\n\nFull audit:\n{audit_path}')

def extract_ranked_items(rerank_output: Any) -> list[dict[str, Any]]:
    if rerank_output is None:
        raise RuntimeError('PyLate rank.rerank returned None.')
    output = rerank_output
    if isinstance(output, tuple):
        output = list(output)
    if not isinstance(output, list):
        raise TypeError(f'Unexpected PyLate reranking output type: {type(output).__name__}')
    if len(output) == 1 and isinstance(output[0], (list, tuple)):
        output = list(output[0])
    normalized: list[dict[str, Any]] = []
    for item in output:
        if isinstance(item, dict):
            document_id = item.get('id', item.get('document_id'))
            score = item.get('score')
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            document_id = item[0]
            score = item[1]
        else:
            raise TypeError(f'Unexpected reranked item returned by PyLate: {item!r}')
        if document_id is None or score is None:
            raise ValueError(f'Each PyLate result must contain an id and score. Received: {item!r}')
        normalized.append({'id': str(document_id), 'score': float(score)})
    return normalized

def score_query_documents(model: models.ColBERT, query_text: str, documents: list[str], document_ids: list[str]) -> dict[str, float]:
    if len(documents) != len(document_ids):
        raise ValueError('documents and document_ids must have the same length.')
    query_embeddings = model.encode([query_text], batch_size=QUERY_BATCH_SIZE, is_query=True, show_progress_bar=False)
    document_embeddings = model.encode(documents, batch_size=DOCUMENT_BATCH_SIZE, is_query=False, show_progress_bar=False)
    rerank_output = rank.rerank(documents_ids=[document_ids], queries_embeddings=query_embeddings, documents_embeddings=[document_embeddings])
    ranked_items = extract_ranked_items(rerank_output)
    if len(ranked_items) != len(document_ids):
        raise RuntimeError(f'PyLate returned a different number of scores than documents: {len(ranked_items)} scores for {len(document_ids)} documents.')
    scores = {item['id']: item['score'] for item in ranked_items}
    missing_ids = set(document_ids) - set(scores)
    unexpected_ids = set(scores) - set(document_ids)
    if missing_ids or unexpected_ids:
        raise RuntimeError(f'Document ID mismatch after PyLate reranking. Missing={sorted(missing_ids)}; Unexpected={sorted(unexpected_ids)}')
    del query_embeddings
    del document_embeddings
    del rerank_output
    return scores

def rerank_one_system(ranking: pd.DataFrame, model: models.ColBERT, retriever_name: str, logger: logging.Logger) -> pd.DataFrame:
    output_groups: list[pd.DataFrame] = []
    grouped = ranking.groupby('query_id', sort=False)
    progress = tqdm(grouped, total=ranking['query_id'].nunique(), desc=f'{retriever_name} -> {RERANKER_NAME}')
    for query_id, group in progress:
        group = group.copy()
        query_text = str(group['query'].iloc[0])
        documents = group['candidate_text'].fillna('').astype(str).tolist()
        document_ids = group['candidate_id'].astype(str).tolist()
        score_by_id = score_query_documents(model=model, query_text=query_text, documents=documents, document_ids=document_ids)
        group['retriever'] = retriever_name
        group['reranker'] = RERANKER_NAME
        group['original_rank'] = group['rank']
        if 'score' in group.columns:
            group['original_score'] = group['score']
        elif 'rrf_score' in group.columns:
            group['original_score'] = group['rrf_score']
        else:
            group['original_score'] = np.nan
        group['reranker_score'] = group['candidate_id'].astype(str).map(score_by_id)
        if group['reranker_score'].isna().any():
            missing_scores = group.loc[group['reranker_score'].isna(), 'candidate_id'].astype(str).tolist()
            raise RuntimeError(f'Missing GTE-ModernColBERT scores for query {query_id}: {missing_scores}')
        group = group.sort_values(['reranker_score', 'original_rank'], ascending=[False, True], kind='stable').reset_index(drop=True)
        group['reranked_rank'] = np.arange(1, len(group) + 1)
        group['rank_change'] = group['original_rank'] - group['reranked_rank']
        output_groups.append(group)
        del score_by_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not output_groups:
        raise RuntimeError('No query groups were available for reranking.')
    reranked = pd.concat(output_groups, ignore_index=True)
    reranked = reranked.sort_values(['query_id', 'reranked_rank'], kind='stable').reset_index(drop=True)
    logger.info('Reranked rows: %d | Queries: %d', len(reranked), reranked['query_id'].nunique())
    return reranked

def create_evaluation_ranking(reranked: pd.DataFrame, system_name: str) -> pd.DataFrame:
    evaluation = reranked.copy()
    evaluation['system'] = system_name
    evaluation['rank'] = evaluation['reranked_rank']
    evaluation['score'] = evaluation['reranker_score']
    preferred_columns = ['system', 'query_id', 'category', 'query', 'candidate_id', 'dataset_index', 'rank', 'score']
    return evaluation[preferred_columns].copy()

def markdown_value(value: object) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip()

def format_score(value: object) -> str:
    if pd.isna(value):
        return ''
    return f'{float(value):.6f}'

def save_markdown_report(reranked: pd.DataFrame, experiment_dir: Path, experiment_name: str, retriever_name: str, logger: logging.Logger) -> None:
    report_path = experiment_dir / f'{experiment_name}_results.md'
    lines: list[str] = [f'# GTE-ModernColBERT Reranking Results: {experiment_name}', '', f'Model: {MODEL_NAME}', f'Initial retriever: {retriever_name}', f'Top K reranked: {TOP_K}', f'Query limit: {QUERY_LENGTH} tokens', f'CV/document limit: {DOCUMENT_LENGTH} tokens', f'Query batch size: {QUERY_BATCH_SIZE}', f'Document batch size: {DOCUMENT_BATCH_SIZE}', 'Similarity: token-level MaxSim', 'Reranking text: complete candidate_text', 'Truncation policy: preflight token audit; run stops if any query or CV exceeds its limit', '']
    ordered = reranked.sort_values(['category', 'query_id', 'reranked_rank'], kind='stable')
    for (category, query_id), group in ordered.groupby(['category', 'query_id'], sort=False):
        group = group.sort_values('reranked_rank', kind='stable')
        query_text = markdown_value(group['query'].iloc[0])
        lines.extend(['=' * 80, f'## Category: {markdown_value(category)}', '=' * 80, '', f'### Query ID: {query_id}', f'### Query: {query_text}', '', f'TOP {len(group)} GTE-MODERNCOLBERT RERANKED RESULTS', ''])
        for row in group.itertuples(index=False):
            row_data = row._asdict()
            lines.extend(['---', f'### Rank {int(row.reranked_rank)}', f'- GTE-ModernColBERT MaxSim Score: {format_score(row.reranker_score)}', f'- Original Hybrid Rank: {int(row.original_rank)}', f'- Rank Change: {int(row.rank_change):+d}', f'- Original Hybrid Score: {format_score(row.original_score)}', f'- Dataset Index: {int(row.dataset_index)}', f'- Candidate ID: {markdown_value(row.candidate_id)}'])
            if 'Position' in row_data:
                lines.append(f"- Position: {markdown_value(row_data['Position'])}")
            if 'Experience_Years' in row_data:
                lines.append(f"- Experience Years: {markdown_value(row_data['Experience_Years'])}")
            if 'English_Level' in row_data:
                lines.append(f"- English Level: {markdown_value(row_data['English_Level'])}")
            lines.append('')
            if 'Looking_For' in row_data:
                lines.extend(['#### Looking For', markdown_value(row_data['Looking_For']), ''])
            if 'Highlights' in row_data:
                lines.extend(['#### Highlights', markdown_value(row_data['Highlights']), ''])
            if 'CV' in row_data:
                lines.extend(['#### Full CV', markdown_value(row_data['CV']), ''])
            else:
                lines.extend(['#### Complete candidate text', markdown_value(row_data['candidate_text']), ''])
        lines.append('')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Markdown report: %s', report_path)

def save_outputs(reranked: pd.DataFrame, retriever_name: str, logger: logging.Logger) -> None:
    experiment_name = f'{retriever_name}__{RERANKER_NAME}'
    experiment_dir = OUTPUT_DIR / RERANKER_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = experiment_dir / f'{experiment_name}_detailed.csv'
    evaluation_path = experiment_dir / f'{experiment_name}_rankings.csv'
    reranked.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    evaluation = create_evaluation_ranking(reranked=reranked, system_name=experiment_name)
    evaluation.to_csv(evaluation_path, index=False, encoding='utf-8-sig')
    save_markdown_report(reranked=reranked, experiment_dir=experiment_dir, experiment_name=experiment_name, retriever_name=retriever_name, logger=logger)
    logger.info('Detailed output: %s', detailed_path)
    logger.info('Evaluation output: %s', evaluation_path)

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print('=' * 78)
    print('GTE-MODERNCOLBERT-V1 — BM25+NOMIC AND BM25+JINA TOP-50')
    print('=' * 78)
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Dataset: {DATASET_PATH}')
    print(f'Model: {MODEL_NAME}')
    print(f'Device: {device}')
    print(f'Top K: {TOP_K}')
    print(f'Query length: {QUERY_LENGTH}')
    print(f'CV/document length: {DOCUMENT_LENGTH}')
    print(f'Query batch size: {QUERY_BATCH_SIZE}')
    print(f'Document batch size: {DOCUMENT_BATCH_SIZE}')
    print(f'Silent truncation allowed: {not FAIL_ON_TRUNCATION}')
    candidates = load_candidate_dataset(DATASET_PATH)
    hybrid_rankings: dict[str, pd.DataFrame] = {}
    for retriever_name, ranking_path in HYBRID_FILES.items():
        hybrid_rankings[retriever_name] = load_hybrid_ranking(path=ranking_path, candidates=candidates, top_k=TOP_K)
        print(f'Loaded {retriever_name}: {len(hybrid_rankings[retriever_name])} rows')
    print('\nLoading audit tokenizer...')
    audit_tokenizer = load_audit_tokenizer(MODEL_NAME)
    experiment_dir = OUTPUT_DIR / RERANKER_NAME
    for retriever_name, ranking in hybrid_rankings.items():
        audit = create_token_audit(ranking=ranking, tokenizer=audit_tokenizer, retriever_name=retriever_name)
        save_and_validate_token_audit(audit=audit, retriever_name=retriever_name, experiment_dir=experiment_dir)
    del audit_tokenizer
    gc.collect()
    print('\nLoading GTE-ModernColBERT reranker...')
    model = load_gte_moderncolbert(model_name=MODEL_NAME, device=device)
    total_experiments = len(hybrid_rankings)
    for experiment_number, (retriever_name, ranking) in enumerate(hybrid_rankings.items(), start=1):
        log_path = LOG_DIR / f'{retriever_name}__{RERANKER_NAME}.log'
        logger = create_logger(log_path)
        logger.info('Experiment %d/%d', experiment_number, total_experiments)
        logger.info('Retriever: %s', retriever_name)
        logger.info('Reranker: %s', MODEL_NAME)
        logger.info('Device: %s', device)
        logger.info('Top K: %d | Query length: %d | Document length: %d | Query batch size: %d | Document batch size: %d', TOP_K, QUERY_LENGTH, DOCUMENT_LENGTH, QUERY_BATCH_SIZE, DOCUMENT_BATCH_SIZE)
        start_time = time.perf_counter()
        try:
            reranked = rerank_one_system(ranking=ranking, model=model, retriever_name=retriever_name, logger=logger)
            save_outputs(reranked=reranked, retriever_name=retriever_name, logger=logger)
            elapsed = time.perf_counter() - start_time
            logger.info('Completed successfully in %.2f seconds.', elapsed)
        except torch.cuda.OutOfMemoryError as error:
            logger.exception('CUDA out of memory with DOCUMENT_BATCH_SIZE=%d and DOCUMENT_LENGTH=%d. Batch size is already 1. To preserve full CVs, do not lower DOCUMENT_LENGTH below the longest audited CV; use CPU inference or a larger GPU.', DOCUMENT_BATCH_SIZE, DOCUMENT_LENGTH)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise error
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('\n' + '=' * 78)
    print('ALL GTE-MODERNCOLBERT RERANKING RUNS COMPLETED')
    print('=' * 78)
if __name__ == '__main__':
    main()