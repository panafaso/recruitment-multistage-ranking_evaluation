# CV–Job Ranking Benchmark

**Benchmarking multi-stage retrieval and reranking models for CV–job matching**

This repository contains an experimental framework for evaluating information-retrieval and reranking methods for matching candidate profiles with recruiter queries and job requirements.

The project compares sparse, dense, hybrid, and neural reranking approaches over a collection of approximately 20,000 English-language candidate profiles. Its main objective is to study how different retrieval pipelines affect the quality of ranked candidate results.

---

## Project Scope

The benchmark covers the following stages:

- candidate-profile cleaning and preprocessing;
- sparse lexical retrieval;
- dense semantic retrieval;
- hybrid rank fusion;
- neural and LLM-based reranking;
- relevance assessment and information-retrieval evaluation.

The repository focuses on comparative experimentation rather than deployment as a production recruitment system.

---

## Retrieval and Ranking Methods

### Sparse retrieval

Lexical retrieval methods include:

- BM25;
- BM25+;
- learned sparse retrieval experiments such as SPLADE.

### Dense retrieval

Dense candidate representations are generated using transformer-based embedding models, including experiments with:

- BGE-M3;
- multilingual E5;
- MXBAI Embed Large;
- Jina embedding models;
- Nomic Embed Text.

### Hybrid retrieval

Sparse and dense rankings can be combined through **Reciprocal Rank Fusion (RRF)** in order to benefit from both exact lexical matching and semantic similarity.

### Reranking

Top-ranked candidates can be reranked using:

- cross-encoder rerankers;
- instruction-tuned reranking models;
- local large language models;
- listwise or pairwise reranking strategies.

---

## Evaluation

The project follows a Cranfield-style information-retrieval evaluation methodology.

Candidate profiles are assessed against recruiter or job-related queries using graded relevance labels:

| Label | Meaning |
|---|---|
| `0` | Not relevant |
| `1` | Partially relevant |
| `2` | Highly relevant |

The evaluation includes standard ranking metrics such as:

- Precision@k;
- Recall@k;
- Mean Reciprocal Rank;
- Mean Average Precision;
- nDCG@k.

Experiments are performed at different cut-off values, including Top-5, Top-10, Top-20, and Top-50, depending on the retrieval or reranking stage.

---

## Repository Structure

```text
cv-job-ranking-benchmark/
│
├── .gitignore
├── README.md
│
├── cv_cleaning_database/
│   ├── data-cleaning scripts
│   ├── language-filtering scripts
│   └── preprocessing utilities
│
├── embeddings/
│   ├── embedding-generation scripts
│   └── model-specific utilities
│
├── dense_retrieval/
│   ├── dense-retrieval scripts
│   ├── Top-k CSV outputs
│   └── Markdown result summaries
│
├── sparse_retrieval/
│   └── BM25 and learned-sparse experiments
│
├── hybrid_retrieval/
│   └── rank-fusion experiments
│
├── reranking/
│   └── cross-encoder and LLM reranking experiments
│
└── evaluation/
    ├── relevance judgments
    ├── metric computation
    └── comparative results
```

The exact structure may evolve as additional models and experiments are incorporated.

---

## Data

The experiments use an English-language candidate-profile dataset containing approximately 20,000 profiles.

The complete dataset and generated embedding matrices are **not included** in this repository because they may be large and may require separate compliance with the dataset's licensing and data-protection conditions.

The repository instead contains:

- preprocessing code;
- retrieval and reranking scripts;
- configuration logic;
- aggregate evaluation outputs;
- selected ranking results where appropriate.

Users should obtain the original dataset independently and review its applicable licence before use.

---

## Privacy and Responsible Use

Recruitment data may include personal or professionally sensitive information. This project therefore follows a local-first and privacy-conscious experimental approach.

The repository does not intentionally publish:

- names, email addresses, or telephone numbers;
- complete CV texts containing personal data;
- private candidate records;
- model credentials or API keys;
- full embedding matrices;
- downloaded model weights.

This code is intended for research and evaluation purposes. It should not be used as the sole basis for employment decisions.

Any real-world use should include:

- human oversight;
- bias and fairness assessment;
- data-quality controls;
- appropriate transparency;
- data-protection impact assessment where required;
- compliance with applicable data-protection and employment law.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/cv-job-ranking-benchmark.git
cd cv-job-ranking-benchmark
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

A dedicated `requirements.txt` file should list only the packages used by the project.

---

## Typical Experimental Workflow

A typical experiment consists of the following steps:

```text
Candidate profiles
        ↓
Cleaning and preprocessing
        ↓
Sparse and/or dense retrieval
        ↓
Top-k candidate pool
        ↓
Hybrid fusion
        ↓
Cross-encoder or LLM reranking
        ↓
Evaluation against relevance judgments
```

Example execution pattern:

```bash
python path/to/embedding_script.py
python path/to/retrieval_script.py
python path/to/reranking_script.py
python path/to/evaluation_script.py
```

The precise commands depend on the model and experiment being reproduced.

---

## Outputs

The repository may include:

- Top-k ranked candidate lists in CSV format;
- human-readable Markdown summaries;
- aggregate evaluation tables;
- comparisons between sparse, dense, hybrid, and reranked pipelines.

Large generated artifacts such as `.npy`, `.faiss`, model checkpoints, and complete datasets are excluded through `.gitignore`.

---

## Current Research Questions

The project investigates questions such as:

1. How do sparse and dense retrieval methods compare in CV–job matching?
2. Does hybrid RRF improve ranking quality over individual retrieval methods?
3. How much does cross-encoder reranking improve Top-k precision and nDCG?
4. Can local LLMs provide reliable listwise reranking?
5. Which models offer the best balance between ranking quality, latency, memory use, and privacy?
6. How does model performance vary across technical, legal, administrative, HR, and multilingual queries?

---

## Limitations

- Relevance judgments may contain subjective assessment.
- Performance depends on query design and candidate-profile quality.
- Ranking accuracy does not imply fairness or legal compliance.
- Results from one candidate dataset may not generalise to other labour markets.
- LLM-based reranking may be sensitive to prompt design and model quantisation.
- Candidate matching should support, not replace, human recruitment decisions.

---

## Roadmap

Planned or ongoing work includes:

- additional learned-sparse retrieval experiments;
- broader dense-model comparison;
- hybrid sparse–dense fusion;
- Top-20 reranking experiments;
- local LLM reranking;
- latency and GPU-memory benchmarking;
- expanded relevance judgments;
- statistical comparison of ranking results;
- privacy, fairness, and EU regulatory analysis.

---

## Research Context

This repository supports research on AI-powered CV and job-vacancy matching, with particular emphasis on:

- information retrieval;
- semantic search;
- neural reranking;
- local and privacy-preserving AI;
- responsible AI in recruitment;
- GDPR and EU AI Act considerations.

---

## Citation

When referring to this repository, please cite it as:

```text
Panagiotis Fasoulidis. CV–Job Ranking Benchmark:
Benchmarking Multi-Stage Retrieval and Reranking for CV–Job Matching. 2026.
```

A formal `CITATION.cff` file may be added in a future release.

---

## Licence

The software licence for this repository will be specified in the `LICENSE` file.

Third-party datasets, pretrained models, and libraries remain subject to their own licences and terms of use.

---

## Author

**Panagiotis Fasoulidis**

Research interests include information retrieval, natural-language processing, privacy-preserving AI, data protection, and AI governance.
