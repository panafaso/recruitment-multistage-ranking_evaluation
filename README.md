CV–Job Ranking Benchmark

Benchmarking multi-stage retrieval and reranking models for CV–job matching

This repository contains an experimental framework for evaluating information-retrieval and reranking methods for matching candidate profiles with recruiter queries and job requirements.

The project compares sparse, dense, hybrid, and neural reranking approaches over a collection of approximately 20,000 English-language candidate profiles. Its main objective is to study how different retrieval pipelines affect the quality of ranked candidate results.

Project Scope

The benchmark covers the following stages:

candidate-profile cleaning and preprocessing;

sparse lexical retrieval;

dense semantic retrieval;

hybrid rank fusion;

neural and LLM-based reranking;

relevance assessment and information-retrieval evaluation.

The repository focuses on comparative experimentation rather than deployment as a production recruitment system.

Retrieval and Ranking Methods

Sparse retrieval

Lexical retrieval methods include:

BM25;

BM25+;

learned sparse retrieval experiments such as SPLADE.

Dense retrieval

Dense candidate representations are generated using transformer-based embedding models, including experiments with:

BGE-M3;

multilingual E5;

MXBAI Embed Large;

Jina embedding models;

Nomic Embed Text.

Hybrid retrieval

Sparse and dense rankings can be combined through Reciprocal Rank Fusion (RRF) in order to benefit from both exact lexical matching and semantic similarity.

Reranking

Top-ranked candidates can be reranked using:

cross-encoder rerankers;

instruction-tuned reranking models;

local large language models;

listwise or pairwise reranking strategies.

Evaluation

The project follows a Cranfield-style information-retrieval evaluation methodology.

Candidate profiles are assessed against recruiter or job-related queries using graded relevance labels:

Label

Meaning

0

Not relevant

1

Partially relevant

2

Highly relevant

The evaluation includes standard ranking metrics such as:

Precision@k;

Recall@k;

Mean Reciprocal Rank;

Mean Average Precision;

nDCG@k.

Experiments are performed at different cut-off values, including Top-5, Top-10, Top-20, and Top-50, depending on the retrieval or reranking stage.
