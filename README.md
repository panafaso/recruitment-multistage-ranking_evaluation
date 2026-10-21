# CV–Job Ranking Benchmark

**A Multi-Stage Information Retrieval and Reranking Benchmark for AI-Assisted Candidate Matching**

## Overview

This repository presents an experimental framework for evaluating contemporary Information Retrieval (IR) and neural reranking approaches for recruiter-oriented candidate matching.

The research investigates how lexical, semantic, hybrid and neural ranking methods perform when retrieving and ranking candidate profiles against natural-language recruitment requirements. Particular attention is given to multi-stage retrieval pipelines, in which efficient first-stage retrieval is followed by more computationally intensive reranking.

The experiments use a collection of **20,000 English-language candidate profiles** and form part of an MSc thesis on AI-assisted candidate matching for recruitment.

The repository focuses on experimental evaluation and comparative ranking effectiveness rather than the deployment of an automated recruitment decision-making system.

## Research Objectives

The main objectives of this research are to:

- Compare standalone lexical and dense retrieval approaches for candidate matching.
- Examine the effectiveness of hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Evaluate specialised neural rerankers and generative reranking approaches based on Large Language Models (LLMs) and Small Language Models (SLMs).
- Assess ranking effectiveness using established IR evaluation measures.
- Examine the consistency of the initial experimental findings across an extended set of recruitment queries.

## Retrieval and Reranking Methods

### 1. Sparse Retrieval

Lexical retrieval is evaluated using **BM25Plus**, providing a baseline based on term-level matching between recruitment queries and candidate profiles.

### 2. Dense Retrieval

The benchmark investigates pretrained embedding models representing different architectures and semantic retrieval capabilities:

- **[BGE-M3](https://huggingface.co/BAAI/bge-m3)** — BAAI
- **[mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)** — Mixedbread AI
- **[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)** — Nomic AI
- **[jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)** — Jina AI

The models generate semantic representations of candidate profiles and recruitment queries for similarity-based retrieval.

### 3. Hybrid Retrieval

Hybrid retrieval combines BM25Plus with dense retrieval rankings through **Reciprocal Rank Fusion (RRF)**.

The evaluated combinations include BM25Plus with BGE-M3, Nomic Embed Text v1.5 and Jina Embeddings v5.

This approach investigates whether combining lexical correspondence with semantic similarity improves candidate ranking effectiveness.

### 4. Specialised Neural Reranking

The benchmark examines several pretrained neural reranking architectures:

- **[BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)** — BAAI
- **[Llama-Nemotron-Rerank-1B-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2)** — NVIDIA
- **[Ettin-reranker-150m-v1](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)** — Sentence Transformers
- **[GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)** — LightOn
- **[Jina Reranker v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5)** — Jina AI

These models represent cross-encoder, late-interaction and listwise ranking approaches. They are applied to previously retrieved candidate lists to assess whether more expressive query–candidate interactions improve ranking quality.

### 5. Generative Reranking with SLMs

Small Language Models are investigated through a pairwise reranking approach based on **PRP-Graph**, using iterative preference comparisons and graph-based ranking aggregation.

The evaluated SLMs include:

- **[Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)** — Mistral AI
- **[Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)** — Microsoft
- **[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)** — Hugging Face

SLM experiments were conducted using Google Colab. The extended 100-query evaluation focuses on Ministral-3-3B and Phi-4-mini.

### 6. Generative Reranking with LLMs

The research also includes an LLM-based PRP-Graph evaluation designed for larger instruction-tuned language models.

The selected models are:

- **[EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct)** — EuroLLM / UTTER Project
- **[Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b)** — IBM
- **[Granite 4.1 30B](https://huggingface.co/ibm-granite/granite-4.1-30b)** — IBM
- **[Teuken-7B-Instruct-v0.6](https://huggingface.co/openGPT-X/Teuken-7B-instruct-v0.6)** — OpenGPT-X
- **[Mistral Small 3.1 24B Instruct](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)** — Mistral AI

These experiments are planned for execution on the **LUMI supercomputer**. Their evaluation results will be added following completion.

## Experimental Evaluation

The research follows a **Cranfield-style Information Retrieval evaluation framework**, using fixed candidate collections, recruitment queries and graded relevance judgments.

Two experimental settings are considered:

| Characteristic | Initial evaluation | Extended evaluation |
|---|---:|---:|
| Candidate profiles | 20,000 | 20,000 |
| Recruitment queries | 16 | 100 |
| Relevance judgments | 2,335 | 5,949 |
| Relevance grades | 0, 1, 2 | 0, 1, 2 |

Relevance judgments follow a three-level grading scheme:

| Grade | Interpretation |
|---|---|
| **0** | Not relevant |
| **1** | Partially relevant |
| **2** | Strongly relevant |

Ranking effectiveness is primarily assessed using:

- **Normalized Discounted Cumulative Gain (nDCG)** — graded ranking quality.
- **Mean Average Precision (MAP)** — effectiveness in ranking strongly relevant candidates.
- **Mean Reciprocal Rank (MRR)** — position of the first strongly relevant candidate.

MAP and MRR use strict binary relevance, treating only grade-2 profiles as relevant.

Results are evaluated at different ranking depths, including Top-5, Top-10 and Top-20.

Pretrained neural retrieval and reranking models are evaluated without additional recruitment-specific fine-tuning.

## Extended 100-Query Evaluation

The extended evaluation expands the initial experimental setting to 100 recruitment queries covering a broader range of occupations, professional competencies, experience levels and qualification requirements.

It examines the consistency of the initial findings across standalone and hybrid retrieval, specialised neural reranking, and SLM-based generative reranking.

The main completed findings identify:

- **Hybrid BM25Plus–Jina v5** as the strongest first-stage retrieval configuration at nDCG@10.
- **Llama-Nemotron-Rerank-1B-v2** as the leading specialised neural reranker at nDCG@10.
- **Ministral-3-3B-Instruct-2512** as the strongest SLM among those included in the extended evaluation.

The extended results provide additional empirical evidence regarding the consistency of the leading approaches within the evaluated candidate collection.

The corresponding experimental code, queries, relevance judgments and ranking outputs are maintained in the [`100_queries_evaluation/`](100_queries_evaluation/) directory.

## Dataset

The experiments use a processed English-language subset of the publicly available recruitment candidate-profile dataset:

[Recruitment Dataset — Candidate Profiles (English), Hugging Face](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english)

The searchable collection contains 20,000 candidate profiles. Recruitment queries and relevance judgments were prepared for the purposes of the experimental evaluation.

The original dataset's licensing conditions and applicable data protection considerations should be reviewed before redistributing candidate-profile content.
