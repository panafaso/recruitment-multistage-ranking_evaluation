from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "cv_embeddings_jina_v5_text_small_20k_2048.npy"
)

MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "jina_v5_text_small_results_top50.md"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "jina_v5_text_small_top50.csv"
)

MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"

EXPECTED_ROWS = 20_000
EXPECTED_EMBEDDING_DIMENSION = 1024

# This concerns only the short retrieval queries.
# The saved CV embeddings were already created with 2048 tokens.
QUERY_MAX_SEQ_LENGTH = 512

TOP_K = 50


query_categories = {
    "Software Engineering": [
        "Senior software engineers with 5+ years experience in backend systems cloud infrastructure and microservices"
    ],

    "Frontend Development": [
        "Frontend React developers with TypeScript Redux Next.js and responsive web application experience"
    ],

    "Machine Learning / AI": [
        "Junior Python machine learning engineers with TensorFlow PyTorch NLP and FastAPI experience"
    ],

    "Cybersecurity": [
        "Cybersecurity engineers with penetration testing OWASP vulnerability assessment and SIEM experience"
    ],

    "Legal / Compliance": [
        "Experienced legal consultants with 7+ years experience in data protection AML compliance and cybersecurity"
    ],

    "Project Management": [
        "Project managers with Agile Scrum Jira stakeholder management and up to 4 years experience"
    ],

    "HR / Recruitment": [
        "HR recruiters with IT hiring onboarding sourcing and candidate interview experience"
    ],

    "Customer Support / Success": [
        "Customer success managers with 3-6 years of experience and B2B SaaS client retention onboarding and communication skills"
    ],

    "Marketing": [
        "Digital marketing specialists with SEO Google Analytics content marketing and campaign management experience"
    ],

    "Sales / Business Development": [
        "Business development managers with B2B sales negotiation partnership building and international client experience"
    ],

    "UI / UX Design": [
        "UI UX designers with Figma prototyping user research and mobile application design experience"
    ],

    "Blockchain / Fintech": [
        "Fintech software engineers with blockchain cryptocurrency smart contract and payment systems experience"
    ],

    "Operations Management": [
        "Operations managers with experience in process optimization budgeting team coordination reporting and business operations"
    ],

    "Soft Skills": [
        "Multilingual professionals speaking English Greek French and German with client facing communication experience",
        "Professionals with leadership mentoring stakeholder management and cross functional collaboration skills",
    ],

    "Junior / Trainee Queries": [
        "Junior professionals and trainees looking for entry level opportunities with internship project or early career experience"
    ],
}


def safe_value(value: object) -> str:
    """Convert missing dataframe values to an empty string."""

    return "" if pd.isna(value) else str(value)


def log(line: str, file) -> None:
    """Print and write one line to the Markdown file."""

    print(line)
    file.write(line + "\n")


def create_query_id(
    category: str,
    query_number: int,
    number_of_queries: int,
) -> str:
    """Create a stable query identifier."""

    query_id = (
        category.lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )

    query_id = re.sub(
        r"_+",
        "_",
        query_id,
    ).strip("_")

    if number_of_queries > 1:
        query_id = f"{query_id}_{query_number}"

    return query_id


def load_dataset() -> pd.DataFrame:
    """Load and validate the candidate dataset."""

    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    ).reset_index(drop=True)

    df["candidate_id"] = df.index.astype(str)

    print("Dataset rows:", len(df))
    print("Dataset columns:", df.columns.tolist())

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"but found {len(df):,}."
        )

    if "candidate_text" not in df.columns:
        raise ValueError(
            "candidate_text column not found"
        )

    return df


def load_document_embeddings(
    dataset_rows: int,
) -> np.ndarray:
    """Load and validate the saved Jina CV embeddings."""

    print(
        "\nLoading Jina v5 Text Small document embeddings..."
    )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Document embeddings file not found:\n"
            f"{EMBEDDINGS_PATH}"
        )

    document_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        "Embeddings shape:",
        document_embeddings.shape,
    )

    if document_embeddings.ndim != 2:
        raise ValueError(
            "The document embeddings array must be "
            "two-dimensional."
        )

    if len(document_embeddings) != dataset_rows:
        raise ValueError(
            f"Dataset has {dataset_rows} rows, but embeddings have "
            f"{len(document_embeddings)} rows."
        )

    if document_embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
            f"found {document_embeddings.shape[1]}."
        )

    document_embeddings = document_embeddings.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        document_embeddings
    ).all():
        raise ValueError(
            "Document embeddings contain NaN or infinite values."
        )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
    )

    if not np.allclose(
        document_norms,
        1.0,
        atol=5e-3,
        rtol=0.0,
    ):
        raise ValueError(
            "Saved document embeddings are outside the "
            "expected normalized range. "
            f"Minimum norm: {document_norms.min():.6f}, "
            f"maximum norm: {document_norms.max():.6f}."
        )

    print(
        "Document norm range:",
        f"{document_norms.min():.6f}",
        "to",
        f"{document_norms.max():.6f}",
    )

    return document_embeddings


def load_model(
    device: str,
) -> SentenceTransformer:
    """Load Jina v5 Text Small for query encoding."""

    print(
        "\nLoading Jina v5 Text Small model..."
    )

    model_kwargs = {}

    if device == "cuda":
        model_kwargs["dtype"] = torch.bfloat16

    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=device,
        model_kwargs=model_kwargs,
    )

    model.max_seq_length = QUERY_MAX_SEQ_LENGTH

    print(
        "Effective query maximum sequence length:",
        model.max_seq_length,
    )

    if model.max_seq_length != QUERY_MAX_SEQ_LENGTH:
        raise ValueError(
            "The effective query sequence length does not "
            "match the requested value."
        )

    print(
        "Jina v5 Text Small model ready!"
    )

    return model


def retrieve_jina(
    query: str,
    model: SentenceTransformer,
    document_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one query and retrieve the top candidate profiles."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive."
        )

    if top_k > len(document_embeddings):
        raise ValueError(
            "top_k cannot exceed the number of documents."
        )

    with torch.inference_mode():
        query_embedding = model.encode(
            inputs=query,
            task="retrieval",
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.ndim != 1:
        query_embedding = query_embedding.reshape(-1)

    if (
        query_embedding.shape[0]
        != EXPECTED_EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "Unexpected query embedding dimension. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
            f"found {query_embedding.shape[0]}."
        )

    if not np.isfinite(
        query_embedding
    ).all():
        raise ValueError(
            "Query embedding contains NaN or infinite values."
        )

    query_norm = float(
        np.linalg.norm(query_embedding)
    )

    if not np.isclose(
        query_norm,
        1.0,
        atol=5e-3,
        rtol=0.0,
    ):
        raise ValueError(
            "Query embedding is outside the expected "
            f"normalized range. Norm: {query_norm:.6f}"
        )

    if (
        document_embeddings.shape[1]
        != query_embedding.shape[0]
    ):
        raise ValueError(
            "Embedding dimension mismatch: "
            f"documents have dimension "
            f"{document_embeddings.shape[1]}, "
            f"but the query has dimension "
            f"{query_embedding.shape[0]}."
        )

    # Dot product between normalized vectors is cosine similarity.
    scores = (
        document_embeddings
        @ query_embedding
    )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Similarity scores contain NaN or infinite values."
        )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    top_scores = scores[
        top_indices
    ]

    return top_indices, top_scores


def write_results_md(
    df: pd.DataFrame,
    query: str,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    file,
) -> None:
    """Write one query's Top-K results to Markdown."""

    log(
        f"\n### Query: {query}",
        file,
    )

    log(
        f"\nTOP {TOP_K} JINA V5 TEXT SMALL RESULTS\n",
        file,
    )

    for rank, (
        idx,
        score,
    ) in enumerate(
        zip(
            top_indices,
            top_scores,
        ),
        start=1,
    ):
        idx = int(idx)
        row = df.iloc[idx]

        log("---", file)
        log(f"### Rank {rank}", file)

        log(
            f"- Jina v5 Similarity: {float(score):.6f}",
            file,
        )

        log(
            f"- Dataset Index: {idx}",
            file,
        )

        log(
            f"- Candidate ID: {row['candidate_id']}",
            file,
        )

        log(
            "- Position: "
            f"{safe_value(row.get('Position'))}",
            file,
        )

        log(
            "- Experience Years: "
            f"{safe_value(row.get('Experience Years'))}",
            file,
        )

        log(
            "- English Level: "
            f"{safe_value(row.get('English Level'))}",
            file,
        )

        log(
            "\n#### Looking For",
            file,
        )

        log(
            safe_value(
                row.get("Looking For")
            ),
            file,
        )

        log(
            "\n#### Highlights",
            file,
        )

        log(
            safe_value(
                row.get("Highlights")
            ),
            file,
        )

        log(
            "\n#### CV snippet",
            file,
        )

        log(
            safe_value(
                row.get("CV")
            ),
            file,
        )

        log("", file)


def validate_results(
    results_df: pd.DataFrame,
) -> None:
    """Validate the complete Top-50 ranking output."""

    expected_queries = sum(
        len(queries)
        for queries in query_categories.values()
    )

    expected_rows = (
        expected_queries
        * TOP_K
    )

    actual_queries = results_df[
        "query_id"
    ].nunique()

    actual_rows = len(
        results_df
    )

    duplicate_count = results_df.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    rows_per_query = results_df.groupby(
        "query_id"
    ).size()

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        "Expected queries:",
        expected_queries,
    )

    print(
        "Actual queries:",
        actual_queries,
    )

    print(
        "Expected rows:",
        expected_rows,
    )

    print(
        "Actual rows:",
        actual_rows,
    )

    print(
        "Duplicate query/candidate pairs:",
        duplicate_count,
    )

    print("\nRows per query:")
    print(
        rows_per_query.to_string()
    )

    if actual_queries != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} queries, "
            f"but found {actual_queries}."
        )

    if actual_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"but found {actual_rows}."
        )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate candidate IDs were found "
            "inside one or more query rankings."
        )

    if not (
        rows_per_query
        == TOP_K
    ).all():
        raise ValueError(
            f"Not every query contains exactly "
            f"{TOP_K} results."
        )

    if not np.isfinite(
        results_df["score"].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "The final result table contains "
            "non-finite scores."
        )

    expected_ranks = list(
        range(
            1,
            TOP_K + 1,
        )
    )

    for query_id, group in results_df.groupby(
        "query_id",
        sort=False,
    ):
        actual_ranks = group[
            "rank"
        ].tolist()

        if actual_ranks != expected_ranks:
            raise ValueError(
                f"Incorrect rank sequence for query "
                f"{query_id}."
            )

        query_scores = group[
            "score"
        ].to_numpy(
            dtype=np.float64
        )

        if np.any(
            query_scores[:-1]
            < query_scores[1:]
        ):
            raise ValueError(
                f"Scores are not descending for query "
                f"{query_id}."
            )

    print(
        "\nValidation successful."
    )


def main() -> None:
    """Run Jina v5 Text Small retrieval."""

    MARKDOWN_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    document_embeddings = (
        load_document_embeddings(
            dataset_rows=len(df),
        )
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 80)
    print("JINA V5 TEXT SMALL DENSE RETRIEVAL")
    print("=" * 80)
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Dataset rows:", len(df))
    print("Document embedding dimension:", EXPECTED_EMBEDDING_DIMENSION)
    print("Document max sequence length used previously: 2048")
    print("Query max sequence length:", QUERY_MAX_SEQ_LENGTH)
    print("Task:", "retrieval")
    print("Document prompt:", "document")
    print("Query prompt:", "query")
    print("Top K:", TOP_K)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA version:",
            torch.version.cuda,
        )

    print("=" * 80)

    model = load_model(
        device=device,
    )

    all_results: list[
        dict[str, object]
    ] = []

    total_queries = sum(
        len(queries)
        for queries in query_categories.values()
    )

    query_counter = 0

    with MARKDOWN_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        log(
            "# Jina v5 Text Small Dense Retrieval Results",
            file,
        )

        log(
            f"\nModel: `{MODEL_NAME}`",
            file,
        )

        log(
            f"Dataset rows: {len(df)}",
            file,
        )

        log(
            f"Top K per query: {TOP_K}",
            file,
        )

        log(
            "Similarity: dot product between normalized embeddings",
            file,
        )

        log(
            "Document embeddings normalized: yes",
            file,
        )

        log(
            "Query embeddings normalized: yes",
            file,
        )

        log(
            "Document maximum sequence length: 2048",
            file,
        )

        log(
            f"Query maximum sequence length: "
            f"{QUERY_MAX_SEQ_LENGTH}",
            file,
        )

        log(
            "Task: retrieval",
            file,
        )

        log(
            "Document prompt: document",
            file,
        )

        log(
            "Query prompt: query",
            file,
        )

        log(
            "Embedding dimension: 1024",
            file,
        )

        log(
            "Retrieval text: candidate_text",
            file,
        )

        log(
            "Candidate ID: stable dataframe index after reset_index",
            file,
        )

        for category, queries in query_categories.items():
            log(
                "\n" + "=" * 80,
                file,
            )

            log(
                f"## Category: {category}",
                file,
            )

            log(
                "=" * 80,
                file,
            )

            for query_number, query in enumerate(
                queries,
                start=1,
            ):
                query_counter += 1

                query_id = create_query_id(
                    category=category,
                    query_number=query_number,
                    number_of_queries=len(queries),
                )

                print(
                    f"\nQuery "
                    f"{query_counter}/{total_queries}: "
                    f"{category}"
                )

                top_indices, top_scores = retrieve_jina(
                    query=query,
                    model=model,
                    document_embeddings=document_embeddings,
                    top_k=TOP_K,
                )

                write_results_md(
                    df=df,
                    query=query,
                    top_indices=top_indices,
                    top_scores=top_scores,
                    file=file,
                )

                for rank, (
                    candidate_idx,
                    score,
                ) in enumerate(
                    zip(
                        top_indices,
                        top_scores,
                    ),
                    start=1,
                ):
                    candidate_idx = int(
                        candidate_idx
                    )

                    row = df.iloc[
                        candidate_idx
                    ]

                    all_results.append({
                        "query_id": query_id,
                        "category": category,
                        "query": query,
                        "candidate_id": str(
                            row["candidate_id"]
                        ),
                        "dataset_index": candidate_idx,
                        "rank": int(rank),
                        "score": float(score),
                    })

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        by=[
            "query_id",
            "rank",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    validate_results(
        results_df
    )

    results_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nJina v5 Text Small retrieval completed!"
    )

    print(
        "Markdown results saved to:",
        MARKDOWN_OUTPUT_PATH,
    )

    print(
        "CSV rankings saved to:",
        CSV_OUTPUT_PATH,
    )

    print(
        "Number of queries:",
        results_df[
            "query_id"
        ].nunique(),
    )

    print(
        "Number of rows:",
        len(results_df),
    )


if __name__ == "__main__":
    main()