from pathlib import Path
import re

import numpy as np
import pandas as pd
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
    / "cv_embeddings_mxbai_embed_large_20k.npy"
)

MARKDOWN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "mxbai_results_top50.md"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "dense_retrieval"
    / "mxbai_top50.csv"
)

MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

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


print("Loading dataset...")

df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

# Stable candidate ID shared across every retriever.
df["candidate_id"] = df.index.astype(str)

print("Dataset rows:", len(df))
print("Dataset columns:", df.columns.tolist())

if "candidate_text" not in df.columns:
    raise ValueError("candidate_text column not found")


print("\nLoading MXBAI embeddings...")

document_embeddings = np.load(EMBEDDINGS_PATH)

if document_embeddings.ndim != 2:
    raise ValueError(
        "The document embeddings array must be two-dimensional."
    )

document_embeddings = document_embeddings.astype(
    np.float32,
    copy=False,
)

print("Embeddings shape:", document_embeddings.shape)

if len(df) != len(document_embeddings):
    raise ValueError(
        f"Dataset has {len(df)} rows, but embeddings have "
        f"{len(document_embeddings)} rows."
    )


print("\nLoading MXBAI model...")

model = SentenceTransformer(MODEL_NAME)

model.max_seq_length = 512

print("MXBAI model ready!")



def safe_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def log(line, file):
    print(line)
    file.write(line + "\n")


def create_query_id(
    category,
    query_number,
    number_of_queries,
):
    """
    Create a stable query identifier.

    Examples:
        Software Engineering -> software_engineering
        Machine Learning / AI -> machine_learning_ai
        Soft Skills query 1 -> soft_skills_1
        Soft Skills query 2 -> soft_skills_2
    """

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

def retrieve_mxbai(
    query,
    top_k=TOP_K,
):
    query_embedding = model.encode(
        query,
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

    # Dot product equals cosine similarity only when
    # both document and query embeddings are normalized.
    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    return top_indices, scores[top_indices]


def write_results_md(
    query,
    top_indices,
    top_scores,
    file,
):
    log(f"\n### Query: {query}", file)
    log(f"\nTOP {TOP_K} MXBAI RESULTS\n", file)

    for rank, (idx, score) in enumerate(
        zip(top_indices, top_scores),
        start=1,
    ):
        row = df.iloc[idx]

        log("---", file)
        log(f"### Rank {rank}", file)
        log(
            f"- MXBAI Similarity: {score:.6f}",
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
            f"- Position: "
            f"{safe_value(row.get('Position'))}",
            file,
        )

        log(
            f"- Experience Years: "
            f"{safe_value(row.get('Experience Years'))}",
            file,
        )

        log(
            f"- English Level: "
            f"{safe_value(row.get('English Level'))}",
            file,
        )

        log("\n#### Looking For", file)
        log(
            safe_value(row.get("Looking For")),
            file,
        )

        log("\n#### Highlights", file)
        log(
            safe_value(row.get("Highlights")),
            file,
        )

        log("\n#### CV snippet", file)
        log(
            safe_value(row.get("CV")),
            file,
        )

        log("", file)


def validate_results(results_df):
    expected_queries = sum(
        len(queries)
        for queries in query_categories.values()
    )

    expected_rows = expected_queries * TOP_K

    actual_queries = results_df[
        "query_id"
    ].nunique()

    actual_rows = len(results_df)

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
    print(rows_per_query.to_string())

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
        rows_per_query == TOP_K
    ).all():
        raise ValueError(
            f"Not every query contains exactly "
            f"{TOP_K} results."
        )

    print("\nValidation successful.")

if __name__ == "__main__":

    MARKDOWN_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    total_queries = sum(
        len(queries)
        for queries in query_categories.values()
    )

    query_counter = 0

    with open(
        MARKDOWN_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        log(
            "# MXBAI Dense Retrieval Results",
            file,
        )

        log(
            f"\nModel: {MODEL_NAME}",
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
            "Similarity: dot product between "
            "normalized embeddings",
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
            "Retrieval text: candidate_text",
            file,
        )

        log(
            "Candidate ID: stable dataframe index "
            "after reset_index",
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

                top_indices, top_scores = retrieve_mxbai(
                    query=query,
                    top_k=TOP_K,
                )

                write_results_md(
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
                    row = df.iloc[candidate_idx]

                    all_results.append({
                        "query_id": query_id,
                        "category": category,
                        "query": query,
                        "candidate_id": str(
                            row["candidate_id"]
                        ),
                        "dataset_index": int(
                            candidate_idx
                        ),
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

    results_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nMXBAI retrieval completed!")

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
        results_df["query_id"].nunique(),
    )

    print(
        "Number of rows:",
        len(results_df),
    )

    validate_results(
        results_df
    )