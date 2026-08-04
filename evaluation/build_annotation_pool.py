from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "candidate_profiles_english_clean_20k_with_text.parquet"
)

RANKING_FILES = {

    "bm25plus_stopwords": (
    PROJECT_ROOT
    / "sparse_retrieval"
    / "bm25plus_top50.csv"
),

    "bge": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "bge_m3_top50.csv"
    ),
    "mxbai": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "mxbai_top50.csv"
    ),
    "jina": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "jina_v5_text_small_top50.csv"
    ),
    "nomic": (
        PROJECT_ROOT
        / "dense_retrieval"
        / "nomic_embed_text_v1_5_top50.csv"
    ),
}


OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "annotation_pool_top50_for_annotation.csv"
)

EXPECTED_TOP_K = 50

CANDIDATE_COLUMNS = [
    "Position",
    "Moreinfo",
    "Looking For",
    "Highlights",
    "English Level",
    "Experience Years",
    "CV",
    "candidate_text",
]

def validate_file_exists(path: Path, name: str) -> None:
    """Raise an error when a required input file does not exist."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} file was not found:\n{path}"
        )



def clean_profile_value(value) -> str:
    """Convert missing values to an empty string and normalise whitespace."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_candidate_profile(row: pd.Series) -> str:
    """Create one readable profile for manual relevance annotation."""

    profile_sections = [
        ("Position", row.get("Position")),
        ("Moreinfo", row.get("Moreinfo")),
        ("Looking For", row.get("Looking For")),
        ("Highlights", row.get("Highlights")),
        ("English Level", row.get("English Level")),
        ("Experience Years", row.get("Experience Years")),
        ("CV", row.get("CV")),
    ]

    formatted_sections = []

    for label, value in profile_sections:
        cleaned_value = clean_profile_value(value)

        if cleaned_value:
            formatted_sections.append(
                f"{label}:\n{cleaned_value}"
            )

    return "\n\n".join(formatted_sections)

def load_ranking(path: Path, system_name: str) -> pd.DataFrame:
    """
    Load and standardise one ranking file.

    Each output row represents one candidate retrieved for one query.
    """

    ranking = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    required_columns = {
        "query_id",
        "category",
        "query",
        "candidate_id",
        "dataset_index",
        "rank",
    }

    missing_columns = required_columns - set(ranking.columns)

    if missing_columns:
        raise ValueError(
            f"{system_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_count = ranking.duplicated(
        subset=["query_id", "candidate_id"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{system_name} contains {duplicate_count} duplicate "
            "query/candidate pairs."
        )

    results_per_query = ranking.groupby("query_id").size()

    if not (results_per_query == EXPECTED_TOP_K).all():
        problematic_queries = results_per_query[
            results_per_query != EXPECTED_TOP_K
        ]

        raise ValueError(
            f"{system_name} does not contain exactly "
            f"{EXPECTED_TOP_K} candidates for every query:\n"
            f"{problematic_queries}"
        )

    # Keep only the common fields needed for pooling.
    ranking = ranking[
        [
            "query_id",
            "category",
            "query",
            "candidate_id",
            "dataset_index",
            "rank",
        ]
    ].copy()

    ranking["system"] = system_name

    return ranking


validate_file_exists(
    DATASET_PATH,
    "Candidate dataset",
)

print("\nRetrieval systems included in this annotation pool:")
for system_name, ranking_path in RANKING_FILES.items():
    print(f"  - {system_name}: {ranking_path}")
    validate_file_exists(
        ranking_path,
        system_name,
    )

candidates = pd.read_parquet(
    DATASET_PATH
).reset_index(drop=True)

missing_candidate_columns = (
    set(CANDIDATE_COLUMNS)
    - set(candidates.columns)
)

if missing_candidate_columns:
    raise ValueError(
        "The candidate dataset is missing columns: "
        f"{sorted(missing_candidate_columns)}"
    )


candidates["candidate_id"] = (
    candidates.index.astype(str)
)

candidates["dataset_index"] = candidates.index


candidate_metadata = candidates[
    [
        "candidate_id",
        "dataset_index",
        *CANDIDATE_COLUMNS,
    ]
].copy()


candidate_metadata["candidate_profile"] = (
    candidate_metadata.apply(
        build_candidate_profile,
        axis=1,
    )
)

empty_candidate_profiles = (
    candidate_metadata["candidate_profile"]
    .str.strip()
    .eq("")
    .sum()
)

if empty_candidate_profiles > 0:
    raise ValueError(
        f"Found {empty_candidate_profiles} candidates with an empty "
        "candidate_profile."
    )

ranking_frames = []

for system_name, ranking_path in RANKING_FILES.items():
    ranking = load_ranking(
        path=ranking_path,
        system_name=system_name,
    )

    ranking_frames.append(ranking)


all_rankings = pd.concat(
    ranking_frames,
    ignore_index=True,
)


reference_queries = set(
    ranking_frames[0]["query_id"].unique()
)

for system_name, ranking in zip(
    RANKING_FILES.keys(),
    ranking_frames,
):
    current_queries = set(
        ranking["query_id"].unique()
    )

    if current_queries != reference_queries:
        missing_queries = (
            reference_queries - current_queries
        )

        extra_queries = (
            current_queries - reference_queries
        )

        raise ValueError(
            f"Query mismatch in {system_name}.\n"
            f"Missing queries: {sorted(missing_queries)}\n"
            f"Extra queries: {sorted(extra_queries)}"
        )


rank_table = all_rankings.pivot(
    index=[
        "query_id",
        "candidate_id",
    ],
    columns="system",
    values="rank",
).reset_index()

rank_table = rank_table.rename(
    columns={
        system_name: f"{system_name}_rank"
        for system_name in RANKING_FILES
    }
)


RANK_COLUMNS = [
    f"{system_name}_rank"
    for system_name in RANKING_FILES
]

for rank_column in RANK_COLUMNS:
    if rank_column not in rank_table.columns:
        rank_table[rank_column] = pd.NA


query_metadata = (
    all_rankings[
        [
            "query_id",
            "category",
            "query",
        ]
    ]
    .drop_duplicates()
)


query_metadata_conflicts = (
    query_metadata
    .groupby("query_id")
    .size()
)

if not (query_metadata_conflicts == 1).all():
    raise ValueError(
        "Some query IDs have conflicting category or query text."
    )


retrieval_sources = (
    all_rankings
    .groupby(
        [
            "query_id",
            "candidate_id",
        ]
    )["system"]
    .agg(
        lambda systems: ";".join(
            sorted(set(systems))
        )
    )
    .reset_index(
        name="retrieved_by_systems"
    )
)

retrieval_sources["retrieval_count"] = (
    retrieval_sources["retrieved_by_systems"]
    .str.split(";")
    .str.len()
)


pool = rank_table.merge(
    query_metadata,
    on="query_id",
    how="left",
    validate="many_to_one",
)

pool = pool.merge(
    retrieval_sources,
    on=[
        "query_id",
        "candidate_id",
    ],
    how="left",
    validate="one_to_one",
)

pool = pool.merge(
    candidate_metadata,
    on="candidate_id",
    how="left",
    validate="many_to_one",
)


if "dataset_index" not in pool.columns:
    raise ValueError(
        "dataset_index was not added from the candidate dataset."
    )

missing_dataset_indices = pool["dataset_index"].isna().sum()

if missing_dataset_indices > 0:
    raise ValueError(
        f"Found {missing_dataset_indices} pool rows without "
        "a matching dataset_index."
    )

pool["relevance"] = ""
pool["annotation_notes"] = ""

pool["best_retrieval_rank"] = (
    pool[RANK_COLUMNS]
    .min(
        axis=1,
        skipna=True,
    )
)

pool = pool.sort_values(
    by=[
        "query_id",
        "best_retrieval_rank",
        "candidate_id",
    ],
    ascending=[
        True,
        True,
        True,
    ],
).reset_index(drop=True)


final_columns = [
    "query_id",
    "category",
    "query",
    "candidate_id",
    "dataset_index",

    "candidate_profile",

    *CANDIDATE_COLUMNS,

    "retrieved_by_systems",
    "retrieval_count",

    *RANK_COLUMNS,

    "relevance",
    "annotation_notes",
]

pool = pool[final_columns]


duplicate_pool_rows = pool.duplicated(
    subset=[
        "query_id",
        "candidate_id",
    ]
).sum()

if duplicate_pool_rows > 0:
    raise ValueError(
        f"The final pool contains {duplicate_pool_rows} "
        "duplicate query/candidate pairs."
    )


missing_profiles = (
    pool["candidate_profile"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

if missing_profiles > 0:
    raise ValueError(
        f"Candidate metadata could not be found for "
        f"{missing_profiles} pool rows."
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pool.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

pool_sizes = (
    pool
    .groupby("query_id")
    .size()
)

print("=" * 70)
print("ANNOTATION POOL CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Retrieval systems: {len(RANKING_FILES)}")
print(f"Queries: {pool['query_id'].nunique()}")
print(f"Unique query/candidate pairs: {len(pool)}")

print(
    "Average unique candidates per query: "
    f"{pool_sizes.mean():.2f}"
)

print(
    "Minimum unique candidates for one query: "
    f"{pool_sizes.min()}"
)

print(
    "Maximum unique candidates for one query: "
    f"{pool_sizes.max()}"
)

print(f"Duplicate pairs: {duplicate_pool_rows}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nPool size per query:")
print(pool_sizes.to_string())