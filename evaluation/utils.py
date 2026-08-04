from pathlib import Path

import pandas as pd


QRELS_COLUMNS = {
    "query_id",
    "candidate_id",
    "relevance",
}

RANKING_COLUMNS = {
    "query_id",
    "candidate_id",
    "rank",
}


def validate_file_exists(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def load_qrels(path: Path) -> pd.DataFrame:
    qrels = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    missing = QRELS_COLUMNS - set(qrels.columns)

    if missing:
        raise ValueError(
            f"qrels.csv is missing: {sorted(missing)}"
        )

    qrels = qrels[
        [
            "query_id",
            "candidate_id",
            "relevance",
        ]
    ].copy()

    if qrels["relevance"].isna().any():
        raise ValueError(
            "qrels.csv contains missing relevance labels."
        )

    qrels["relevance"] = pd.to_numeric(
        qrels["relevance"],
        errors="raise",
    ).astype(int)

    invalid_labels = (
        set(qrels["relevance"].unique())
        - {0, 1, 2}
    )

    if invalid_labels:
        raise ValueError(
            "Invalid relevance labels: "
            f"{sorted(invalid_labels)}"
        )

    duplicates = qrels.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"qrels.csv contains {duplicates} duplicates."
        )

    return qrels


def load_ranking(
    path: Path,
    system_name: str,
) -> pd.DataFrame:
    ranking = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    missing = RANKING_COLUMNS - set(ranking.columns)

    if missing:
        raise ValueError(
            f"{system_name} is missing: "
            f"{sorted(missing)}"
        )

    ranking = ranking[
        [
            "query_id",
            "candidate_id",
            "rank",
        ]
    ].copy()

    ranking["rank"] = pd.to_numeric(
        ranking["rank"],
        errors="raise",
    ).astype(int)

    duplicates = ranking.duplicated(
        subset=[
            "query_id",
            "candidate_id",
        ]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"{system_name} contains {duplicates} duplicates."
        )

    if (ranking["rank"] < 1).any():
        raise ValueError(
            f"{system_name} contains ranks below 1."
        )

    return ranking.sort_values(
        by=[
            "query_id",
            "rank",
        ]
    ).reset_index(drop=True)