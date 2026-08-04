from __future__ import annotations

from typing import Any

import pandas as pd


TEXT_COLUMNS = [
    "Position",
    "Moreinfo",
    "Looking For",
    "Highlights",
    "English Level",
    "Experience Years",
    "CV",
]


def clean_value(value: Any) -> str:
    """
    Convert a dataframe value into clean text.

    Missing and empty values are ignored.
    """
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


def build_candidate_text(row: pd.Series) -> str:
    """
    Combine the selected candidate fields into one semantic document.

    Excluded from semantic text:
    - Primary Keyword
    - CV_lang
    - id
    - __index_level_0__
    """
    parts: list[str] = []

    for column in TEXT_COLUMNS:
        value = clean_value(row.get(column))

        if value:
            parts.append(f"{column}: {value}")

    return "\n".join(parts)