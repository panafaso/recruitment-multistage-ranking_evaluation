"""Step 1: Load the frozen 100-query validation set.

All subsequent retrievers should call load_queries() rather than recreate IDs
from category names. The original model parameters are NOT changed here.
"""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
QUERY_CSV = ROOT / "queries_100.csv"
QUERY_JSON = ROOT / "queries_100.json"

def load_queries(path=QUERY_CSV):
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"query_id", "category", "query"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required CSV columns: {required}")
    expected_ids = [f"Q{i:03d}" for i in range(1, 101)]
    if [row["query_id"] for row in rows] != expected_ids:
        raise ValueError("Expected Q001-Q100 in order without missing or duplicate IDs.")
    if any(not row["query"].strip() for row in rows):
        raise ValueError("A recruitment query is blank.")
    if len({row["query"].casefold().strip() for row in rows}) != 100:
        raise ValueError("Duplicate query texts were found.")
    for i, row in enumerate(rows):
        expected_category = (
            "Corporate / Technology / Business" if i < 70
            else "Cross-Domain / Entry-Level"
        )
        if row["category"] != expected_category:
            raise ValueError(f"Unexpected category at {row['query_id']}")
    return rows

if __name__ == "__main__":
    queries = load_queries()
    print(f"Validated {len(queries)} queries")
    print(f"Q001: {queries[0]['query']}")
    print(f"Q100: {queries[-1]['query']}")
    print("70 corporate/technology/business + 30 cross-domain/entry-level")
