import pandas as pd
import re

df = pd.read_parquet(
    r"C:\Users\Panagiotis\Documents\recruitment-multistage-ranking\data\candidate_profiles_english_clean_20k.parquet"
)

pattern = re.compile(r"[\u0400-\u04FF]")

rows_with_cyrillic = []

for idx, row in df.iterrows():

    text = " ".join(
        str(v)
        for v in row.values
        if v is not None
    )

    if pattern.search(text):
        rows_with_cyrillic.append(idx)

print("Rows with Cyrillic:", len(rows_with_cyrillic))

if rows_with_cyrillic:
    print(rows_with_cyrillic[:20])