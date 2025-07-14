# format_converter
# This script will:
# 1. Read the Excel sheet
# 2. Combine fields into a clean document format
# 3. Save each abstract as a JSON object (one per row) into a .jsonl file
import pandas as pd
import json
from pathlib import Path

input_file = Path("/data/raw/aacr.xlsx")
output_file = Path("/data/processed/abstracts.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(input_file)
with output_file.open("w", encoding="utf-8") as f_out:
    for i, row in df.iterrows():
        doc = {
            "title": row["title"],
            "session": row["session"],
            "authors": [a.strip() for a in str(row["authors"]).split(",")],
            "text": str(row["abstract"]).strip(),
            "meta": {
                "id": f"row_{i:03d}",
                "tokens": len(str(row["abstract"]).split())  # rough count
            }
        }
        json.dump(doc, f_out, ensure_ascii=False)
        f_out.write("\n")
