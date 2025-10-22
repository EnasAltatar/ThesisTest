import pandas as pd
from pathlib import Path

# --- configuration ---
INPUT_FILE  = "AI RECOMMENDATIONS.csv"
OUTPUT_FILE = "AI_RECOMMENDATIONS_CLEAN.csv"

# --- read raw file safely ---
with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = [line.strip() for line in f if line.strip()]

rows = []
for line in lines:
    parts = line.split(",")
    # skip garbage or empty headers
    if len(parts) < 8:
        continue
    # heuristic: valid lines usually start with a case_id (not '-')
    if parts[0].startswith("-") or parts[0].startswith("##") or parts[0].startswith("Task"):
        continue
    rows.append(parts[:12])

# --- rebuild clean dataframe ---
cols = [
    "case_id",
    "diagnosis_subtype",
    "regimen",
    "source_key",
    "stage",
    "model_id",
    "temperature",
    "status",
    "error_message",
    "recommendation_text",
    "review_json",
    "created_at_utc",
]

df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
df = df.dropna(how="all")

# --- trim spaces & clean newlines ---
for col in df.columns:
    df[col] = df[col].astype(str).str.replace("\n", " ").str.strip()

# --- reorder columns for human-evaluation tool ---
ordered = [
    "case_id",
    "source_key",
    "stage",
    "model_id",
    "status",
    "temperature",
    "recommendation_text",
    "diagnosis_subtype",
    "regimen",
    "error_message",
    "created_at_utc",
]
df = df.reindex(columns=ordered, fill_value="")

# --- save clean version ---
out_path = Path(OUTPUT_FILE)
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ Clean CSV saved to {out_path.resolve()} ({len(df)} rows)")
