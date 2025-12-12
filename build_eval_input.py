"""
build_eval_input.py

Transforms KHCC_AI_Notes.xlsx into a blinded evaluation format.

INPUT:
  - khcc_cases_200.xlsx (optional: original cases with metadata)
  - KHCC_AI_Notes.xlsx (AI-generated notes with patient_summary column)

OUTPUT:
  - khcc_eval_input.xlsx (blinded notes for human evaluation)

Format:
  - Each case generates 5 rows (one per model: GPT, Claude, DeepSeek, CADSS, Human)
  - Randomly assigned labels (A, B, C, D, E)
  - Includes patient_summary for context
"""

import os
import random
import pandas as pd
from pathlib import Path

# --- Configuration ---
CASES_FILE = os.getenv("CASES_FILE", "khcc_cases_200.xlsx")
NOTES_FILE = os.getenv("NOTES_FILE", "KHCC_AI_Notes.xlsx")
OUT_FILE = os.getenv("OUT_FILE", "khcc_eval_input.xlsx")

RANDOM_SEED = 42  # For reproducible label shuffling

# Model mappings
MODEL_LABELS = {
    "gpt": "GPT-4o-mini",
    "claude": "Claude Sonnet 4",
    "deepseek": "DeepSeek Chat",
    "cadss": "CADSS – Collaborative AI consensus note",
    "human": "Human KHCC Pharmacist",
}

# --- Load data ---
print(f"📂 Loading AI notes from: {NOTES_FILE}")
df_notes = pd.read_excel(NOTES_FILE)

print(f"   Columns found: {df_notes.columns.tolist()}")
print(f"   Rows: {len(df_notes)}")

# Validate required columns
required_cols = ["case_id", "patient_summary", "gpt_note", "claude_note", 
                 "deepseek_note", "cadss_note", "human_note"]
missing_cols = [col for col in required_cols if col not in df_notes.columns]

if missing_cols:
    raise ValueError(f"❌ Missing required columns in {NOTES_FILE}: {missing_cols}")

# --- Transform to evaluation format ---
print(f"\n🔄 Transforming to evaluation format...")

random.seed(RANDOM_SEED)
eval_rows = []

for idx, row in df_notes.iterrows():
    case_id = row["case_id"]
    patient_summary = row["patient_summary"]
    
    # Check if patient_summary is empty or NaN
    if pd.isna(patient_summary) or not patient_summary or patient_summary.strip() == "":
        print(f"   ⚠️  WARNING: Case {case_id} has empty patient_summary!")
        patient_summary = "[No summary available]"
    
    # Create 5 notes for this case
    notes = [
        {"source": "gpt", "text": row["gpt_note"]},
        {"source": "claude", "text": row["claude_note"]},
        {"source": "deepseek", "text": row["deepseek_note"]},
        {"source": "cadss", "text": row["cadss_note"]},
        {"source": "human", "text": row["human_note"]},
    ]
    
    # Shuffle and assign labels A-E
    random.shuffle(notes)
    labels = ["A", "B", "C", "D", "E"]
    
    for label, note in zip(labels, notes):
        eval_rows.append({
            "case_id": case_id,
            "note_label": label,
            "note_text": note["text"],
            "patient_summary": patient_summary,  # Include the summary here
            "note_source": note["source"],
            "note_source_full": MODEL_LABELS[note["source"]],
        })
    
    if (idx + 1) % 10 == 0:
        print(f"   Processed {idx + 1}/{len(df_notes)} cases...")

# --- Create output dataframe ---
df_eval = pd.DataFrame(eval_rows)

print(f"\n📊 Evaluation format created:")
print(f"   Total rows: {len(df_eval)} ({len(df_notes)} cases × 5 notes)")
print(f"   Columns: {df_eval.columns.tolist()}")

# Check patient_summary column
summary_empty = df_eval["patient_summary"].isna().sum()
if summary_empty > 0:
    print(f"   ⚠️  WARNING: {summary_empty} rows have empty patient_summary!")
else:
    print(f"   ✅ All rows have patient_summary data")

# --- Save output ---
out_path = Path(OUT_FILE)
df_eval.to_excel(out_path, index=False)

print(f"\n✅ Evaluation input saved to: {out_path.resolve()}")
print(f"   Ready for blind evaluation!\n")

# --- Summary statistics ---
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Cases processed: {len(df_notes)}")
print(f"Total evaluation rows: {len(df_eval)}")
print(f"Labels per case: 5 (A, B, C, D, E)")
print(f"Models: GPT, Claude, DeepSeek, CADSS, Human")
print(f"\nSample patient_summary (first case):")
print("-" * 70)
first_summary = df_eval.iloc[0]["patient_summary"]
if first_summary and not pd.isna(first_summary):
    print(first_summary[:300] + "..." if len(str(first_summary)) > 300 else first_summary)
else:
    print("[EMPTY - This is a problem!]")
print("=" * 70)
