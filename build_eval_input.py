"""
build_eval_input.py  – FINAL

Takes the wide AI-notes file and prepares a tall evaluation file.

Inputs (env vars, with defaults):
- CASES_FILE: Excel file with KHCC cases       (optional, used only if needed)
- NOTES_FILE: Excel file with AI notes (wide)  [required]
- OUT_FILE:   Output Excel for evaluation, default "khcc_eval_input.xlsx"

Expected NOTES_FILE columns (any of these names will be accepted):
- case_id:        ["case_id", "Case_ID", "Case Id", "CaseID"]
- patient_summary:["patient_summary", "Patient_Summary", "summary"]
- GPT note:       ["gpt_note", "OpenAI_Note", "GPT_Note", "ChatGPT_Note"]
- Claude note:    ["claude_note", "Claude_Note"]
- DeepSeek note:  ["deepseek_note", "DeepSeek_Note"]
- CADSS note:     ["cadss_note", "CADSS_Note", "Agent_Note"]
- Human note:     ["human_note", "Original_Note", "Pharmacist_Note"]

Output format (one row per note per case):
- case_id          (string / int)
- patient_summary  (string, if available)
- note_label       (A/B/C/D/E)
- note_source      ("gpt", "claude", "cadss", "deepseek", "human")
- note_text        (full note text)
"""

"""
build_eval_input.py
-------------------
Takes wide AI-notes file (one row per case with 5 note columns)
and produces a long blinded file for the Streamlit evaluation app.

Input (default):  KHCC_AI_Notes.xlsx
Output (default): khcc_eval_input.xlsx
"""

import os
import hashlib
import random
from pathlib import Path

import pandas as pd


# --------- Config from env (with defaults) ----------
NOTES_FILE = os.getenv("NOTES_FILE", "KHCC_AI_Notes.xlsx")
OUT_FILE = os.getenv("OUT_FILE", "khcc_eval_input.xlsx")


def deterministic_shuffle(case_id, labels):
    """
    Shuffle labels in a deterministic way per case_id,
    so the blinding is stable across runs.
    """
    seed_src = str(case_id)
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed)
    labels = list(labels)
    rnd.shuffle(labels)
    return labels


def main():
    print(f"Loading notes from: {NOTES_FILE}")
    if not Path(NOTES_FILE).is_file():
        raise FileNotFoundError(f"Cannot find input file: {NOTES_FILE}")

    df = pd.read_excel(NOTES_FILE)

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}

    def col(*names, required=True):
        for n in names:
            if n in cols:
                return cols[n]
        if required:
            raise KeyError(
                f"Required column not found. Tried names: {', '.join(names)}. "
                f"Available columns: {list(df.columns)}"
            )
        return None

    case_col = col("case_id", "case id", "cid")
    gpt_col = col("gpt_note", "gpt note", "chatgpt_note", "gpt")
    claude_col = col("claude_note", "claude note", "claude")
    deepseek_col = col("deepseek_note", "deepseek note", "ds_note", "deepseek")
    cadss_col = col("cadss_note", "cadss note", "consensus_note", "cadss")
    human_col = col("human_note", "human note", "reference_note", "pharmacist_note")

    patient_summary_col = col("patient_summary", "summary", required=False)

    # Map logical sources → column names
    source_defs = [
        ("GPT-4o-mini", gpt_col),
        ("Claude-3.5-Sonnet", claude_col),
        ("DeepSeek-Chat", deepseek_col),
        ("CADSS", cadss_col),
        ("Human pharmacist (KHCC baseline)", human_col),
    ]

    out_rows = []

    for _, row in df.iterrows():
        case_id = row[case_col]
        case_id_str = str(case_id).strip()

        # Skip rows with missing case id
        if not case_id_str:
            continue

        # Build list of available sources for this case
        per_case_sources = []
        for pretty_name, col_name in source_defs:
            text = row.get(col_name, None)
            if pd.isna(text) or (isinstance(text, str) and not text.strip()):
                # Skip completely empty notes
                continue
            per_case_sources.append(
                {
                    "source_pretty": pretty_name,
                    "source_key": col_name,
                    "note_text": str(text),
                }
            )

        if not per_case_sources:
            # No notes for this case, skip
            continue

        # Determine labels A–E (only as many as we have notes)
        base_labels = ["A", "B", "C", "D", "E"][: len(per_case_sources)]
        shuffled_labels = deterministic_shuffle(case_id_str, base_labels)

        patient_summary = (
            str(row[patient_summary_col]) if patient_summary_col and not pd.isna(row[patient_summary_col]) else ""
        )

        # Assign each note a label according to shuffled_labels
        for label, src in zip(shuffled_labels, per_case_sources):
            out_rows.append(
                {
                    "case_id": case_id_str,
                    "note_label": label,             # A / B / C / D / E
                    "note_text": src["note_text"],
                    "note_source": src["source_key"],        # internal column name (hidden)
                    "note_source_full": src["source_pretty"],  # human-readable (for analysis only)
                    "patient_summary": patient_summary,
                }
            )

    out_df = pd.DataFrame(out_rows)

    # Sort by case then label just for neatness
    out_df.sort_values(by=["case_id", "note_label"], inplace=True)

    out_df.to_excel(OUT_FILE, index=False)
    print(f"Saved blinded evaluation file to: {OUT_FILE}")
    print(f"Total rows: {len(out_df)}; unique cases: {out_df['case_id'].nunique()}")


if __name__ == "__main__":
    main()
