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

import os
from pathlib import Path

import pandas as pd


# --------- ENV / CONFIG ---------
CASES_FILE = os.getenv("CASES_FILE", "khcc_cases_200.xlsx")   # optional
NOTES_FILE = os.getenv("NOTES_FILE", "KHCC_AI_Notes.xlsx")    # required
OUT_FILE   = os.getenv("OUT_FILE", "khcc_eval_input.xlsx")    # output we CREATE


LETTER_MAP = {
    "gpt": "A",
    "claude": "B",
    "cadss": "C",
    "deepseek": "D",
    "human": "E",
}


def find_col(df: pd.DataFrame, candidates, required: bool = True):
    """
    Return the first matching column name from `candidates` that exists in df.
    Raise a clear error if required and not found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the columns {candidates} were found in {NOTES_FILE}")
    return None


def main():
    # --- sanity check: notes file must exist ---
    notes_path = Path(NOTES_FILE)
    if not notes_path.exists():
        raise FileNotFoundError(f"Cannot find AI notes file: {NOTES_FILE}")

    print(f"Loading AI notes from: {notes_path}")

    notes_df = pd.read_excel(notes_path)

    print("Columns in AI notes file:", list(notes_df.columns))

    # --- identify column names robustly ---
    case_col = find_col(notes_df, ["case_id", "Case_ID", "Case Id", "CaseID"])
    summary_col = find_col(
        notes_df,
        ["patient_summary", "Patient_Summary", "summary"],
        required=False,
    )

    gpt_col      = find_col(notes_df, ["gpt_note", "OpenAI_Note", "GPT_Note", "ChatGPT_Note"])
    claude_col   = find_col(notes_df, ["claude_note", "Claude_Note"])
    deepseek_col = find_col(notes_df, ["deepseek_note", "DeepSeek_Note"])
    cadss_col    = find_col(notes_df, ["cadss_note", "CADSS_Note", "Agent_Note"])
    human_col    = find_col(notes_df, ["human_note", "Original_Note", "Pharmacist_Note"])

    # --- build tall evaluation rows ---
    rows = []

    for _, row in notes_df.iterrows():
        cid = row[case_col]
        summary = ""
        if summary_col is not None:
            val = row[summary_col]
            summary = "" if pd.isna(val) else str(val)

        def add_note(source_key: str, col_name: str):
            note_val = row.get(col_name, "")
            if pd.isna(note_val):
                note_text = ""
            else:
                note_text = str(note_val).strip()

            # Skip completely empty notes
            if not note_text:
                return

            rows.append(
                {
                    "case_id": cid,
                    "patient_summary": summary,
                    "note_label": LETTER_MAP[source_key],   # A/B/C/D/E
                    "note_source": source_key,              # gpt / claude / cadss / deepseek / human
                    "note_text": note_text,
                }
            )

        add_note("gpt", gpt_col)
        add_note("claude", claude_col)
        add_note("cadss", cadss_col)
        add_note("deepseek", deepseek_col)   # NOTE D
        add_note("human", human_col)         # NOTE E

    eval_df = pd.DataFrame(rows)

    if eval_df.empty:
        raise RuntimeError("No evaluation rows were created – check your input columns.")

    out_path = Path(OUT_FILE)
    eval_df.to_excel(out_path, index=False)
    print(f"Saved evaluation input file with {len(eval_df)} rows to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
