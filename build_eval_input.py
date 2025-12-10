"""
build_eval_input.py

Transforms the raw KHCC notes file into a blinded evaluation input for the
Streamlit app + a separate key mapping file.

INPUT
------
khcc_eval_input.xlsx

Expected columns (case-insensitive):
- Case_ID
- OpenAI_Note
- Claude_Note
- DeepSeek_Note
- CADSS_Note
- Original_Note
- (optional) Patient_Summary / Case_Summary / Summary

OUTPUT
------
1) khcc_eval_cases.xlsx
   Columns:
   - case_id
   - patient_summary
   - note_a  (OpenAI / ChatGPT)
   - note_b  (Claude)
   - note_c  (CADSS / agent)
   - note_d  (DeepSeek)
   - note_e  (Human / original)

   This is the file you upload into the Streamlit evaluation app.

2) khcc_eval_note_key.xlsx
   Columns:
   - case_id
   - note_label   (A–E)
   - note_source  (openai / claude / cadss / deepseek / human)

   This is for your own analysis later when you join with the exported
   evaluations (which only contain case_id + note_label).
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------
INPUT_FILE = Path("khcc_eval_input.xlsx")
EVAL_CASES_FILE = Path("khcc_eval_cases.xlsx")
NOTE_KEY_FILE = Path("khcc_eval_note_key.xlsx")


def _find_col(cols_lower_map, *candidates):
    """
    Helper: given a dict {lower_name: original_name} and a list of candidate
    lower-case names, return the first existing original name.
    Raise KeyError if none are found.
    """
    for cand in candidates:
        if cand in cols_lower_map:
            return cols_lower_map[cand]
    raise KeyError(f"None of the required columns found: {candidates}")


def _clean_note_series(s: pd.Series) -> pd.Series:
    """
    Convert to string, strip, and remove rows where generation failed
    (e.g., starts with '[ERROR] RetryError').
    """
    s = s.fillna("").astype(str).str.strip()
    mask_error = s.str.startswith("[ERROR]")
    s.loc[mask_error] = ""
    return s


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    print(f"Reading input file: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    # Build mapping from lower-case column name -> original column name
    cols_lower = {c.lower(): c for c in df.columns}

    # Mandatory columns (with flexible naming)
    case_col = _find_col(cols_lower, "case_id", "case", "id")

    openai_col = _find_col(
        cols_lower,
        "openai_note",
        "gpt_note",
        "chatgpt_note",
        "gpt4o_note",
    )
    claude_col = _find_col(cols_lower, "claude_note")
    deepseek_col = _find_col(cols_lower, "deepseek_note", "deepseek_v3_note")
    cadss_col = _find_col(
        cols_lower,
        "cadss_note",
        "agent_note",
        "ai_agent_note",
        "collaborative_note",
    )
    human_col = _find_col(
        cols_lower,
        "original_note",
        "human_note",
        "pharmacist_note",
        "clinical_pharmacist_note",
    )

    # Optional patient summary column; if missing, fallback to human note
    try:
        summary_col = _find_col(
            cols_lower,
            "patient_summary",
            "case_summary",
            "summary",
            "case_description",
        )
    except KeyError:
        summary_col = human_col
        print(
            "⚠️  No explicit patient summary column found; "
            "using the human/original note as patient_summary."
        )

    # Clean notes (remove [ERROR] rows)
    df[openai_col] = _clean_note_series(df[openai_col])
    df[claude_col] = _clean_note_series(df[claude_col])
    df[deepseek_col] = _clean_note_series(df[deepseek_col])
    df[cadss_col] = _clean_note_series(df[cadss_col])
    df[human_col] = _clean_note_series(df[human_col])

    # ------------------------------------------------------------------
    # Build evaluation cases file (wide, one row per case)
    # ------------------------------------------------------------------
    eval_df = pd.DataFrame(
        {
            "case_id": df[case_col].astype(str).str.strip(),
            "patient_summary": df[summary_col],
            "note_a": df[openai_col],
            "note_b": df[claude_col],
            "note_c": df[cadss_col],
            "note_d": df[deepseek_col],
            "note_e": df[human_col],
        }
    )

    # Optional: drop rows where all notes are empty
    all_empty = (
        eval_df[["note_a", "note_b", "note_c", "note_d", "note_e"]]
        .apply(lambda row: all(not str(v).strip() for v in row), axis=1)
    )
    before = len(eval_df)
    eval_df = eval_df[~all_empty].reset_index(drop=True)
    dropped = before - len(eval_df)
    if dropped:
        print(f"Dropped {dropped} rows where all notes were empty.")

    print(f"Writing evaluation cases file: {EVAL_CASES_FILE}")
    eval_df.to_excel(EVAL_CASES_FILE, index=False)

    # ------------------------------------------------------------------
    # Build note key mapping (long format)
    # ------------------------------------------------------------------
    key_rows = []

    mapping = [
        ("A", "openai", openai_col),
        ("B", "claude", claude_col),
        ("C", "cadss", cadss_col),
        ("D", "deepseek", deepseek_col),
        ("E", "human", human_col),
    ]

    for label, source, col in mapping:
        tmp = pd.DataFrame(
            {
                "case_id": df[case_col].astype(str).str.strip(),
                "note_label": label,
                "note_source": source,
            }
        )
        key_rows.append(tmp)

    key_df = pd.concat(key_rows, ignore_index=True).drop_duplicates()

    print(f"Writing note key mapping file: {NOTE_KEY_FILE}")
    key_df.to_excel(NOTE_KEY_FILE, index=False)

    print("\nDone.")
    print(f"- {len(eval_df)} cases written to: {EVAL_CASES_FILE}")
    print(f"- {len(key_df)} mapping rows written to: {NOTE_KEY_FILE}")


if __name__ == "__main__":
    main()
