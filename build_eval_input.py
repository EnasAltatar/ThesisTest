"""
build_eval_input.py — FINAL VERSION FOR KHCC

Takes:
  1) CASES_FILE (not strictly needed now, but kept for future use)
  2) NOTES_FILE = KHCC_AI_Notes.xlsx (one row per case, with 5 notes)

Produces:
  khcc_eval_input.xlsx   (long, blinded, for Streamlit evaluation_app.py)
"""

import os
import random
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------
# Config from environment
# -------------------------------------------------------------------
CASES_FILE = os.getenv("CASES_FILE", "khcc_cases_200.xlsx")
CASES_SHEET = os.getenv("CASES_SHEET", 0)

NOTES_FILE = os.getenv("NOTES_FILE", "KHCC_AI_Notes.xlsx")
NOTES_SHEET = os.getenv("NOTES_SHEET", 0)

OUT_FILE = os.getenv("OUT_FILE", "khcc_eval_input.xlsx")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2025"))

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def col(df: pd.DataFrame, candidates, required=True):
    """
    Try several candidate column names and return the one that exists.
    Raise a clear error if none found and `required` is True.
    """
    if isinstance(candidates, str):
        candidates = [candidates]

    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(
            f"Required column not found. Tried names: {', '.join(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print(f"Running build_eval_input.py on {CASES_FILE} and {NOTES_FILE}")

    # ---- Load cases file (currently only used for potential future extensions) ----
    if Path(CASES_FILE).exists():
        try:
            cases_df = pd.read_excel(CASES_FILE, sheet_name=CASES_SHEET)
            print(f"Loaded cases file with {len(cases_df)} rows.")
        except Exception as e:
            print(f"Warning: could not read cases file ({CASES_FILE}): {e}")
            cases_df = None
    else:
        print(f"Warning: cases file {CASES_FILE} not found. Continuing without it.")
        cases_df = None

    # ---- Load notes file (KHCC_AI_Notes.xlsx) ----
    print(f"Loading notes from: {NOTES_FILE}")
    notes_df = pd.read_excel(NOTES_FILE, sheet_name=NOTES_SHEET)

    # Figure out the column names (supports both old & new naming)
    cid_col = col(notes_df, ["case_id", "Case_ID", "Case Id"])

    gpt_col = col(
        notes_df,
        ["gpt_note", "gpt note", "chatgpt_note", "gpt", "OpenAI_Note", "openai_note"],
    )
    claude_col = col(
        notes_df,
        ["claude_note", "claude note", "Claude_Note", "anthropic_note"],
    )
    deepseek_col = col(
        notes_df,
        ["deepseek_note", "deepseek note", "DeepSeek_Note"],
    )
    cadss_col = col(
        notes_df,
        ["cadss_note", "cadss note", "CADSS_Note"],
    )
    human_col = col(
        notes_df,
        ["human_note", "Original_Note", "original_note", "Reference_Note"],
    )

    print("Detected columns:")
    print(f"  case_id column     -> {cid_col}")
    print(f"  GPT/OpenAI column  -> {gpt_col}")
    print(f"  Claude column      -> {claude_col}")
    print(f"  DeepSeek column    -> {deepseek_col}")
    print(f"  CADSS column       -> {cadss_col}")
    print(f"  Human column       -> {human_col}")

    # Normalise Case IDs to string
    notes_df[cid_col] = notes_df[cid_col].astype(str).str.strip()

    # Mapping from internal source code to full description
    SOURCE_META = {
        "gpt": "OpenAI – ChatGPT-4o-mini (LLM)",
        "claude": "Anthropic – Claude 3.5 Sonnet (LLM)",
        "deepseek": "DeepSeek-Chat (LLM)",
        "cadss": "CADSS – Collaborative AI consensus note",
        "human": "Human clinical oncology pharmacist (KHCC)",
    }

    rng = random.Random(RANDOM_SEED)

    long_rows = []

    for _, row in notes_df.iterrows():
        raw_case = str(row[cid_col]).strip()
        # This is the label shown in the Streamlit app
        case_id = f"Case {raw_case}"

        notes_list = [
            ("gpt", str(row.get(gpt_col, "")) if pd.notna(row.get(gpt_col, "")) else ""),
            (
                "claude",
                str(row.get(claude_col, ""))
                if pd.notna(row.get(claude_col, ""))
                else "",
            ),
            (
                "deepseek",
                str(row.get(deepseek_col, ""))
                if pd.notna(row.get(deepseek_col, ""))
                else "",
            ),
            (
                "cadss",
                str(row.get(cadss_col, ""))
                if pd.notna(row.get(cadss_col, ""))
                else "",
            ),
            (
                "human",
                str(row.get(human_col, ""))
                if pd.notna(row.get(human_col, ""))
                else "",
            ),
        ]

        # Filter out completely empty notes (should not normally happen, but safe)
        notes_list = [(src, txt) for src, txt in notes_list if txt.strip() != ""]

        if not notes_list:
            print(f"Warning: no notes found for case {raw_case}; skipping.")
            continue

        # Randomise order per case, but reproducibly (same RANDOM_SEED)
        indices = list(range(len(notes_list)))
        rng.shuffle(indices)

        LABELS = ["A", "B", "C", "D", "E"]

        for label_idx, note_idx in enumerate(indices):
            if label_idx >= len(LABELS):
                break  # safety; should be exactly 5

            src, txt = notes_list[note_idx]
            long_rows.append(
                {
                    "case_id": case_id,
                    "note_label": LABELS[label_idx],
                    "note_text": txt,
                    # patient_summary not used by current app, but kept for compatibility
                    "patient_summary": "",
                    "note_source": src,
                    "note_source_full": SOURCE_META.get(src, src),
                }
            )

    out_df = pd.DataFrame(long_rows)

    if out_df.empty:
        raise RuntimeError("No rows generated for evaluation input – check your sources.")

    out_df.to_excel(OUT_FILE, index=False)
    print(f"Saved blinded evaluation file: {OUT_FILE} with {len(out_df)} rows.")


if __name__ == "__main__":
    main()
