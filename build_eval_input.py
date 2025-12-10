# build_eval_input.py
#
# Create khcc_eval_input.xlsx for the blinded evaluation app.
#
# Inputs (in repo root by default):
#   - KHCC_AI_Notes.xlsx   (AI + human notes, structure exactly as you pasted)
#   - khcc_cases_200.xlsx  (OPTIONAL; used only to attach patient_summary)
#
# Output:
#   - khcc_eval_input.xlsx

import os
from pathlib import Path

import pandas as pd

AI_NOTES_FILE = os.getenv("AI_NOTES_FILE", "KHCC_AI_Notes.xlsx")
CASES_FILE = os.getenv("CASES_FILE", "khcc_cases_200.xlsx")
OUT_FILE = Path("khcc_eval_input.xlsx")


def main():
    # --------------------------------------------------
    # 1) Load AI + human notes
    # --------------------------------------------------
    print(f"Loading AI notes from: {AI_NOTES_FILE}")
    ai = pd.read_excel(AI_NOTES_FILE)

    required = [
        "Case_ID",
        "OpenAI_Note",
        "Claude_Note",
        "DeepSeek_Note",
        "CADSS_Note",
        "Original_Note",
    ]
    missing = [c for c in required if c not in ai.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {AI_NOTES_FILE}: {missing}. "
            f"Columns found: {list(ai.columns)}"
        )

    # Clean NaNs to empty strings
    for c in required:
        ai[c] = ai[c].fillna("")

    # OPTIONAL: drop rows where a generation failed and only contains "[ERROR...]"
    # (You can comment this block out if you want to keep them.)
    error_mask = ai["OpenAI_Note"].astype(str).str.startswith("[ERROR]")
    if error_mask.any():
        print(f"Dropping {error_mask.sum()} rows with OpenAI_Note starting with [ERROR].")
        ai = ai[~error_mask].copy()

    # Rename to stable schema used by the evaluation app
    ai = ai.rename(
        columns={
            "Case_ID": "case_id",
            "OpenAI_Note": "chatgpt_note",
            "Claude_Note": "claude_note",
            "DeepSeek_Note": "deepseek_note",
            "CADSS_Note": "cadss_note",
            "Original_Note": "human_note",
        }
    )

    ai["case_id"] = ai["case_id"].astype(str).str.strip()

    # --------------------------------------------------
    # 2) Try to attach patient summaries (optional)
    # --------------------------------------------------
    if Path(CASES_FILE).exists():
        print(f"Loading cases from: {CASES_FILE} to get patient summaries")
        cases = pd.read_excel(CASES_FILE)

        # Find the Case_ID-like column
        def norm(name: str) -> str:
            return (
                str(name)
                .strip()
                .lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
            )

        case_cols = [
            c
            for c in cases.columns
            if norm(c) in {"caseid", "case", "noteid"}
        ]
        if not case_cols:
            raise ValueError(
                f"Could not identify case-id column in {CASES_FILE}. "
                f"Columns: {list(cases.columns)}"
            )

        cases = cases.rename(columns={case_cols[0]: "case_id"})
        cases["case_id"] = cases["case_id"].astype(str).str.strip()

        # Try to find a patient summary column
        summary_cols = [
            c
            for c in cases.columns
            if norm(c)
            in {
                "patientsummary",
                "summary",
                "notewithoutrecommendations",
                "note_without_recommendations",
            }
        ]
        if summary_cols:
            cases = cases.rename(columns={summary_cols[0]: "patient_summary"})
            print(f"Using '{summary_cols[0]}' as patient_summary.")
            ai = ai.merge(
                cases[["case_id", "patient_summary"]],
                on="case_id",
                how="left",
            )
        else:
            print(
                "WARNING: No patient summary column found in cases file; "
                "patient_summary will be blank."
            )

    # If we still don't have patient_summary, create an empty one
    if "patient_summary" not in ai.columns:
        ai["patient_summary"] = ""

    # --------------------------------------------------
    # 3) Reorder columns and save
    # --------------------------------------------------
    cols = [
        "case_id",
        "patient_summary",
        "chatgpt_note",
        "claude_note",
        "deepseek_note",  # you can ignore this in the app if you want only A/B/C
        "cadss_note",
        "human_note",
    ]
    ai = ai[cols]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ai.to_excel(OUT_FILE, index=False)
    print(f"Saved {len(ai)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
