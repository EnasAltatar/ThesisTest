"""
preprocess_khcc_data.py
----------------------------------------
Official preprocessing pipeline for KHCC clinical pharmacist notes.

Author: Enas Altatar
Supervisors: Dr. Bassam Hammo, Dr. Abdullah Qusef
Date: 2025

Purpose:
--------
This script converts the raw KHCC extract (merged_labs_pharmacy.xlsx)
into a clean, structured format where:

    1 row = 1 clinical pharmacist note

Each note aggregates:
- The full pharmacist recommendation text
- All associated laboratory results
- Metadata (MRN, visit, location, timestamps)
- A unique case_id for downstream AI generation

This script produces two output files:
1) data/interim/pharmacy_notes_with_labs.xlsx
2) data/processed/khcc_cases_for_ai.xlsx   (ready for LLM generation)

Inputs:
-------
data/raw/merged_labs_pharmacy.xlsx

Outputs:
--------
See above.

Usage:
------
python scripts/preprocess_khcc_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
RAW_PATH = Path("data/raw/merged_labs_pharmacy.xlsx")
INTERIM_PATH = Path("data/interim/pharmacy_notes_with_labs.xlsx")
PROCESSED_PATH = Path("data/processed/khcc_cases_for_ai.xlsx")

INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Step 1 — Load KHCC extract
# -------------------------------------------------------------------
def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw KHCC pharmacist extract."""
    print(f"Loading raw dataset: {path}")
    df = pd.read_excel(path)
    print(f"Loaded {df.shape[0]} rows.")
    return df


# -------------------------------------------------------------------
# Step 2 — Aggregate labs per note
# -------------------------------------------------------------------
def aggregate_labs(group: pd.DataFrame) -> str:
    """Convert all lab rows into a single semicolon-separated string."""
    labs = []
    for _, row in group.iterrows():
        name = str(row.get("TEST_NAME", "")).strip()
        result = str(row.get("TEST_RESULT", "")).strip()
        if not name:
            continue
        labs.append(f"{name}: {result}")
    return "; ".join(labs)


# -------------------------------------------------------------------
# Step 3 — Produce note-level table
# -------------------------------------------------------------------
NOTE_KEYS = [
    "MRN", "Document_Number", "DOCUMENT_TYPE", "Entry_Date",
    "Visit", "VISIT_LOCATION", "SERVICE",
    "Parent_Number", "Parent_Type", "HOSPITAL_LOCATION",
    "AUTHOR_SERVICE", "Visit_Number",
    "Has_Clinical_Recommendation", "Note"
]

def build_note_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per pharmacist note with aggregated lab results."""
    print("Building note-level table...")
    grouped = df.groupby(["MRN", "Document_Number"], dropna=False)

    rows = []
    for (mrn, doc), group in grouped:
        base = group.iloc[0][NOTE_KEYS].copy()
        labs_text = aggregate_labs(group)
        base["LABS_TEXT"] = labs_text
        base["N_LABS"] = len(group["TEST_NAME"].dropna())
        rows.append(base)

    notes_df = pd.DataFrame(rows)
    print(f"Created {notes_df.shape[0]} unique notes.")
    return notes_df


# -------------------------------------------------------------------
# Step 4 — Final formatting for LLM generator
# -------------------------------------------------------------------
def format_for_llm(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final table matching the expected AI generator schema.
    Fields like diagnosis_subtype or regimen are missing in KHCC file
    but can be extracted later from the Note text or left blank.
    """

    print("Formatting for LLM generator...")

    out = pd.DataFrame()
    out["case_id"] = notes_df.apply(
        lambda r: f"{r['MRN']}_{r['Document_Number']}", axis=1
    )

    # Fields required by generator (empty for now)
    out["diagnosis_subtype"] = ""
    out["regimen"] = ""
    out["cycle"] = ""
    out["bsa"] = ""
    out["lvef"] = ""
    out["crcl"] = ""
    out["ast"] = ""
    out["alt"] = ""
    out["tbil"] = ""
    out["comorbidities"] = ""
    out["meds"] = ""

    # Keep original note text & labs for extraction stage
    out["note_text"] = notes_df["Note"]
    out["labs_text"] = notes_df["LABS_TEXT"]

    print("Formatting complete.")
    return out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    df = load_raw_data(RAW_PATH)

    # Ensure only clinical recommendation notes are included
    df = df[df["Has_Clinical_Recommendation"].astype(str).str.lower() == "yes"]

    notes_df = build_note_level_table(df)
    notes_df.to_excel(INTERIM_PATH, index=False)
    print(f"Saved interim note-level file: {INTERIM_PATH}")

    ai_ready = format_for_llm(notes_df)
    ai_ready.to_excel(PROCESSED_PATH, index=False)
    print(f"Saved AI-ready file: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
