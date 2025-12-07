import re
from pathlib import Path

import pandas as pd

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

# Repository root = parent of this file's folder
REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = REPO_ROOT / "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = REPO_ROOT / "khcc_preprocessed.xlsx"

# List of common breast-cancer chemo / targeted meds to look for in notes
CHEMO_DRUGS = [
    # Anthracyclines
    "doxorubicin",
    "adriamycin",
    "epirubicin",
    # Taxanes
    "paclitaxel",
    "taxol",
    "docetaxel",
    "taxotere",
    # Alkylating
    "cyclophosphamide",
    "ifosfamide",
    # Antimetabolites
    "methotrexate",
    "5-fu",
    "5 fluorouracil",
    "capecitabine",
    "xeloda",
    # HER2-targeted
    "trastuzumab",
    "herceptin",
    "pertuzumab",
    "perjeta",
    "lapatinib",
    "neratinib",
    # CDK4/6 inhibitors
    "palbociclib",
    "ribociclib",
    "abemaciclib",
    # Hormonal
    "tamoxifen",
    "letrozole",
    "anastrozole",
    "exemestane",
]

# Very simple comorbidity keyword list (can be expanded later)
COMORBIDITY_KEYWORDS = [
    "diabetes",
    "dm",
    "hypertension",
    "htn",
    "ischemic heart disease",
    "ihd",
    "coronary artery disease",
    "cad",
    "heart failure",
    "renal impairment",
    "ckd",
    "copd",
    "asthma",
    "hypothyroidism",
    "hyperthyroidism",
    "obesity",
]


# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------


def clean_text(text: str) -> str:
    """Basic cleaning: ensure string, strip, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove line breaks and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_lab_entry(row: pd.Series) -> str:
    """
    Build a one-line description for a single lab row.
    Example: "HB: 10.5 (result) | taken: 2024-01-01 | reported: 2024-01-02 | Δdays: 1"
    """
    test_name = clean_text(row.get("TEST_NAME", ""))
    result = clean_text(row.get("TEST_RESULT", ""))
    taken = clean_text(row.get("DATE/TIME SPECIMEN TAKEN", ""))
    reported = clean_text(row.get("DATE REPORT COMPLETED", ""))
    diff = row.get("date_diff_days", "")

    parts = []
    if test_name:
        parts.append(test_name)
    if result:
        parts.append(f"result={result}")
    if taken:
        parts.append(f"specimen={taken}")
    if reported:
        parts.append(f"reported={reported}")
    if diff != "" and not pd.isna(diff):
        parts.append(f"Δdays={diff}")

    return " | ".join(parts)


def extract_bsa(note_lower: str) -> str:
    """
    Extract BSA patterns such as:
    - "BSA 1.65"
    - "BSA=1.73 m2"
    Returns the first match as string, or "" if not found.
    """
    # Look for a number around "bsa"
    match = re.search(r"bsa[^0-9]*([1-3]\.\d{1,2})", note_lower)
    if match:
        return match.group(1)
    return ""


def extract_cycle(note_lower: str) -> str:
    """
    Extract cycle info such as:
    - "cycle 1"
    - "C1D1"
    - "C2 D8"
    """
    # C1D1 / C2D8 / C3 D1 etc.
    match = re.search(r"\bC(\d+)\s*D?(\d+)?", note_lower)
    if match:
        c = match.group(1)
        d = match.group(2)
        if d:
            return f"C{c}D{d}"
        return f"C{c}"

    # "cycle 3"
    match = re.search(r"cycle\s*(\d+)", note_lower)
    if match:
        return f"cycle {match.group(1)}"

    return ""


def find_keywords(note_lower: str, keywords: list) -> str:
    """
    Return a '; '-joined list of keywords that appear in the note.
    """
    found = []
    for kw in keywords:
        if kw.lower() in note_lower:
            found.append(kw)
    # Keep unique order
    seen = set()
    unique = []
    for x in found:
        if x.lower() not in seen:
            unique.append(x)
            seen.add(x.lower())
    return "; ".join(unique)


# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------


def main() -> None:
    print(f"Loading input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    print(f"Original rows: {len(df)}")

    # Ensure expected column exists
    expected_cols = [
        "MRN",
        "Document_Number",
        "DOCUMENT_TYPE",
        "Entry_Date",
        "Visit",
        "VISIT_LOCATION",
        "SERVICE",
        "Parent_Number",
        "Parent_Type",
        "HOSPITAL_LOCATION",
        "AUTHOR_SERVICE",
        "Note",
        "Visit_Number",
        "Has_Clinical_Recommendation",
        "DATE/TIME SPECIMEN TAKEN",
        "DATE REPORT COMPLETED",
        "TEST_NAME",
        "TEST_RESULT",
        "date_diff_days",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # --- Clean note text ---
    df["NOTE_TEXT"] = df["Note"].apply(clean_text)
    df["NOTE_LOWER"] = df["NOTE_TEXT"].str.lower()
    df["NOTE_LENGTH"] = df["NOTE_TEXT"].str.len().fillna(0).astype(int)

    # Build one-line lab description for each row (may be empty)
    df["LAB_ENTRY"] = df.apply(build_lab_entry, axis=1)
    df["HAS_LAB_ENTRY"] = df["LAB_ENTRY"].str.len() > 0

    # Grouping keys to identify one clinical pharmacist note
    group_keys = [
        "MRN",
        "Document_Number",
        "DOCUMENT_TYPE",
        "Entry_Date",
        "Visit",
        "VISIT_LOCATION",
        "SERVICE",
        "Parent_Number",
        "Parent_Type",
        "HOSPITAL_LOCATION",
        "AUTHOR_SERVICE",
        "Visit_Number",
        "Has_Clinical_Recommendation",
        "NOTE_TEXT",
    ]

    # Aggregate labs per note
    def agg_labs(sub: pd.DataFrame) -> pd.Series:
        # Keep only non-empty lab entries
        labs = [x for x in sub["LAB_ENTRY"].tolist() if x]
        labs_text = " || ".join(labs) if labs else ""
        num_labs = len(labs)

        # For now, simple flags; you can refine later
        has_labs = num_labs > 0

        # Derived from NOTE_LOWER (same for all rows in the group)
        note_lower = sub["NOTE_LOWER"].iloc[0]

        bsa = extract_bsa(note_lower)
        cycle = extract_cycle(note_lower)
        chemo_meds = find_keywords(note_lower, CHEMO_DRUGS)
        comorbidities = find_keywords(note_lower, COMORBIDITY_KEYWORDS)

        return pd.Series(
            {
                "LABS_TEXT": labs_text,
                "NUM_LABS": num_labs,
                "HAS_LABS": has_labs,
                "NOTE_LOWER": note_lower,
                "BSA_EXTRACTED": bsa,
                "CYCLE_INFO": cycle,
                "CHEMO_MEDS": chemo_meds,
                "COMORBIDITIES": comorbidities,
            }
        )

    grouped = df.groupby(group_keys, dropna=False, as_index=False).apply(agg_labs)

    # After groupby+apply, pandas creates a MultiIndex; reset it
    if isinstance(grouped.index, pd.MultiIndex):
        grouped.reset_index(drop=True, inplace=True)

    # Order columns nicely
    col_order = (
        group_keys
        + [
            "NOTE_LOWER",
            "NOTE_LENGTH",
            "HAS_LABS",
            "NUM_LABS",
            "LABS_TEXT",
            "BSA_EXTRACTED",
            "CYCLE_INFO",
            "CHEMO_MEDS",
            "COMORBIDITIES",
        ]
    )

    # NOTE_LENGTH is not created inside agg_labs; add from original
    # Map NOTE_LENGTH from original df (same per note)
    note_len_map = (
        df.drop_duplicates(subset=group_keys)[group_keys + ["NOTE_LENGTH"]]
        .set_index(group_keys)["NOTE_LENGTH"]
    )
    grouped["NOTE_LENGTH"] = grouped.set_index(group_keys).index.map(note_len_map)

    # Ensure all columns exist
    for c in col_order:
        if c not in grouped.columns:
            grouped[c] = ""

    grouped = grouped[col_order]

    print(f"Preprocessed rows (one per note): {len(grouped)}")
    print(f"Saving to: {OUTPUT_FILE}")

    grouped.to_excel(OUTPUT_FILE, index=False)

    print("Done. Preprocessed file written successfully.")


if __name__ == "__main__":
    main()
