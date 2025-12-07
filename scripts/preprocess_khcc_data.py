import re
from pathlib import Path

import pandas as pd

# ---------- CONFIG ----------
INPUT_FILE = "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = "khcc_preprocessed.xlsx"

# List of common breast cancer chemo / targeted meds to look for in notes
CHEMO_DRUGS = [
    "doxorubicin", "adriamycin",
    "epirubicin",
    "cyclophosphamide",
    "docetaxel",
    "paclitaxel", "taxol", "abraxane",
    "carboplatin", "cisplatin",
    "trastuzumab", "herceptin",
    "pertuzumab", "perjeta",
    "lapatinib",
    "capecitabine", "xeloda",
    "vinorelbine",
    "gemcitabine",
    "fulvestrant",
    "letrozole", "anastrozole", "exemestane",
]

# Simple dictionary for comorbidities (you can add more)
COMORBIDITIES = {
    "DM": r"\b(dm|diabetes)\b",
    "HTN": r"\b(htn|hypertension)\b",
    "IHD": r"\b(ihd|ischemic heart disease|coronary artery disease|cad)\b",
    "CKD": r"\b(ckd|chronic kidney disease|renal impairment|renal failure)\b",
    "Asthma": r"\b(asthma)\b",
    "Hypothyroidism": r"\b(hypothyroid|hypothyroidism)\b",
}


# ---------- HELPERS ----------

def clean_note(text: str) -> str:
    """Basic cleaning of clinical note text."""
    if pd.isna(text):
        return ""
    # Normalize line breaks / spaces
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    # Remove duplicated spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_bsa(text: str):
    """
    Extract BSA if written like:
    - BSA 1.73
    - BSA=1.8 m2
    - BSA: 1.65 m^2
    """
    if not text:
        return None
    m = re.search(r"\bBSA\b[^0-9]{0,10}([1-2]\.\d{1,2})", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_cycle(text: str):
    """
    Extract cycle number like:
    - Cycle 3
    - cycle #4
    - C3
    """
    if not text:
        return None

    # Pattern: "Cycle 3"
    m = re.search(r"\b[Cc]ycle\D{0,3}(\d+)", text)
    if not m:
        # Pattern: "C3", "C 4", etc. (avoid catching 'CKD', etc.)
        m = re.search(r"\bC\s?(\d+)\b", text)
    if not m:
        return None

    try:
        return int(m.group(1))
    except ValueError:
        return None


def extract_chemo_meds(text: str):
    """Return ;–separated list of chemo drugs mentioned in the note."""
    if not text:
        return None
    lower = text.lower()
    found = []
    for drug in CHEMO_DRUGS:
        if drug.lower() in lower:
            found.append(drug)
    found = sorted(set(found))
    return "; ".join(found) if found else None


def extract_comorbid_flags(text: str):
    """
    Return dict {label: bool} for each comorbidity based on regex patterns.
    Uses the CLEANED note (note_clean).
    """
    flags = {}
    for label, pattern in COMORBIDITIES.items():
        flags[label] = bool(re.search(pattern, text, flags=re.IGNORECASE))
    return flags


def build_comorbid_list(row, labels):
    """Combine individual comorbidity flags into one text column."""
    present = [lbl for lbl in labels if row[f"comorb_{lbl}"]]
    return "; ".join(present) if present else None


def extract_other_meds(text: str):
    """
    Very rough extraction of 'other meds' – looks for phrases like:
    - 'other meds: ...'
    - 'home meds: ...'
    - 'currently on: ...'
    This is a placeholder that you can refine later.
    """
    if not text:
        return None
    patterns = [
        r"(home meds?:\s*)(.+?)(?=[.;]|$)",
        r"(other meds?:\s*)(.+?)(?=[.;]|$)",
        r"(currently on:\s*)(.+?)(?=[.;]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(2).strip()
    return None


# ---------- MAIN PIPELINE ----------

def main():
    here = Path(__file__).resolve().parent
    input_path = here / INPUT_FILE
    output_path = here / OUTPUT_FILE

    print(f"Loading: {input_path}")
    df = pd.read_excel(input_path)

    # --- keep your original columns as they are ---

    # 1) Clean note text
    df["note_clean"] = df["Note"].apply(clean_note)

    # 2) Length features
    df["note_len_chars"] = df["note_clean"].str.len()
    df["note_len_words"] = df["note_clean"].str.split().str.len()

    # 3) Simple semantic flags from previous version (you already have these – keep or remove)
    df["has_dose_change"] = df["note_clean"].str.contains(
        r"dose|reduce dose|increase dose|adjust dose|mg/m2",
        case=False,
        regex=True,
    )
    df["has_start_stop"] = df["note_clean"].str.contains(
        r"start|initiate|stop|discontinue|hold",
        case=False,
        regex=True,
    )
    df["has_interaction"] = df["note_clean"].str.contains(
        r"interaction|contraindication|avoid with|combination with",
        case=False,
        regex=True,
    )
    df["has_monitoring_plan"] = df["note_clean"].str.contains(
        r"monitor|follow up|check lab|repeat lab|ECG|LFT|RFT",
        case=False,
        regex=True,
    )

    # 4) NEW: BSA, Cycle, Chemo meds, Comorbidities, Other meds

    print("Extracting BSA...")
    df["BSA_m2"] = df["note_clean"].apply(extract_bsa)

    print("Extracting cycle number...")
    df["cycle_number"] = df["note_clean"].apply(extract_cycle)

    print("Extracting chemo medications...")
    df["chemo_meds"] = df["note_clean"].apply(extract_chemo_meds)

    print("Extracting comorbidities flags...")
    for label in COMORBIDITIES.keys():
        df[f"comorb_{label}"] = df["note_clean"].apply(
            lambda txt, lbl=label: extract_comorbid_flags(txt)[lbl]
        )

    # Combine comorbidities into one list column
    df["comorbidities_list"] = df.apply(
        lambda row: build_comorbid_list(row, list(COMORBIDITIES.keys())),
        axis=1,
    )

    print("Extracting other/home meds (very rough)...")
    df["other_meds_free_text"] = df["note_clean"].apply(extract_other_meds)

    # 5) Save
    print(f"Saving to: {output_path}")
    df.to_excel(output_path, index=False)
    print("Done ✅")


if __name__ == "__main__":
    main()
