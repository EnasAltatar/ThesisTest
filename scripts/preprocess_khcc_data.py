import re
from pathlib import Path

import pandas as pd

# ------------ CONFIG ------------
INPUT_FILE = "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = "khcc_preprocessed.xlsx"

# Common breast-cancer chemo / targeted meds
CHEMO_DRUGS = [
    "doxorubicin", "adriamycin",
    "epirubicin",
    "cyclophosphamide",
    "docetaxel",
    "paclitaxel", "taxol", "abraxane",
    "carboplatin",
    "cisplatin",
    "oxaliplatin",
    "5-fluorouracil", "5fu", "fluorouracil",
    "capecitabine", "xeloda",
    "trastuzumab", "herceptin",
    "pertuzumab", "perjeta",
    "t-dm1", "kadcyla",
    "lapatinib",
    "neratinib",
    "tucatinib",
    "palbociclib", "ribociclib", "abemaciclib",
    "everolimus",
    "olaparib", "talazoparib",
]

# Supportive / concomitant meds we may want to capture
SUPPORTIVE_DRUGS = [
    "dexamethasone",
    "ondansetron", "granisetron", "palonosetron",
    "aprepitant", "fosaprepitant",
    "metoclopramide",
    "lorazepam",
    "pegfilgrastim", "filgrastim", "g-csf",
    "epoetin", "erythropoietin",
    "bisphosphonate", "zoledronic", "zoledronate",
    "denosumab",
]

# Map lab test names (from TEST_NAME) to synthetic-style column names
LAB_TEST_MAP = {
    "WBC_x10^9_L": ["WBC", "WHITE BLOOD"],
    "ANC_x10^9_L": ["ANC", "ABSOLUTE NEUTROPHIL"],
    "Hemoglobin_g_dL": ["HGB", "HEMOGLOBIN"],
    "Platelets_x10^9_L": ["PLT", "PLATELET"],
    "Creatinine_mg_dL": ["CREATININE"],
    "AST_U_L": ["AST", "SGOT"],
    "ALT_U_L": ["ALT", "SGPT"],
    "ALP_U_L": ["ALK PHOS", "ALP", "ALKALINE PHOSPHATASE"],
    "Total_Bilirubin_mg_dL": ["TOTAL BILI", "T BILI", "TBIL", "BILIRUBIN TOTAL"],
    "Albumin_g_dL": ["ALBUMIN"],
    "eGFR_mL_min_1_73m2": ["EGFR", "GFR"],
}

# Simple comorbidity keyword map
COMORBIDITY_KEYWORDS = {
    "Diabetes": ["DIABETES", " DM ", "T2DM", "T1DM"],
    "Hypertension": ["HYPERTENSION", " HTN "],
    "IschemicHeartDisease": ["IHD", "ISCHEMIC HEART", "CORONARY ARTERY", "CAD"],
    "HeartFailure": ["HEART FAILURE", "CHF"],
    "CKD": ["CKD", "CHRONIC KIDNEY"],
    "Asthma": ["ASTHMA"],
    "COPD": ["COPD", "CHRONIC OBSTRUCTIVE"],
    "Depression": ["DEPRESSION", "DEPRESSIVE"],
    "Anxiety": ["ANXIETY", "GAD"],
    "Dyslipidemia": ["DYSLIPIDEMIA", "HYPERLIPIDEMIA"],
}


def normalize(text: str) -> str:
    return (text or "").replace("\n", " ").replace("\r", " ").strip()


def extract_numeric(value: str):
    """
    Try to pull a single numeric value (int/float) from a lab result string.
    Returns float or None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def map_test_name(test_name: str):
    if not isinstance(test_name, str):
        return None
    upper = test_name.upper()
    for col, keywords in LAB_TEST_MAP.items():
        for kw in keywords:
            if kw in upper:
                return col
    return None


def extract_labs_for_group(group: pd.DataFrame):
    """
    From all lab rows tied to a single pharmacist note, pick one value
    per important test (closest in time; smallest absolute date_diff_days).
    Also build a compact Labs_Combined string for reference.
    """
    labs = {col: None for col in LAB_TEST_MAP.keys()}
    combined_parts = []

    for _, row in group.iterrows():
        tname = row.get("TEST_NAME")
        tres = row.get("TEST_RESULT")
        date_diff = row.get("date_diff_days")

        # Combined text, regardless of whether we parse it
        if pd.notna(tname) or pd.notna(tres):
            combined_parts.append(f"{tname}: {tres}")

        mapped_col = map_test_name(tname)
        if not mapped_col:
            continue

        val = extract_numeric(tres)
        if val is None:
            continue

        # If there is more than one row for the same test, keep the one
        # closest in time to the note (smallest |date_diff_days|).
        cur = labs.get(mapped_col)
        if cur is None:
            labs[mapped_col] = (val, abs(date_diff) if pd.notna(date_diff) else 0)
        else:
            _, cur_diff = cur
            new_diff = abs(date_diff) if pd.notna(date_diff) else 0
            if new_diff < cur_diff:
                labs[mapped_col] = (val, new_diff)

    # Strip the helper date_diff and leave only numeric values
    labs_numeric = {
        col: (val_diff[0] if val_diff is not None else None)
        for col, val_diff in labs.items()
    }
    labs_numeric["Labs_Combined"] = "; ".join(combined_parts) if combined_parts else None
    return labs_numeric


def extract_cycle(note: str):
    note_u = note.upper()
    m = re.search(r"CYCLE\s*(\d+)", note_u)
    if not m:
        m = re.search(r"\bC(\d+)\b", note_u)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def extract_bsa(note: str):
    m = re.search(r"\bBSA\b[^0-9]*(\d+(?:\.\d+)?)", note, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def extract_ecog(note: str):
    m = re.search(r"ECOG[^0-9]*(\d)", note, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def extract_drugs(note: str, drug_list):
    text_u = note.upper()
    found = []
    for drug in drug_list:
        if drug.upper() in text_u:
            found.append(drug)
    # Deduplicate and keep original order
    if not found:
        return None
    unique = []
    for d in found:
        if d not in unique:
            unique.append(d)
    return ", ".join(unique)


def extract_comorbidities(note: str):
    text_u = " " + note.upper() + " "
    found = []
    for label, patterns in COMORBIDITY_KEYWORDS.items():
        for pat in patterns:
            if pat in text_u:
                found.append(label)
                break
    return ", ".join(found) if found else None


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)

    # Ensure Note is string
    df["Note"] = df["Note"].fillna("").astype(str)

    # Define the columns that uniquely identify a pharmacist note
    key_cols = ["MRN", "Document_Number", "Entry_Date", "Visit_Number", "Note"]

    grouped = df.groupby(key_cols, dropna=False)

    rows = []

    for key, group in grouped:
        mrn, doc_no, entry_date, visit_no, note = key

        base_row = {
            "MRN": mrn,
            "Document_Number": doc_no,
            "Entry_Date": entry_date,
            "Visit_Number": visit_no,
            "VISIT_LOCATION": group.iloc[0].get("VISIT_LOCATION"),
            "SERVICE": group.iloc[0].get("SERVICE"),
            "HOSPITAL_LOCATION": group.iloc[0].get("HOSPITAL_LOCATION"),
            "AUTHOR_SERVICE": group.iloc[0].get("AUTHOR_SERVICE"),
            "Has_Clinical_Recommendation": group.iloc[0].get("Has_Clinical_Recommendation"),
            "Note": normalize(note),
        }

        # Lab features
        lab_features = extract_labs_for_group(group)

        # Text-derived clinical features
        bsa = extract_bsa(note)
        cycle = extract_cycle(note)
        ecog = extract_ecog(note)
        chemo = extract_drugs(note, CHEMO_DRUGS)
        supportive = extract_drugs(note, SUPPORTIVE_DRUGS)
        comorbid = extract_comorbidities(note)

        text_features = {
            "BSA_m2": bsa,
            "Cycle_Number": cycle,
            "ECOG_Performance": ecog,
            "Chemo_Drugs_In_Note": chemo,
            "Supportive_Meds_In_Note": supportive,
            "Comorbidities_In_Note": comorbid,
        }

        merged_row = {**base_row, **lab_features, **text_features}
        rows.append(merged_row)

    out_df = pd.DataFrame(rows)

    out_df.to_excel(output_path, index=False)
    print(f"Saved preprocessed data to: {output_path}")


if __name__ == "__main__":
    main()
