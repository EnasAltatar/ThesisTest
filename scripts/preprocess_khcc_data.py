import re
from pathlib import Path

import pandas as pd

# ---------- CONFIG ----------

INPUT_FILE = "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = "khcc_preprocessed.xlsx"

# Columns that must NOT appear in the final output
COLUMNS_TO_DROP = [
    "Subtype",
    "PriorLines",
    "PriorTherapies",
    "BestPriorResponse",
    "DosePlan",
    "Schedule",
    "PreMeds",
    "PostMeds",
    "IntendedDoseIntensity_pct",
    "ActualDoseIntensity_pct",
    "DoseAdjustmentNote",
    "InteractionsCheck",
    "Contraindications",
    "SupportiveCareInstructions",
    "Rationale",
]

# ---------- LAB TEST PATTERNS ----------

LAB_PATTERNS = {
    "WBC_x10^9_L": [r"\bWBC\b", r"WHITE BLOOD CELL"],
    # FIXED: ANC now only from specific ANC tests
    "ANC_x10^9_L": [
        r"\bANC\b",
        r"ABSOLUTE NEUTROPHIL",
        r"NEUTROPHIL.*ABSOLUTE",
        r"NEUTS ABS",
    ],
    "Hemoglobin_g_dL": [r"\bHGB\b", r"HEMOGLOBIN"],
    "Platelets_x10^9_L": [r"\bPLT\b", r"PLATELET"],
    "Creatinine_mg_dL": [r"\bCREATININE\b", r"\bCREA\b"],
    "eGFR_mL_min_1.73m2": [r"\beGFR\b", r"ESTIMATED GFR"],
    "BUN_mg_dL": [r"\bBUN\b", r"UREA NITROGEN"],
    "AST_U_L": [r"\bAST\b", r"ASPARTATE TRANSAMINASE"],
    "ALT_U_L": [r"\bALT\b", r"ALANINE TRANSAMINASE"],
    "ALP_U_L": [r"\bALP\b", r"ALKALINE PHOSPHATASE"],
    "TotalBilirubin_mg_dL": [
        r"TOTAL BILIRUBIN",
        r"\bTBIL\b",
        r"\bT\.? BILI\b",
    ],
    "Albumin_g_dL": [r"\bALBUMIN\b"],
    "Troponin_ng_L": [r"TROPONIN"],
    "LVEF_percent": [r"\bLVEF\b", r"EJECTION FRACTION"],
}

# Patterns for URINE tests (for the Urine_Results column)
URINE_PATTERNS = [
    r"\bURINE\b",
    r"URINALYSIS",
    r"URINE ANALYSIS",
    r"URINALYSIS, ROUTINE",
]


# ---------- TEXT CLEANING / UTILITIES ----------


def clean_note_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # Normalize whitespace and lowercase for pattern matching
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_first_match(text: str, pattern: str, flags=0):
    if not text:
        return pd.NA
    m = re.search(pattern, text, flags)
    if not m:
        return pd.NA
    # If there is a capturing group, return it; else return full match
    return m.group(1) if m.lastindex else m.group(0)


# ---------- CLINICAL EXTRACTION HELPERS ----------


def extract_stage(text: str):
    # e.g. "Stage II", "stage 3A"
    return extract_first_match(
        text,
        r"stage\s*([0-4]{1}[A-C]?)",
        flags=re.IGNORECASE,
    )


def extract_tnm(text: str):
    t = extract_first_match(text, r"\bT([0-4][a-c]?)\b", flags=re.IGNORECASE)
    n = extract_first_match(text, r"\bN([0-3][a-c]?)\b", flags=re.IGNORECASE)
    m = extract_first_match(text, r"\bM([0-1])\b", flags=re.IGNORECASE)
    return t, n, m


def extract_grade(text: str):
    # e.g. "Grade 3", "grade II"
    grade = extract_first_match(text, r"grade\s*([1-3Ii]{1})", flags=re.IGNORECASE)
    return grade


def parse_receptor_status(text: str, receptor: str):
    """
    receptor: 'ER', 'PR', 'HER2'
    Returns one of: 'Positive', 'Negative', 'Equivocal', 'NA'
    """
    if not text:
        return pd.NA

    # Simplify text for robust matching
    lower = text.lower()

    if receptor.lower() == "her2":
        # HER2 3+ positive, 0/1+ negative, 2+ equivocal
        pos_patterns = [r"her2\s*3\+", r"her2\s*positive", r"her-2\s*neu\s*3\+"]
        neg_patterns = [r"her2\s*[01]\+", r"her2\s*negative", r"her-2\s*neu\s*[01]\+"]
        eq_patterns = [r"her2\s*2\+", r"equivocal"]
    else:
        # ER / PR
        pos_patterns = [rf"\b{receptor}\s*positive\b", rf"{receptor}\s*\+\b"]
        neg_patterns = [rf"\b{receptor}\s*negative\b", rf"{receptor}\s*-\b"]
        eq_patterns = []

    for p in pos_patterns:
        if re.search(p, lower, re.IGNORECASE):
            return "Positive"
    for p in neg_patterns:
        if re.search(p, lower, re.IGNORECASE):
            return "Negative"
    for p in eq_patterns:
        if re.search(p, lower, re.IGNORECASE):
            return "Equivocal"

    return pd.NA


def extract_height_weight(text: str):
    """
    Try to pull height (cm) and weight (kg) from the note.
    """
    if not text:
        return pd.NA, pd.NA

    # Height in cm: "Height: 163 cm" or "Ht 163 cm"
    h = extract_first_match(
        text,
        r"(?:height|ht)\s*[:=]?\s*([1-2]\d{2})\s*cm",
        flags=re.IGNORECASE,
    )

    # Weight in kg: "Weight 70 kg", "Wt: 72 kg"
    w = extract_first_match(
        text,
        r"(?:weight|wt)\s*[:=]?\s*(\d{2,3})\s*kg",
        flags=re.IGNORECASE,
    )

    try:
        h_val = float(h) if h is not pd.NA else pd.NA
    except Exception:
        h_val = pd.NA

    try:
        w_val = float(w) if w is not pd.NA else pd.NA
    except Exception:
        w_val = pd.NA

    return h_val, w_val


def calculate_bsa_m2(height_cm, weight_kg):
    """
    Du Bois formula, only if both height and weight exist.
    """
    if pd.isna(height_cm) or pd.isna(weight_kg):
        return pd.NA
    try:
        bsa = 0.007184 * (float(height_cm) ** 0.725) * (float(weight_kg) ** 0.425)
        return round(bsa, 3)
    except Exception:
        return pd.NA


def extract_cycle_number(text: str):
    # e.g. "Cycle 3", "C3D1", "C2"
    c = extract_first_match(text, r"(?:cycle|c)(\d{1,2})\b", flags=re.IGNORECASE)
    return c


def extract_regimen(text: str):
    """
    Very rough: pull common chemo regimens if explicitly mentioned.
    """
    if not text:
        return pd.NA

    regimens = [
        "AC-T",
        "AC T",
        "FEC",
        "FEC-D",
        "FEC D",
        "TC",
        "TCH",
        "TCHP",
        "CMF",
        "EC",
        "EC-P",
    ]
    for reg in regimens:
        if re.search(rf"\b{re}\b".replace(" ", r"\s*"), text, re.IGNORECASE):
            return reg

    return pd.NA


# ---------- COMORBIDITIES ----------

COMORBIDITY_PATTERNS = {
    "Comorbidity_DM": [r"diabetes", r"dm\b"],
    "Comorbidity_HTN": [r"hypertension", r"htn\b"],
    "Comorbidity_CAD": [r"coronary artery disease", r"ischemic heart", r"cad\b"],
    "Comorbidity_CKD": [r"chronic kidney disease", r"ckd\b", r"renal failure"],
    "Comorbidity_Asthma": [r"\basthma\b"],
    "Comorbidity_Depression": [r"\bdepression\b", r"depressive disorder"],
}


def extract_comorbidities(text: str):
    result = {}
    lower = text.lower() if text else ""
    for col, patterns in COMORBIDITY_PATTERNS.items():
        found = any(re.search(p, lower, re.IGNORECASE) for p in patterns)
        result[col] = 1 if found else 0
    return result


def compute_charlson_like(comorbid_dict: dict):
    # Simple sum of present comorbidities – a rough Charlson-like index
    return sum(comorbid_dict.values())


# ---------- LAB HELPERS ----------


def get_lab_value(labs_df: pd.DataFrame, patterns):
    """
    Find the first lab whose TEST_NAME matches any pattern.
    """
    if labs_df is None or labs_df.empty:
        return pd.NA

    mask = pd.Series(False, index=labs_df.index)
    for p in patterns:
        mask = mask | labs_df["TEST_NAME"].str.contains(p, case=False, na=False, regex=True)

    selected = labs_df[mask]
    if selected.empty:
        return pd.NA

    # If multiple rows, take the last one (assuming most recent or most relevant)
    val = selected.iloc[-1]["TEST_RESULT"]
    return val


def combine_labs_readable(labs_df: pd.DataFrame):
    """
    Combine all lab tests into a single string per note, formatted for readability.
    Example: "WBC: 6.5 | Hb: 11.2 | Platelets: 250 ..."
    """
    if labs_df is None or labs_df.empty:
        return pd.NA

    parts = []
    for _, row in labs_df.iterrows():
        name = str(row.get("TEST_NAME", "")).strip()
        res = str(row.get("TEST_RESULT", "")).strip()
        if not name and not res:
            continue
        parts.append(f"{name}: {res}")
    if not parts:
        return pd.NA
    return " | ".join(parts)


def get_urine_results(labs_df: pd.DataFrame):
    """
    Build a Urine_Results column from all lab rows that are clearly urine tests.
    """
    if labs_df is None or labs_df.empty:
        return pd.NA

    mask = pd.Series(False, index=labs_df.index)
    for p in URINE_PATTERNS:
        mask = mask | labs_df["TEST_NAME"].str.contains(p, case=False, na=False, regex=True)

    urine_rows = labs_df[mask]
    if urine_rows.empty:
        return pd.NA

    parts = []
    for _, row in urine_rows.iterrows():
        name = str(row.get("TEST_NAME", "")).strip()
        res = str(row.get("TEST_RESULT", "")).strip()
        if not name and not res:
            continue
        parts.append(f"{name}: {res}")

    if not parts:
        return pd.NA
    return " | ".join(parts)


# ---------- MAIN PREPROCESSING ----------


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)

    # Ensure columns used later exist
    required_cols = [
        "MRN",
        "Document_Number",
        "Entry_Date",
        "Note",
        "TEST_NAME",
        "TEST_RESULT",
        "Has_Clinical_Recommendation",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    # Group by clinical note (Document_Number)
    grouped = df.groupby("Document_Number", dropna=False)

    rows = []

    for doc_num, group in grouped:
        # 1) Identify the clinical pharmacist note row
        # We assume there is exactly one note per Document_Number with Has_Clinical_Recommendation == "Yes"
        note_rows = group[group["Has_Clinical_Recommendation"] == "Yes"]
        if note_rows.empty:
            # If not found, skip this group
            continue

        note_row = note_rows.iloc[0]
        note_text = clean_note_text(note_row.get("Note", ""))

        # 2) Labs belonging to this "note"
        labs_df = group[group["TEST_NAME"].notna()].copy()

        # ---------- BASE METADATA ----------
        out = {
            "MRN": note_row.get("MRN"),
            "Document_Number": doc_num,
            "Entry_Date": note_row.get("Entry_Date"),
            "Visit_Number": note_row.get("Visit_Number"),
            "Note": note_row.get("Note"),
        }

        # ---------- CLINICAL FIELDS FROM NOTE ----------
        out["Stage"] = extract_stage(note_text)
        t, n, m = extract_tnm(note_text)
        out["TNM_T"] = t
        out["TNM_N"] = n
        out["TNM_M"] = m
        out["Grade"] = extract_grade(note_text)

        out["ER_Status"] = parse_receptor_status(note_text, "ER")
        out["PR_Status"] = parse_receptor_status(note_text, "PR")
        out["HER2_Status"] = parse_receptor_status(note_text, "HER2")

        height_cm, weight_kg = extract_height_weight(note_text)
        out["Height_cm"] = height_cm
        out["Weight_kg"] = weight_kg
        out["BSA_m2"] = calculate_bsa_m2(height_cm, weight_kg)

        out["CycleNumber"] = extract_cycle_number(note_text)
        out["Regimen"] = extract_regimen(note_text)

        # Comorbidities
        comorbid = extract_comorbidities(note_text)
        out.update(comorbid)
        out["CharlsonLikeScore"] = compute_charlson_like(comorbid)

        # ---------- LAB VALUES ----------
        if labs_df is not None and not labs_df.empty:
            for col, patterns in LAB_PATTERNS.items():
                out[col] = get_lab_value(labs_df, patterns)

            out["Labs_Combined"] = combine_labs_readable(labs_df)
            out["Urine_Results"] = get_urine_results(labs_df)
        else:
            for col in LAB_PATTERNS.keys():
                out[col] = pd.NA
            out["Labs_Combined"] = pd.NA
            out["Urine_Results"] = pd.NA

        # Placeholder cols (will be dropped at the end – kept here only
        # to avoid breaking any code that might still refer to them)
        out["PriorLines"] = pd.NA
        out["PriorTherapies"] = pd.NA
        out["BestPriorResponse"] = pd.NA
        out["DosePlan"] = pd.NA
        out["Schedule"] = pd.NA
        out["PreMeds"] = pd.NA
        out["PostMeds"] = pd.NA
        out["IntendedDoseIntensity_pct"] = pd.NA
        out["ActualDoseIntensity_pct"] = pd.NA
        out["DoseAdjustmentNote"] = pd.NA
        out["InteractionsCheck"] = pd.NA
        out["Contraindications"] = pd.NA
        out["SupportiveCareInstructions"] = pd.NA
        out["Rationale"] = pd.NA

        rows.append(out)

    if not rows:
        raise ValueError("No rows with Has_Clinical_Recommendation == 'Yes' were found.")

    out_df = pd.DataFrame(rows)

    # Drop the columns you explicitly don't want
    out_df = out_df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Save
    out_path = Path(OUTPUT_FILE)
    out_df.to_excel(out_path, index=False)
    print(f"Saved preprocessed data to {out_path.resolve()}")


if __name__ == "__main__":
    main()
