import re
from pathlib import Path

import pandas as pd

# ------------ CONFIG ------------

INPUT_FILE = "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = "khcc_preprocessed.xlsx"

# ------------ LAB TEST MAPPING ------------

LAB_PATTERNS = {
    "WBC_x10^9_L": r"\b(WBC|WHITE BLOOD CELL)\b",
    "ANC_x10^9_L": r"\b(ANC|ABSOLUTE NEUTROPHIL COUNT|NEUTROPHILS\s*ABS)\b",
    "Hemoglobin_g_dL": r"\b(HGB|HEMOGLOBIN)\b",
    "Platelets_x10^9_L": r"\b(PLT|PLATELET)\b",
    "Creatinine_mg_dL": r"\b(CREATININE|CREAT)\b",
    "eGFR_mL_min_1.73m2": r"\b(EGFR|GFR)\b",
    "BUN_mg_dL": r"\b(BUN|UREA NITROGEN)\b",
    "AST_U_L": r"\b(AST|SGOT)\b",
    "ALT_U_L": r"\b(ALT|SGPT)\b",
    "ALP_U_L": r"\b(ALP|ALKALINE PHOSPHATASE)\b",
    "TotalBilirubin_mg_dL": r"\b(TOTAL BILIRUBIN|TBIL)\b",
    "Albumin_g_dL": r"\b(ALBUMIN)\b",
    "LVEF_percent": r"\b(LVEF|EJECTION FRACTION)\b",
    "Troponin_ng_L": r"\b(TROPONIN)\b",
}

URINE_PATTERN = r"(URINE|URINALYSIS|U/A|U/A\.)"

# ------------ HELPERS ------------


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # Normalize whitespace
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_stage(text: str):
    # Examples: "Stage II", "stage 2B", etc.
    m = re.search(r"\bstage\s*([0-4][ABCabc]?)\b", text, flags=re.I)
    return m.group(1).upper() if m else "NA"


def extract_tnm(text: str):
    # Very tolerant TNM extraction, does not assume presence
    t_match = re.search(r"\bT([0-4][abc]?)\b", text, flags=re.I)
    n_match = re.search(r"\bN([0-3][abc]?)\b", text, flags=re.I)
    m_match = re.search(r"\bM([0-1])\b", text, flags=re.I)

    t = t_match.group(1).upper() if t_match else "NA"
    n = n_match.group(1).upper() if n_match else "NA"
    m = m_match.group(1).upper() if m_match else "NA"
    return t, n, m


def extract_grade(text: str):
    m = re.search(r"\bgrade\s*([1-3])\b", text, flags=re.I)
    return m.group(1) if m else "NA"


def extract_receptor_status(text: str, marker: str):
    """
    Returns 'Positive', 'Negative', or 'NA' based only on explicit mention.
    """
    # Positive
    if re.search(rf"\b{marker}\s*(pos(itive)?|\+)\b", text, flags=re.I):
        return "Positive"
    # Negative
    if re.search(rf"\b{marker}\s*(neg(ative)?|-)\b", text, flags=re.I):
        return "Negative"
    return "NA"


COMORBIDITY_KEYWORDS = {
    "Comorbidity_DM": ["diabetes", "dm", "type 2 dm", "t2dm"],
    "Comorbidity_HTN": ["hypertension", "htn", "high blood pressure"],
    "Comorbidity_CAD": ["coronary artery disease", "cad", "ischemic heart"],
    "Comorbidity_CKD": ["chronic kidney", "ckd", "renal insufficiency"],
    "Comorbidity_Asthma": ["asthma"],
    "Comorbidity_Depression": ["depression", "depressive disorder"],
}


def extract_comorbidities(text: str):
    """
    Returns a dict with keys Comorbidity_* set to 'Yes' if mentioned, otherwise 'NA'.
    We NEVER assume 'No' when not mentioned.
    """
    text_low = text.lower()
    result = {}
    for col, kws in COMORBIDITY_KEYWORDS.items():
        result[col] = "Yes" if any(kw in text_low for kw in kws) else "NA"
    return result


def extract_cycle_number(text: str):
    m = re.search(r"\bcycle\s*(\d+)\b", text, flags=re.I)
    return int(m.group(1)) if m else "NA"


def extract_height_weight_bsa(text: str):
    height = "NA"
    weight = "NA"
    bsa = "NA"

    # Height in cm
    m_h = re.search(r"\bheight[:\s]*([1-2]\d{2})\s*cm\b", text, flags=re.I)
    if m_h:
        height = float(m_h.group(1))

    # Weight in kg
    m_w = re.search(r"\bweight[:\s]*([3-9]\d(?:\.\d+)?)\s*kg\b", text, flags=re.I)
    if m_w:
        weight = float(m_w.group(1))

    # BSA if explicitly written
    m_b = re.search(r"\bBSA[:\s]*([0-9]\.\d{2})\b", text, flags=re.I)
    if m_b:
        bsa = float(m_b.group(1))
    else:
        # Optionally compute if both height and weight present
        if isinstance(height, (int, float)) and isinstance(weight, (int, float)):
            # Mosteller formula
            bsa = round(((height * weight) / 3600.0) ** 0.5, 2)

    return height, weight, bsa


def get_lab_value(labs_df: pd.DataFrame, pattern: str, note_date: pd.Timestamp):
    """
    Pick the lab with TEST_NAME matching 'pattern' (regex, case-insensitive).
    Prefer results on or before the note date, closest in time.
    If none before, take the closest overall.
    """
    if labs_df.empty:
        return "NA"

    mask = labs_df["TEST_NAME"].astype(str).str.contains(pattern, case=False, regex=True, na=False)

    # For ANC, avoid picking percentage tests like "ANC %"
    if "ANC" in pattern:
        mask = mask & ~labs_df["TEST_NAME"].astype(str).str.contains("%")

    sub = labs_df[mask].copy()
    if sub.empty:
        return "NA"

    # Use DATE/TIME SPECIMEN TAKEN if available
    if "DATE/TIME SPECIMEN TAKEN" in sub.columns:
        # Make sure it is datetime
        sub["DATE/TIME SPECIMEN TAKEN"] = pd.to_datetime(
            sub["DATE/TIME SPECIMEN TAKEN"], errors="coerce"
        )
        # rows with date
        with_date = sub.dropna(subset=["DATE/TIME SPECIMEN TAKEN"]).copy()
        if not with_date.empty and pd.notna(note_date):
            # prefer on/before note date, closest first
            before = with_date[with_date["DATE/TIME SPECIMEN TAKEN"] <= note_date]
            if not before.empty:
                before = before.sort_values(
                    "DATE/TIME SPECIMEN TAKEN", ascending=False
                )
                return before.iloc[0]["TEST_RESULT"]

            # otherwise nearest overall
            with_date["delta"] = (
                with_date["DATE/TIME SPECIMEN TAKEN"] - note_date
            ).abs()
            with_date = with_date.sort_values("delta", ascending=True)
            return with_date.iloc[0]["TEST_RESULT"]

    # Fallback: just take the last one
    sub = sub.sort_index()
    return sub.iloc[-1]["TEST_RESULT"]


def build_urine_summary(labs_df: pd.DataFrame):
    if labs_df.empty:
        return "NA"
    mask = labs_df["TEST_NAME"].astype(str).str.contains(
        URINE_PATTERN, case=False, regex=True, na=False
    )
    urine = labs_df[mask]
    if urine.empty:
        return "NA"
    parts = []
    for _, row in urine.iterrows():
        name = str(row.get("TEST_NAME", "")).strip()
        res = str(row.get("TEST_RESULT", "")).strip()
        if name or res:
            parts.append(f"{name}: {res}")
    return "; ".join(parts) if parts else "NA"


# ------------ MAIN ------------


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading: {input_path} ...")
    df = pd.read_excel(input_path)

    # Normalize Has_Clinical_Recommendation
    if "Has_Clinical_Recommendation" not in df.columns:
        raise ValueError("Column 'Has_Clinical_Recommendation' not found in input file.")

    df["Has_Clinical_Recommendation"] = (
        df["Has_Clinical_Recommendation"].astype(str).str.strip().str.lower()
    )

    # Ensure date columns are datetime if present
    for col in ["Entry_Date", "DATE/TIME SPECIMEN TAKEN", "DATE REPORT COMPLETED"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    all_rows = []

    # Group by Document_Number (each represents one note + related labs)
    for doc_num, group in df.groupby("Document_Number", dropna=True):
        # 1) identify the clinical pharmacist note row (Has_Clinical_Recommendation == 'yes')
        note_mask = group["Has_Clinical_Recommendation"].eq("yes")
        note_rows = group[note_mask]

        if note_rows.empty:
            # No flagged note for this document_number → skip
            continue

        # In case of multiple, take the latest Entry_Date
        note_row = note_rows.sort_values("Entry_Date").iloc[-1]
        note_text_raw = note_row.get("Note", "")
        note_text = clean_text(note_text_raw)

        note_date = note_row.get("Entry_Date", pd.NaT)

        # 2) labs for this group (rows that actually have TEST_NAME)
        labs_df = group[group["TEST_NAME"].notna()].copy()

        # --- basic fields from original row ---
        row = {
            "MRN": note_row.get("MRN"),
            "Document_Number": note_row.get("Document_Number"),
            "Visit_Number": note_row.get("Visit_Number"),
            "Entry_Date": note_date,
            "VISIT_LOCATION": note_row.get("VISIT_LOCATION"),
            "SERVICE": note_row.get("SERVICE"),
            "HOSPITAL_LOCATION": note_row.get("HOSPITAL_LOCATION"),
            "AUTHOR_SERVICE": note_row.get("AUTHOR_SERVICE"),
            "Note": note_text_raw,
            "Note_Clean": note_text,
        }

        # --- clinical info extracted from text ---

        # Stage / TNM / grade
        row["Stage"] = extract_stage(note_text)
        t, n, m = extract_tnm(note_text)
        row["TNM_T"] = t
        row["TNM_N"] = n
        row["TNM_M"] = m
        row["Grade"] = extract_grade(note_text)

        # Receptor status
        row["ER_Status"] = extract_receptor_status(note_text, "ER")
        row["PR_Status"] = extract_receptor_status(note_text, "PR")
        row["HER2_Status"] = extract_receptor_status(note_text, "HER2")

        # Height / weight / BSA
        height, weight, bsa = extract_height_weight_bsa(note_text)
        row["Height_cm"] = height
        row["Weight_kg"] = weight
        row["BSA_m2"] = bsa

        # Comorbidities (Yes / NA)
        comorbs = extract_comorbidities(note_text)
        row.update(comorbs)

        # Charlson-like score – not derivable from free text safely → NA
        row["CharlsonLikeScore"] = "NA"

        # Prior lines / therapies / responses etc – we **do not invent** them
        # They will be dropped later if present from older versions.

        # Regimen (only if explicitly named – otherwise NA)
        # VERY conservative: look for common regimen names as full words
        reg_match = re.search(
            r"\b(AC|FEC|EC|TC|TCHP|THP|PACLitaxel|DOCETaxel|CMF)\b", note_text, flags=re.I
        )
        row["Regimen"] = reg_match.group(1).upper() if reg_match else "NA"

        # Cycle number
        row["CycleNumber"] = extract_cycle_number(note_text)

        # --- Lab extraction into columns ---
        for col_name, pattern in LAB_PATTERNS.items():
            row[col_name] = get_lab_value(labs_df, pattern, note_date)

        # Urine summary column
        row["UrineTests"] = build_urine_summary(labs_df)

        all_rows.append(row)

    if not all_rows:
        raise ValueError("No rows with Has_Clinical_Recommendation == 'yes' were found.")

    out_df = pd.DataFrame(all_rows)

    # Columns to drop (if they exist from older versions)
    cols_to_drop = [
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
    existing_to_drop = [c for c in cols_to_drop if c in out_df.columns]
    if existing_to_drop:
        out_df = out_df.drop(columns=existing_to_drop)

    # Save
    output_path = Path(OUTPUT_FILE)
    out_df.to_excel(output_path, index=False)
    print(f"Saved preprocessed file to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
