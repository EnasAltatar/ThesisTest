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

# ------------ GENERIC HELPERS ------------


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_recommendations(text: str) -> str:
    """
    Return the note text up to (but not including) the recommendations section.
    We look for common English & Arabic markers like:
      - Recommendations:
      - Pharmacist recommendation(s)
      - Plan:
      - التوصيات
      - الخطة العلاجية
    If no marker is found, the original text is returned unchanged.
    """
    if pd.isna(text):
        return ""
    s = str(text)
    lower = s.lower()

    markers = [
        "clinical pharmacist recommendation",
        "clinical pharmacist recommendations",
        "pharmacist recommendation",
        "pharmacist recommendations",
        "recommendation:",
        "recommendations:",
        "plan:",
        "treatment plan:",
        "التوصيات",
        "الخطة العلاجية",
        "خطة العلاج",
    ]

    positions = []
    for m in markers:
        idx = lower.find(m)
        if idx != -1:
            positions.append(idx)

    if not positions:
        return s

    cut = min(positions)
    return s[:cut].rstrip()


# ------------ CLINICAL INFO EXTRACTORS ------------


def extract_stage(text: str) -> str:
    m = re.search(r"\bstage\s*([IVX]{1,3}[ABCabc]?)\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\bstage\s*([0-4][ABCabc]?)\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    return "NA"


def extract_tnm(text: str):
    m = re.search(
        r"\b[cp]?T(?P<T>[0-4][a-cA-C]?)N(?P<N>[0-3][a-cA-C]?)M(?P<M>[01])\b",
        text,
        flags=re.I,
    )
    if m:
        return m.group("T").upper(), m.group("N").upper(), m.group("M").upper()

    t_match = re.search(r"\bT([0-4][a-cA-C]?)\b", text, flags=re.I)
    n_match = re.search(r"\bN([0-3][a-cA-C]?)\b", text, flags=re.I)
    m_match = re.search(r"\bM([01])\b", text, flags=re.I)

    t = t_match.group(1).upper() if t_match else "NA"
    n = n_match.group(1).upper() if n_match else "NA"
    m = m_match.group(1).upper() if m_match else "NA"
    return t, n, m


def extract_grade(text: str) -> str:
    m = re.search(r"\bgrade\s*([1-3])\b", text, flags=re.I)
    return m.group(1) if m else "NA"


def _status_from_token(token: str) -> str:
    token = token.strip().lower()
    if token in {"+", "pos", "positive", "3+", "2+", "1+"}:
        return "Positive"
    if token in {"-", "neg", "negative", "0"}:
        return "Negative"
    return "NA"


def extract_receptor_statuses(text: str):
    ER = "NA"
    PR = "NA"
    HER2 = "NA"
    lower = text.lower()

    m = re.search(r"\bER\s*[:=]?\s*([+\-]|pos(?:itive)?|neg(?:ative)?|[0-3]\+)\b", lower)
    if m:
        ER = _status_from_token(m.group(1))

    m = re.search(r"\bPR\s*[:=]?\s*([+\-]|pos(?:itive)?|neg(?:ative)?|[0-3]\+)\b", lower)
    if m:
        PR = _status_from_token(m.group(1))

    m = re.search(
        r"\bHER2\s*[:=]?\s*([+\-]|pos(?:itive)?|neg(?:ative)?|[0-3]\+)\b", lower
    )
    if m:
        HER2 = _status_from_token(m.group(1))

    if "triple negative" in lower:
        ER = ER if ER != "NA" else "Negative"
        PR = PR if PR != "NA" else "Negative"
        HER2 = HER2 if HER2 != "NA" else "Negative"

    return ER, PR, HER2


COMORBIDITY_KEYWORDS = {
    "Comorbidity_DM": ["diabetes", "dm", "type 2 dm", "t2dm"],
    "Comorbidity_HTN": ["hypertension", "htn"],
    "Comorbidity_CAD": ["coronary artery disease", "cad", "ischemic heart"],
    "Comorbidity_CKD": ["chronic kidney", "ckd", "renal insufficiency"],
    "Comorbidity_Asthma": ["asthma"],
    "Comorbidity_Depression": ["depression", "depressive disorder"],
}


def extract_comorbidities(text: str):
    lower = text.lower()
    pmh_match = re.search(
        r"(pmh|past medical history|comorbidit(?:y|ies))\s*[:\-]\s*(.+?)(?:assessment:|plan:|medications:|$)",
        lower,
    )
    section = pmh_match.group(2) if pmh_match else lower

    result = {}
    for col, kws in COMORBIDITY_KEYWORDS.items():
        result[col] = "Yes" if any(kw in section for kw in kws) else "NA"
    return result


def extract_cycle_number(text: str):
    m = re.search(r"\bcycle\s*#?\s*(\d+)\b", text, flags=re.I)
    if m:
        return int(m.group(1))

    m = re.search(r"\bC(\d+)D\d+\b", text, flags=re.I)
    if m:
        return int(m.group(1))

    return "NA"


def extract_height_weight_bsa(text: str):
    height = "NA"
    weight = "NA"
    bsa = "NA"

    m_h = re.search(
        r"\b(height|ht)\s*[:=]?\s*([1-2]\d{2})\s*cm\b", text, flags=re.I
    )
    if m_h:
        height = float(m_h.group(2))

    m_w = re.search(
        r"\b(weight|wt)\s*[:=]?\s*([3-9]\d(?:\.\d+)?)\s*kg\b", text, flags=re.I
    )
    if m_w:
        weight = float(m_w.group(2))

    m_b = re.search(r"\bBSA\s*[:=]?\s*([0-9]\.\d{2})\b", text, flags=re.I)
    if m_b:
        bsa = float(m_b.group(1))
    else:
        if isinstance(height, (int, float)) and isinstance(weight, (int, float)):
            bsa = round(((height * weight) / 3600.0) ** 0.5, 2)

    return height, weight, bsa


CHEMO_DRUG_KEYWORDS = [
    "doxorubicin",
    "epirubicin",
    "cyclophosphamide",
    "docetaxel",
    "paclitaxel",
    "carboplatin",
    "trastuzumab",
    "pertuzumab",
    "fulvestrant",
    "letrozole",
    "anastrozole",
    "exemestane",
    "tamoxifen",
    "lapatinib",
    "olaparib",
    "palbociclib",
    "ribociclib",
    "abemaciclib",
]


def extract_regimen(text: str) -> str:
    lower = text.lower()
    found = []
    for kw in CHEMO_DRUG_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            found.append(kw.capitalize())
    seen = set()
    unique = []
    for d in found:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    if not unique:
        return "NA"
    return " + ".join(unique)


# ------------ LAB HELPERS ------------


def get_lab_value(labs_df: pd.DataFrame, pattern: str, note_date: pd.Timestamp):
    if labs_df.empty:
        return "NA"

    mask = labs_df["TEST_NAME"].astype(str).str.contains(
        pattern, case=False, regex=True, na=False
    )

    if "ANC" in pattern:
        mask = mask & ~labs_df["TEST_NAME"].astype(str).str.contains("%")

    sub = labs_df[mask].copy()
    if sub.empty:
        return "NA"

    if "DATE/TIME SPECIMEN TAKEN" in sub.columns:
        sub["DATE/TIME SPECIMEN TAKEN"] = pd.to_datetime(
            sub["DATE/TIME SPECIMEN TAKEN"], errors="coerce"
        )
        with_date = sub.dropna(subset=["DATE/TIME SPECIMEN TAKEN"]).copy()
        if not with_date.empty and pd.notna(note_date):
            before = with_date[with_date["DATE/TIME SPECIMEN TAKEN"] <= note_date]
            if not before.empty:
                before = before.sort_values(
                    "DATE/TIME SPECIMEN TAKEN", ascending=False
                )
                return before.iloc[0]["TEST_RESULT"]

            with_date["delta"] = (
                with_date["DATE/TIME SPECIMEN TAKEN"] - note_date
            ).abs()
            with_date = with_date.sort_values("delta", ascending=True)
            return with_date.iloc[0]["TEST_RESULT"]

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

    if "Has_Clinical_Recommendation" not in df.columns:
        raise ValueError("Column 'Has_Clinical_Recommendation' not found in input file.")

    df["Has_Clinical_Recommendation"] = (
        df["Has_Clinical_Recommendation"].astype(str).str.strip().str.lower()
    )

    for col in ["Entry_Date", "DATE/TIME SPECIMEN TAKEN", "DATE REPORT COMPLETED"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    all_rows = []

    for doc_num, group in df.groupby("Document_Number", dropna=True):
        note_mask = group["Has_Clinical_Recommendation"].eq("yes")
        note_rows = group[note_mask]
        if note_rows.empty:
            continue

        note_row = note_rows.sort_values("Entry_Date").iloc[-1]

        # --- note columns ---
        note_text_raw = note_row.get("Note", "")
        note_text_clean = clean_text(note_text_raw)
        note_without_rec_raw = remove_recommendations(note_text_raw)

        note_date = note_row.get("Entry_Date", pd.NaT)
        labs_df = group[group["TEST_NAME"].notna()].copy()

        row = {
            "MRN": note_row.get("MRN"),
            "Document_Number": note_row.get("Document_Number"),
            "Visit_Number": note_row.get("Visit_Number"),
            "Entry_Date": note_date,
            "VISIT_LOCATION": note_row.get("VISIT_LOCATION"),
            "SERVICE": note_row.get("SERVICE"),
            "HOSPITAL_LOCATION": note_row.get("HOSPITAL_LOCATION"),
            "AUTHOR_SERVICE": note_row.get("AUTHOR_SERVICE"),
            # new note columns:
            "Note_Original": note_text_raw,
            "Note_Without_Recommendations": note_without_rec_raw,
            # keep a cleaned version (full note) for regex extraction
            "Note_Clean": note_text_clean,
        }

        # Use the cleaned full note for all downstream extractions
        analysis_text = note_text_clean

        row["Stage"] = extract_stage(analysis_text)
        t, n, m = extract_tnm(analysis_text)
        row["TNM_T"] = t
        row["TNM_N"] = n
        row["TNM_M"] = m
        row["Grade"] = extract_grade(analysis_text)

        ER, PR, HER2 = extract_receptor_statuses(analysis_text)
        row["ER_Status"] = ER
        row["PR_Status"] = PR
        row["HER2_Status"] = HER2

        height, weight, bsa = extract_height_weight_bsa(analysis_text)
        row["Height_cm"] = height
        row["Weight_kg"] = weight
        row["BSA_m2"] = bsa

        row.update(extract_comorbidities(analysis_text))

        row["CharlsonLikeScore"] = "NA"

        row["Regimen"] = extract_regimen(analysis_text)
        row["CycleNumber"] = extract_cycle_number(analysis_text)

        for col_name, pattern in LAB_PATTERNS.items():
            row[col_name] = get_lab_value(labs_df, pattern, note_date)

        row["UrineTests"] = build_urine_summary(labs_df)

        all_rows.append(row)

    if not all_rows:
        raise ValueError("No rows with Has_Clinical_Recommendation == 'yes' were found.")

    out_df = pd.DataFrame(all_rows)

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

    output_path = Path(OUTPUT_FILE)
    out_df.to_excel(output_path, index=False)
    print(f"Saved preprocessed file to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
