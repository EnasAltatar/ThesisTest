import math
import re
from pathlib import Path

import pandas as pd


# -------------- CONFIG --------------

# These filenames are relative to the repo root
INPUT_FILE = "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = "khcc_preprocessed.xlsx"


# -------------- LAB MAPPING --------------

# Map many possible TEST_NAME strings to the unified column names
LAB_TEST_MAP = {
    "WBC_x10^9_L": [
        "WBC", "WHITE BLOOD CELL", "WBC COUNT"
    ],
    "ANC_x10^9_L": [
        "ANC", "ABSOLUTE NEUTROPHIL", "NEUTROPHIL COUNT"
    ],
    "Hemoglobin_g_dL": [
        "HGB", "HEMOGLOBIN"
    ],
    "Platelets_x10^9_L": [
        "PLT", "PLATELET"
    ],
    "Creatinine_mg_dL": [
        "CREAT", "CREATININE"
    ],
    "eGFR_mL_min_1.73m2": [
        "EGFR", "ESTIMATED GFR"
    ],
    "BUN_mg_dL": [
        "BUN", "UREA"
    ],
    "AST_U_L": [
        "AST", "SGOT"
    ],
    "ALT_U_L": [
        "ALT", "SGPT"
    ],
    "ALP_U_L": [
        "ALP", "ALK PHOS", "ALKALINE PHOSPHATASE"
    ],
    "TotalBilirubin_mg_dL": [
        "TBIL", "TOTAL BILI", "TOTAL BILIRUBIN"
    ],
    "Albumin_g_dL": [
        "ALB", "ALBUMIN"
    ],
    "Troponin_ng_L": [
        "TROPONIN", "TROP I", "TROP T"
    ],
}


# -------------- TEXT PATTERNS --------------

STAGE_PATTERN = re.compile(r"STAGE\s*([0IVX]+[AB]?)", re.IGNORECASE)
TNM_COMBINED_PATTERN = re.compile(r"\bT(\d[abAB]?)N(\d[abAB]?)M(\d[abAB]?)\b", re.IGNORECASE)
TNM_SPLIT_T_PATTERN = re.compile(r"\bT(\d[abAB]?)\b", re.IGNORECASE)
TNM_SPLIT_N_PATTERN = re.compile(r"\bN(\d[abAB]?)\b", re.IGNORECASE)
TNM_SPLIT_M_PATTERN = re.compile(r"\bM(\d[abAB]?)\b", re.IGNORECASE)

GRADE_PATTERN = re.compile(r"(GRADE\s*([123]))|(G([123]))", re.IGNORECASE)

HEIGHT_PATTERN = re.compile(r"(HEIGHT|HT)\s*[:=]?\s*(\d{2,3})\s*cm", re.IGNORECASE)
WEIGHT_PATTERN = re.compile(r"(WEIGHT|WT)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*kg", re.IGNORECASE)

CYCLE_PATTERN = re.compile(r"CYCLE\s*(\d+)", re.IGNORECASE)

# Keywords for comorbidities (very simple bag-of-words style)
COMORBIDITY_KEYWORDS = {
    "Comorbidity_DM": ["DIABETES", "DM", "T2DM", "TYPE 2 DIABETES"],
    "Comorbidity_HTN": ["HYPERTENSION", "HTN", "HIGH BLOOD PRESSURE"],
    "Comorbidity_CAD": ["CORONARY ARTERY DISEASE", "CAD", "ISCHEMIC HEART"],
    "Comorbidity_CKD": ["CHRONIC KIDNEY", "CKD", "RENAL FAILURE"],
    "Comorbidity_Asthma": ["ASTHMA"],
    "Comorbidity_Depression": ["DEPRESSION", "DEPRESSIVE DISORDER"],
}

# Very small list just to flag that a regimen is mentioned.
# We are not trying to be perfect here – this is mainly to populate Regimen
CHEMO_KEYWORDS = [
    "AC ", "EC ", "FEC", "TC ", "TCH", "HERCEPTIN", "PERTUZUMAB",
    "PACLITAXEL", "DOCETAXEL", "DOXORUBICIN", "EPIRUBICIN",
    "CARBOPLATIN", "CYCLOPHOSPHAMIDE"
]


# -------------- HELPER FUNCTIONS --------------


def map_lab_name(raw_name: str):
    if not isinstance(raw_name, str):
        return None
    upper = raw_name.upper()
    for target_col, patterns in LAB_TEST_MAP.items():
        for pat in patterns:
            if pat in upper:
                return target_col
    return None


def extract_stage(note: str):
    m = STAGE_PATTERN.search(note)
    if m:
        return m.group(1).upper()
    return pd.NA


def extract_tnm(note: str):
    t_val = n_val = m_val = pd.NA
    m_combined = TNM_COMBINED_PATTERN.search(note)
    if m_combined:
        t_val, n_val, m_val = m_combined.groups()
    else:
        mt = TNM_SPLIT_T_PATTERN.search(note)
        mn = TNM_SPLIT_N_PATTERN.search(note)
        mm = TNM_SPLIT_M_PATTERN.search(note)
        if mt:
            t_val = mt.group(1)
        if mn:
            n_val = mn.group(1)
        if mm:
            m_val = mm.group(1)
    return t_val, n_val, m_val


def extract_grade(note: str):
    m = GRADE_PATTERN.search(note)
    if not m:
        return pd.NA
    # Either group(2) or group(4) will contain the number
    return (m.group(2) or m.group(4)).strip()


def extract_receptor_status(note: str, marker: str):
    up = note.upper()
    if marker + " NEG" in up or f"{marker}-" in up:
        return "Negative"
    if marker + " POS" in up or f"{marker}+" in up or marker + " POSITIVE" in up:
        return "Positive"
    return pd.NA


def derive_subtype(er, pr, her2):
    er_pos = str(er).upper().startswith("POS")
    pr_pos = str(pr).upper().startswith("POS")
    her2_pos = str(her2).upper().startswith("POS") or "3+" in str(her2)
    if (er_pos or pr_pos) and not her2_pos:
        return "HR+/HER2-"
    if (er_pos or pr_pos) and her2_pos:
        return "HR+/HER2+"
    if (not er_pos and not pr_pos) and her2_pos:
        return "HR-/HER2+"
    if (not er_pos and not pr_pos) and not her2_pos:
        return "Triple-negative"
    return pd.NA


def extract_height_weight(note: str):
    h = w = pd.NA
    mh = HEIGHT_PATTERN.search(note)
    mw = WEIGHT_PATTERN.search(note)
    if mh:
        try:
            h = float(mh.group(2))
        except ValueError:
            pass
    if mw:
        try:
            w = float(mw.group(2))
        except ValueError:
            pass
    return h, w


def compute_bsa(height_cm, weight_kg, existing_bsa):
    """
    If BSA already present (from labs), keep it.
    Otherwise compute using Mosteller if both height & weight exist.
    """
    if pd.notna(existing_bsa):
        return existing_bsa
    if pd.isna(height_cm) or pd.isna(weight_kg):
        return pd.NA
    try:
        bsa = math.sqrt((height_cm * weight_kg) / 3600.0)
        return round(bsa, 2)
    except Exception:
        return pd.NA


def extract_cycle_number(note: str, visit: str):
    for text in (str(visit), str(note)):
        m = CYCLE_PATTERN.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return pd.NA
    return pd.NA


def extract_comorbidities(note: str):
    up = note.upper()
    result = {}
    for col, keywords in COMORBIDITY_KEYWORDS.items():
        result[col] = any(kw in up for kw in keywords)
    # Charlson-like score = simple count of True flags
    score = sum(1 for v in result.values() if v)
    result["CharlsonLikeScore"] = score if score > 0 else pd.NA
    return result


def extract_regimen(note: str):
    up = note.upper()
    found = []
    for kw in CHEMO_KEYWORDS:
        if kw.strip() in up and kw.strip() not in found:
            found.append(kw.strip())
    if not found:
        return pd.NA
    return " + ".join(found)


# -------------- MAIN PIPELINE --------------


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / INPUT_FILE
    output_path = repo_root / OUTPUT_FILE

    print(f"Loading: {input_path}")
    df = pd.read_excel(input_path)

    # Ensure text columns are strings
    if "Note" in df.columns:
        df["Note"] = df["Note"].fillna("").astype(str)
    else:
        raise RuntimeError("Expected a 'Note' column in the input file.")

    # ---------- Build labs wide table ----------
    labs = df[["MRN", "Document_Number", "TEST_NAME", "TEST_RESULT"]].copy()
    labs = labs.dropna(subset=["TEST_NAME", "TEST_RESULT"])
    labs["Lab_Var"] = labs["TEST_NAME"].apply(map_lab_name)
    labs = labs.dropna(subset=["Lab_Var"])

    if not labs.empty:
        # pivot to wide format
        labs_wide = (
            labs.pivot_table(
                index=["MRN", "Document_Number"],
                columns="Lab_Var",
                values="TEST_RESULT",
                aggfunc="first",
            )
            .reset_index()
        )

        # also keep a human-readable labs text field
        labs_text = (
            labs.groupby(["MRN", "Document_Number"])
            .apply(
                lambda x: "; ".join(
                    f"{n}: {r}" for n, r in zip(x["TEST_NAME"], x["TEST_RESULT"])
                )
            )
            .rename("Labs_Text")
            .reset_index()
        )
    else:
        labs_wide = df[["MRN", "Document_Number"]].drop_duplicates()
        labs_text = labs_wide.copy()
        labs_text["Labs_Text"] = ""

    # ---------- One row per clinical note ----------
    # Take the first row per (MRN, Document_Number) for the non-lab metadata
    base_cols = [
        col
        for col in df.columns
        if col
        not in {
            "TEST_NAME",
            "TEST_RESULT",
        }
    ]
    base = (
        df.sort_values("Entry_Date")
        .groupby(["MRN", "Document_Number"], as_index=False)[base_cols]
        .first()
    )

    merged = (
        base.merge(labs_wide, on=["MRN", "Document_Number"], how="left")
        .merge(labs_text, on=["MRN", "Document_Number"], how="left")
    )

    # ---------- Derived oncology variables from Note text ----------

    print("Extracting stage / TNM / grade / receptors from notes ...")
    merged["Stage"] = merged["Note"].apply(extract_stage)

    tnm_values = merged["Note"].apply(extract_tnm)
    merged["TNM_T"] = tnm_values.apply(lambda x: x[0])
    merged["TNM_N"] = tnm_values.apply(lambda x: x[1])
    merged["TNM_M"] = tnm_values.apply(lambda x: x[2])

    merged["Grade"] = merged["Note"].apply(extract_grade)

    merged["ER_Status"] = merged["Note"].apply(lambda x: extract_receptor_status(x, "ER"))
    merged["PR_Status"] = merged["Note"].apply(lambda x: extract_receptor_status(x, "PR"))
    merged["HER2_Status"] = merged["Note"].apply(
        lambda x: extract_receptor_status(x, "HER2")
    )

    merged["Subtype"] = merged.apply(
        lambda row: derive_subtype(
            row["ER_Status"], row["PR_Status"], row["HER2_Status"]
        ),
        axis=1,
    )

    # ---------- Anthropometrics & BSA ----------
    print("Extracting height / weight / BSA ...")
    hw = merged["Note"].apply(extract_height_weight)
    merged["Height_cm"] = hw.apply(lambda x: x[0])
    merged["Weight_kg"] = hw.apply(lambda x: x[1])

    # If a BSA column already exists from labs, keep it; otherwise compute
    if "BSA_m2" not in merged.columns:
        merged["BSA_m2"] = pd.NA

    merged["BSA_m2"] = merged.apply(
        lambda row: compute_bsa(row["Height_cm"], row["Weight_kg"], row["BSA_m2"]),
        axis=1,
    )

    # ---------- Cycle number ----------
    print("Extracting cycle numbers ...")
    merged["CycleNumber"] = merged.apply(
        lambda row: extract_cycle_number(row.get("Note", ""), row.get("Visit", "")),
        axis=1,
    )

    # ---------- Comorbidities ----------
    print("Extracting comorbidities ...")
    comorb_series = merged["Note"].apply(extract_comorbidities)
    for col in COMORBIDITY_KEYWORDS.keys():
        merged[col] = comorb_series.apply(lambda d: d[col])
    merged["CharlsonLikeScore"] = comorb_series.apply(
        lambda d: d["CharlsonLikeScore"]
    )

    # ---------- Regimen & medication-related placeholders ----------
    print("Extracting regimen (rough) and creating placeholder columns ...")
    merged["Regimen"] = merged["Note"].apply(extract_regimen)

    # Place-holder columns that will later be filled by AI / downstream scripts.
    # We create them now so the schema matches your synthetic dataset.
    placeholder_cols = [
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
    for col in placeholder_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    # ---------- Reorder columns (optional, to resemble synthetic schema) ----------

    preferred_order = [
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
        "Note",
        # oncology / patient characteristics
        "Subtype",
        "Stage",
        "TNM_T",
        "TNM_N",
        "TNM_M",
        "Grade",
        "ER_Status",
        "PR_Status",
        "HER2_Status",
        "WBC_x10^9_L",
        "ANC_x10^9_L",
        "Hemoglobin_g_dL",
        "Platelets_x10^9_L",
        "Creatinine_mg_dL",
        "eGFR_mL_min_1.73m2",
        "BUN_mg_dL",
        "AST_U_L",
        "ALT_U_L",
        "ALP_U_L",
        "TotalBilirubin_mg_dL",
        "Albumin_g_dL",
        "LVEF_percent",       # may stay empty – rarely available in this table
        "Troponin_ng_L",
        "Height_cm",
        "Weight_kg",
        "BSA_m2",
        "Comorbidity_DM",
        "Comorbidity_HTN",
        "Comorbidity_CAD",
        "Comorbidity_CKD",
        "Comorbidity_Asthma",
        "Comorbidity_Depression",
        "CharlsonLikeScore",
        "PriorLines",
        "PriorTherapies",
        "BestPriorResponse",
        "Regimen",
        "CycleNumber",
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
        "Labs_Text",
    ]

    # Add any columns we did not explicitly list to the end
    ordered_cols = [c for c in preferred_order if c in merged.columns] + [
        c for c in merged.columns if c not in preferred_order
    ]
    merged = merged[ordered_cols]

    print(f"Saving preprocessed data to: {output_path}")
    merged.to_excel(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
