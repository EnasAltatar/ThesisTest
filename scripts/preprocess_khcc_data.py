import re
import json
from pathlib import Path
import pandas as pd
from openai import OpenAI
import os

# -------------------------
# CONFIG
# -------------------------

INPUT_FILE = "merged_labs_pharmacy.xlsx"
OUTPUT_FILE = "khcc_preprocessed.xlsx"
MODEL_NAME = "gpt-4o-mini"

# Use PREPROCESS secret for the API key
client = OpenAI(api_key=os.getenv("PREPROCESS"))

# -------------------------
# HELPERS
# -------------------------

def clean_text(t: str) -> str:
    """Light cleaning for LLM (does NOT affect stored original text)."""
    if pd.isna(t):
        return ""
    t = str(t).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


def mask_patient_id(note_text, mrn):
    """
    Keep the original note as-is, but:
    - Replace the exact MRN (if present) with *****.
    - Also mask any 6+ digit standalone numbers as ***** (extra safety).
    No other modifications to wording or formatting.
    """
    if pd.isna(note_text):
        return ""
    text = str(note_text)

    # Mask exact MRN if available
    if pd.notna(mrn):
        mrn_str = str(mrn).strip()
        if mrn_str:
            text = text.replace(mrn_str, "*****")

    # Mask other long numeric IDs (6+ digits)
    text = re.sub(r"\b\d{6,}\b", "*****", text)

    return text


# -------------------------
# LLM EXTRACTION
# -------------------------

def extract_with_llm(note_text: str):
    """
    Ask ChatGPT to:
    - Split note into assessment vs recommendations, VERBATIM.
    - Extract structured fields ONLY if explicitly present.
    - Return strict JSON.
    """
    prompt = f"""
You are a clinical-text extraction expert. Follow these exact rules:

RULES:
1. DO NOT rewrite or paraphrase ANY text.
2. DO NOT summarize.
3. Extract ONLY information explicitly written inside the note.
4. If information is missing → return "NA".
5. Split the note into two sections:
   - "assessment_section": all content BEFORE recommendations.
   - "recommendations_section": ONLY clinical recommendations.
6. RECOMMENDATIONS BEGIN when the text contains explicit instructions
   like continue, hold, start, stop, advise, reduce, increase, repeat,
   consider, administer, etc.
7. Return both sections VERBATIM — do NOT modify text.
8. Always return VALID JSON ONLY.

NOTE:
\"\"\"{note_text}\"\"\"


Return ONLY JSON in this exact structure:

{{
  "assessment_section": "...",
  "recommendations_section": "...",
  "stage": "...",
  "tnm_t": "...",
  "tnm_n": "...",
  "tnm_m": "...",
  "grade": "...",
  "er_status": "...",
  "pr_status": "...",
  "her2_status": "...",
  "height_cm": "...",
  "weight_kg": "...",
  "bsa_m2": "...",
  "regimen": "...",
  "cycle_number": "...",
  "comorbidities": {{
      "dm": "...",
      "htn": "...",
      "cad": "...",
      "ckd": "...",
      "asthma": "...",
      "depression": "..."
  }}
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        print("LLM extraction error:", e)
        return None


# -------------------------
# LAB MAPPING
# -------------------------

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


def get_lab_value(labs_df: pd.DataFrame, pattern: str, note_date: pd.Timestamp):
    """Get closest lab result to note_date for the given pattern."""
    if labs_df.empty:
        return "NA"

    mask = labs_df["TEST_NAME"].astype(str).str.contains(
        pattern, case=False, regex=True, na=False
    )

    # Avoid ANC percentage tests
    if "ANC" in pattern:
        mask = mask & ~labs_df["TEST_NAME"].astype(str).str.contains("%")

    sub = labs_df[mask].copy()
    if sub.empty:
        return "NA"

    if "DATE/TIME SPECIMEN TAKEN" in sub.columns and pd.notna(note_date):
        sub["DATE/TIME SPECIMEN TAKEN"] = pd.to_datetime(
            sub["DATE/TIME SPECIMEN TAKEN"], errors="coerce"
        )
        dated = sub.dropna(subset=["DATE/TIME SPECIMEN TAKEN"])
        if not dated.empty:
            dated["dt"] = (dated["DATE/TIME SPECIMEN TAKEN"] - note_date).abs()
            dated = dated.sort_values("dt")
            return dated.iloc[0]["TEST_RESULT"]

    # fallback (if no dates)
    return sub.iloc[-1]["TEST_RESULT"]


def build_urine_summary(labs_df: pd.DataFrame) -> str:
    """Build a short text summary of urine-related tests."""
    mask = labs_df["TEST_NAME"].astype(str).str.contains(
        URINE_PATTERN, case=False, regex=True, na=False
    )
    urine = labs_df[mask]
    if urine.empty:
        return "NA"
    parts = [
        f"{row['TEST_NAME']}: {row['TEST_RESULT']}"
        for _, row in urine.iterrows()
    ]
    return "; ".join(parts)


def build_all_labs_text(labs_df: pd.DataFrame) -> str:
    """
    Build a single text column containing ALL lab tests for that document,
    preserving them in human-readable form (this is what you asked to keep).
    """
    if labs_df.empty:
        return ""
    rows = []
    for _, r in labs_df.sort_values("DATE/TIME SPECIMEN TAKEN").iterrows():
        dt = r.get("DATE/TIME SPECIMEN TAKEN", "")
        if pd.notna(dt):
            try:
                dt_str = pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M")
            except Exception:
                dt_str = str(dt)
        else:
            dt_str = ""
        tn = str(r.get("TEST_NAME", "")).strip()
        tr = str(r.get("TEST_RESULT", "")).strip()
        if dt_str:
            rows.append(f"{dt_str} - {tn}: {tr}")
        else:
            rows.append(f"{tn}: {tr}")
    return " | ".join(rows)


# -------------------------
# MAIN PIPELINE
# -------------------------

def main():
    df = pd.read_excel(INPUT_FILE)

    # Normalize recommendation flag
    df["Has_Clinical_Recommendation"] = (
        df["Has_Clinical_Recommendation"].astype(str).str.strip().str.lower()
    )

    # Dates
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"], errors="coerce")
    if "DATE/TIME SPECIMEN TAKEN" in df.columns:
        df["DATE/TIME SPECIMEN TAKEN"] = pd.to_datetime(
            df["DATE/TIME SPECIMEN TAKEN"], errors="coerce"
        )

    final_rows = []

    # Group by clinical note (Document_Number)
    for doc, group in df.groupby("Document_Number"):
        note_rows = group[group["Has_Clinical_Recommendation"] == "yes"]
        if note_rows.empty:
            continue

        # Use the latest row for that document as the base row
        note_row = note_rows.sort_values("Entry_Date").iloc[-1]
        note_raw = note_row.get("Note", "")
        mrn_val = note_row.get("MRN")

        # De-identify ONLY the patient ID
        note_masked = mask_patient_id(note_raw, mrn_val)
        note_for_llm = clean_text(note_masked)
        note_date = note_row.get("Entry_Date")

        # LLM extraction for assessment vs recommendations + structure
        llm = extract_with_llm(note_for_llm)
        if not llm:
            continue

        # All lab rows for this document
        labs_df = group[group["TEST_NAME"].notna()].copy()

        # Start from ALL ORIGINAL COLUMNS of the selected row
        base_row = note_row.to_dict()

        # Overwrite Note with masked version and also add explicit original column
        base_row["Note"] = note_masked
        base_row["Note_Original"] = note_masked

        # New split columns
        base_row["Note_Without_Recommendations"] = llm.get("assessment_section", "")
        base_row["Recommendations_Only"] = llm.get("recommendations_section", "")

        # Structured fields (only if present in note; otherwise "NA")
        base_row["Stage"] = llm.get("stage", "NA")
        base_row["TNM_T"] = llm.get("tnm_t", "NA")
        base_row["TNM_N"] = llm.get("tnm_n", "NA")
        base_row["TNM_M"] = llm.get("tnm_m", "NA")
        base_row["Grade"] = llm.get("grade", "NA")
        base_row["ER_Status"] = llm.get("er_status", "NA")
        base_row["PR_Status"] = llm.get("pr_status", "NA")
        base_row["HER2_Status"] = llm.get("her2_status", "NA")
        base_row["Height_cm"] = llm.get("height_cm", "NA")
        base_row["Weight_kg"] = llm.get("weight_kg", "NA")
        base_row["BSA_m2"] = llm.get("bsa_m2", "NA")
        base_row["Regimen"] = llm.get("regimen", "NA")
        base_row["CycleNumber"] = llm.get("cycle_number", "NA")

        # Comorbidities (nested JSON)
        comorbid = llm.get("comorbidities", {}) or {}
        base_row["Comorbidity_DM"] = comorbid.get("dm", "NA")
        base_row["Comorbidity_HTN"] = comorbid.get("htn", "NA")
        base_row["Comorbidity_CAD"] = comorbid.get("cad", "NA")
        base_row["Comorbidity_CKD"] = comorbid.get("ckd", "NA")
        base_row["Comorbidity_Asthma"] = comorbid.get("asthma", "NA")
        base_row["Comorbidity_Depression"] = comorbid.get("depression", "NA")

        # Text column containing ALL lab tests
        base_row["All_Labs_Text"] = build_all_labs_text(labs_df)

        # Urine tests
        base_row["UrineTests"] = build_urine_summary(labs_df)

        # Numeric lab fields
        for col, pattern in LAB_PATTERNS.items():
            base_row[col] = get_lab_value(labs_df, pattern, note_date)

        final_rows.append(base_row)

    out_df = pd.DataFrame(final_rows)

    # Save final file (all original columns + new ones)
    out_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
