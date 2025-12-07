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

# IMPORTANT: using PREPROCESS secret
client = OpenAI(api_key=os.getenv("PREPROCESS"))

# -------------------------
# CLEAN TEXT
# -------------------------

def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


# -------------------------
# LLM — VERBATIM SPLIT + STRUCTURED EXTRACTION
# -------------------------

def extract_with_llm(note_text):
    """
    ChatGPT returns:
    - assessment_section (verbatim)
    - recommendations_section (verbatim)
    - structured fields (only if explicitly present)
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
6. RECOMMENDATIONS BEGIN when the text contains explicit instructions (continue, hold, start, stop, advise, reduce, increase, repeat, consider, administer, etc.).
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
            max_tokens=1500
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


def get_lab_value(labs_df, pattern, note_date):
    if labs_df.empty:
        return "NA"

    mask = labs_df["TEST_NAME"].astype(str).str.contains(
        pattern, case=False, regex=True, na=False
    )

    # Avoid percentage ANC tests
    if "ANC" in pattern:
        mask = mask & ~labs_df["TEST_NAME"].astype(str).str.contains("%")

    sub = labs_df[mask].copy()
    if sub.empty:
        return "NA"

    if "DATE/TIME SPECIMEN TAKEN" in sub.columns:
        sub["DATE/TIME SPECIMEN TAKEN"] = pd.to_datetime(
            sub["DATE/TIME SPECIMEN TAKEN"], errors="coerce"
        )
        dated = sub.dropna(subset=["DATE/TIME SPECIMEN TAKEN"])
        if not dated.empty:
            dated["dt"] = (dated["DATE/TIME SPECIMEN TAKEN"] - note_date).abs()
            dated = dated.sort_values("dt")
            return dated.iloc[0]["TEST_RESULT"]

    # fallback
    return sub.iloc[-1]["TEST_RESULT"]


def build_urine_summary(labs_df):
    mask = labs_df["TEST_NAME"].astype(str).str.contains(
        URINE_PATTERN, case=False, regex=True, na=False
    )
    urine = labs_df[mask]
    if urine.empty:
        return "NA"
    return "; ".join(f"{r['TEST_NAME']}: {r['TEST_RESULT']}" for _, r in urine.iterrows())


# -------------------------
# MAIN PIPELINE
# -------------------------

def main():
    df = pd.read_excel(INPUT_FILE)

    df["Has_Clinical_Recommendation"] = (
        df["Has_Clinical_Recommendation"].astype(str).str.strip().str.lower()
    )

    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"], errors="coerce")
    df["DATE/TIME SPECIMEN TAKEN"] = pd.to_datetime(
        df["DATE/TIME SPECIMEN TAKEN"], errors="coerce"
    )

    final_rows = []

    for doc, group in df.groupby("Document_Number"):
        note_rows = group[group["Has_Clinical_Recommendation"] == "yes"]
        if note_rows.empty:
            continue

        note_row = note_rows.sort_values("Entry_Date").iloc[-1]
        note = note_row.get("Note", "")
        note_clean = clean_text(note)
        note_date = note_row.get("Entry_Date")

        # ---- LLM extraction ----
        llm = extract_with_llm(note_clean)
        if not llm:
            continue

        labs_df = group[group["TEST_NAME"].notna()].copy()

        row = {
            "MRN": note_row.get("MRN"),
            "Document_Number": doc,
            "Visit_Number": note_row.get("Visit_Number"),
            "Entry_Date": note_date,
            "Note_Original": note,
            "Note_Without_Recommendations": llm.get("assessment_section", ""),
            "Recommendations_Only": llm.get("recommendations_section", ""),
            # Structured fields
            "Stage": llm.get("stage", "NA"),
            "TNM_T": llm.get("tnm_t", "NA"),
            "TNM_N": llm.get("tnm_n", "NA"),
            "TNM_M": llm.get("tnm_m", "NA"),
            "Grade": llm.get("grade", "NA"),
            "ER_Status": llm.get("er_status", "NA"),
            "PR_Status": llm.get("pr_status", "NA"),
            "HER2_Status": llm.get("her2_status", "NA"),
            "Height_cm": llm.get("height_cm", "NA"),
            "Weight_kg": llm.get("weight_kg", "NA"),
            "BSA_m2": llm.get("bsa_m2", "NA"),
            "Regimen": llm.get("regimen", "NA"),
            "CycleNumber": llm.get("cycle_number", "NA"),
            # Comorbidities
            "Comorbidity_DM": llm["comorbidities"].get("dm", "NA"),
            "Comorbidity_HTN": llm["comorbidities"].get("htn", "NA"),
            "Comorbidity_CAD": llm["comorbidities"].get("cad", "NA"),
            "Comorbidity_CKD": llm["comorbidities"].get("ckd", "NA"),
            "Comorbidity_Asthma": llm["comorbidities"].get("asthma", "NA"),
            "Comorbidity_Depression": llm["comorbidities"].get("depression", "NA"),
            "UrineTests": build_urine_summary(labs_df),
        }

        for col, pattern in LAB_PATTERNS.items():
            row[col] = get_lab_value(labs_df, pattern, note_date)

        final_rows.append(row)

    out = pd.DataFrame(final_rows)
    out.to_excel(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
