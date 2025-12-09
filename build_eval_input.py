import pandas as pd

# ========= 1) Load input files =========
# Adjust filenames here if yours are different
KHCC_CASES_FILE = "khcc_cases_200.xlsx"
AI_NOTES_FILE   = "KHCC_AI_Notes.xlsx"
OUTPUT_FILE     = "khcc_eval_input.xlsx"

print(f"Loading cases from: {KHCC_CASES_FILE}")
cases = pd.read_excel(KHCC_CASES_FILE)

print(f"Loading AI notes from: {AI_NOTES_FILE}")
ai = pd.read_excel(AI_NOTES_FILE)

# ========= 2) Standardize key column name =========
# Both tables must have the same key to merge on
cases = cases.rename(columns={"Case_ID": "case_id"})
ai    = ai.rename(columns={"Case_ID": "case_id"})

# Optional: strip whitespace from case_id
cases["case_id"] = cases["case_id"].astype(str).str.strip()
ai["case_id"]    = ai["case_id"].astype(str).str.strip()

# ========= 3) Merge on case_id =========
merged = cases.merge(ai, on="case_id", how="left")

print(f"Merged rows: {len(merged)}")

# ========= 4) Build the exact columns your evaluation app expects =========
# If your patient summary column has a different name, change it here.
# Common options from your preprocessing:
#   "Note_Without_Recommendations"  (recommended)
#   "Original_Note"
PATIENT_SUMMARY_COL = "Note_Without_Recommendations"

# If you don't have a phase column yet, you can set everything to "1" for now.
# Later you can change this to real phases (1, 2, 3) if you want.
if "phase" in merged.columns:
    phase_series = merged["phase"]
else:
    phase_series = 1  # all phase 1 for now

eval_df = pd.DataFrame({
    "case_id": merged["case_id"],
    "phase": phase_series,
    "patient_summary": merged[PATIENT_SUMMARY_COL],

    # Map AI notes from KHCC_AI_Notes.xlsx
    "chatgpt_note": merged["OpenAI_Note"],   # OpenAI column
    "claude_note":  merged["Claude_Note"],   # Claude column
    "cadss_note":   merged["CADSS_Note"],    # CADSS column

    # Human pharmacist note from the original data
    "human_note":   merged["Original_Note"],  # ground-truth pharmacist note
})

# ========= 5) Save for the evaluation app =========
eval_df.to_excel(OUTPUT_FILE, index=False)
print(f"Saved evaluation input file: {OUTPUT_FILE}")

