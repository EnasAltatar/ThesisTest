#!/usr/bin/env python3
"""
Synthetic Breast Cancer Dataset Generator
----------------------------------------
This generated dataset of breast cancer cases is fully synthetic, anonymized dataset along with a data dictionary. It is intended for experimentation, and methods research. NOT for clinical decision-making.

"""

import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd


def bounded_normal(mean, sd, low, high, size, decimals=1):
    vals = np.random.normal(mean, sd, size)
    vals = np.clip(vals, low, high)
    return np.round(vals, decimals)


def pick_regimen(subtype):
    hr = [
        "AC-T (Doxorubicin/Cyclophosphamide → Paclitaxel)",
        "TC (Docetaxel/Cyclophosphamide)",
        "CDK4/6 inhibitor + AI (Palbociclib + Letrozole)",
        "Fulvestrant + CDK4/6 inhibitor",
        "Paclitaxel weekly",
    ]
    her2 = [
        "TCHP (Docetaxel/Carboplatin/Trastuzumab/Pertuzumab)",
        "THP (Paclitaxel/Trastuzumab/Pertuzumab)",
        "AC → TH (Doxorubicin/Cyclophosphamide → Paclitaxel + Trastuzumab)",
        "Docetaxel + Trastuzumab",
    ]
    tnbc = [
        "AC-T (Doxorubicin/Cyclophosphamide → Paclitaxel)",
        "Carboplatin + Paclitaxel",
        "Pembrolizumab + Chemotherapy",
        "EC → Paclitaxel (Epirubicin/Cyclophosphamide → Paclitaxel)",
    ]
    if subtype == "HR+ (ER and/or PR positive, HER2-)":
        return random.choice(hr)
    elif subtype == "HER2+":
        return random.choice(her2)
    else:
        return random.choice(tnbc)


def schedule_for_regimen(reg):
    if "AC-T" in reg:
        return "AC q2w ×4 then Paclitaxel weekly ×12"
    if "TCHP" in reg:
        return "q3w ×6 cycles"
    if "THP" in reg:
        return "Paclitaxel weekly ×12; Trastuzumab/Pertuzumab q3w"
    if "CDK4/6" in reg:
        return "28-day cycles, days 1–21 on / 7 off (CDK4/6), AI daily"
    if "TC (" in reg:
        return "q3w ×4 cycles"
    if "Pembrolizumab" in reg:
        return "Pembrolizumab q3w + chemo per protocol"
    if "Carboplatin + Paclitaxel" in reg:
        return "q3w ×4–6 cycles"
    if "EC → Paclitaxel" in reg:
        return "EC q3w ×4 then Paclitaxel weekly ×12"
    if "Docetaxel + Trastuzumab" in reg:
        return "Docetaxel q3w ×4–6 + Trastuzumab q3w"
    if "AC → TH" in reg:
        return "AC q3w ×4 then Paclitaxel + Trastuzumab q3w"
    if "Paclitaxel weekly" in reg:
        return "Weekly ×12"
    if "Fulvestrant" in reg:
        return "Fulvestrant days 1,15,29 then q4w + CDK4/6 21/28"
    return "Per protocol"


def dose_plan_for_regimen(reg):
    if "AC-T" in reg:
        return "Doxorubicin 60 mg/m² + Cyclophosphamide 600 mg/m²; then Paclitaxel 80 mg/m² weekly"
    if "TCHP" in reg:
        return "Docetaxel 75 mg/m² + Carboplatin AUC6 + Trastuzumab + Pertuzumab"
    if "THP" in reg:
        return "Paclitaxel 80 mg/m² + Trastuzumab + Pertuzumab"
    if "CDK4/6" in reg:
        return "Palbociclib 125 mg d1–21 q28 + Letrozole 2.5 mg daily"
    if "TC (" in reg:
        return "Docetaxel 75 mg/m² + Cyclophosphamide 600 mg/m²"
    if "Pembrolizumab" in reg:
        return "Pembrolizumab 200 mg q3w + chemo per protocol"
    if "Carboplatin + Paclitaxel" in reg:
        return "Carboplatin AUC5 + Paclitaxel 175 mg/m² q3w"
    if "EC → Paclitaxel" in reg:
        return "Epirubicin 90 mg/m² + Cyclophosphamide 600 mg/m²; then Paclitaxel 80 mg/m² weekly"
    if "Docetaxel + Trastuzumab" in reg:
        return "Docetaxel 75 mg/m² + Trastuzumab q3w"
    if "AC → TH" in reg:
        return "Doxorubicin 60 mg/m² + Cyclophosphamide 600 mg/m²; then Paclitaxel 175 mg/m² + Trastuzumab q3w"
    if "Paclitaxel weekly" in reg:
        return "Paclitaxel 80 mg/m² weekly"
    if "Fulvestrant" in reg:
        return "Fulvestrant 500 mg IM days 1,15,29 then q4w + CDK4/6 21/28"
    return "Per protocol"


def marker_status(subtype):
    if subtype == "HR+ (ER and/or PR positive, HER2-)":
        er = np.random.choice(["Positive", "Positive", "Strong Positive"])
        pr = np.random.choice(["Positive", "Positive", "Negative"], p=[0.6, 0.25, 0.15])
        her2 = "Negative"
    elif subtype == "HER2+":
        er = np.random.choice(["Positive", "Negative"], p=[0.45, 0.55])
        pr = np.random.choice(["Positive", "Negative"], p=[0.35, 0.65])
        her2 = "Positive (IHC 3+ or FISH+)"
    else:
        er = "Negative"
        pr = "Negative"
        her2 = "Negative"
    return er, pr, her2


def tnm_for_stage(stage):
    if stage == "I":
        T = np.random.choice(["T1", "T2"], p=[0.7, 0.3])
        N = "N0"
        M = "M0"
    elif stage == "II":
        T = np.random.choice(["T2", "T3"], p=[0.7, 0.3])
        N = np.random.choice(["N0", "N1"], p=[0.4, 0.6])
        M = "M0"
    elif stage == "III":
        T = np.random.choice(["T3", "T4"], p=[0.6, 0.4])
        N = np.random.choice(["N1", "N2", "N3"], p=[0.5, 0.3, 0.2])
        M = "M0"
    else:
        T = np.random.choice(["T2", "T3", "T4"], p=[0.4, 0.35, 0.25])
        N = np.random.choice(["N0", "N1", "N2", "N3"], p=[0.2, 0.4, 0.25, 0.15])
        M = "M1"
    return T, N, M


def build_dataset(N, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    subtypes = np.random.choice(
        ["HR+ (ER and/or PR positive, HER2-)", "HER2+", "Triple-negative"],
        size=N,
        p=[0.68, 0.18, 0.14],
    )

    markers = np.array([marker_status(s) for s in subtypes])
    ER = markers[:, 0]
    PR = markers[:, 1]
    HER2 = markers[:, 2]

    stages = np.random.choice(["I", "II", "III", "IV"], size=N, p=[0.28, 0.38, 0.22, 0.12])
    TNM = np.array([tnm_for_stage(s) for s in stages])
    T, Nn, M = TNM[:, 0], TNM[:, 1], TNM[:, 2]

    grade = np.random.choice(["G1 (well differentiated)", "G2 (moderately)", "G3 (poorly)"], size=N, p=[0.25, 0.5, 0.25])

    WBC = bounded_normal(6.5, 2.0, 2.0, 20.0, N, 1)
    ANC = bounded_normal(3.8, 1.5, 0.2, 15.0, N, 1)
    Hgb = bounded_normal(12.5, 1.5, 7.0, 17.0, N, 1)
    Platelets = np.round(np.clip(np.random.normal(280, 80, N), 50, 800))

    Creatinine = bounded_normal(0.9, 0.3, 0.3, 2.5, N, 2)
    eGFR = bounded_normal(90, 20, 20, 120, N, 0)
    BUN = bounded_normal(14, 6, 3, 40, N, 0)

    AST = bounded_normal(24, 10, 5, 150, N, 0)
    ALT = bounded_normal(23, 10, 5, 150, N, 0)
    ALP = bounded_normal(90, 35, 30, 400, N, 0)
    Bilirubin = bounded_normal(0.7, 0.3, 0.1, 3.0, N, 2)
    Albumin = bounded_normal(4.1, 0.4, 2.0, 5.5, N, 2)

    LVEF = bounded_normal(62, 6, 35, 75, N, 0)
    Troponin = np.round(np.abs(np.random.normal(6, 8, N)), 1)

    Height_cm = bounded_normal(162, 8, 140, 190, N, 1)
    Weight_kg = bounded_normal(70, 15, 40, 120, N, 1)
    BSA = np.round(np.sqrt(Height_cm * Weight_kg / 3600), 2)

    dm = np.random.binomial(1, 0.22, N)
    htn = np.random.binomial(1, 0.38, N)
    cad = np.random.binomial(1, 0.10, N)
    ckd = (eGFR < 60).astype(int)
    asthma = np.random.binomial(1, 0.08, N)
    depression = np.random.binomial(1, 0.12, N)
    charlson = dm + htn + cad + ckd + asthma + depression + np.random.binomial(1, 0.15, N)

    prior_lines = np.random.choice([0, 1, 2, 3], size=N, p=[0.55, 0.25, 0.15, 0.05])
    prior_therapies = []
    prior_response = []
    for pl in prior_lines:
        if pl == 0:
            prior_therapies.append("None")
            prior_response.append("None")
        else:
            tx = random.sample(
                ["Neoadjuvant chemo", "Adjuvant chemo", "Endocrine therapy", "Anti-HER2 therapy", "Immunotherapy", "PARP inhibitor", "Radiation"],
                k=min(pl, 3),
            )
            prior_therapies.append("; ".join(tx))
            prior_response.append(random.choice(["CR", "PR", "SD", "PD"]))

    regimens = [pick_regimen(s) for s in subtypes]
    schedules = [schedule_for_regimen(r) for r in regimens]
    dose_plans = [dose_plan_for_regimen(r) for r in regimens]

    cycle_number = np.random.choice(range(1, 9), size=N, p=[0.20, 0.15, 0.13, 0.12, 0.11, 0.10, 0.10, 0.09])

    intended = np.round(np.random.uniform(90, 100, N), 0)
    actual = intended.copy()
    for i in range(N):
        reduction = 0
        if ANC[i] < 1.5 or Platelets[i] < 100:
            reduction += np.random.choice([10, 15, 20])
        if eGFR[i] < 50 or Creatinine[i] > 1.5:
            reduction += np.random.choice([10, 15])
        if AST[i] > 60 or ALT[i] > 60 or Bilirubin[i] > 1.5:
            reduction += np.random.choice([10, 15])
        if "Trastuzumab" in regimens[i] and LVEF[i] < 55:
            reduction += 10
        actual[i] = max(50, intended[i] - reduction)

    premeds = []
    for r in regimens:
        items = []
        if any(x in r for x in ["Docetaxel", "Paclitaxel"]):
            items += ["Dexamethasone", "H1/H2 antihistamine"]
        items += ["5-HT3 antagonist", "NK1 antagonist", "Dexamethasone (antiemetic)"]
        if "Carboplatin" in r:
            items += ["Hydration protocol"]
        premeds.append("; ".join(sorted(set(items))))

    postmeds = []
    for r in regimens:
        items = ["Oral antiemetic PRN"]
        if any(x in r for x in ["AC-T", "EC →", "Carboplatin"]):
            items += ["G-CSF support per risk"]
        if "CDK4/6" in r:
            items += ["CBC every cycle"]
        postmeds.append("; ".join(sorted(set(items))))

    dose_adj_note = []
    interactions = []
    contra = []
    supportive = []
    rationale = []

    for i in range(N):
        notes = []
        inters = []
        contr = []
        supp = []
        why = []

        if actual[i] < intended[i]:
            notes.append(f"Reduce to {int(actual[i])}% due to labs/toxicity.")
            why.append("Dose intensity reduced based on cytopenias/organ function.")
        else:
            notes.append("Proceed with full dose.")
            why.append("Counts and organ function acceptable.")

        if dm[i] == 1 and "Dexamethasone" in premeds[i]:
            inters.append("Monitor hyperglycemia with steroid premedication.")
        if "CDK4/6" in regimens[i]:
            inters.append("Avoid strong CYP3A inhibitors/inducers with CDK4/6 inhibitor.")
        if "Trastuzumab" in regimens[i] and LVEF[i] < 55:
            inters.append("Cardiotoxicity monitoring with HER2 therapy.")

        if ANC[i] < 1.0:
            contr.append("Hold cycle: ANC <1.0 ×10^9/L.")
        if float(Bilirubin[i]) > 2.0:
            contr.append("Hepatic impairment—consider dose modification or hold.")
        if LVEF[i] < 50 and "Trastuzumab" in regimens[i]:
            contr.append("Relative contraindication: low LVEF for HER2 therapy.")

        supp_base = ["Antiemetic plan", "Oral care to prevent mucositis", "Infection precautions education"]
        if Platelets[i] < 100:
            supp_base.append("Bleeding precautions")
        if Hgb[i] < 10.0:
            supp_base.append("Assess for iron deficiency; consider transfusion per symptoms")
        if ANC[i] < 1.5:
            supp_base.append("G-CSF primary/secondary prophylaxis as indicated")
        if eGFR[i] < 50:
            supp_base.append("Renal dosing adjustments and hydration advice")
        supportive.append("; ".join(sorted(set(supp_base))))

        dose_adj_note.append("; ".join(notes))
        interactions.append("; ".join(inters) if inters else "No significant interactions identified.")
        contra.append("; ".join(contr) if contr else "None.")
        rationale.append("; ".join(why))

    df = pd.DataFrame({
        "CaseID": [f"P-{i:04d}" for i in range(1, N + 1)],
        "Subtype": subtypes,
        "Stage": stages,
        "TNM_T": T,
        "TNM_N": Nn,
        "TNM_M": M,
        "Grade": grade,
        "ER_Status": ER,
        "PR_Status": PR,
        "HER2_Status": HER2,
        "WBC_x10^9_L": WBC,
        "ANC_x10^9_L": ANC,
        "Hemoglobin_g_dL": Hgb,
        "Platelets_x10^9_L": Platelets,
        "Creatinine_mg_dL": Creatinine,
        "eGFR_mL_min_1.73m2": eGFR,
        "BUN_mg_dL": BUN,
        "AST_U_L": AST,
        "ALT_U_L": ALT,
        "ALP_U_L": ALP,
        "TotalBilirubin_mg_dL": Bilirubin,
        "Albumin_g_dL": Albumin,
        "LVEF_percent": LVEF,
        "Troponin_ng_L": Troponin,
        "Height_cm": Height_cm,
        "Weight_kg": Weight_kg,
        "BSA_m2": BSA,
        "Comorbidity_DM": dm,
        "Comorbidity_HTN": htn,
        "Comorbidity_CAD": cad,
        "Comorbidity_CKD": ckd,
        "Comorbidity_Asthma": asthma,
        "Comorbidity_Depression": depression,
        "CharlsonLikeScore": charlson,
        "PriorLines": prior_lines,
        "PriorTherapies": prior_therapies,
        "BestPriorResponse": prior_response,
        "Regimen": regimens,
        "CycleNumber": cycle_number,
        "DosePlan": dose_plans,
        "Schedule": schedules,
        "PreMeds": premeds,
        "PostMeds": postmeds,
        "IntendedDoseIntensity_pct": intended,
        "ActualDoseIntensity_pct": actual,
        "DoseAdjustmentNote": dose_adj_note,
        "InteractionsCheck": interactions,
        "Contraindications": contra,
        "SupportiveCareInstructions": supportive,
        "Rationale": rationale,
    })

    dictionary = pd.DataFrame([
        ["CaseID", "Unique synthetic identifier", "string", "P-0001 ..."],
        ["Subtype", "Breast cancer subtype", "categorical", "HR+ (ER and/or PR positive, HER2-); HER2+; Triple-negative"],
        ["Stage", "Overall stage at presentation", "categorical", "I, II, III, IV"],
        ["TNM_T", "Tumor (T) category", "categorical", "T1–T4 (synthetic)"],
        ["TNM_N", "Node (N) category", "categorical", "N0–N3 (synthetic)"],
        ["TNM_M", "Metastasis (M) category", "categorical", "M0, M1"],
        ["Grade", "Histologic grade", "categorical", "G1, G2, G3"],
        ["ER_Status", "Estrogen receptor status", "categorical", "Negative; Positive; Strong Positive"],
        ["PR_Status", "Progesterone receptor status", "categorical", "Negative; Positive"],
        ["HER2_Status", "HER2 status", "categorical", "Negative; Positive (IHC 3+ or FISH+)"],
        ["WBC_x10^9_L", "White blood cells", "numeric", "2.0–20.0"],
        ["ANC_x10^9_L", "Absolute neutrophil count", "numeric", "0.2–15.0"],
        ["Hemoglobin_g_dL", "Hemoglobin", "numeric", "7.0–17.0"],
        ["Platelets_x10^9_L", "Platelet count", "numeric", "50–800"],
        ["Creatinine_mg_dL", "Serum creatinine", "numeric", "0.3–2.5"],
        ["eGFR_mL_min_1.73m2", "Estimated GFR", "numeric", "20–120"],
        ["BUN_mg_dL", "Blood urea nitrogen", "numeric", "3–40"],
        ["AST_U_L", "AST", "numeric", "5–150"],
        ["ALT_U_L", "ALT", "numeric", "5–150"],
        ["ALP_U_L", "Alkaline phosphatase", "numeric", "30–400"],
        ["TotalBilirubin_mg_dL", "Total bilirubin", "numeric", "0.1–3.0"],
        ["Albumin_g_dL", "Albumin", "numeric", "2.0–5.5"],
        ["LVEF_percent", "Left ventricular ejection fraction", "numeric", "35–75"],
        ["Troponin_ng_L", "High-sensitivity troponin", "numeric", "0–~60 (synthetic)"],
        ["Height_cm", "Height", "numeric", "140–190"],
        ["Weight_kg", "Weight", "numeric", "40–120"],
        ["BSA_m2", "Body surface area (Mosteller)", "numeric", "≈1.3–2.4"],
        ["Comorbidity_DM", "Diabetes mellitus", "binary", "0/1"],
        ["Comorbidity_HTN", "Hypertension", "binary", "0/1"],
        ["Comorbidity_CAD", "Coronary artery disease", "binary", "0/1"],
        ["Comorbidity_CKD", "Chronic kidney disease (eGFR<60)", "binary", "0/1"],
        ["Comorbidity_Asthma", "Asthma", "binary", "0/1"],
        ["Comorbidity_Depression", "Depression", "binary", "0/1"],
        ["CharlsonLikeScore", "Approximate multimorbidity score", "integer", "0–6+"],
        ["PriorLines", "Number of prior systemic therapy lines", "integer", "0–3"],
        ["PriorTherapies", "Summary of prior therapies", "string", "None; or semicolon-separated list"],
        ["BestPriorResponse", "Best response to any prior therapy", "categorical", "None, CR, PR, SD, PD"],
        ["Regimen", "Current chemotherapy/targeted regimen", "string", "Commonly used regimens (synthetic)"],
        ["CycleNumber", "Current cycle number", "integer", "1–8"],
        ["DosePlan", "Dose specifications (per regimen)", "string", "e.g., Doxorubicin 60 mg/m² + ..."],
        ["Schedule", "Administration schedule", "string", "e.g., q3w ×6 cycles"],
        ["PreMeds", "Pre-medication requirements", "string", "semicolon-separated list"],
        ["PostMeds", "Post-medication requirements", "string", "semicolon-separated list"],
        ["IntendedDoseIntensity_pct", "Planned dose intensity (%)", "numeric", "90–100"],
        ["ActualDoseIntensity_pct", "Delivered dose intensity (%)", "numeric", "50–100"],
        ["DoseAdjustmentNote", "Pharmacist dose adjustment note", "string", "short free text"],
        ["InteractionsCheck", "Interaction checks", "string", "short free text"],
        ["Contraindications", "Contraindications flagged", "string", "short free text"],
        ["SupportiveCareInstructions", "Supportive care plan", "string", "short free text"],
        ["Rationale", "Justification for decisions", "string", "short free text"],
    ], columns=["Column", "Description", "Type", "Values/Range"])

    return df, dictionary


def save_outputs(df, dictionary, outdir, n):
    dataset_csv = f"{outdir}/synthetic_breast_cancer_{n}.csv"
    dataset_xlsx = f"{outdir}/synthetic_breast_cancer_{n}.xlsx"
    dict_csv = f"{outdir}/synthetic_breast_cancer_dictionary.csv"
    dict_xlsx = f"{outdir}/synthetic_breast_cancer_dictionary.xlsx"

    df.to_csv(dataset_csv, index=False)
    try:
        df.to_excel(dataset_xlsx, index=False)
    except Exception as e:
        print(f"[WARN] Could not write Excel dataset: {e}")

    dictionary.to_csv(dict_csv, index=False)
    try:
        dictionary.to_excel(dict_xlsx, index=False)
    except Exception as e:
        print(f"[WARN] Could not write Excel dictionary: {e}")

    print("Wrote:")
    print(" -", dataset_csv)
    print(" -", dataset_xlsx)
    print(" -", dict_csv)
    print(" -", dict_xlsx)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic breast cancer dataset and dictionary.")
    parser.add_argument("--n", type=int, default=1000, help="Number of cases to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory (default: current directory)")
    args = parser.parse_args()

    df, dictionary = build_dataset(args.n, seed=args.seed)
    save_outputs(df, dictionary, args.outdir, args.n)


if __name__ == "__main__":
    main()
