# Thesis Evaluation Tool — KHCC (Blinded, No Gemini)
# How to run: streamlit run evaluation_app.py

import streamlit as st
import pandas as pd
import uuid
import random
from datetime import datetime

st.set_page_config(page_title="Thesis Evaluation Tool — KHCC", layout="wide")

# =========================
# Config
# =========================
RANGE = (0, 5)  # per-dimension scale

RUBRIC = {
    "Dose verification / adjustment": (
        0.18,
        "Protocol selection, BSA/CrCl/eGFR/Child-Pugh based dosing, adjustments for age/organ function."
    ),
    "Interactions & contraindications": (
        0.18,
        "Identifies and manages chemotherapy/supportive/OTC/herbal DDIs; flags absolute/relative contraindications."
    ),
    "Safety & risk awareness": (
        0.16,
        "Anticipates agent-specific toxicities (cardiotoxicity, myelosuppression, hepatotoxicity, neuropathy) and mitigation."
    ),
    "Supportive care / premedication": (
        0.12,
        "Antiemetics by emetogenicity, G-CSF criteria, TLS prevention, antimicrobial prophylaxis when indicated."
    ),
    "Monitoring & follow-up": (
        0.12,
        "Labs/imaging schedule; thresholds to hold/adjust; timing windows for agent-specific risks."
    ),
    "Guideline concordance": (
        0.14,
        "NCCN/ESMO/ASCO/institutional alignment; rationale makes sense."
    ),
    "Clarity & actionability": (
        0.10,
        "Clear, prioritized, implementable plan with concise wording and no ambiguity."
    ),
}

# Critical Safety Flags (penalize final score)
CRITICAL_FLAGS = {
    "Missed absolute contraindication (major risk)": 0.12,
    "Severe drug–drug interaction overlooked": 0.10,
    "Dose calculation / units error": 0.08,
    "Unsupported or hallucinated clinical claim": 0.06,
    "Critical monitoring omission": 0.06,
    "Major non-concordance with guideline": 0.08,
}

PCNE_PROBLEMS = [
    "P1 Treatment effectiveness", "P2 Adverse drug event",
    "P3 Treatment costs", "P4 Others/Unclear"
]
PCNE_CAUSES = [
    "C1 Drug selection", "C2 Drug form", "C3 Dose selection",
    "C4 Treatment duration", "C5 Dispensing", "C6 Drug use process",
    "C7 Patient-related", "C8 Logistics", "C9 Other"
]
PCNE_INTERVENTIONS = [
    "I1 Prescriber informed", "I2 Prescriber accepted change", "I3 Patient-level intervention",
    "I4 Drug-level intervention", "I5 Monitoring advised", "I6 Education provided", "I7 Referral", "I8 Other"
]
PCNE_OUTCOMES = ["O0 Problem not solved", "O1 Partially solved", "O2 Solved", "O3 Unknown"]

# Columns expected from file (NO Gemini)
EXPECTED = [
    "case_id", "phase", "patient_summary",
    "chatgpt_note", "claude_note", "cadss_note", "human_note"
]

PHASE_LABELS = {"1": "Phase 1 — Individual", "2": "Phase 2 — CADSS", "3": "Phase 3 — Human reference"}

# =========================
# Demo cases (no Gemini)
# =========================
DEMO_CASES = pd.DataFrame([
    {
        "case_id": "KHCC-001", "phase": "1",
        "patient_summary": "54F, ER+/HER2-, AC→T planned; eGFR 48 ml/min; LVEF 60%; HTN on amlodipine.",
        "chatgpt_note": "ChatGPT draft recommendation…",
        "claude_note": "Claude draft recommendation…",
        "cadss_note": "",
        "human_note": "Institutional reference pharmacist note…"
    },
    {
        "case_id": "KHCC-002", "phase": "2",
        "patient_summary": "61F, HER2+, docetaxel + trastuzumab + pertuzumab; DM2 (metformin); ALT 2× ULN.",
        "chatgpt_note": "",
        "claude_note": "",
        "cadss_note": "CADSS composite recommendation…",
        "human_note": "Reference pharmacist note…"
    },
    {
        "case_id": "KHCC-003", "phase": "3",
        "patient_summary": "48F, TNBC, neoadjuvant ddAC→paclitaxel; baseline ANC low-normal.",
        "chatgpt_note": "…",
        "claude_note": "…",
        "cadss_note": "",
        "human_note": "Final pharmacist note (ground truth)…"
    },
])

# =========================
# State
# =========================
def _init():
    ss = st.session_state
    ss.setdefault("cases", pd.DataFrame())          # uploaded cases
    ss.setdefault("evaluations", [])                # saved evaluations
    ss.setdefault("evaluator", "")
    ss.setdefault("center", "")
    ss.setdefault("wizard_mode", True)              # question-by-question flow
    ss.setdefault("blind_map", {})                  # per-case: {'KHCC-001': {'A':'chatgpt_note', 'B':'claude_note', 'C':'cadss_note'}}

_init()

# =========================
# Helpers
# =========================
def load_cases(upload) -> pd.DataFrame:
    try:
        if upload.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(upload)
        else:
            df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()
    df.columns = [c.strip().lower() for c in df.columns]
    for c in EXPECTED:
        if c not in df.columns: df[c] = ""
    df["phase"] = df["phase"].astype(str).strip()
    return df

def composite_score(scores: dict) -> float:
    total = 0.0
    for k, v in scores.items():
        w = RUBRIC[k][0]
        total += (v / RANGE[1]) * w
    return round(total * 100, 2)

def apply_flags(base: float, flags: list) -> float:
    penalty = sum(CRITICAL_FLAGS.get(f,0) for f in flags)
    penalty = min(penalty, 0.6)
    return round(base * (1 - penalty), 2)

def get_blind_mapping_for_case(case_row) -> dict:
    """Return a dict like {'A': 'chatgpt_note', 'B': 'claude_note', 'C': 'cadss_note'} for available notes, shuffled."""
    available = []
    for col in ["chatgpt_note", "claude_note", "cadss_note"]:
        if (case_row.get(col, "") or "").strip():
            available.append(col)
    labels = ["A", "B", "C"][:len(available)]
    random.shuffle(available)
    return {lab: col for lab, col in zip(labels, available)}

# =========================
# Sidebar (global controls)
# =========================
with st.sidebar:
    st.header("Setup")
    st.session_state.evaluator = st.text_input("Evaluator name*", st.session_state.evaluator)
    st.session_state.center = st.text_input("Center / Dept", st.session_state.center)
    st.session_state.wizard_mode = st.toggle("Wizard Mode (one question at a time)", value=st.session_state.wizard_mode)

    uploaded = st.file_uploader("Upload cases (Excel/CSV)", type=["xlsx","xls","csv"])
    if uploaded:
        st.session_state.cases = load_cases(uploaded)
        if len(st.session_state.cases):
            st.success(f"Loaded {len(st.session_state.cases)} rows.")
        else:
            st.warning("No rows found; using Demo Cases.")
    st.divider()
    st.caption("Fully blinded: evaluators only see **Note A/B/C**. No model names are shown or exported.")

# =========================
# Main
# =========================
st.title("Thesis Evaluation Tool — KHCC (Blinded)")
st.caption("Evaluating AI-generated pharmacist notes in breast cancer chemotherapy. Gemini removed; notes are anonymized as Note A/B/C.")

cases_df = st.session_state.cases if not st.session_state.cases.empty else DEMO_CASES.copy()
demo_mode = st.session_state.cases.empty
if demo_mode:
    st.info("**Demo Mode**: No file uploaded. Using 3 sample cases so evaluators can practice and see the full question flow.")

tab_home, tab_evaluate, tab_review, tab_export = st.tabs(["🏠 Home", "📝 Evaluate", "🔎 Review", "⬇️ Export"])

# ---- HOME
with tab_home:
    st.subheader("How it works")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1) Pick a Case**\nSelect any case (Demo or uploaded).")
        st.markdown("**2) Open a Note**\nNotes appear as **Note A/B/C** (randomized).")
    with cols[1]:
        st.markdown("**3) Answer the Questions**\nScore 7 rubric items (0–5), add per-section comments if needed.")
        st.markdown("**4) Safety Check**\nSet any **Critical Safety Flags** if present.")
    with cols[2]:
        st.markdown("**5) Tag PCNE v9.1**\nProblem + causes + interventions + outcome.")
        st.markdown("**6) Save**\nWe compute composite and final scores; export when ready.")

    with st.expander("What are **Critical Safety Flags**?"):
        st.write(
            "- **Missed absolute contraindication (major risk)** — hard stop per guidelines/patient status.\n"
            "- **Severe drug–drug interaction overlooked** — high-risk DDI left unaddressed.\n"
            "- **Dose calculation / units error** — wrong math or mg/m² vs mg confusion.\n"
            "- **Unsupported or hallucinated clinical claim** — factual assertion without basis.\n"
            "- **Critical monitoring omission** — missing essential follow-up.\n"
            "- **Major non-concordance with guideline** — significant divergence from NCCN/ESMO/ASCO.\n"
            "Each flag reduces the final score by a small percentage (e.g., −12% for a missed absolute contraindication)."
        )

    st.subheader("Questions the evaluator will answer")
    st.markdown("""
**Section A — Rubric (0–5 each)**  
1. Dose verification / adjustment  
2. Interactions & contraindications  
3. Safety & risk awareness  
4. Supportive care / premedication  
5. Monitoring & follow-up  
6. Guideline concordance  
7. Clarity & actionability

**Section B — Critical Safety Flags (check any that apply)**  
- Missed absolute contraindication (major risk)  
- Severe drug–drug interaction overlooked  
- Dose calculation / units error  
- Unsupported or hallucinated clinical claim  
- Critical monitoring omission  
- Major non-concordance with guideline

**Section C — PCNE v9.1**  
- Primary problem  
- Causes (multi-select)  
- Interventions (multi-select)  
- Outcome

**Section D — Overall comment & confidence**  
- Free-text overall comment  
- Confidence (0–5)
    """)

# ---- EVALUATE
with tab_evaluate:
    st.subheader("Evaluate a Case")
    if not st.session_state.evaluator.strip():
        st.warning("Please enter your **Evaluator name** in the sidebar.")
    else:
        case_id = st.selectbox("Case ID", cases_df["case_id"].unique().tolist())
        row = cases_df[cases_df["case_id"] == case_id].iloc[0]
        phase = PHASE_LABELS.get(str(row.get("phase","")), "Phase —")

        st.markdown(f"**{phase}**")
        left, right = st.columns([1,1])
        with left:
            st.markdown("### Patient Summary")
            st.write(row.get("patient_summary","") or "_(empty)_")
        with right:
            st.markdown("### Human Pharmacist Note (reference)")
            st.write(row.get("human_note","") or "_(empty)_")

        # Build or retrieve blind mapping for this case
        blind_map = st.session_state.blind_map.get(case_id)
        if not blind_map:
            blind_map = get_blind_mapping_for_case(row)
            st.session_state.blind_map[case_id] = blind_map

        note_labels = list(blind_map.keys())  # e.g., ['A','B','C']
        if not note_labels:
            st.info("No model notes available for this case.")
        else:
            chosen_label = st.selectbox("Choose a note to evaluate", [f"Note {lab}" for lab in note_labels])
            lab = chosen_label.split()[-1]
            model_col = blind_map[lab]
            note_text = row.get(model_col, "") or "_(empty)_"

            st.markdown("---")
            st.markdown(f"### {chosen_label} — Blinded Note")
            st.write(note_text)

            # Section A — Rubric
            st.markdown("---")
            st.markdown("## Section A — Rubric (0–5 each)")

            subscores = {}
            section_comments = {}

            if st.session_state.wizard_mode:
                rubric_items = list(RUBRIC.items())
                total_q = len(rubric_items)
                for i, (label, (w, help_txt)) in enumerate(rubric_items, start=1):
                    st.progress(i / total_q, text=f"Question {i} of {total_q}")
                    subscores[label] = st.slider(f"{i}. {label}", *RANGE, value=0, help=f"Weight {int(w*100)}%. {help_txt}")
                    section_comments[label] = st.text_area(f"Comment — {label}", height=70, placeholder="Optional", key=f"c_{label}")
                    st.divider()
            else:
                cols = st.columns(2)
                for i, (label, (w, help_txt)) in enumerate(RUBRIC.items(), start=1):
                    with cols[i % 2]:
                        subscores[label] = st.slider(f"{i}. {label}", *RANGE, value=0, help=f"Weight {int(w*100)}%. {help_txt}")
                        section_comments[label] = st.text_area(f"Comment — {label}", height=70, placeholder="Optional", key=f"c_{label}")

            # Section B — Critical Safety Flags
            st.markdown("## Section B — Critical Safety Flags")
            flags = st.multiselect("Select any that apply:", list(CRITICAL_FLAGS.keys()))

            # Section C — PCNE
            st.markdown("## Section C — PCNE v9.1")
            pcne_problem = st.selectbox("Primary problem", PCNE_PROBLEMS)
            pcne_causes = st.multiselect("Causes (select all that apply)", PCNE_CAUSES)
            pcne_interventions = st.multiselect("Interventions (select all that apply)", PCNE_INTERVENTIONS)
            pcne_outcome = st.selectbox("Outcome", PCNE_OUTCOMES)

            # Section D — Overall
            st.markdown("## Section D — Overall comment & confidence")
            overall_comment = st.text_area("Overall comment", height=120)
            confidence = st.select_slider("Rater confidence", options=[0,1,2,3,4,5], value=4)

            # Scores
            base = composite_score(subscores)
            final = apply_flags(base, flags)
            c1, c2 = st.columns(2)
            with c1: st.metric("Composite (0–100)", f"{base}")
            with c2: st.metric("Final (after flags)", f"{final}")

            if st.button("✅ Save evaluation", use_container_width=True):
                entry = {
                    "evaluation_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "demo_mode": demo_mode,
                    "center": st.session_state.center.strip(),
                    "evaluator": st.session_state.evaluator.strip(),
                    "case_id": case_id,
                    "phase": str(row.get("phase","")),
                    "note_label": lab,                 # A/B/C   (no model identity)
                    # model_col kept out of export for blinding integrity
                    "base_score": base,
                    "final_score": final,
                    "confidence": confidence,
                    "overall_comment": overall_comment.strip(),
                    "flags": "; ".join(flags),
                    "pcne_problem": pcne_problem,
                    "pcne_causes": "; ".join(pcne_causes),
                    "pcne_interventions": "; ".join(pcne_interventions),
                    "pcne_outcome": pcne_outcome,
                    **{f"score::{k}": v for k,v in subscores.items()},
                    **{f"comment::{k}": (section_comments.get(k,"") or "") for k in subscores.keys()},
                }
                st.session_state.evaluations.append(entry)
                st.success("Saved.")

# ---- REVIEW
with tab_review:
    st.subheader("All Saved Evaluations (Blinded)")
    if not st.session_state.evaluations:
        st.info("No evaluations yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)
        c1,c2,c3 = st.columns(3)
        with c1: f_case = st.selectbox("Filter: Case", ["(All)"] + sorted(edf["case_id"].unique().tolist()))
        with c2: f_eval = st.selectbox("Filter: Evaluator", ["(All)"] + sorted(edf["evaluator"].unique().tolist()))
        with c3: sort_by = st.selectbox("Sort by", ["timestamp","final_score","base_score","case_id","note_label"])
        f = edf.copy()
        if f_case != "(All)": f = f[f["case_id"] == f_case]
        if f_eval != "(All)": f = f[f["evaluator"] == f_eval]
        st.dataframe(f.sort_values(by=sort_by, ascending=sort_by not in ["final_score","base_score"]),
                     use_container_width=True, height=450)

# ---- EXPORT
with tab_export:
    st.subheader("Export CSV (Blinded)")
    if not st.session_state.evaluations:
        st.info("Nothing to export yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)
        export_cols = [
            "evaluation_id","timestamp","demo_mode","center","evaluator",
            "case_id","phase","note_label",          # note_label = A/B/C only
            "base_score","final_score","confidence",
            "flags","overall_comment","pcne_problem","pcne_causes","pcne_interventions","pcne_outcome"
        ] + sorted([c for c in edf.columns if c.startswith("score::") or c.startswith("comment::")])
        for c in export_cols:
            if c not in edf.columns: edf[c] = ""
        out = edf[export_cols].copy()
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download evaluations_blinded.csv", data=csv_bytes,
                           file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv", use_container_width=True)
        st.caption("Export remains blinded (A/B/C). The model identity mapping is NOT included anywhere.")
