# --- Thesis Evaluation Tool (Methodology v2025-10) ---
# Phases:
#   Phase 1: Individual models (ChatGPT, Claude, Gemini) per case
#   Phase 2: CADSS (synthesizer + GPT-4o-mini | DeepSeek-Reasoner | Claude-3.5-Sonnet)
#   Phase 3: Human reference (for comparison only; not scored by evaluators)
#
# Rubrics: Modified Stanford-style rubric + PCNE v9.1 tagging, with safety/hallucination overlays
# Blinding: Optional "Blind Mode" to mask model identity during scoring
#
# How to run:
#   pip install -r requirements.txt
#   streamlit run app.py

import streamlit as st
import pandas as pd
import uuid
from datetime import datetime

st.set_page_config(page_title="Thesis Evaluation Tool (KHCC)", layout="wide")

# ========== CONFIG ==========
# Expected columns in the cases file (case-insensitive)
EXPECTED_COLUMNS = [
    "case_id",
    "phase",                   # "1"=individual, "2"=cadss, "3"=human
    "patient_summary",
    "chatgpt_note",
    "claude_note",
    "gemini_note",
    "cadss_note",
    "human_note",
    # Optional metadata (free-form): regimen, stage, her2, er_pr, egfr, lvef, comorbidities, labs_json, etc.
]

# Modified Stanford-style subscales (weights sum to 1.0)
RUBRIC = {
    # label: (weight, help text)
    "Dose verification / adjustment": (
        0.18,
        "Correct protocol selection; BSA/eGFR/Child-Pugh based dosing; adjustments for organ function & age."
    ),
    "Interactions & contraindications": (
        0.18,
        "Identifies DDIs (chemo, supportive, OTC/herbals), absolute/relative contraindications; manages overlap."
    ),
    "Safety & risk awareness": (
        0.16,
        "Anticipates key toxicities (e.g., cardiotoxicity, myelosuppression, hepatotoxicity, neuropathy); mitigation steps."
    ),
    "Supportive care / premedication": (
        0.12,
        "Antiemetics by emetogenicity, G-CSF criteria, TLS prevention, antimicrobial prophylaxis if indicated."
    ),
    "Monitoring & follow-up": (
        0.12,
        "Labs/imaging schedule; thresholds to hold/adjust; monitoring windows for agent-specific risks."
    ),
    "Guideline concordance": (
        0.14,
        "Consistency with institutional/NCCN/ESMO/ASCO; cites logic or references when relevant."
    ),
    "Clarity & actionability": (
        0.10,
        "Clear, prioritized, implementable plan; concise rationale; structured, avoids ambiguity."
    ),
}
RANGE = (0, 5)  # 0–5 per subscale

# Safety / quality overlays (binary or counts -> penalty)
# Each selected flag applies a percentage penalty to the composite score (post-weighting)
OVERLAY_PENALTIES = {
    "Absolute contraindication missed": 0.12,
    "Severe DDI overlooked": 0.10,
    "Dose calculation/units error": 0.08,
    "Unsupported/hallucinated claim": 0.06,
    "Critical monitoring omission": 0.06,
    "Non-concordant with guideline (major)": 0.08,
}

# PCNE v9.1 options (primary problem + causes + interventions + outcome)
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
    "I4 Drug-level intervention", "I5 Monitoring advised", "I6 Education provided",
    "I7 Referral", "I8 Other"
]
PCNE_OUTCOMES = [
    "O0 Problem not solved", "O1 Problem partially solved", "O2 Problem solved", "O3 Outcome unknown"
]

MODEL_MAP = {
    "ChatGPT": "chatgpt_note",
    "Claude": "claude_note",
    "Gemini": "gemini_note",
    "CADSS (AI agent)": "cadss_note",
    "Human (reference)": "human_note",
}

PHASE_LABELS = {
    "1": "Phase 1 — Individual model",
    "2": "Phase 2 — CADSS (collaborative AI)",
    "3": "Phase 3 — Human reference",
}

# ========== STATE ==========
def _init_state():
    ss = st.session_state
    ss.setdefault("cases", pd.DataFrame())
    ss.setdefault("evaluations", [])     # list of dict rows
    ss.setdefault("rubric", RUBRIC.copy())
    ss.setdefault("evaluator_name", "")
    ss.setdefault("center_name", "")
    ss.setdefault("blind_mode", True)    # mask model identity while scoring
    ss.setdefault("visible_models", ["ChatGPT", "Claude", "Gemini", "CADSS (AI agent)"])  # can be filtered in UI

_init_state()

# ========== HELPERS ==========
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
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for c in df.columns:
        df[c] = df[c].fillna("")
    # Normalize phase to string "1/2/3"
    df["phase"] = df["phase"].astype(str).str.strip()
    return df

def composite_score(subscores: dict, rubric: dict) -> float:
    # subscores: {label: 0-5}
    total = 0.0
    for k, v in subscores.items():
        w = rubric.get(k, (0.0, ""))[0] if isinstance(rubric.get(k), tuple) else rubric[k]
        denom = RANGE[1] if RANGE[1] else 5
        total += (v / denom) * w
    return round(total * 100, 2)

def apply_overlays(base_score: float, selected_overlays: list) -> float:
    penalty = 0.0
    for item in selected_overlays:
        penalty += OVERLAY_PENALTIES.get(item, 0.0)
    penalty = min(penalty, 0.6)  # cap at 60% reduction
    return round(base_score * (1.0 - penalty), 2)

def masked(name: str, do_mask: bool, alt: str = "Model (blinded)") -> str:
    return alt if do_mask else name

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("Setup")
    st.session_state.evaluator_name = st.text_input("Evaluator name*", st.session_state.evaluator_name)
    st.session_state.center_name = st.text_input("Center / Dept", st.session_state.center_name)

    st.markdown("**Scoring options**")
    st.session_state.blind_mode = st.toggle("Blind Mode (hide model identity during scoring)",
                                            value=st.session_state.blind_mode)
    visible_models = st.multiselect(
        "Models to include",
        options=list(MODEL_MAP.keys()),
        default=st.session_state.visible_models,
        help="Controls which model columns are shown for selection; Human is for reference only."
    )
    if visible_models:
        st.session_state.visible_models = visible_models

    uploaded = st.file_uploader("Upload cases (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if uploaded:
        st.session_state.cases = load_cases(uploaded)
        if len(st.session_state.cases):
            st.success(f"Loaded {len(st.session_state.cases)} rows.")
        else:
            st.warning("No rows found; please check your file.")
    st.markdown("---")

    with st.expander("Rubric weights (optional)"):
        # Lightweight editor: keep labels fixed; edit weights
        new_weights = {}
        total_preview = 0.0
        for label, (w, help_txt) in st.session_state.rubric.items():
            nw = st.number_input(f"{label} weight", min_value=0.0, max_value=1.0, value=float(w), step=0.01)
            new_weights[label] = (nw, help_txt)
            total_preview += nw
        st.caption(f"Sum of weights: **{total_preview:.2f}** (should be 1.00)")
        if st.button("Apply weights"):
            if abs(total_preview - 1.0) > 1e-6:
                st.error("Weights must sum to 1.00")
            else:
                st.session_state.rubric = new_weights
                st.success("Updated rubric weights.")

# ========== MAIN ==========
st.title("Thesis Evaluation Tool — KHCC (Breast Cancer Chemotherapy)")
st.caption("Phases: Phase 1 (Individual), Phase 2 (CADSS), Phase 3 (Human reference). Modified Stanford rubric + PCNE v9.1. Safety overlays & blinded scoring supported.")

tab_cases, tab_eval, tab_compare, tab_review, tab_export, tab_help = st.tabs(
    ["📄 Cases", "📝 Evaluate", "📊 Compare (per case)", "🔎 Review / Filter", "⬇️ Export", "❓ Help"]
)

# --- CASES
with tab_cases:
    st.subheader("Cases")
    df = st.session_state.cases
    if df.empty:
        st.info("Upload a cases file in the sidebar to begin.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            q = st.text_input("Search (case_id, patient_summary)", "")
        with c2:
            show_cols = st.multiselect(
                "Show columns",
                options=[c for c in df.columns if c in [*EXPECTED_COLUMNS, *df.columns.tolist()]],
                default=["case_id", "phase", "patient_summary", "human_note"]
            )
        view = df.copy()
        if q.strip():
            qq = q.lower()
            view = view[
                view["case_id"].str.lower().str.contains(qq) |
                view["patient_summary"].str.lower().str.contains(qq)
            ]
        st.dataframe(view[show_cols], use_container_width=True, height=420)

# --- EVALUATE
with tab_eval:
    st.subheader("Evaluate a case")
    df = st.session_state.cases
    if df.empty:
        st.info("Upload cases first.")
    elif not st.session_state.evaluator_name.strip():
        st.warning("Please enter your Evaluator name in the sidebar.")
    else:
        # Select case & phase-aware model
        case_id = st.selectbox("Select Case ID", df["case_id"].unique().tolist())
        row = df[df["case_id"] == case_id].iloc[0]
        phase = row.get("phase", "")
        phase_label = PHASE_LABELS.get(str(phase), f"Phase {phase or '—'}")
        st.badge(phase_label)

        # Model choices (hide Human for scoring)
        model_choices = [m for m in st.session_state.visible_models if m != "Human (reference)"]
        model_choice = st.selectbox("Which note to evaluate?", model_choices)
        model_col = MODEL_MAP[model_choice]

        # Display notes (with blinding)
        left, right = st.columns(2)
        with left:
            st.markdown("**Patient Summary**")
            st.write(row.get("patient_summary", "") or "_(empty)_")
            st.markdown(f"**{masked(model_choice, st.session_state.blind_mode)} Note**")
            st.write(row.get(model_col, "") or "_(empty)_")
        with right:
            st.markdown("**Human Pharmacist Note (reference)**")
            st.write(row.get("human_note", "") or "_(empty)_")

        st.markdown("---")
        st.markdown("### Rubric scoring (0–5 each)")
        subscores = {}
        per_section_comments = {}
        cols = st.columns(2)
        for i, (label, (w, help_txt)) in enumerate(st.session_state.rubric.items()):
            with cols[i % 2]:
                subscores[label] = st.slider(label, *RANGE, value=0, help=f"Weight {int(w*100)}%. {help_txt}")
                per_section_comments[label] = st.text_area(
                    f"Comment — {label}",
                    height=70,
                    placeholder="Brief justification for this section (optional)."
                )

        st.markdown("### Safety / quality overlays (apply penalties if present)")
        overlay_sel = st.multiselect(
            "Select any issues observed",
            options=list(OVERLAY_PENALTIES.keys()),
            help="Each selected item reduces the final score by the listed penalty."
        )

        st.markdown("### PCNE v9.1 tagging")
        pcne_problem = st.selectbox("Primary problem", PCNE_PROBLEMS)
        pcne_causes = st.multiselect("Causes (select all that apply)", PCNE_CAUSES)
        pcne_interv = st.multiselect("Interventions (planned/performed)", PCNE_INTERVENTIONS)
        pcne_outcome = st.selectbox("Outcome", PCNE_OUTCOMES)

        st.markdown("### Overall comment & confidence")
        overall_comment = st.text_area("Overall comment / rationale", height=120)
        confidence = st.select_slider("Rater confidence in this evaluation", options=[0,1,2,3,4,5], value=4)

        base_score = composite_score(subscores, st.session_state.rubric)
        final_score = apply_overlays(base_score, overlay_sel)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Composite (0–100)", f"{base_score}")
        with c2:
            st.metric("Final (after overlays)", f"{final_score}")

        if st.button("✅ Save evaluation", use_container_width=True):
            entry = {
                "evaluation_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "center": st.session_state.center_name.strip(),
                "evaluator": st.session_state.evaluator_name.strip(),
                "blind_mode": st.session_state.blind_mode,
                "case_id": case_id,
                "phase": str(phase),
                "model": model_choice,
                "base_score": base_score,
                "final_score": final_score,
                "confidence": confidence,
                "overlays": "; ".join(overlay_sel),
                "overall_comment": overall_comment.strip(),
                # per-subscale scores & comments
                **{f"score::{k}": v for k, v in subscores.items()},
                **{f"comment::{k}": (per_section_comments.get(k, "") or "") for k in subscores.keys()},
                # pcne
                "pcne_problem": pcne_problem,
                "pcne_causes": "; ".join(pcne_causes) if pcne_causes else "",
                "pcne_interventions": "; ".join(pcne_interv) if pcne_interv else "",
                "pcne_outcome": pcne_outcome,
            }
            st.session_state.evaluations.append(entry)
            st.success("Saved.")

# --- COMPARE (per case)
with tab_compare:
    st.subheader("Compare scores for a single case")
    if not st.session_state.evaluations:
        st.info("No evaluations yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)
        case_opt = st.selectbox("Case", sorted(edf["case_id"].unique().tolist()))
        f = edf[edf["case_id"] == case_opt]
        # Simple table; (you can graph in analysis notebook)
        view_cols = ["timestamp", "evaluator", "phase", "model", "base_score", "final_score", "overlays", "confidence"]
        st.dataframe(f[view_cols].sort_values("final_score", ascending=False), use_container_width=True, height=420)

# --- REVIEW
with tab_review:
    st.subheader("All evaluations")
    if not st.session_state.evaluations:
        st.info("No data yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_case = st.selectbox("Filter: Case", ["(All)"] + sorted(edf["case_id"].unique().tolist()))
        with c2:
            f_phase = st.selectbox("Filter: Phase", ["(All)", "1", "2", "3"])
        with c3:
            f_model = st.selectbox("Filter: Model", ["(All)"] + sorted(edf["model"].unique().tolist()))
        with c4:
            f_eval = st.selectbox("Filter: Evaluator", ["(All)"] + sorted(edf["evaluator"].unique().tolist()))
        f = edf.copy()
        if f_case != "(All)":
            f = f[f["case_id"] == f_case]
        if f_phase != "(All)":
            f = f[f["phase"] == f_phase]
        if f_model != "(All)":
            f = f[f["model"] == f_model]
        if f_eval != "(All)":
            f = f[f["evaluator"] == f_eval]
        st.dataframe(f.sort_values(["case_id","model","timestamp"]), use_container_width=True, height=450)

# --- EXPORT
with tab_export:
    st.subheader("Export")
    if not st.session_state.evaluations:
        st.info("Nothing to export yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)
        # Tidy column order suggestion
        meta_cols = [
            "evaluation_id","timestamp","center","evaluator","blind_mode",
            "case_id","phase","model","confidence","base_score","final_score",
            "overlays","overall_comment","pcne_problem","pcne_causes","pcne_interventions","pcne_outcome"
        ]
        # Gather per-section (scores/comments)
        section_cols = sorted([c for c in edf.columns if c.startswith("score::") or c.startswith("comment::")])
        export_cols = meta_cols + section_cols
        # Ensure all exist
        for c in export_cols:
            if c not in edf.columns:
                edf[c] = ""
        export_df = edf[export_cols].copy()
        st.dataframe(export_df.tail(10), use_container_width=True, height=300)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download evaluations CSV",
            data=csv_bytes,
            file_name=f"khcc_evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Columns are ready for analysis (inter-rater, per-phase / per-model comparisons, overlay impact).")

# --- HELP
with tab_help:
    st.subheader("Data format & phases")
    st.markdown("""
**Columns (case-insensitive):**
- `case_id` (required)
- `phase` — "1" (Individual model), "2" (CADSS), "3" (Human reference)
- `patient_summary` — brief structured summary used for scoring context
- Model note columns: `chatgpt_note`, `claude_note`, `gemini_note`, `cadss_note`, `human_note`

> You can include extra metadata columns (e.g., regimen, receptor status, eGFR, LVEF). They will be ignored in the UI but can help evaluators decide.

**Blinded scoring**  
Toggle **Blind Mode** in the sidebar to hide model names during scoring. Human note is shown for reference only.

**Rubric & overlays**  
Scores are 0–5 per subscale; the app applies weights (sum=1.0) → 0–100 composite.  
Safety/quality overlays apply penalties post-composite (e.g., missed contraindication).

**PCNE v9.1**  
Tag primary problem, causes, interventions, and outcome for each evaluation.

**Troubleshooting**
- Run with `streamlit run app.py`
- Install deps via `pip install -r requirements.txt`
- If port busy: `streamlit run app.py --server.port 8502`
    """)
