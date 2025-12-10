# Thesis Evaluation Tool — KHCC (Blinded, Long-format, No Gemini)
# Run: streamlit run evaluation_app.py

# -------------------
# Imports
# -------------------
import re
import random
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

# -------------------
# Page config
# -------------------
st.set_page_config(
    page_title="Thesis Evaluation Tool — KHCC (Blinded)",
    layout="wide"
)

# -------------------
# Config / Constants
# -------------------
RANGE = (0, 5)  # per-dimension scale

RUBRIC = {
    "Dose verification / adjustment": (
        0.18,
        "Protocol selection; BSA/CrCl/eGFR/Child-Pugh based dosing; adjustments for age/organ function."
    ),
    "Interactions & contraindications": (
        0.18,
        "Identifies & manages chemo/supportive/OTC/herbal DDIs; flags absolute/relative contraindications."
    ),
    "Safety & risk awareness": (
        0.16,
        "Anticipates agent-specific toxicities (cardiotox, myelosuppression, hepatotox, neuropathy) & mitigation."
    ),
    "Supportive care / premedication": (
        0.12,
        "Antiemetics by emetogenicity; G-CSF criteria; TLS prevention; antimicrobial prophylaxis when indicated."
    ),
    "Monitoring & follow-up": (
        0.12,
        "Labs/imaging schedule; thresholds to hold/adjust; timing windows for agent-specific risks."
    ),
    "Guideline concordance": (
        0.14,
        "Alignment with NCCN/ESMO/ASCO/institutional guidance; rationale is sound."
    ),
    "Clarity & actionability": (
        0.10,
        "Clear, prioritized, implementable plan with concise wording and no ambiguity."
    ),
}

# Critical Safety Flags (penalize final score after composite)
CRITICAL_FLAGS = {
    "Missed absolute contraindication (major risk)": 0.12,
    "Severe drug–drug interaction overlooked": 0.10,
    "Dose calculation / units error": 0.08,
    "Unsupported or hallucinated clinical claim": 0.06,
    "Critical monitoring omission": 0.06,
    "Major non-concordance with guideline": 0.08,
}

# PCNE v9.1
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

# Expected columns for the long-format input
# (each row is one note for one case)
EXPECTED = ["case_id", "patient_summary", "note_source", "note_text"]

# -------------------
# Sanitizer: remove model/brand names from any note text that might leak
# -------------------
_SANITIZE_TERMS = [
    r"chatgpt(?:[-\s]*4(?:o|o[-\s]*mini)?)?",
    r"gpt[-\s]*4(?:o|o[-\s]*mini)?",
    r"claude(?:[-\s]*3(?:\.?5)?(?:[-\s]*sonnet)?)?",
    r"deepseek(?:[-\s]*v3|[-\s]*reasoner)?",
    r"cadss", r"collaborative\s*ai",
    r"gemini(?:[-\s]*1\.\d+)?",
    r"openai", r"anthropic", r"google\s*deepmind"
]
_SANITIZE_RE = re.compile(r"(?i)\b(" + r"|".join(_SANITIZE_TERMS) + r")\b")

def sanitize_note(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"(?i)\b(model|assistant)\s+says?:", "", text)
    text = _SANITIZE_RE.sub("[redacted]", text)
    return text.strip()

# -------------------
# Demo cases (long format, neutral text, no model names)
# -------------------
DEMO_CASES = pd.DataFrame([
    {
        "case_id": "KHCC-001",
        "patient_summary": "54F, ER+/HER2-, AC→T planned; eGFR 48 ml/min; LVEF 60%; HTN on amlodipine.",
        "note_source": "chatgpt",
        "note_text": "Draft pharmacist recommendation (System A)…"
    },
    {
        "case_id": "KHCC-001",
        "patient_summary": "54F, ER+/HER2-, AC→T planned; eGFR 48 ml/min; LVEF 60%; HTN on amlodipine.",
        "note_source": "claude",
        "note_text": "Draft pharmacist recommendation (System B)…"
    },
    {
        "case_id": "KHCC-001",
        "patient_summary": "54F, ER+/HER2-, AC→T planned; eGFR 48 ml/min; LVEF 60%; HTN on amlodipine.",
        "note_source": "human",
        "note_text": "Institutional reference pharmacist note…"
    },
    {
        "case_id": "KHCC-002",
        "patient_summary": "61F, HER2+, docetaxel + trastuzumab + pertuzumab; DM2 (metformin); ALT 2× ULN.",
        "note_source": "cadss",
        "note_text": "Composite pharmacist recommendation (System C)…"
    },
])

# -------------------
# Session state
# -------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("cases", pd.DataFrame())          # uploaded cases (long format)
    ss.setdefault("evaluations", [])                # saved evaluations (dict list)
    ss.setdefault("evaluator", "")
    ss.setdefault("center", "")
    ss.setdefault("wizard_mode", True)
    # per-case mapping: {'KHCC-001': {'A': {...row dict...}, 'B': {...}}}
    ss.setdefault("blind_map", {})

_init_state()

# -------------------
# Helpers
# -------------------
def load_cases(upload) -> pd.DataFrame:
    """Load long-format cases file (case_id, patient_summary, note_source, note_text)."""
    try:
        if upload.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(upload)
        else:
            df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    for c in EXPECTED:
        if c not in df.columns:
            df[c] = ""

    # Ensure string types and sanitize text
    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["patient_summary"] = df["patient_summary"].astype(str)
    df["note_source"] = df["note_source"].astype(str).str.strip().str.lower()
    df["note_text"] = df["note_text"].astype(str)

    return df


def composite_score(scores: dict) -> float:
    total = 0.0
    for k, v in scores.items():
        w = RUBRIC[k][0]
        total += (v / RANGE[1]) * w
    return round(total * 100, 2)


def apply_flags(base: float, flags: list) -> float:
    penalty = sum(CRITICAL_FLAGS.get(f, 0) for f in flags)
    penalty = min(penalty, 0.6)
    return round(base * (1 - penalty), 2)


def get_blind_mapping_for_case(case_rows: pd.DataFrame) -> dict:
    """
    For one case_id, take all notes, shuffle them, and assign A/B/C...
    Returns dict: {'A': row_dict, 'B': row_dict, ...}
    """
    if case_rows.empty:
        return {}

    shuffled = case_rows.sample(frac=1).reset_index(drop=True)
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    mapping = {}

    for i in range(len(shuffled)):
        row = shuffled.iloc[i]
        label = labels[i]
        mapping[label] = {
            "case_id": row["case_id"],
            "patient_summary": row["patient_summary"],
            "note_source": row["note_source"],
            "note_text": row["note_text"],
        }

    return mapping

# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.header("Setup")
    st.session_state.evaluator = st.text_input(
        "Evaluator name*",
        st.session_state.evaluator
    )
    st.session_state.center = st.text_input(
        "Center / Dept",
        st.session_state.center
    )
    st.session_state.wizard_mode = st.toggle(
        "Wizard Mode (one question at a time)",
        value=st.session_state.wizard_mode
    )

    uploaded = st.file_uploader(
        "Upload evaluation file (khcc_eval_input.xlsx)",
        type=["xlsx", "xls", "csv"]
    )
    if uploaded:
        st.session_state.cases = load_cases(uploaded)
        if len(st.session_state.cases):
            st.success(f"Loaded {len(st.session_state.cases)} notes.")
        else:
            st.warning("No rows found; using Demo Cases.")

    st.divider()
    st.caption(
        "Fully blinded: evaluators only see **Note A/B/C**. "
        "True source (AI model vs human) is hidden in the UI but stored in exports."
    )

# -------------------
# Main layout
# -------------------
st.title("Thesis Evaluation Tool — KHCC (Blinded)")
st.caption(
    "Evaluating AI-generated and human pharmacist notes for breast cancer chemotherapy "
    "in a blinded fashion."
)

cases_df = st.session_state.cases if not st.session_state.cases.empty else DEMO_CASES.copy()
demo_mode = st.session_state.cases.empty
if demo_mode:
    st.info(
        "**Demo Mode**: No file uploaded. Using sample cases so evaluators can "
        "practice and see the full question flow."
    )

tab_home, tab_evaluate, tab_review, tab_export = st.tabs(
    ["🏠 Home", "📝 Evaluate", "🔎 Review", "⬇️ Export"]
)

# -------------------
# HOME
# -------------------
with tab_home:
    st.subheader("How it works")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1) Pick a Case**\nSelect any case (demo or uploaded).")
        st.markdown(
            "**2) Pick a Note**\nFor each case, notes appear as **Note A/B/C/...** "
            "(randomized per case)."
        )
    with cols[1]:
        st.markdown(
            "**3) Answer the Questions**\nScore 7 rubric items (0–5), add "
            "per-section comments if needed."
        )
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

    st.subheader("Exact questions evaluators answer")
    st.markdown(
        """
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

**Section D — Overall**  
- Overall comment (free text)  
- Rater confidence (0–5)
        """
    )

# -------------------
# EVALUATE
# -------------------
with tab_evaluate:
    st.subheader("Evaluate a Case / Note")
    if not st.session_state.evaluator.strip():
        st.warning("Please enter your **Evaluator name** in the sidebar.")
    else:
        case_ids = sorted(cases_df["case_id"].unique().tolist())
        if not case_ids:
            st.info("No cases available.")
        else:
            case_id = st.selectbox("Case ID", case_ids)
            case_rows = cases_df[cases_df["case_id"] == case_id]

            if case_rows.empty:
                st.info("No notes available for this case.")
            else:
                # Use patient summary from first row
                patient_summary = case_rows.iloc[0]["patient_summary"]

                left, right = st.columns([1, 1])
                with left:
                    st.markdown("### Patient Summary")
                    st.write(sanitize_note(patient_summary) or "_(empty)_")

                with right:
                    st.markdown("### Notes for this case")
                    st.write(
                        "Notes are labelled **A/B/C/...** and randomized per case. "
                        "You will evaluate them one at a time."
                    )

                # Build/retrieve blind mapping for this case
                blind_map = st.session_state.blind_map.get(case_id)
                if not blind_map:
                    blind_map = get_blind_mapping_for_case(case_rows)
                    st.session_state.blind_map[case_id] = blind_map

                note_labels = list(blind_map.keys())
                if not note_labels:
                    st.info("No notes available for this case.")
                else:
                    chosen_label = st.selectbox(
                        "Choose a note to evaluate",
                        [f"Note {lab}" for lab in note_labels]
                    )
                    lab = chosen_label.split()[-1]
                    note_row = blind_map[lab]

                    note_text = sanitize_note(note_row["note_text"]) or "_(empty)_"
                    note_source = note_row["note_source"]         # HIDDEN in UI
                    underlying_summary = note_row["patient_summary"]

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
                        for i, (label_txt, (w, help_txt)) in enumerate(rubric_items, start=1):
                            st.progress(i / total_q, text=f"Question {i} of {total_q}")
                            subscores[label_txt] = st.slider(
                                f"{i}. {label_txt}", *RANGE, value=0,
                                help=f"Weight {int(w*100)}%. {help_txt}"
                            )
                            section_comments[label_txt] = st.text_area(
                                f"Comment — {label_txt}",
                                height=70,
                                placeholder="Optional",
                                key=f"c_{label_txt}"
                            )
                            st.divider()
                    else:
                        cols = st.columns(2)
                        for i, (label_txt, (w, help_txt)) in enumerate(RUBRIC.items(), start=1):
                            with cols[i % 2]:
                                subscores[label_txt] = st.slider(
                                    f"{i}. {label_txt}", *RANGE, value=0,
                                    help=f"Weight {int(w*100)}%. {help_txt}"
                                )
                                section_comments[label_txt] = st.text_area(
                                    f"Comment — {label_txt}",
                                    height=70,
                                    placeholder="Optional",
                                    key=f"c_{label_txt}"
                                )

                    # Section B — Critical Safety Flags
                    st.markdown("## Section B — Critical Safety Flags")
                    flags = st.multiselect(
                        "Select any that apply:",
                        list(CRITICAL_FLAGS.keys())
                    )

                    # Section C — PCNE
                    st.markdown("## Section C — PCNE v9.1")
                    pcne_problem = st.selectbox("Primary problem", PCNE_PROBLEMS)
                    pcne_causes = st.multiselect("Causes (select all that apply)", PCNE_CAUSES)
                    pcne_interventions = st.multiselect("Interventions (select all that apply)", PCNE_INTERVENTIONS)
                    pcne_outcome = st.selectbox("Outcome", PCNE_OUTCOMES)

                    # Section D — Overall & confidence
                    st.markdown("## Section D — Overall comment & confidence")
                    overall_comment = st.text_area("Overall comment", height=120)
                    confidence = st.select_slider(
                        "Rater confidence",
                        options=[0, 1, 2, 3, 4, 5],
                        value=4
                    )

                    # Scores
                    base = composite_score(subscores)
                    final = apply_flags(base, flags)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Composite (0–100)", f"{base}")
                    with c2:
                        st.metric("Final (after flags)", f"{final}")

                    if st.button("✅ Save evaluation", use_container_width=True):
                        entry = {
                            "evaluation_id": str(uuid.uuid4()),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "demo_mode": demo_mode,
                            "center": st.session_state.center.strip(),
                            "evaluator": st.session_state.evaluator.strip(),
                            "case_id": case_id,
                            "note_label": lab,               # A/B/C only — what the rater saw
                            "note_source": note_source,       # true source (chatgpt/claude/cadss/human/...)
                            "patient_summary": underlying_summary,
                            "note_text": note_row["note_text"],
                            "base_score": base,
                            "final_score": final,
                            "confidence": confidence,
                            "overall_comment": overall_comment.strip(),
                            "flags": "; ".join(flags),
                            "pcne_problem": pcne_problem,
                            "pcne_causes": "; ".join(pcne_causes),
                            "pcne_interventions": "; ".join(pcne_interventions),
                            "pcne_outcome": pcne_outcome,
                            **{f"score::{k}": v for k, v in subscores.items()},
                            **{f"comment::{k}": (section_comments.get(k, "") or "")
                               for k in subscores.keys()},
                        }
                        st.session_state.evaluations.append(entry)
                        st.success("Saved.")

# -------------------
# REVIEW
# -------------------
with tab_review:
    st.subheader("All Saved Evaluations (Blinded)")
    if not st.session_state.evaluations:
        st.info("No evaluations yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_case = st.selectbox(
                "Filter: Case",
                ["(All)"] + sorted(edf["case_id"].unique().tolist())
            )
        with c2:
            f_eval = st.selectbox(
                "Filter: Evaluator",
                ["(All)"] + sorted(edf["evaluator"].unique().tolist())
            )
        with c3:
            f_src = st.selectbox(
                "Filter: Note source",
                ["(All)"] + sorted(edf["note_source"].unique().tolist())
            )
        with c4:
            sort_by = st.selectbox(
                "Sort by",
                ["timestamp", "final_score", "base_score", "case_id", "note_label", "note_source"]
            )

        f = edf.copy()
        if f_case != "(All)":
            f = f[f["case_id"] == f_case]
        if f_eval != "(All)":
            f = f[f["evaluator"] == f_eval]
        if f_src != "(All)":
            f = f[f["note_source"] == f_src]

        st.dataframe(
            f.sort_values(by=sort_by, ascending=sort_by not in ["final_score", "base_score"]),
            use_container_width=True,
            height=450
        )

# -------------------
# EXPORT
# -------------------
with tab_export:
    st.subheader("Export CSV (Blinded in UI, but includes sources for analysis)")
    if not st.session_state.evaluations:
        st.info("Nothing to export yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)
        export_cols = [
            "evaluation_id", "timestamp", "demo_mode", "center", "evaluator",
            "case_id",
            "note_label",          # A/B/C as seen by rater
            "note_source",         # true origin (chatgpt/claude/deepseek/cadss/human)
            "patient_summary",
            "note_text",
            "base_score", "final_score", "confidence",
            "flags", "overall_comment",
            "pcne_problem", "pcne_causes", "pcne_interventions", "pcne_outcome",
        ] + sorted(
            [c for c in edf.columns if c.startswith("score::") or c.startswith("comment::")]
        )

        for c in export_cols:
            if c not in edf.columns:
                edf[c] = ""

        out = edf[export_cols].copy()
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download evaluations.csv",
            data=csv_bytes,
            file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(
            "Download includes the true note source **for analysis only**; "
            "evaluators never see model identities during scoring."
        )
