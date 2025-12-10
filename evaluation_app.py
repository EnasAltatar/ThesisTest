# evaluation_app.py
# Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)
# Run with:  streamlit run evaluation_app.py

import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

# -------------------
# Page config
# -------------------
st.set_page_config(
    page_title="Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)",
    layout="wide",
)

# -------------------
# Config / Constants
# -------------------
RANGE = (0, 5)  # slider range per rubric dimension

RUBRIC = {
    "Dose verification / adjustment": (
        0.18,
        "Protocol selection; BSA/CrCl/eGFR/Child-Pugh based dosing; adjustments "
        "for age/organ function.",
    ),
    "Interactions & contraindications": (
        0.18,
        "Identifies & manages chemo/supportive/OTC/herbal interactions; flags absolute/relative contraindications.",
    ),
    "Safety & risk awareness": (
        0.16,
        "Anticipates agent-specific toxicities (cardiotoxicity, myelosuppression, hepatotoxicity, neuropathy) & mitigation.",
    ),
    "Supportive care / premedication": (
        0.12,
        "Antiemetics by emetogenicity; G-CSF criteria; TLS prevention; antimicrobial prophylaxis when indicated.",
    ),
    "Monitoring & follow-up": (
        0.12,
        "Labs/imaging schedule; thresholds to hold/adjust; timing windows for agent-specific risks.",
    ),
    "Guideline concordance": (
        0.14,
        "Alignment with NCCN/ESMO/ASCO/institutional guidance; rationale is sound.",
    ),
    "Clarity & actionability": (
        0.10,
        "Clear, prioritized, implementable plan with concise wording and no ambiguity.",
    ),
}

CRITICAL_FLAGS = {
    "Missed absolute contraindication (major risk)": 0.12,
    "Severe drug–drug interaction overlooked": 0.10,
    "Dose calculation / units error": 0.08,
    "Unsupported or hallucinated clinical claim": 0.06,
    "Critical monitoring omission": 0.06,
    "Major non-concordance with guideline": 0.08,
}

PCNE_PROBLEMS = [
    "P1 Treatment effectiveness",
    "P2 Adverse drug event",
    "P3 Treatment costs",
    "P4 Others/Unclear",
]
PCNE_CAUSES = [
    "C1 Drug selection",
    "C2 Drug form",
    "C3 Dose selection",
    "C4 Treatment duration",
    "C5 Dispensing",
    "C6 Drug use process",
    "C7 Patient-related",
    "C8 Logistics",
    "C9 Other",
]
PCNE_INTERVENTIONS = [
    "I1 Prescriber informed",
    "I2 Prescriber accepted change",
    "I3 Patient-level intervention",
    "I4 Drug-level intervention",
    "I5 Monitoring advised",
    "I6 Education provided",
    "I7 Referral",
    "I8 Other",
]
PCNE_OUTCOMES = ["O0 Problem not solved", "O1 Partially solved", "O2 Solved", "O3 Unknown"]

# Expected columns in khcc_eval_input.xlsx
EXPECTED = [
    "case_id",
    "patient_summary",
    "note_a",
    "note_b",
    "note_c",
    "note_d",
    "note_e",
]

# -------------------
# Sanitizer — remove model/vendor names
# -------------------
_SANITIZE_TERMS = [
    r"chatgpt(?:[-\s]*4(?:o|o[-\s]*mini)?)?",
    r"gpt[-\s]*4(?:o|o[-\s]*mini)?",
    r"claude(?:[-\s]*3(?:\.?5)?(?:[-\s]*sonnet)?)?",
    r"deepseek(?:[-\s]*v3|[-\s]*reasoner)?",
    r"cadss",
    r"collaborative\s*ai",
    r"gemini(?:[-\s]*1\.\d+)?",
    r"openai",
    r"anthropic",
    r"google\s*deepmind",
]
_SANITIZE_RE = re.compile(r"(?i)\b(" + r"|".join(_SANITIZE_TERMS) + r")\b")


def sanitize_note(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"(?i)\b(model|assistant)\s+says?:", "", text)
    text = _SANITIZE_RE.sub("[redacted]", text)
    return text.strip()


# -------------------
# Demo Data (if no file uploaded)
# -------------------
DEMO_CASES = pd.DataFrame(
    [
        {
            "case_id": "DEMO-001",
            "patient_summary": "54F, ER+/HER2-, AC→T planned; eGFR 48 ml/min; LVEF 60%; HTN on amlodipine.",
            "note_a": "Demo pharmacist recommendation (Note A)…",
            "note_b": "Demo pharmacist recommendation (Note B)…",
            "note_c": "Demo pharmacist recommendation (Note C)…",
            "note_d": "Demo pharmacist recommendation (Note D)…",
            "note_e": "Demo pharmacist recommendation (Note E)…",
        }
    ]
)

# -------------------
# Session state init
# -------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("cases", pd.DataFrame())
    ss.setdefault("evaluations", [])
    ss.setdefault("evaluator", "")
    ss.setdefault("center", "")
    ss.setdefault("wizard_mode", True)


_init_state()

# -------------------
# Helpers
# -------------------
def load_cases(upload) -> pd.DataFrame:
    try:
        if upload.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(upload)
        else:
            df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # ensure expected columns exist
    for c in EXPECTED:
        if c not in df.columns:
            df[c] = ""

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["patient_summary"] = df["patient_summary"].astype(str)

    for col in ["note_a", "note_b", "note_c", "note_d", "note_e"]:
        df[col] = df[col].astype(str).apply(sanitize_note)

    df = df[EXPECTED].copy()
    return df


def composite_score(scores: dict) -> float:
    total = 0.0
    for k, v in scores.items():
        w = RUBRIC[k][0]
        total += (v / RANGE[1]) * w
    return round(total * 100, 2)


def apply_flags(base: float, flags: list) -> float:
    penalty = sum(CRITICAL_FLAGS.get(f, 0) for f in flags)
    penalty = min(penalty, 0.6)  # cap penalty
    return round(base * (1 - penalty), 2)


# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.header("Setup")
    st.session_state.evaluator = st.text_input(
        "Evaluator name*", st.session_state.evaluator
    )
    st.session_state.center = st.text_input(
        "Center / Dept", st.session_state.center
    )
    st.session_state.wizard_mode = st.toggle(
        "Wizard Mode (one question at a time)",
        value=st.session_state.wizard_mode,
    )

    uploaded = st.file_uploader(
        "Upload cases (khcc_eval_input.xlsx)", type=["xlsx", "xls", "csv"]
    )
    if uploaded:
        st.session_state.cases = load_cases(uploaded)
        if len(st.session_state.cases):
            st.success(f"Loaded {len(st.session_state.cases)} rows.")
        else:
            st.warning("No rows found; using demo data instead.")

    st.divider()
    st.caption(
        "Blinded design: evaluators see **Note A–E** only. "
        "Model/human identities are not shown or exported."
    )

# -------------------
# Main layout
# -------------------
st.title("Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)")
st.caption("Evaluating pharmacist-style recommendations for breast cancer chemotherapy.")

cases_df = (
    st.session_state.cases
    if not st.session_state.cases.empty
    else DEMO_CASES.copy()
)
demo_mode = st.session_state.cases.empty
if demo_mode:
    st.info("Demo Mode: no file uploaded. Using sample data for practice only.")

tab_home, tab_evaluate, tab_review, tab_export = st.tabs(
    ["🏠 Home", "📝 Evaluate", "🔎 Review", "⬇️ Export"]
)

# -------------------
# HOME
# -------------------
with tab_home:
    st.subheader("How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1) Pick a Case**\nSelect any case ID.")
        st.markdown("**2) Pick a Note**\nNotes are labeled **A–E**.")
    with c2:
        st.markdown("**3) Score the note**\nUse the 7-dimension rubric (0–5 each).")
        st.markdown("**4) Add safety flags**\nIf any critical issue exists.")
    with c3:
        st.markdown(
            "**5) Tag PCNE categories**\nProblem, causes, interventions, outcome."
        )
        st.markdown("**6) Save & Export**\nDownload blinded CSV when done.")

    st.subheader("Rubric dimensions")
    for name, (w, desc) in RUBRIC.items():
        st.markdown(f"- **{name}** ({int(w*100)}%): {desc}")

# -------------------
# EVALUATE
# -------------------
with tab_evaluate:
    st.subheader("Evaluate a Case / Note")
    if not st.session_state.evaluator.strip():
        st.warning("Please enter your **Evaluator name** in the sidebar.")
    else:
        case_id = st.selectbox("Case ID", cases_df["case_id"].unique().tolist())
        row = cases_df[cases_df["case_id"] == case_id].iloc[0]

        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Patient Summary")
            st.write(row.get("patient_summary", "") or "_(empty)_")

        with right:
            st.markdown("### Instructions")
            st.write(
                "Evaluate one note at a time. Each note is labelled **A–E**. "
                "Please try to score all available notes for this case."
            )

        # Which note labels exist for this case?
        available_labels = []
        for lab in ["A", "B", "C", "D", "E"]:
            col = f"note_{lab.lower()}"
            if str(row.get(col, "")).strip():
                available_labels.append(lab)

        if not available_labels:
            st.info("No notes available for this case.")
        else:
            chosen_label = st.selectbox(
                "Choose a note to evaluate",
                [f"Note {lab}" for lab in available_labels],
            )
            lab = chosen_label.split()[-1]  # "A".."E"
            note_col = f"note_{lab.lower()}"
            note_text = row.get(note_col, "") or "_(empty)_"

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
                    subscores[label] = st.slider(
                        f"{i}. {label}",
                        *RANGE,
                        value=0,
                        help=f"Weight {int(w*100)}%. {help_txt}",
                    )
                    section_comments[label] = st.text_area(
                        f"Comment — {label}",
                        height=70,
                        placeholder="Optional",
                        key=f"c_{label}",
                    )
                    st.divider()
            else:
                cols = st.columns(2)
                for i, (label, (w, help_txt)) in enumerate(RUBRIC.items(), start=1):
                    with cols[i % 2]:
                        subscores[label] = st.slider(
                            f"{i}. {label}",
                            *RANGE,
                            value=0,
                            help=f"Weight {int(w*100)}%. {help_txt}",
                        )
                        section_comments[label] = st.text_area(
                            f"Comment — {label}",
                            height=70,
                            placeholder="Optional",
                            key=f"c_{label}",
                        )

            # Section B — Critical Safety Flags
            st.markdown("## Section B — Critical Safety Flags")
            flags = st.multiselect("Select any that apply:", list(CRITICAL_FLAGS.keys()))

            # Section C — PCNE
            st.markdown("## Section C — PCNE v9.1")
            pcne_problem = st.selectbox("Primary problem", PCNE_PROBLEMS)
            pcne_causes = st.multiselect(
                "Causes (select all that apply)", PCNE_CAUSES
            )
            pcne_interventions = st.multiselect(
                "Interventions (select all that apply)", PCNE_INTERVENTIONS
            )
            pcne_outcome = st.selectbox("Outcome", PCNE_OUTCOMES)

            # Section D — Overall & confidence
            st.markdown("## Section D — Overall comment & confidence")
            overall_comment = st.text_area("Overall comment", height=120)
            confidence = st.select_slider(
                "Rater confidence", options=[0, 1, 2, 3, 4, 5], value=4
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
                    "note_label": lab,  # A–E ONLY (no model/human name)
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
                    **{
                        f"comment::{k}": (section_comments.get(k, "") or "")
                        for k in subscores.keys()
                    },
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
        c1, c2, c3 = st.columns(3)
        with c1:
            f_case = st.selectbox(
                "Filter: Case",
                ["(All)"] + sorted(edf["case_id"].unique().tolist()),
            )
        with c2:
            f_eval = st.selectbox(
                "Filter: Evaluator",
                ["(All)"] + sorted(edf["evaluator"].unique().tolist()),
            )
        with c3:
            sort_by = st.selectbox(
                "Sort by",
                [
                    "timestamp",
                    "final_score",
                    "base_score",
                    "case_id",
                    "note_label",
                ],
            )

        f = edf.copy()
        if f_case != "(All)":
            f = f[f["case_id"] == f_case]
        if f_eval != "(All)":
            f = f[f["evaluator"] == f_eval]

        st.dataframe(
            f.sort_values(
                by=sort_by,
                ascending=sort_by not in ["final_score", "base_score"],
            ),
            use_container_width=True,
            height=450,
        )

# -------------------
# EXPORT
# -------------------
with tab_export:
    st.subheader("Export CSV (Blinded)")
    if not st.session_state.evaluations:
        st.info("Nothing to export yet.")
    else:
        edf = pd.DataFrame(st.session_state.evaluations)

        export_cols = [
            "evaluation_id",
            "timestamp",
            "demo_mode",
            "center",
            "evaluator",
            "case_id",
            "note_label",  # A–E only
            "base_score",
            "final_score",
            "confidence",
            "flags",
            "overall_comment",
            "pcne_problem",
            "pcne_causes",
            "pcne_interventions",
            "pcne_outcome",
        ] + sorted(
            [
                c
                for c in edf.columns
                if c.startswith("score::") or c.startswith("comment::")
            ]
        )

        # ensure all columns exist
        for c in export_cols:
            if c not in edf.columns:
                edf[c] = ""

        out = edf[export_cols].copy()
        csv_bytes = out.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download evaluations_blinded.csv",
            data=csv_bytes,
            file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Export is fully blinded. It contains only Note A–E labels, "
            "no model or human identifiers."
        )
