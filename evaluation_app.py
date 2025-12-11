import io
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Basic page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)",
    layout="wide",
)

# ---------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------
if "cases_df" not in st.session_state:
    st.session_state["cases_df"] = None

if "eval_records" not in st.session_state:
    st.session_state["eval_records"] = []  # list of dicts

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_cases_from_upload(uploaded_file):
    """Read khcc_eval_input.xlsx into a DataFrame with basic validation."""
    try:
        df = pd.read_excel(uploaded_file)

        required_cols = {"case_id", "note_label", "note_text"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(
                f"The uploaded file is missing required columns: {', '.join(missing)}"
            )
            return None

        # Ensure string types
        df["case_id"] = df["case_id"].astype(str)
        df["note_label"] = df["note_label"].astype(str)

        # Optional columns
        for col in ["patient_summary", "note_source", "note_source_full"]:
            if col not in df.columns:
                df[col] = ""

        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None


def get_cases_list(df: pd.DataFrame) -> List[str]:
    return sorted(df["case_id"].unique(), key=lambda x: (int(x) if x.isdigit() else x))


def get_labels_for_case(df: pd.DataFrame, case_id: str) -> List[str]:
    subset = df[df["case_id"] == case_id]
    labels = sorted(subset["note_label"].unique())
    return labels


def get_note_for_case_label(df: pd.DataFrame, case_id: str, label: str) -> Dict[str, Any]:
    subset = df[(df["case_id"] == case_id) & (df["note_label"] == label)]
    if subset.empty:
        return {}
    row = subset.iloc[0]
    return {
        "note_text": row["note_text"],
        "patient_summary": row.get("patient_summary", ""),
        "note_source": row.get("note_source", ""),
        "note_source_full": row.get("note_source_full", ""),
    }


# ---------------------------------------------------------
# UI sections
# ---------------------------------------------------------
def sidebar_setup():
    with st.sidebar:
        st.header("Setup")

        evaluator_name = st.text_input("Evaluator name*", "")
        center_dept = st.text_input("Center / Dept", "")

        wizard_mode = st.toggle("Wizard Mode (one question at a time)", value=False)

        st.markdown("---")
        st.subheader("Upload cases (khcc_eval_input.xlsx)")

        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["xlsx"],
            help="Use the blinded file generated from build_eval_input.py",
        )

        if uploaded_file is not None:
            df = load_cases_from_upload(uploaded_file)
            if df is not None:
                st.session_state["cases_df"] = df
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.success(
                    f"Loaded {len(df)} rows "
                    f"from {uploaded_file.name} "
                    f"({df['case_id'].nunique()} unique cases)."
                )

        return evaluator_name, center_dept, wizard_mode


def home_page():
    st.title("Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)")

    st.write(
        "Evaluating pharmacist-style recommendations for breast cancer chemotherapy. "
        "Each case has up to five blinded notes (A–E): four AI systems and one human reference note."
    )

    st.markdown(
        """
### Instructions (Summary)

1. **Upload** the blinded evaluation file (`khcc_eval_input.xlsx`) in the sidebar.  
2. Go to **Evaluate**:
   * Select a **Case ID**.
   * Select a **Note label (A–E)**.
   * Read the note text (the author is hidden).
   * Fill in the **PCNE V9.1 codes** (Problems, Causes, Interventions, Outcomes).
   * Score the note using the **six-domain holistic rubric** (1–5 for each domain).
   * Add optional comments and save.
3. In **Review**, you can see all evaluations recorded in this session.
4. In **Export**, download your evaluations as an Excel or CSV file for analysis.

All evaluations are anonymous with respect to the models; the mapping of A–E to systems is only known in the analysis phase.
        """
    )


def evaluate_page(evaluator_name: str, center_dept: str, wizard_mode: bool):
    st.title("Evaluate a Case / Note")

    if not evaluator_name:
        st.warning("Please enter your **name** in the sidebar before evaluating.")
        return

    df = st.session_state.get("cases_df", None)
    if df is None:
        st.info("Please upload `khcc_eval_input.xlsx` in the sidebar to begin.")
        return

    # --- Case & Note selection ---
    case_list = get_cases_list(df)
    if not case_list:
        st.error("No cases found in the uploaded file.")
        return

    col_case, col_label = st.columns([2, 1])

    with col_case:
        selected_case = st.selectbox("Case ID", case_list, key="eval_case")

    labels = get_labels_for_case(df, selected_case)
    if not labels:
        st.warning("No notes available for this case.")
        return

    with col_label:
        selected_label = st.selectbox("Note label", labels, key="eval_label")

    note_info = get_note_for_case_label(df, selected_case, selected_label)
    if not note_info:
        st.warning("No note found for this Case / Label combination.")
        return

    # ------------- Display Note (no patient summary section) -------------
    st.subheader("Note Text (blinded)")
    st.markdown(
        "<div style='border:1px solid #ddd; border-radius:6px; padding:1rem; "
        "background-color:#fafafa; white-space:pre-wrap;'>"
        f"{note_info['note_text']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ------------- PCNE V9.1 Coding -------------
    st.header("PCNE V9.1 Coding")

    st.markdown(
        """
Use PCNE to code observed drug-related problems (DRPs) in this note.
You may select **multiple codes** per domain if needed.
        """
    )

    pcne_col1, pcne_col2 = st.columns(2)

    with pcne_col1:
        pcne_problems = st.multiselect(
            "Problems (P)",
            options=[
                "P1.1 – No effect of drug treatment",
                "P1.2 – Effect of drug treatment not optimal",
                "P1.3 – Untreated indication",
                "P2.1 – Adverse drug event occurs",
                "P2.2 – Potential adverse drug event",
                "P3 – Other / process-related",
            ],
        )

        pcne_causes = st.multiselect(
            "Causes (C)",
            options=[
                "C1 – Drug selection",
                "C2 – Dose selection",
                "C3 – Treatment duration",
                "C4 – Drug form",
                "C5 – Route of administration",
                "C6 – Drug use process",
                "C9 – Monitoring",
            ],
        )

    with pcne_col2:
        pcne_interventions = st.multiselect(
            "Interventions (I)",
            options=[
                "I0 – No intervention",
                "I1 – At prescriber level",
                "I2 – At patient level",
                "I3 – At drug level",
            ],
        )

        pcne_outcomes = st.multiselect(
            "Outcomes (O)",
            options=[
                "O0 – Unknown",
                "O1 – Solved",
                "O2 – Partially solved",
                "O3 – Not solved",
            ],
        )

    pcne_comment = st.text_area(
        "PCNE comments / description of key DRPs (optional)", height=120
    )

    st.markdown("---")

    # ------------- Modified Stanford Holistic Rubric -------------
    st.header("Holistic Rubric (1–5)")

    st.markdown(
        """
Rate the **clinical quality** of this note across six domains.  
1 = Poor, 3 = Acceptable / borderline, 5 = Excellent.
        """
    )

    rubric_col1, rubric_col2 = st.columns(2)

    with rubric_col1:
        clinical_reasoning = st.slider(
            "1. Clinical Reasoning Accuracy",
            min_value=1,
            max_value=5,
            value=3,
            help="Logical coherence and pharmacologic correctness of the recommendation.",
        )
        safety_risk = st.slider(
            "2. Safety and Risk Sensitivity",
            min_value=1,
            max_value=5,
            value=3,
            help="Ability to detect contraindications, toxicity risks, and monitoring needs.",
        )
        completeness = st.slider(
            "3. Completeness and Relevance",
            min_value=1,
            max_value=5,
            value=3,
            help="Coverage of dose, safety, interactions, supportive care, and monitoring.",
        )

    with rubric_col2:
        guideline_adherence = st.slider(
            "4. Guideline and Protocol Adherence",
            min_value=1,
            max_value=5,
            value=3,
            help="Alignment with KHCC protocols and international oncology standards.",
        )
        communication_clarity = st.slider(
            "5. Clinical Communication Clarity",
            min_value=1,
            max_value=5,
            value=3,
            help="Structure, tone, and readability of the note.",
        )
        overall_value = st.slider(
            "6. Overall Clinical Value",
            min_value=1,
            max_value=5,
            value=3,
            help="How confident you would feel using this note in real practice.",
        )

    unacceptable = st.checkbox(
        "This note is **clinically unacceptable** / unsafe overall",
        value=False,
    )

    overall_comment = st.text_area(
        "Overall comments on this note (optional)", height=150
    )

    st.markdown("---")

    if st.button("✅ Save evaluation for this note"):
        record = {
            "evaluator_name": evaluator_name,
            "center_dept": center_dept,
            "case_id": selected_case,
            "note_label": selected_label,
            # hidden meta (for analysis only; may be empty if you deleted it)
            "note_source": note_info.get("note_source", ""),
            "note_source_full": note_info.get("note_source_full", ""),
            # PCNE
            "pcne_problems": "; ".join(pcne_problems),
            "pcne_causes": "; ".join(pcne_causes),
            "pcne_interventions": "; ".join(pcne_interventions),
            "pcne_outcomes": "; ".join(pcne_outcomes),
            "pcne_comment": pcne_comment,
            # rubric
            "clinical_reasoning": clinical_reasoning,
            "safety_risk": safety_risk,
            "completeness": completeness,
            "guideline_adherence": guideline_adherence,
            "communication_clarity": communication_clarity,
            "overall_value": overall_value,
            "clinically_unacceptable": unacceptable,
            "overall_comment": overall_comment,
        }

        st.session_state["eval_records"].append(record)
        st.success(f"Saved evaluation for Case {selected_case}, note {selected_label}.")


def review_page():
    st.title("Review Saved Evaluations")

    records = st.session_state.get("eval_records", [])
    if not records:
        st.info("No evaluations recorded in this session yet.")
        return

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)


def export_page():
    st.title("Export Evaluations")

    records = st.session_state.get("eval_records", [])
    if not records:
        st.info("No evaluations to export yet.")
        return

    df = pd.DataFrame(records)

    st.subheader("Preview")
    st.dataframe(df, use_container_width=True)

    # Excel export
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="evaluations")
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download as Excel (.xlsx)",
        data=buffer,
        file_name="khcc_evaluations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # CSV export
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_bytes,
        file_name="khcc_evaluations.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# Main router
# ---------------------------------------------------------
def main():
    evaluator_name, center_dept, wizard_mode = sidebar_setup()

    tabs = st.tabs(["🏠 Home", "🧪 Evaluate", "📋 Review", "📤 Export"])

    with tabs[0]:
        home_page()
    with tabs[1]:
        evaluate_page(evaluator_name, center_dept, wizard_mode)
    with tabs[2]:
        review_page()
    with tabs[3]:
        export_page()


if __name__ == "__main__":
    main()
