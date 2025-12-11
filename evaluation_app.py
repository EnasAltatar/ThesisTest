import io
from datetime import datetime

import pandas as pd
import streamlit as st


# ==============================
# Helpers
# ==============================

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip column names."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def detect_core_columns(df: pd.DataFrame):
    """
    Try to detect case_id, note_label, note_text columns
    from flexible column names.
    """
    cols = {c.lower(): c for c in df.columns}

    def find(opts):
        for o in opts:
            if o in cols:
                return cols[o]
        return None

    case_col = find(["case_id", "case id", "case"])
    label_col = find(["note_label", "note label", "label", "noteid", "note_id", "note"])
    text_col = find(["note_text", "note text", "note_body", "note body", "content", "text"])

    # If label was mapped to "note" and text_col is None, we may have:
    # label separate, note text in another; or the opposite.
    # If text_col is None but there is a column literally named "note",
    # we assume that is text, and look for label in something else.
    if text_col is None and "note" in cols:
        # Try to keep "label" separate if exists
        if "label" in cols and label_col != cols["label"]:
            label_col = cols["label"]
            text_col = cols["note"]
        else:
            text_col = cols["note"]

    return case_col, label_col, text_col


def init_session_state():
    if "cases_df" not in st.session_state:
        st.session_state["cases_df"] = None
    if "eval_records" not in st.session_state:
        st.session_state["eval_records"] = []


# ==============================
# UI Components
# ==============================

def sidebar_setup():
    st.sidebar.header("Setup")

    evaluator_name = st.sidebar.text_input("Evaluator name*", value=st.session_state.get("evaluator_name", ""))
    center_name = st.sidebar.text_input("Center / Dept", value=st.session_state.get("center_name", ""))

    st.session_state["evaluator_name"] = evaluator_name
    st.session_state["center_name"] = center_name

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload cases file")
    st.sidebar.caption("Upload **khcc_eval_input.xlsx** generated from your pipeline.")

    uploaded = st.sidebar.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        key="cases_file_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
            df = normalise_columns(df)
            case_col, label_col, text_col = detect_core_columns(df)

            missing = []
            if case_col is None:
                missing.append("case_id")
            if label_col is None:
                missing.append("note_label (A–E)")
            if text_col is None:
                missing.append("note_text (note content)")

            if missing:
                st.sidebar.error(
                    "❌ Could not detect required columns:\n\n"
                    + ", ".join(missing)
                    + "\n\nMake sure your file has columns like: "
                      "`Case_ID`, `note_label`, `note_text` / `Note`."
                )
                st.session_state["cases_df"] = None
            else:
                # Keep only what we need + any extra columns (but never show note_source)
                keep_cols = [case_col, label_col, text_col]
                keep_cols_set = set(keep_cols)
                extra_cols = [c for c in df.columns if c not in keep_cols_set]

                df = df[keep_cols + extra_cols].copy()
                df.rename(
                    columns={
                        case_col: "case_id",
                        label_col: "note_label",
                        text_col: "note_text",
                    },
                    inplace=True,
                )

                # Normalise case_id & note_label types
                df["case_id"] = df["case_id"].astype(str).str.strip()
                df["note_label"] = df["note_label"].astype(str).str.strip().str.upper()

                # Drop note_source from view (if it exists)
                if "note_source" in df.columns:
                    # we keep it in df (so it appears in export if needed),
                    # but we will never display it in the UI.
                    pass

                st.session_state["cases_df"] = df
                st.sidebar.success(f"Loaded {len(df)} rows.")

        except Exception as e:
            st.sidebar.error(f"Error reading Excel file: {e}")
            st.session_state["cases_df"] = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Export evaluations")
    if st.session_state["eval_records"]:
        export_df = pd.DataFrame(st.session_state["eval_records"])
        buf = io.BytesIO()
        export_df.to_excel(buf, index=False)
        buf.seek(0)
        st.sidebar.download_button(
            "⬇️ Download evaluations (Excel)",
            data=buf,
            file_name=f"khcc_eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.sidebar.caption("No evaluations yet.")


def evaluate_page():
    df = st.session_state.get("cases_df")
    if df is None:
        st.info("Please upload **khcc_eval_input.xlsx** from the sidebar to start.")
        return

    st.header("Evaluate a Case / Note")

    # ---------------------------
    # Case selection
    # ---------------------------
    case_ids = sorted(df["case_id"].unique(), key=lambda x: (len(x), x))
    selected_case = st.selectbox("Case ID", case_ids, key="selected_case")

    # Filter notes for that case
    case_df = df[df["case_id"] == selected_case].copy()
    if case_df.empty:
        st.warning("No notes available for this case.")
        return

    # ---------------------------
    # Note selection (A–E)
    # ---------------------------
    st.subheader("Select note to evaluate")
    labels_available = sorted(case_df["note_label"].unique())
    selected_label = st.radio(
        "Note label",
        options=labels_available,
        key="selected_note_label",
        horizontal=True,
    )

    note_row = case_df[case_df["note_label"] == selected_label].iloc[0]
    note_text = str(note_row["note_text"])

    # ---------------------------
    # Display note (no patient summary)
    # ---------------------------
    st.markdown("### Note Text")
    st.text_area(
        "Clinical pharmacist note (read-only)",
        value=note_text,
        height=350,
        key="note_display",
        disabled=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # =============================
    # PCNE V9.1 Coding
    # =============================
    st.subheader("PCNE V9.1 – Drug-Related Problems")

    st.markdown("#### A. Problem (P)")
    p_options = {
        "P1.1 – No effect of drug treatment despite correct use": "P1.1",
        "P1.2 – Effect of drug treatment not optimal": "P1.2",
        "P1.3 – Untreated indication": "P1.3",
        "P2.1 – Adverse drug event occurs": "P2.1",
        "P2.2 – Potential adverse drug event": "P2.2",
        "P3 – Other / process-related": "P3",
    }
    p_selected = st.multiselect(
        "Select all problems that apply",
        options=list(p_options.keys()),
        key="pcne_p",
    )

    st.markdown("#### B. Cause (C)")
    c_options = {
        "C1 – Drug selection": "C1",
        "C2 – Dose selection": "C2",
        "C3 – Treatment duration": "C3",
        "C4 – Drug form": "C4",
        "C5 – Route of administration": "C5",
        "C6 – Drug use process": "C6",
        "C9 – Monitoring": "C9",
    }
    c_selected = st.multiselect(
        "Select underlying causes",
        options=list(c_options.keys()),
        key="pcne_c",
    )

    st.markdown("#### C. Intervention (I)")
    i_options = {
        "I0 – No intervention": "I0",
        "I1 – At prescriber level": "I1",
        "I2 – At patient level": "I2",
        "I3 – At drug level": "I3",
    }
    i_selected = st.multiselect(
        "Select interventions (if any)",
        options=list(i_options.keys()),
        key="pcne_i",
    )

    st.markdown("#### D. Outcome (O)")
    o_options = {
        "O0 – Unknown": "O0",
        "O1 – Solved": "O1",
        "O2 – Partially solved": "O2",
        "O3 – Not solved": "O3",
    }
    o_selected = st.radio(
        "Expected outcome if this note were implemented",
        options=list(o_options.keys()),
        key="pcne_o",
    )

    pcne_free_text = st.text_area(
        "PCNE comments (optional)",
        value="",
        key="pcne_comment",
        height=80,
    )

    st.markdown("---")

    # =============================
    # Modified Stanford Rubric – 6 Domains
    # =============================
    st.subheader("Modified Stanford Holistic Evaluation Rubric (1–5)")

    def score_slider(label, key):
        return st.slider(
            label,
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key=key,
        )

    s_reasoning = score_slider("1. Clinical Reasoning Accuracy", "score_reasoning")
    s_safety = score_slider("2. Safety and Risk Sensitivity", "score_safety")
    s_completeness = score_slider("3. Completeness and Relevance", "score_completeness")
    s_guidelines = score_slider("4. Guideline and Protocol Adherence", "score_guidelines")
    s_clarity = score_slider("5. Clinical Communication Clarity", "score_clarity")
    s_overall = score_slider("6. Overall Clinical Value", "score_overall")

    overall_comment = st.text_area(
        "Overall comments on this note (optional)",
        value="",
        key="overall_comment",
        height=120,
    )

    # =============================
    # Save evaluation
    # =============================
    st.markdown("---")
    if st.button("✅ Save evaluation for this note"):
        if not st.session_state.get("evaluator_name"):
            st.error("Please enter your **evaluator name** in the sidebar before saving.")
            return

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "evaluator_name": st.session_state.get("evaluator_name", ""),
            "center": st.session_state.get("center_name", ""),
            "case_id": selected_case,
            "note_label": selected_label,
            # PCNE – store code lists (comma-separated) + labels chosen
            "pcne_p_codes": ",".join([p_options[p] for p in p_selected]) if p_selected else "",
            "pcne_p_labels": "; ".join(p_selected),
            "pcne_c_codes": ",".join([c_options[c] for c in c_selected]) if c_selected else "",
            "pcne_c_labels": "; ".join(c_selected),
            "pcne_i_codes": ",".join([i_options[i] for i in i_selected]) if i_selected else "",
            "pcne_i_labels": "; ".join(i_selected),
            "pcne_o_code": o_options[o_selected] if o_selected else "",
            "pcne_o_label": o_selected,
            "pcne_comment": pcne_free_text,
            # Rubric scores
            "score_reasoning": s_reasoning,
            "score_safety": s_safety,
            "score_completeness": s_completeness,
            "score_guidelines": s_guidelines,
            "score_clarity": s_clarity,
            "score_overall": s_overall,
            "overall_comment": overall_comment,
        }

        st.session_state["eval_records"].append(record)
        st.success(f"Saved evaluation for {selected_case} / note {selected_label}.")

        # Optionally clear some fields (we leave sliders as last scores)
        st.session_state["pcne_comment"] = ""
        st.session_state["overall_comment"] = ""


# ==============================
# Main
# ==============================

def main():
    st.set_page_config(
        page_title="Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)",
        layout="wide",
    )
    init_session_state()

    st.title("Thesis Evaluation Tool — KHCC (Blinded, 5 Notes)")
    st.caption(
        "Evaluating pharmacist-style recommendations for breast cancer chemotherapy.\n\n"
        "All evaluations are **blinded**; note identity (model vs human) is hidden."
    )

    sidebar_setup()
    evaluate_page()


if __name__ == "__main__":
    main()
