# scripts/generate_ai_notes.py
# KHCC cases → AI clinical pharmacist notes (GPT / Claude / DeepSeek / CADSS)
# One row per Case_ID, columns:
# Case_ID | OpenAI_Note | Claude_Note | DeepSeek_Note | CADSS_Note | Original_Note

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

# -------------------------------------------------
# Load env / configuration
# -------------------------------------------------
load_dotenv()

# Input table (your khcc_cases_200.xlsx)
INPUT_EXCEL  = os.getenv("INPUT_EXCEL", "khcc_cases_200.xlsx")
# None = first sheet
INPUT_SHEET  = os.getenv("INPUT_SHEET", None)

OUT_DIR      = Path(os.getenv("OUT_DIR", "outputs"))
OUT_FILE     = OUT_DIR / os.getenv("OUT_FILE", "KHCC_AI_Notes.xlsx")

# Which sources to run (comma-separated: gpt,claude,deepseek,cadss)
SOURCES      = [
    s.strip()
    for s in os.getenv("SOURCES", "gpt,claude,deepseek,cadss").split(",")
    if s.strip()
]

# pacing & limits
ROW_LIMIT        = int(os.getenv("ROW_LIMIT", "0") or "0")  # 0 = all rows
PER_CASE_SLEEP   = float(os.getenv("PER_CASE_SLEEP", "0.2"))

# temperatures
GEN_TEMPERATURE  = float(os.getenv("GEN_TEMPERATURE", "0.2"))
REV_TEMPERATURE  = float(os.getenv("REV_TEMPERATURE", "0.4"))
SYN_TEMPERATURE  = float(os.getenv("SYN_TEMPERATURE", "0.2"))

# models
GPT_MODEL        = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL     = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_R       = os.getenv("DEEPSEEK_R", "deepseek-reasoner")

# CADSS roles
CADSS_GENERATOR  = os.getenv("CADSS_GENERATOR", "gpt")       # gpt|claude|deepseek
CADSS_REVIEWER   = os.getenv("CADSS_REVIEWER",  "deepseek")  # gpt|claude|deepseek
CADSS_SYNTH      = os.getenv("CADSS_SYNTH",     "claude")    # claude|gpt|deepseek

# -------------------------------------------------
# API clients
# -------------------------------------------------
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

import anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

import httpx
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE    = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")

# -------------------------------------------------
# Prompt configuration
# -------------------------------------------------

# Columns we DO NOT send to the models
EXCLUDED_FOR_PROMPT = {
    "Case_ID",
    "case_id",
    "Note",                  # original pharmacist note
    "Note_Original",
    "Original_Note",
    "Recommendations_Only",
    # any AI outputs if they exist in the file
    "OpenAI_Note",
    "Claude_Note",
    "DeepSeek_Note",
    "CADSS_Note",
}

# Safety limit on extremely long notes
MAX_NOTE_LEN = int(os.getenv("MAX_NOTE_LEN", "2500") or "2500")


# -------------------------------------------------
# Low-level model callers (with retry)
# -------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 4))
def call_gpt_text(prompt: str, temperature: float = 0.2) -> str:
    if not oai:
        raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing)")
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 4))
def call_claude_text(prompt: str, temperature: float = 0.2) -> str:
    if not ac:
        raise RuntimeError("Anthropic client not configured (ANTHROPIC_API_KEY missing)")
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temperature,
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 4))
def call_deepseek_text(prompt: str, model: str, temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek key not configured (DEEPSEEK_API_KEY missing)")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        resp = client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# -------------------------------------------------
# Prompt builders
# -------------------------------------------------
def build_case_prompt(row: pd.Series) -> str:
    """Builds the clinical context for ONE case using all non-excluded columns,
    including Note_Without_Recommendations.
    """
    case_id = row.get("Case_ID") or row.get("case_id") or ""

    lines: list[str] = []
    lines.append("You are a clinical oncology pharmacist at King Hussein Cancer Center (KHCC).")
    lines.append("")
    if case_id:
        lines.append(f"Case ID: {case_id}")
    lines.append(
        "Use the anonymized patient data below to write a single, comprehensive "
        "clinical pharmacist note (assessment + recommendations)."
    )
    lines.append("The note should:")
    lines.append("- Summarize the clinical situation and key problems.")
    lines.append("- Evaluate chemotherapy regimen, doses, labs, comorbidities and co-medications.")
    lines.append("- State clear pharmacist recommendations (dose changes, holds, supportive care, monitoring).")
    lines.append("- Follow KHCC documentation style and NEVER mention that you are an AI model.")
    lines.append("")
    lines.append("=== Structured and semi-structured data ===")

    for col, val in row.items():
        if col in EXCLUDED_FOR_PROMPT:
            continue
        if pd.isna(val):
            continue
        text = str(val).strip()
        if not text:
            continue

        # special handling for the long free-text assessment
        if col == "Note_Without_Recommendations" and len(text) > MAX_NOTE_LEN:
            text = text[:MAX_NOTE_LEN] + "… [truncated]"

        lines.append(f"- {col}: {text}")

    lines.append("")
    lines.append(
        "Now write the clinical pharmacist note in free-text form as you would document in the KHCC system. "
        "Include both assessment and recommendations inside one continuous note."
    )
    return "\n".join(lines)


def safety_gate(note: str, row: pd.Series) -> Tuple[bool, str]:
    """Very small safety sanity-check. Flags obvious red situations but never blocks generation."""
    reasons = []

    # try to read LVEF from any reasonable column name
    lvef_val = None
    for c in ("LVEF_percent", "lvef", "LVEF"):
        if c in row.index and pd.notna(row[c]):
            try:
                lvef_val = float(row[c])
                break
            except Exception:
                continue

    if "trastuzumab" in note.lower() and (lvef_val is None or lvef_val < 50):
        reasons.append("HER2 therapy mentioned with LVEF <50% or missing.")
    if any(w in note.lower() for w in ["double dose", "tripled dose"]):
        reasons.append("Suspicious dose wording (double/triple dose).")

    return (len(reasons) == 0, "; ".join(reasons))


def build_review_prompt(case_prompt: str, draft_note: str) -> str:
    """Prompt for the CADSS reviewer."""
    return f"""Role: Clinical safety reviewer and protocol auditor (KHCC).

Case (summary):
{case_prompt}

Draft clinical pharmacist note:
\"\"\"{draft_note}\"\"\"

Return STRICT JSON with keys:
- "issues": [{{"type":"dose|interaction|safety|monitoring|completeness|protocol","detail":"..."}}]
- "required_changes": ["..."],
- "pcne_summary": {{
    "problems": ["P1.2","P2.2"],
    "causes": ["C2.1","C9.1"],
    "interventions": ["I1.2","I3.1"],
    "outcomes": ["O1","O0"]
  }}

Rules:
- Focus on concrete errors/omissions and KHCC protocol adherence.
- If the draft is safe and complete, issues = [] and required_changes = ["No change"].
- Output JSON only (no explanations outside JSON).
"""


def build_synthesis_prompt(case_prompt: str, draft_note: str, critique_json: str) -> str:
    """Prompt for the CADSS synthesizer."""
    return f"""Role: Pharmacist synthesizer. Produce the FINAL KHCC-compliant clinical pharmacist note.

Case (summary):
{case_prompt}

Initial draft:
\"\"\"{draft_note}\"\"\"

Reviewer critique (JSON):
{critique_json}

Write the final integrated note as free text (assessment + recommendations together).
Apply all valid required_changes. Do NOT include JSON, section titles, or model names. Note only.
"""


def cadss_flow(case_prompt: str) -> Tuple[str, str]:
    """Generator → Reviewer → Synthesizer flow, returning final note + provenance string."""
    # generator
    if CADSS_GENERATOR == "gpt":
        draft = call_gpt_text(case_prompt, GEN_TEMPERATURE)
        gen_model = GPT_MODEL
    elif CADSS_GENERATOR == "claude":
        draft = call_claude_text(case_prompt, GEN_TEMPERATURE)
        gen_model = CLAUDE_MODEL
    else:
        draft = call_deepseek_text(case_prompt, DEEPSEEK_MODEL, GEN_TEMPERATURE)
        gen_model = DEEPSEEK_MODEL

    # reviewer
    rev_prompt = build_review_prompt(case_prompt, draft)
    if CADSS_REVIEWER == "gpt":
        critique = call_gpt_text(rev_prompt, REV_TEMPERATURE)
        rev_model = GPT_MODEL
    elif CADSS_REVIEWER == "claude":
        critique = call_claude_text(rev_prompt, REV_TEMPERATURE)
        rev_model = CLAUDE_MODEL
    else:
        critique = call_deepseek_text(rev_prompt, DEEPSEEK_R, REV_TEMPERATURE)
        rev_model = DEEPSEEK_R

    # synthesizer
    synth_prompt = build_synthesis_prompt(case_prompt, draft, critique)
    if CADSS_SYNTH == "claude":
        final = call_claude_text(synth_prompt, SYN_TEMPERATURE)
        syn_model = CLAUDE_MODEL
    elif CADSS_SYNTH == "gpt":
        final = call_gpt_text(synth_prompt, SYN_TEMPERATURE)
        syn_model = GPT_MODEL
    else:
        final = call_deepseek_text(synth_prompt, DEEPSEEK_MODEL, SYN_TEMPERATURE)
        syn_model = DEEPSEEK_MODEL

    provenance = f"{gen_model}|{rev_model}|{syn_model}"
    return final, provenance


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Read input file
    if INPUT_EXCEL.lower().endswith(".xlsx"):
        if INPUT_SHEET:
            df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
        else:
            df = pd.read_excel(INPUT_EXCEL)
    else:
        df = pd.read_csv(INPUT_EXCEL)

    if ROW_LIMIT and ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    # Make sure we have a lowercase case_id column for convenience
    if "case_id" not in df.columns:
        if "Case_ID" in df.columns:
            df["case_id"] = df["Case_ID"]
        else:
            df["case_id"] = [f"Case {i+1}" for i in range(len(df))]

    results = []

    # 2) Loop over cases
    for _, row in df.iterrows():
        case_id = row.get("Case_ID") or row.get("case_id")

        # Try to recover original pharmacist note from any reasonable column name
        original_note = (
            row.get("Note")
            or row.get("Original_Note")
            or row.get("Note_Original")
            or ""
        )

        out_row = {
            "Case_ID": case_id,
            "OpenAI_Note": "",
            "Claude_Note": "",
            "DeepSeek_Note": "",
            "CADSS_Note": "",
            "Original_Note": original_note,
        }

        case_prompt = build_case_prompt(row)

        # ---- GPT (OpenAI) ----
        if "gpt" in SOURCES:
            try:
                note = call_gpt_text(case_prompt, GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                if ok:
                    out_row["OpenAI_Note"] = note
                else:
                    out_row["OpenAI_Note"] = f"[SAFETY FLAG] {why}\n\n{note}"
            except Exception as e:
                out_row["OpenAI_Note"] = f"[ERROR] {e}"

        # ---- Claude ----
        if "claude" in SOURCES:
            try:
                note = call_claude_text(case_prompt, GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                if ok:
                    out_row["Claude_Note"] = note
                else:
                    out_row["Claude_Note"] = f"[SAFETY FLAG] {why}\n\n{note}"
            except Exception as e:
                out_row["Claude_Note"] = f"[ERROR] {e}"

        # ---- DeepSeek ----
        if "deepseek" in SOURCES:
            try:
                note = call_deepseek_text(case_prompt, DEEPSEEK_MODEL, GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                if ok:
                    out_row["DeepSeek_Note"] = note
                else:
                    out_row["DeepSeek_Note"] = f"[SAFETY FLAG] {why}\n\n{note}"
            except Exception as e:
                out_row["DeepSeek_Note"] = f"[ERROR] {e}"

        # ---- CADSS (multi-agent) ----
        if "cadss" in SOURCES:
            try:
                final_note, provenance = cadss_flow(case_prompt)
                ok, why = safety_gate(final_note, row)
                if ok:
                    out_row["CADSS_Note"] = final_note
                else:
                    out_row["CADSS_Note"] = (
                        f"[SAFETY FLAG] {why} (models: {provenance})\n\n{final_note}"
                    )
            except Exception as e:
                out_row["CADSS_Note"] = f"[ERROR] {e}"

        results.append(out_row)

        # gentle pacing to avoid rate limits
        time.sleep(PER_CASE_SLEEP)

    # 3) Save output as Excel
    out_df = pd.DataFrame(
        results,
        columns=[
            "Case_ID",
            "OpenAI_Note",
            "Claude_Note",
            "DeepSeek_Note",
            "CADSS_Note",
            "Original_Note",
        ],
    )
    out_df.to_excel(OUT_FILE, index=False)
    print(f"Saved {len(out_df)} cases to {OUT_FILE}")


if __name__ == "__main__":
    main()
