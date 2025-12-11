"""
generate_ai_notes.py — FINAL VERSION FOR KHCC (de-identified, robust Claude/CADSS)

Produces AI clinical pharmacist recommendations for each KHCC case and saves
them in an analysis-ready Excel file.

Required input columns (case-insensitive):
- Case_ID        -> case_id
- Original_Note  -> original_note (optional in prompts, used only as reference)

Optional clinical columns:
- sex, age, diagnosis_subtype, regimen, lvef, crcl,
  ast, alt, tbil, comorbidities, meds
"""

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------
load_dotenv()

INPUT_EXCEL = os.getenv("INPUT_EXCEL", "khcc_cases_200.xlsx")
INPUT_SHEET = os.getenv("INPUT_SHEET", 0)

OUT_DIR = os.getenv("OUT_DIR", "outputs_khcc")
OUT_FILE = os.getenv("OUT_FILE")
if not OUT_FILE:
    OUT_FILE = str(Path(OUT_DIR) / "KHCC_AI_Notes.xlsx")

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_R = os.getenv("DEEPSEEK_R", "deepseek-reasoner")

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
REV_TEMPERATURE = float(os.getenv("REV_TEMPERATURE", "0.3"))
SYN_TEMPERATURE = float(os.getenv("SYN_TEMPERATURE", "0.2"))

PER_CASE_SLEEP = float(os.getenv("PER_CASE_SLEEP", "0.3"))
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "0"))

# ---------------------------------------------------------------------
# SDK clients
# ---------------------------------------------------------------------
from openai import OpenAI
import anthropic
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = "https://api.deepseek.com"

oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ---------------------------------------------------------------------
# API wrappers with retry
# ---------------------------------------------------------------------
def _ensure_openai() -> None:
    if oai is None:
        raise RuntimeError(
        "OpenAI client not initialized. Check OPENAI_API_KEY in your GitHub secrets / .env."
        )


def _ensure_claude() -> None:
    if ac is None:
        raise RuntimeError(
        "Anthropic client not initialized. Check ANTHROPIC_API_KEY (CLAUDE_API_KEY secret)."
        )


def _ensure_deepseek() -> None:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
        "DeepSeek API key missing. Check DEEPSEEK_API_KEY in your GitHub secrets / .env."
        )


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_gpt(prompt: str, temp: float = 0.2) -> str:
    _ensure_openai()
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return r.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_claude(prompt: str, temp: float = 0.2) -> str:
    _ensure_claude()
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_deepseek(prompt: str, model: str, temp: float = 0.2) -> str:
    _ensure_deepseek()
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
    }
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        resp = client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------
# PROMPTS  (all de-identified)
# ---------------------------------------------------------------------
DEIDENT_INSTRUCTION = (
    "Very important: DO NOT include any patient identifiers such as name, "
    "MRN, hospital number, national ID, case ID, or date of birth. "
    "Refer to the person only as 'the patient'."
)


def build_patient_summary(row: pd.Series) -> str:
    """Generate a structured, de-identified summary for the models."""
    return (
        f"Sex: {row.get('sex', '')}, Age: {row.get('age', '')}. "
        f"Diagnosis: {row.get('diagnosis_subtype', '')}. "
        f"Regimen: {row.get('regimen', '')}. "
        f"LVEF: {row.get('lvef', '')}. "
        f"CrCl: {row.get('crcl', '')}. "
        f"AST/ALT/TBili: {row.get('ast', '')}/"
        f"{row.get('alt', '')}/"
        f"{row.get('tbil', '')}. "
        f"Comorbidities: {row.get('comorbidities', '')}. "
        f"Medications: {row.get('meds', '')}."
    )


def build_case_prompt(row: pd.Series) -> str:
    return f"""
You are a clinical oncology pharmacist at a tertiary cancer center.

Patient Summary (already de-identified):
{build_patient_summary(row)}

{DEIDENT_INSTRUCTION}

Provide a concise, structured pharmacist recommendation that covers:
1) Dose verification and adjustments
2) Safety issues (organ function, toxicity risks)
3) Drug–drug and drug–disease interactions
4) Supportive care (e.g., antiemetics, growth factors, adjuncts)
5) Monitoring (labs, imaging, clinical follow-up)
6) Final plan / key recommendations

Write the note as if documenting in a KHCC chemotherapy verification note.
""".strip()


def build_review_prompt(case_prompt: str, draft: str) -> str:
    return f"""
You are reviewing a clinical oncology pharmacist note for accuracy and safety.

{DEIDENT_INSTRUCTION}

Case (context):
{case_prompt}

Draft note:
\"\"\"{draft}\"\"\"

Identify any problems and required changes. Return STRICT JSON with keys:
{{
  "issues": [
    "short description of each problem..."
  ],
  "required_changes": [
    "what must be changed to make the note clinically correct and safe..."
  ]
}}
""".strip()


def build_synthesis_prompt(case_prompt: str, draft: str, critique: str) -> str:
    return f"""
You are an expert clinical oncology pharmacist.

Task:
Use the reviewer feedback to produce a FINAL optimized KHCC-style
clinical pharmacist note for chemotherapy verification.

{DEIDENT_INSTRUCTION}

Case (context):
{case_prompt}

Original draft:
\"\"\"{draft}\"\"\"

Reviewer critique (JSON):
{critique}

Return the FINAL NOTE ONLY, in clear sections, without any explanations or JSON.
""".strip()


# ---------------------------------------------------------------------
# CADSS (generator → reviewer → synthesis) with fallbacks
# ---------------------------------------------------------------------
def cadss_flow(case_prompt: str) -> str:
    """
    Multi-step CADSS flow:
    1) GPT generates a draft.
    2) DeepSeek reviewer creates JSON critique.
    3) Claude synthesizes final note.
    If reviewer or Claude fails, we fall back gracefully.
    """

    # 1) Generator (GPT)
    draft = call_gpt(case_prompt, GEN_TEMPERATURE)

    # 2) Reviewer (DeepSeek Reasoner) with safe fallback JSON
    try:
        rev_json = call_deepseek(
            build_review_prompt(case_prompt, draft),
            model=DEEPSEEK_R,
            temp=REV_TEMPERATURE,
        )
    except Exception as e:
        rev_json = (
            '{'
            f'"issues": ["Reviewer model error: {str(e)[:120]}"], '
            '"required_changes": []'
            '}'
        )

    # 3) Synthesizer (Claude) with GPT fallback if Claude fails
    try:
        final = call_claude(
            build_synthesis_prompt(case_prompt, draft, rev_json),
            SYN_TEMPERATURE,
        )
        return final
    except Exception as e:
        # Fallback to GPT synthesizer so we still get a usable CADSS note
        fallback_note = call_gpt(
            build_synthesis_prompt(case_prompt, draft, rev_json),
            SYN_TEMPERATURE,
        )
        return (
            "[FALLBACK_FROM_CLAUDE] "
            "Claude synthesis failed; note synthesized by GPT instead.\n\n"
            + fallback_note
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    print(f"Loading KHCC cases from: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)

    # Normalize column names to lowercase
    df.columns = [str(c).strip().lower() for c in df.columns]

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    # Ensure output directory exists
    out_dir_path = Path(OUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        try:
            cid = row["case_id"]
        except KeyError:
            raise KeyError(
                "Column 'case_id' not found after lowercasing headers. "
                "Make sure your Excel file has a 'Case_ID' column."
            )

        print(f"Processing case_id={cid} (row {idx + 1}/{len(df)})")

        case_prompt = build_case_prompt(row)
        patient_summary = build_patient_summary(row)

        # GPT note
        try:
            gpt_note = call_gpt(case_prompt)
        except Exception as e:
            gpt_note = f"[ERROR_FROM_GPT] {e}"

        # Claude note
        try:
            claude_note = call_claude(case_prompt)
        except Exception as e:
            claude_note = f"[ERROR_FROM_CLAUDE] {e}"

        # DeepSeek note
        try:
            ds_note = call_deepseek(case_prompt, model=DEEPSEEK_MODEL)
        except Exception as e:
            ds_note = f"[ERROR_FROM_DEEPSEEK] {e}"

        # CADSS (multi-model) note
        try:
            cadss_note = cadss_flow(case_prompt)
        except Exception as e:
            cadss_note = f"[ERROR_FROM_CADSS] {e}"

        # Human reference (not used in prompts)
        human_note = row.get("original_note", "")

        records.append(
            {
                "case_id": cid,
                "patient_summary": patient_summary,
                "gpt_note": gpt_note,
                "claude_note": claude_note,
                "deepseek_note": ds_note,
                "cadss_note": cadss_note,
                "human_note": human_note,
            }
        )

        time.sleep(PER_CASE_SLEEP)

    out_df = pd.DataFrame(records)
    out_path = Path(OUT_FILE)
    out_df.to_excel(out_path, index=False)
    print(f"Saved AI notes to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
