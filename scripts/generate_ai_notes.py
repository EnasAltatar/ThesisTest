"""
generate_ai_notes.py — FINAL KHCC VERSION (with de-identification + Claude fallback)

Produces AI clinical pharmacist recommendations for each KHCC case and saves
them in an Excel file that will later be transformed into the evaluation format.

INPUT (env / defaults)
----------------------
- INPUT_EXCEL: Excel file with KHCC cases (default: "khcc_cases_200.xlsx")
- INPUT_SHEET: Sheet index or name (default: 0)

Required columns in the Excel file (case-insensitive):
- Case_ID            (becomes "case_id")
- Note_Original      (becomes "note_original")

Optional clinical columns (if missing, we just leave them blank in the summary):
- sex, age, diagnosis_subtype, regimen, lvef, crcl,
  ast, alt, tbil, comorbidities, meds

OUTPUT (env / defaults)
-----------------------
- OUT_DIR: directory for outputs (default: "outputs_khcc")
- OUT_FILE: full path to Excel file. If not provided, defaults to
  "<OUT_DIR>/KHCC_AI_Notes.xlsx"

Columns in the output:
- case_id
- patient_summary
- gpt_note
- claude_note
- deepseek_note
- cadss_note
- human_note
"""

import os
import re
import time
from pathlib import Path
from typing import Optional

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
OUT_FILE = os.getenv("OUT_FILE")  # if None, we derive it from OUT_DIR later

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
# IMPORTANT: if Claude keeps returning NotFound, try a more widely available model, e.g.:
#   claude-3-sonnet-20240229
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")

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
# Small helper: de-identify any accidental identifiers in model output
# ---------------------------------------------------------------------
_PATIENT_ID_PATTERNS = [
    r"patient\s*name\s*:\s*.*",
    r"pt\s*name\s*:\s*.*",
    r"name\s*:\s*.*",
    r"patient\s*id\s*:\s*.*",
    r"mrn\s*:\s*.*",
    r"medical\s*record\s*(number|no\.)\s*:\s*.*",
    r"id\s*:\s*.*",
]

_DATE_PATTERN = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"


def deidentify_text(text: str) -> str:
    """Remove typical identifiers like 'Patient Name', IDs, MRN, explicit dates."""
    if not isinstance(text, str):
        return text

    # Remove lines that clearly contain Name / ID / MRN info
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in _PATIENT_ID_PATTERNS):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Strip explicit standalone dates (we still allow relative timing words)
    cleaned = re.sub(_DATE_PATTERN, "[date removed]", cleaned)

    return cleaned.strip()


# ---------------------------------------------------------------------
# API wrappers with retry
# ---------------------------------------------------------------------


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_gpt(prompt: str, temp: float = 0.2) -> str:
    if oai is None:
        raise RuntimeError("OpenAI client not configured (missing OPENAI_API_KEY).")

    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return r.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_claude(prompt: str, temp: float = 0.2) -> str:
    if ac is None:
        raise RuntimeError("Anthropic client not configured (missing ANTHROPIC_API_KEY).")

    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    # anthropic-python returns a list of content blocks
    return "".join(block.text for block in r.content if hasattr(block, "text")).strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_deepseek(prompt: str, model: str, temp: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
    }
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        resp = client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------


def build_patient_summary(row: pd.Series) -> str:
    """
    Short, de-identified summary.

    NOTE:
    Many of your KHCC rows are missing most clinical fields. That’s okay; this
    summary is mainly to give minimal context for the model prompts.
    """
    return (
        f"Sex: {row.get('sex', '') or ''}, "
        f"Age: {row.get('age', '') or ''}. "
        f"Diagnosis: {row.get('diagnosis_subtype', '') or ''}. "
        f"Regimen: {row.get('regimen', '') or ''}. "
        f"LVEF: {row.get('lvef', '') or ''}. "
        f"CrCl: {row.get('crcl', '') or ''}. "
        f"AST/ALT/TBili: {row.get('ast', '') or ''}/"
        f"{row.get('alt', '') or ''}/"
        f"{row.get('tbil', '') or ''}. "
        f"Comorbidities: {row.get('comorbidities', '') or ''}. "
        f"Medications: {row.get('meds', '') or ''}."
    ).strip()


def build_case_prompt(row: pd.Series) -> str:
    """
    Main generation prompt.

    Very explicit that the model must NOT mention identifiers.
    """
    return f"""
You are a clinical oncology pharmacist at a tertiary cancer center.

You will see a de-identified patient summary. Based ONLY on this summary,
write a chemotherapy verification / clinical pharmacist recommendation note.

STRICT DE-IDENTIFICATION RULES:
- Do NOT mention any patient name, initials, medical record number, hospital number,
  national ID, phone number, or bed/room number.
- Do NOT fabricate or include any patient identifiers (for example:
  'Patient Name: ...', 'Patient ID: ...', 'MRN: ...').
- It is acceptable to refer to the person only as "the patient".
- Do NOT invent exact calendar dates. You may use phrases such as "baseline",
  "prior to cycle 1", "every cycle", "at follow-up", etc.

De-identified patient summary:
{build_patient_summary(row)}

Write a structured clinical pharmacist note with clear headings covering:
1) Dose verification and adjustments
2) Safety / organ function / toxicity risks
3) Drug–drug and drug–disease interactions
4) Supportive care
5) Monitoring (labs, imaging, clinical follow-up)
6) Final plan / key recommendations

The style should be concise, professional, and suitable for an oncology
pharmacist note at KHCC.
""".strip()


def build_review_prompt(case_prompt: str, draft: str) -> str:
    return f"""
You are an expert oncology clinical pharmacist reviewing a draft recommendation
note for clinical accuracy and safety.

Case (de-identified):
{case_prompt}

DRAFT NOTE:
\"\"\"{draft}\"\"\"

TASK:
1) Identify any clinical problems, mistakes, missing safety checks, or unclear
   recommendations.
2) Suggest concrete changes to improve dose logic, safety, interactions,
   supportive care, and monitoring.

RESPONSE FORMAT:
Return STRICT JSON with the keys exactly as:
{{
  "issues": ["..."],
  "required_changes": ["..."]
}}

Do NOT include any patient identifiers or dates in your response.
""".strip()


def build_synthesis_prompt(case_prompt: str, draft: str, critique: str) -> str:
    return f"""
You are an expert oncology clinical pharmacist.

You are given:
- A de-identified case description
- A draft pharmacist note
- Reviewer feedback in JSON (issues + required_changes)

Case (de-identified):
{case_prompt}

Draft note:
\"\"\"{draft}\"\"\"

Reviewer feedback JSON:
{critique}

TASK:
Write a FINAL, optimized chemotherapy verification / clinical pharmacist note.

STRICT RULES:
- The note MUST be fully de-identified: NO patient name, ID, MRN, bed number,
  or exact calendar dates.
- Refer to the person only as "the patient".
- Do NOT copy the JSON; apply the feedback as clinical improvements.

Return ONLY the final note text, with clear headings similar to:
1) Dose verification and adjustments
2) Safety ...
... etc.
""".strip()


# ---------------------------------------------------------------------
# CADSS (generator → reviewer → synthesis)
# ---------------------------------------------------------------------


def cadss_flow(case_prompt: str) -> str:
    """
    Multi-step collaborative flow:
      1) Generator: GPT
      2) Reviewer: DeepSeek Reasoner
      3) Synthesizer: Claude (preferred) or GPT fallback
    Always returns a usable note (never raises due to model errors).
    """
    # 1) Generator (GPT)
    draft = call_gpt(case_prompt, GEN_TEMPERATURE)

    # 2) Reviewer (DeepSeek Reasoner)
    try:
        rev_json = call_deepseek(
            build_review_prompt(case_prompt, draft),
            model=DEEPSEEK_R,
            temp=REV_TEMPERATURE,
        )
    except Exception as e:
        # If DeepSeek fails, keep going with a simple text description
        rev_json = f'{{"issues": ["DeepSeek error: {type(e).__name__}"], "required_changes": []}}'

    synth_prompt = build_synthesis_prompt(case_prompt, draft, rev_json)

    # 3) Synthesizer: prefer Claude, fallback to GPT if Claude fails
    try:
        final = call_claude(synth_prompt, SYN_TEMPERATURE)
    except Exception as e:
        # Fallback: GPT synthesizer instead of failing the whole CADSS note
        fallback_note = call_gpt(
            f"Claude (synthesizer) failed with error {type(e).__name__}. "
            f"Please act as the synthesizer and complete this task instead.\n\n{synth_prompt}",
            SYN_TEMPERATURE,
        )
        final = f"[FALLBACK_FROM_CLAUDE] {fallback_note}"

    return deidentify_text(final)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def get_human_note(row: pd.Series) -> str:
    """
    Pull the human KHCC note from the Excel row.

    Your header in Excel is 'Note_Original ' (with a trailing space).
    After lowercasing + strip(), it becomes 'note_original'.
    """
    # try both, just in case
    return (
        row.get("note_original")
        or row.get("original_note")
        or row.get("note__original")  # super defensive
        or ""
    )


def main():
    print(f"Loading KHCC cases from: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)

    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    out_dir_path = Path(OUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if not OUT_FILE:
        out_path = out_dir_path / "KHCC_AI_Notes.xlsx"
    else:
        out_path = Path(OUT_FILE)

    records = []

    for idx, row in df.iterrows():
        if "case_id" not in row.index:
            raise KeyError(
                "Column 'case_id' not found after lowercasing headers. "
                "Make sure your Excel file has a 'Case_ID' column."
            )

        cid = row["case_id"]
        print(f"Processing case_id={cid} (row {idx + 1}/{len(df)})")

        case_prompt = build_case_prompt(row)
        patient_summary = build_patient_summary(row)

        # GPT note
        try:
            gpt_raw = call_gpt(case_prompt, GEN_TEMPERATURE)
            gpt_note = deidentify_text(gpt_raw)
        except Exception as e:
            gpt_note = f"[GPT_ERROR] {type(e).__name__}: {e}"

        # Claude note (standalone)
        try:
            claude_raw = call_claude(case_prompt, GEN_TEMPERATURE)
            claude_note = deidentify_text(claude_raw)
        except Exception as e:
            claude_note = f"[CLAUDE_ERROR] {type(e).__name__}: {e}"

        # DeepSeek note
        try:
            ds_raw = call_deepseek(case_prompt, model=DEEPSEEK_MODEL, temp=GEN_TEMPERATURE)
            deepseek_note = deidentify_text(ds_raw)
        except Exception as e:
            deepseek_note = f"[DEEPSEEK_ERROR] {type(e).__name__}: {e}"

        # CADSS consolidated note
        try:
            cadss_note = cadss_flow(case_prompt)
        except Exception as e:
            cadss_note = f"[CADSS_ERROR] {type(e).__name__}: {e}"

        # Human KHCC note (NOT de-identified here; blinding is handled later
        # when building the evaluation file)
        human_note = get_human_note(row)

        records.append(
            {
                "case_id": cid,
                "patient_summary": patient_summary,
                "gpt_note": gpt_note,
                "claude_note": claude_note,
                "deepseek_note": deepseek_note,
                "cadss_note": cadss_note,
                "human_note": human_note,
            }
        )

        time.sleep(PER_CASE_SLEEP)

    out_df = pd.DataFrame(records)
    out_df.to_excel(out_path, index=False)
    print(f"Saved AI notes to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
