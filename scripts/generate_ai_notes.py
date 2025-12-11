"""
generate_ai_notes.py — FINAL VERSION FOR KHCC

Produces AI clinical pharmacist recommendations for each KHCC case and saves
them in an Excel file that will later be transformed into the blinded
evaluation format.

INPUT (env / defaults)
----------------------
- INPUT_EXCEL: Excel file with KHCC cases (default: "khcc_cases_200.xlsx")
- INPUT_SHEET: Sheet index or name (default: 0)

Required columns in the Excel file (case-insensitive):
- Case_ID           (becomes "case_id")
- Note_Original     (becomes "note_original")

Optional clinical columns (if missing, they are left blank in the summary):
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
OUT_FILE = os.getenv("OUT_FILE")  # may be None
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

oai: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
ac: Optional[anthropic.Anthropic] = (
    anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
)

# ---------------------------------------------------------------------
# Helper: scrub any obvious identifiers
# ---------------------------------------------------------------------


IDENTIFIER_PATTERNS = [
    r"(?i)patient\s*id[:#]?\s*\S+",
    r"(?i)mrn[:#]?\s*\S+",
    r"(?i)medical\s*record\s*number[:#]?\s*\S+",
    r"(?i)national\s*id[:#]?\s*\S+",
    r"(?i)hospital\s*number[:#]?\s*\S+",
    r"(?i)dob[:#]?\s*\S+",
]


def scrub_identifiers(text: str) -> str:
    """Remove common identifier patterns just in case the model invents them."""
    if not isinstance(text, str):
        return text

    cleaned = text
    for pat in IDENTIFIER_PATTERNS:
        cleaned = re.sub(pat, "[REDACTED]", cleaned)

    # Also avoid explicit labels
    cleaned = re.sub(r"(?i)(ID|MRN)\s*[:#]", "[REDACTED]:", cleaned)
    return cleaned


# ---------------------------------------------------------------------
# API wrappers with retry
# ---------------------------------------------------------------------


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_gpt(prompt: str, temp: float = 0.2) -> str:
    if oai is None:
        raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing).")
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return r.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_claude(prompt: str, temp: float = 0.2) -> str:
    if ac is None:
        raise RuntimeError("Anthropic client not configured (ANTHROPIC_API_KEY missing).")
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    # Claude returns a list of content blocks
    return r.content[0].text.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_deepseek(prompt: str, model: str, temp: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY missing.")
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
    """Generate a structured, fully anonymized summary."""
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


BASE_GENERATION_INSTRUCTIONS = """
You are a clinical oncology pharmacist at a tertiary cancer center.

Write a KHCC-style chemotherapy verification note based ONLY on the
clinical information provided. The note MUST:

- NOT include ANY patient identifiers (no name, ID, MRN, national number,
  date of birth, hospital number, bed number, or dates).
- Refer to the person only as "the patient" (do not invent demographic details).
- Focus strictly on chemotherapy verification and supportive care.

Structure the note clearly with the following sections:

1) Dose verification and adjustments
2) Safety issues (organ function, toxicity risks)
3) Drug–drug and drug–disease interactions
4) Supportive care
5) Monitoring plan (labs, imaging, clinical follow-up)
6) Final plan / key recommendations

Use concise, professional clinical language.
""".strip()


def build_case_prompt(row: pd.Series) -> str:
    return (
        BASE_GENERATION_INSTRUCTIONS
        + "\n\nPatient Summary:\n"
        + build_patient_summary(row)
    )


def build_review_prompt(case_prompt: str, draft: str) -> str:
    return f"""
You are reviewing a draft chemotherapy verification note written by an AI
clinical pharmacist.

Your task:
- Identify clinical inaccuracies, missing safety checks, or poor reasoning.
- DO NOT add any patient identifiers (no IDs, names, MRNs, dates, etc.).

Case:
{case_prompt}

Draft note:
\"\"\"{draft}\"\"\"

Return STRICT JSON with the keys exactly as:
{{
  "issues": [...],
  "required_changes": [...]
}}
""".strip()


def build_synthesis_prompt(case_prompt: str, draft: str, critique: str) -> str:
    return f"""
You are an expert clinical oncology pharmacist at a tertiary cancer center.

Use the original case information and the reviewer JSON feedback to produce
a FINAL optimized chemotherapy verification note.

Requirements:
- Completely anonymized (NO names, IDs, MRNs, dates, or other identifiers).
- Refer only to "the patient".
- Keep the same structure:

1) Dose verification and adjustments
2) Safety issues (organ function, toxicity risks)
3) Drug–drug and drug–disease interactions
4) Supportive care
5) Monitoring plan
6) Final plan / key recommendations

Case:
{case_prompt}

Original draft:
\"\"\"{draft}\"\"\"

Reviewer critique (JSON):
{critique}

Return the FINAL NOTE ONLY, without any explanations or JSON.
""".strip()


# ---------------------------------------------------------------------
# CADSS (generator → reviewer → synthesis) with fallback
# ---------------------------------------------------------------------


def cadss_flow(case_prompt: str) -> str:
    """Run generator → reviewer → synthesis; fall back to GPT-only if Claude fails."""
    # 1) Generator (GPT)
    draft = call_gpt(case_prompt, GEN_TEMPERATURE)

    # 2) Reviewer (DeepSeek Reasoner)
    rev_json = call_deepseek(
        build_review_prompt(case_prompt, draft),
        model=DEEPSEEK_R,
        temp=REV_TEMPERATURE,
    )

    # 3) Synthesizer (Claude, with GPT fallback)
    try:
        final = call_claude(
            build_synthesis_prompt(case_prompt, draft, rev_json),
            SYN_TEMPERATURE,
        )
        note = final
    except Exception as e:
        # Fall back to GPT synthesis
        fallback_prompt = f"""
Claude synthesis failed with error: {e!s}

Now you, as GPT, must synthesize a FINAL note using the same requirements.

Case:
{case_prompt}

Original draft:
\"\"\"{draft}\"\"\"

Reviewer critique (JSON):
{rev_json}

Remember: NO patient identifiers of any kind.
Return ONLY the final note.
""".strip()
        gpt_final = call_gpt(fallback_prompt, SYN_TEMPERATURE)
        note = "[FALLBACK_FROM_CLAUDE] " + gpt_final

    return scrub_identifiers(note)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    print(f"Loading KHCC cases from: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)

    # Normalize column names to lowercase to handle Case_ID / Note_Original etc.
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "case_id" not in df.columns:
        raise KeyError(
            "Required column 'Case_ID' (case-insensitive) not found in the Excel file."
        )

    if "note_original" not in df.columns:
        raise KeyError(
            "Required column 'Note_Original' (case-insensitive) not found in the Excel file."
        )

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    # Ensure output directory exists
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    records = []

    for idx, row in df.iterrows():
        cid = row["case_id"]
        print(f"Processing case_id={cid} (row {idx + 1}/{len(df)})")

        case_prompt = build_case_prompt(row)
        patient_summary = build_patient_summary(row)

        # GPT
        try:
            gpt_note_raw = call_gpt(case_prompt, GEN_TEMPERATURE)
            gpt_note = scrub_identifiers(gpt_note_raw)
        except Exception as e:
            gpt_note = f"[ERROR_FROM_GPT] {e}"

        # Claude
        try:
            claude_raw = call_claude(case_prompt, GEN_TEMPERATURE)
            claude_note = scrub_identifiers(claude_raw)
        except Exception as e:
            claude_note = f"[ERROR_FROM_CLAUDE] {e}"

        # DeepSeek
        try:
            ds_raw = call_deepseek(case_prompt, model=DEEPSEEK_MODEL, temp=GEN_TEMPERATURE)
            ds_note = scrub_identifiers(ds_raw)
        except Exception as e:
            ds_note = f"[ERROR_FROM_DEEPSEEK] {e}"

        # CADSS
        try:
            cadss_note = cadss_flow(case_prompt)
        except Exception as e:
            cadss_note = f"[ERROR_FROM_CADSS] {e}"

        # Human note (KHCC reference)
        human_note = row.get("note_original", "")

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
