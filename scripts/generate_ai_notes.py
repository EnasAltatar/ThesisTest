"""
generate_ai_notes.py — FINAL VERSION FOR KHCC

Produces AI clinical pharmacist recommendations for each KHCC case and saves
them in a long, analysis-ready Excel file.

INPUT (env / defaults)
----------------------
- INPUT_EXCEL: Excel file with KHCC cases (default: "khcc_cases_200.xlsx")
- INPUT_SHEET: Sheet index or name (default: 0)

Required columns in the Excel file (case-insensitive):
- Case_ID           (becomes "case_id")
- Original_Note     (becomes "original_note")

Optional clinical columns (if missing, they are just left blank in the summary):
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
import time
from pathlib import Path

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
# If OUT_FILE isn't provided, put it under OUT_DIR
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


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_gpt(prompt: str, temp: float = 0.2) -> str:
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return r.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_claude(prompt: str, temp: float = 0.2) -> str:
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_deepseek(prompt: str, model: str, temp: float = 0.2) -> str:
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
    """Generate a structured summary for the evaluation app."""
    # row.get() on a pandas Series is safe even if the key doesn't exist
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
You are a clinical oncology pharmacist.

Patient Summary:
{build_patient_summary(row)}

Provide structured pharmacist recommendations with:
1) Dose verification
2) Safety issues
3) Interactions
4) Supportive care
5) Monitoring
6) Final plan
""".strip()


def build_review_prompt(case_prompt: str, draft: str) -> str:
    return f"""
Review the following draft clinical pharmacist recommendation for clinical
accuracy and safety.

Case:
{case_prompt}

Draft:
\"\"\"{draft}\"\"\"

Return STRICT JSON with the keys exactly as:
{{
  "issues": [...],
  "required_changes": [...]
}}
""".strip()


def build_synthesis_prompt(case_prompt: str, draft: str, critique: str) -> str:
    return f"""
You are an expert clinical oncology pharmacist.

Task:
Apply the reviewer feedback and produce a final optimized KHCC-style
clinical pharmacist note.

Case:
{case_prompt}

Draft:
\"\"\"{draft}\"\"\"

Reviewer critique (JSON):
{critique}

Return the FINAL NOTE ONLY, without any explanations or JSON.
""".strip()


# ---------------------------------------------------------------------
# CADSS (generator → reviewer → synthesis)
# ---------------------------------------------------------------------


def cadss_flow(case_prompt: str) -> str:
    # 1) Generator (GPT)
    draft = call_gpt(case_prompt, GEN_TEMPERATURE)

    # 2) Reviewer (DeepSeek Reasoner)
    rev_json = call_deepseek(
        build_review_prompt(case_prompt, draft),
        model=DEEPSEEK_R,
        temp=REV_TEMPERATURE,
    )

    # 3) Synthesizer (Claude)
    final = call_claude(
        build_synthesis_prompt(case_prompt, draft, rev_json),
        SYN_TEMPERATURE,
    )

    return final


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    print(f"Loading KHCC cases from: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)

    # Normalize column names to lowercase (fixes 'Case_ID' vs 'case_id', etc.)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    # Ensure output directory exists
    out_dir_path = Path(OUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    records = []

    for idx, row in df.iterrows():
        try:
            cid = row["case_id"]
        except KeyError:
            raise KeyError(
                "Column 'case_id' not found after lowercasing headers. "
                "Make sure your Excel file has a 'Case_ID' (or similar) column."
            )

        print(f"Processing case_id={cid} (row {idx + 1}/{len(df)})")

        case_prompt = build_case_prompt(row)
        patient_summary = build_patient_summary(row)

        # GPT
        try:
            gpt_note = call_gpt(case_prompt)
        except Exception as e:
            gpt_note = f"[ERROR] {e}"

        # Claude
        try:
            claude_note = call_claude(case_prompt)
        except Exception as e:
            claude_note = f"[ERROR] {e}"

        # DeepSeek
        try:
            ds_note = call_deepseek(case_prompt, model=DEEPSEEK_MODEL)
        except Exception as e:
            ds_note = f"[ERROR] {e}"

        # CADSS
        try:
            cadss_note = cadss_flow(case_prompt)
        except Exception as e:
            cadss_note = f"[ERROR] {e}"

        # Human note (optional, used later for comparison)
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
