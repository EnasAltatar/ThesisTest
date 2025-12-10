"""
generate_ai_notes.py — FINAL VERSION FOR KHCC
Produces AI recommendations for each case in long-ready format.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# -----------------------------
# Load env vars
# -----------------------------
load_dotenv()

INPUT_EXCEL = os.getenv("INPUT_EXCEL", "khcc_cases_200.xlsx")
INPUT_SHEET = os.getenv("INPUT_SHEET", 0)
OUT_FILE = os.getenv("OUT_FILE", "KHCC_AI_Notes.xlsx")

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_R = os.getenv("DEEPSEEK_R", "deepseek-reasoner")

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
REV_TEMPERATURE = float(os.getenv("REV_TEMPERATURE", "0.3"))
SYN_TEMPERATURE = float(os.getenv("SYN_TEMPERATURE", "0.2"))

PER_CASE_SLEEP = float(os.getenv("PER_CASE_SLEEP", "0.3"))
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "0"))

SOURCES = ["gpt", "claude", "deepseek", "cadss"]

# -----------------------------
# SDK clients
# -----------------------------
from openai import OpenAI
import anthropic
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = "https://api.deepseek.com"

oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# -----------------------------
# API wrappers with retry
# -----------------------------

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_gpt(prompt, temp=0.2):
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return r.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_claude(prompt, temp=0.2):
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_deepseek(prompt, model, temp=0.2):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temp}
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        resp = client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

# -----------------------------
# PROMPTS
# -----------------------------

def build_patient_summary(row):
    """Generate a structured summary for evaluation app."""
    return (
        f"Sex: {row.get('sex','')}, Age: {row.get('age','')}. "
        f"Diagnosis: {row.get('diagnosis_subtype','')}. "
        f"Regimen: {row.get('regimen','')}. "
        f"LVEF: {row.get('lvef','')}. "
        f"CrCl: {row.get('crcl','')}. "
        f"AST/ALT/TBili: {row.get('ast','')}/{row.get('alt','')}/{row.get('tbil','')}. "
        f"Comorbidities: {row.get('comorbidities','')}. "
        f"Medications: {row.get('meds','')}."
    )

def build_case_prompt(row):
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
"""

def build_review_prompt(case_prompt, draft):
    return f"""
Review the draft for clinical accuracy.

Case:
{case_prompt}

Draft:
\"\"\"{draft}\"\"\"

Return JSON:
{{
 "issues": [...],
 "required_changes": [...]
}}
"""

def build_synthesis_prompt(case_prompt, draft, critique):
    return f"""
Apply reviewer feedback and produce final optimized KHCC-style note.

Case:
{case_prompt}

Draft:
\"\"\"{draft}\"\"\"

Reviewer critique:
{critique}

Return final note only.
"""

# -----------------------------
# CADSS (generator → reviewer → synthesis)
# -----------------------------

def cadss_flow(case_prompt):
    # generator
    draft = call_gpt(case_prompt, GEN_TEMPERATURE)

    # reviewer
    rev_json = call_deepseek(
        build_review_prompt(case_prompt, draft),
        model=DEEPSEEK_R,
        temp=REV_TEMPERATURE
    )

    # synthesizer
    final = call_claude(
        build_synthesis_prompt(case_prompt, draft, rev_json),
        SYN_TEMPERATURE
    )

    return final

# -----------------------------
# MAIN
# -----------------------------

def main():
    print(f"Loading KHCC cases: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    out = []

    for _, row in df.iterrows():
        cid = row["case_id"]
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

        out.append({
            "case_id": cid,
            "patient_summary": patient_summary,
            "gpt_note": gpt_note,
            "claude_note": claude_note,
            "deepseek_note": ds_note,
            "cadss_note": cadss_note,
            "human_note": row.get("Original_Note", "")
        })

        time.sleep(PER_CASE_SLEEP)

    pd.DataFrame(out).to_excel(OUT_FILE, index=False)
    print(f"Saved AI notes to {OUT_FILE}")


if __name__ == "__main__":
    main()
