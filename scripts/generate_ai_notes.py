# scripts/generate_ai_notes.py
# v2 — tidy outputs, stronger prompts, robust per-model logging, configurable CADSS roles

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

# -----------------------------
# Load env/config
# -----------------------------
load_dotenv()

INPUT_EXCEL  = os.getenv("INPUT_EXCEL", "synthetic_breast_cancer_1000.xlsx")
INPUT_SHEET  = os.getenv("INPUT_SHEET", "Sheet1")   # << you said your sheet is Sheet1
OUT_DIR      = Path(os.getenv("OUT_DIR", "outputs"))
OUT_FILE     = OUT_DIR / "AI_Recommendations.csv"
SOURCES      = [s.strip() for s in os.getenv("SOURCES", "gpt,claude,deepseek,cadss").split(",") if s.strip()]

# pacing & limits
ROW_LIMIT        = int(os.getenv("ROW_LIMIT", "0") or "0")  # 0 = all rows
PER_CASE_SLEEP   = float(os.getenv("PER_CASE_SLEEP", "0.2"))

# temps
GEN_TEMPERATURE  = float(os.getenv("GEN_TEMPERATURE", "0.2"))
REV_TEMPERATURE  = float(os.getenv("REV_TEMPERATURE", "0.4"))
SYN_TEMPERATURE  = float(os.getenv("SYN_TEMPERATURE", "0.2"))

# models
GPT_MODEL        = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL     = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_R       = os.getenv("DEEPSEEK_R", "deepseek-reasoner")

# CADSS role routing (change in workflow env to A/B test)
CADSS_GENERATOR  = os.getenv("CADSS_GENERATOR", "gpt")       # gpt|claude|deepseek
CADSS_REVIEWER   = os.getenv("CADSS_REVIEWER",  "deepseek")  # gpt|claude|deepseek
CADSS_SYNTH      = os.getenv("CADSS_SYNTH",     "claude")    # claude|gpt|deepseek

# -----------------------------
# SDK clients
# -----------------------------
# OpenAI
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Anthropic
import anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# DeepSeek (OpenAI-compatible endpoint)
import httpx
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE    = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")


# -----------------------------
# Output schema (tidy; one row per case×source)
# -----------------------------
OUTPUT_COLS = [
    "case_id",
    "diagnosis_subtype",
    "regimen",
    "source_key",          # gpt|claude|deepseek|cadss
    "stage",               # generator|reviewer|synthesizer
    "model_id",
    "temperature",
    "status",              # ok|error
    "error_message",
    "recommendation_text", # final text or draft
    "review_json",         # JSON string only for reviewer rows
    "created_at_utc",
]

def append_record(records: List[dict], row: pd.Series, *,
                  source_key: str,
                  stage: str,
                  model_id: str,
                  temperature: float,
                  status: str = "ok",
                  error_message: str = "",
                  recommendation_text: str = "",
                  review_json: str = "") -> None:
    records.append({
        "case_id": row.get("case_id"),
        "diagnosis_subtype": row.get("diagnosis_subtype"),
        "regimen": row.get("regimen"),
        "source_key": source_key,
        "stage": stage,
        "model_id": model_id,
        "temperature": temperature,
        "status": status,
        "error_message": (error_message or "")[:500],
        "recommendation_text": recommendation_text,
        "review_json": review_json,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })


# -----------------------------
# Stronger prompts
# -----------------------------
def build_case_prompt(row: pd.Series) -> str:
    return f"""You are a clinical oncology pharmacist at a tertiary cancer center (KHCC).

## Patient Summary
- Diagnosis/Subtype: {row.get('diagnosis_subtype')}
- Regimen / Cycle: {row.get('regimen')} / {row.get('cycle')}
- BSA (m²): {row.get('bsa')}
- Cardiac (LVEF%): {row.get('lvef')}
- Renal (CrCl, mL/min): {row.get('crcl')}
- Hepatic (AST/ALT/Tbili): {row.get('ast')}/{row.get('alt')}/{row.get('tbil')}
- Comorbidities: {row.get('comorbidities')}
- Current Medications: {row.get('meds')}

## Task
Return a pharmacist recommendation **in the exact layout below**. Be specific, numeric, and concise. If data are missing, state the assumption explicitly and proceed with a safe, conservative plan.

### 1) Dose Verification / Adjustment
- For **each drug** in the regimen, provide a row with:
  Drug | Ordered dose (mg/m² or mg) | Calculated dose (mg) | Basis (BSA/renal/hepatic) | Adjustment (Y/N) | Rationale

### 2) Interaction & Contraindication Check
- Bullet list: each item = Interaction/Issue | Clinical relevance | Action.

### 3) Safety Considerations (Breast Cancer–specific)
- Cardiac risks (anthracyclines, HER2 agents), hepatic/renal flags, myelosuppression thresholds. Reference **LVEF**, **CrCl**, **AST/ALT/Tbili** against safe use thresholds.

### 4) Supportive Care / Premedication
- Antiemetic regimen (by emetogenicity), growth factors (if indicated), HER2 infusion premeds (if needed), cardioprotection (if indicated).

### 5) Monitoring & Follow-Up
- Labs (timing/thresholds), ECHO/MUGA schedule (if HER2/anthracycline), toxicity monitoring, hold/reduce rules.

### 6) Final Plan (Brief)
- 3–6 lines summarizing go/hold, any dose changes, key monitoring, patient counseling points.

**Constraints**
- No PHI, no model names, no references list.
- Follow KHCC-style clinical documentation and safe practice defaults.
"""

def build_review_prompt(case_prompt: str, draft_note: str) -> str:
    return f"""Role: Clinical safety reviewer and protocol auditor (KHCC).

Case:
{case_prompt}

Draft to review:
\"\"\"{draft_note}\"\"\"

Return **strict JSON** with keys exactly:
- "issues": [{{"type":"dose|interaction|safety|monitoring|completeness|protocol","detail":"..."}}]
- "required_changes": ["..."],  // imperative, one change per bullet
- "pcne_summary": {{
    "problems":["P1.2","P2.2",...],
    "causes":["C2.1","C9.1",...],
    "interventions":["I1.2","I3.1",...],
    "outcomes":["O1","O0",...]
  }}

Rules:
- Focus on **concrete** errors/omissions and KHCC protocol adherence.
- If the draft is safe and complete, "issues" should be empty and "required_changes"=["No change"].
- Output JSON only.
"""

def build_synthesis_prompt(case_prompt: str, draft_note: str, critique_json: str) -> str:
    return f"""Role: Pharmacist synthesizer. Produce the **final KHCC-compliant note**.

Case:
{case_prompt}

Initial draft:
\"\"\"{draft_note}\"\"\"

Reviewer critique (JSON):
{critique_json}

Apply **every** required_change if valid. Keep the exact six sections and concise style from the generator prompt. Do **not** include JSON or model names. Return the final note only.
"""


# -----------------------------
# API callers with retry
# -----------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 4))
def call_gpt_text(prompt: str, temperature: float = 0.2) -> str:
    if not oai:
        raise RuntimeError("OpenAI client not configured")
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 4))
def call_claude_text(prompt: str, temperature: float = 0.2) -> str:
    if not ac:
        raise RuntimeError("Anthropic client not configured")
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temperature,
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 4))
def call_deepseek_text(prompt: str, model: str, temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek key not configured")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        resp = client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# -----------------------------
# Minimal safety post-process
# -----------------------------
def safety_gate(note: str, row: pd.Series) -> Tuple[bool, str]:
    red = []
    try:
        lvef = float(row.get("lvef")) if str(row.get("lvef")).strip() not in ("", "None", "nan") else None
    except Exception:
        lvef = None
    if "trastuzumab" in note.lower() and (lvef is None or lvef < 50):
        red.append("HER2 therapy mentioned with LVEF <50% or missing value.")
    if any(w in note.lower() for w in ["double dose", "tripled dose"]):
        red.append("Suspicious dose language.")
    return (len(red) == 0, "; ".join(red))


# -----------------------------
# CADSS orchestration (role-configurable)
# -----------------------------
def cadss_flow(case_prompt: str) -> Tuple[str, str]:
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

    # return final + a tiny provenance string
    return final, f"{gen_model}|{rev_model}|{syn_model}"


# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # read input table
    if INPUT_EXCEL.lower().endswith(".xlsx"):
        df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    else:
        df = pd.read_csv(INPUT_EXCEL)

    if ROW_LIMIT and ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    # make sure missing cols don't crash prompts
    needed = ["case_id","diagnosis_subtype","regimen","cycle","bsa","lvef","crcl","ast","alt","tbil","comorbidities","meds"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""

    records: List[dict] = []

    for i, row in df.iterrows():
        case_prompt = build_case_prompt(row)

        # GPT generator
        if "gpt" in SOURCES:
            try:
                note = call_gpt_text(case_prompt, temperature=GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                append_record(records, row,
                    source_key="gpt", stage="generator", model_id=GPT_MODEL,
                    temperature=GEN_TEMPERATURE,
                    status="ok" if ok else "error",
                    error_message="" if ok else f"SafetyGate: {why}",
                    recommendation_text=note
                )
            except Exception as e:
                append_record(records, row,
                    source_key="gpt", stage="generator", model_id=GPT_MODEL,
                    temperature=GEN_TEMPERATURE,
                    status="error", error_message=str(e)
                )

        # Claude generator
        if "claude" in SOURCES:
            try:
                note = call_claude_text(case_prompt, temperature=GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                append_record(records, row,
                    source_key="claude", stage="generator", model_id=CLAUDE_MODEL,
                    temperature=GEN_TEMPERATURE,
                    status="ok" if ok else "error",
                    error_message="" if ok else f"SafetyGate: {why}",
                    recommendation_text=note
                )
            except Exception as e:
                append_record(records, row,
                    source_key="claude", stage="generator", model_id=CLAUDE_MODEL,
                    temperature=GEN_TEMPERATURE,
                    status="error", error_message=str(e)
                )

        # DeepSeek generator
        if "deepseek" in SOURCES:
            try:
                note = call_deepseek_text(case_prompt, model=DEEPSEEK_MODEL, temperature=GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                append_record(records, row,
                    source_key="deepseek", stage="generator", model_id=DEEPSEEK_MODEL,
                    temperature=GEN_TEMPERATURE,
                    status="ok" if ok else "error",
                    error_message="" if ok else f"SafetyGate: {why}",
                    recommendation_text=note
                )
            except Exception as e:
                append_record(records, row,
                    source_key="deepseek", stage="generator", model_id=DEEPSEEK_MODEL,
                    temperature=GEN_TEMPERATURE,
                    status="error", error_message=str(e)
                )

        # CADSS pipeline (generator → reviewer → synthesizer)
        if "cadss" in SOURCES:
            try:
                final_note, provenance = cadss_flow(case_prompt)
                ok, why = safety_gate(final_note, row)
                append_record(records, row,
                    source_key="cadss", stage="synthesizer", model_id=provenance,
                    temperature=SYN_TEMPERATURE,
                    status="ok" if ok else "error",
                    error_message="" if ok else f"SafetyGate: {why}",
                    recommendation_text=final_note
                )
            except Exception as e:
                append_record(records, row,
                    source_key="cadss", stage="synthesizer",
                    model_id=f"{CADSS_GENERATOR}|{CADSS_REVIEWER}|{CADSS_SYNTH}",
                    temperature=SYN_TEMPERATURE,
                    status="error", error_message=str(e)
                )

        # gentle pacing to avoid rate limits
        time.sleep(PER_CASE_SLEEP)

    # write tidy CSV in guaranteed column order
    pd.DataFrame(records, columns=OUTPUT_COLS).to_csv(OUT_FILE, index=False)
    print(f"Saved {len(records)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
