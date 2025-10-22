# scripts/generate_ai_notes.py
import os
import time
from pathlib import Path

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

# =============================
# Load env / global config
# =============================
load_dotenv()

INPUT_EXCEL = os.getenv("INPUT_EXCEL", "synthetic_breast_cancer_1000.xlsx")
SHEET_NAME  = os.getenv("SHEET_NAME", "").strip()  # <-- matches workflow env
OUT_DIR     = Path(os.getenv("OUT_DIR", "outputs"))
OUT_FILE    = OUT_DIR / "AI_Recommendations.csv"

SOURCES = [
    s.strip().lower()
    for s in os.getenv("SOURCES", "gpt,claude,deepseek,cadss").split(",")
    if s.strip()
]

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
REV_TEMPERATURE = float(os.getenv("REV_TEMPERATURE", "0.4"))
SYN_TEMPERATURE = float(os.getenv("SYN_TEMPERATURE", "0.2"))

ROW_LIMIT = int(os.getenv("ROW_LIMIT", "0"))  # 0 => no cap (use small value in CI test runs)

# =============================
# Model clients / adapters
# =============================
# OpenAI (GPT)
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Anthropic (Claude)
import anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# DeepSeek (OpenAI-compatible REST)
import httpx
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")

# Models
GPT_MODEL      = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_R     = os.getenv("DEEPSEEK_R", "deepseek-reasoner")

# =============================
# Prompt builders
# =============================
def build_case_prompt(row: pd.Series) -> str:
    return f"""You are a clinical oncology pharmacist at a tertiary cancer center.

Patient Summary:
- Diagnosis/Subtype: {row.get('diagnosis_subtype')}
- Regimen / Cycle: {row.get('regimen')} / {row.get('cycle')}
- BSA (m^2): {row.get('bsa')}
- Cardiac (LVEF%): {row.get('lvef')}
- Renal (CrCl mL/min): {row.get('crcl')}
- Hepatic (AST/ALT/Tbili): {row.get('ast')}/{row.get('alt')}/{row.get('tbil')}
- Comorbidities: {row.get('comorbidities')}
- Medications: {row.get('meds')}

Task:
Write a pharmacist-style recommendation with:
1) Dose verification/adjustment with numbers and rationale,
2) Drug–drug and drug–disease interactions,
3) Safety considerations specific to breast cancer therapy (cardiac/hepatic/renal),
4) Supportive care & pre-medication,
5) Monitoring & follow-up,
6) Brief rationale.

Return a concise, structured note (no model name, no PHI)."""

def build_review_prompt(case_prompt: str, draft_note: str) -> str:
    return f"""You are reviewing a chemotherapy recommendation for safety, protocol compliance, and PCNE framing.

Case (for context):
{case_prompt}

Draft recommendation to review:
\"\"\"{draft_note}\"\"\"

Your tasks:
- Identify concrete safety issues, dose errors, or missing interactions.
- Map issues to PCNE (Problem, Cause, Intervention, Outcome) at a high level.
- List exact required changes.

Return JSON with keys: issues[], required_changes[], pcne_summary.
"""

def build_synthesis_prompt(case_prompt: str, draft_note: str, critique_json: str) -> str:
    return f"""You are the synthesizer. Produce a final KHCC-compliant pharmacist recommendation.

Case:
{case_prompt}

Initial draft:
\"\"\"{draft_note}\"\"\"

Reviewer critique (JSON):
{critique_json}

Instructions:
- Address each required_change exactly.
- Keep the final note concise, structured, and implementable.
- Do NOT include model names or JSON; return final note as clean text only.
"""

# =============================
# API callers (with retry)
# =============================
@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
def call_gpt_text(prompt: str, temperature: float = 0.2) -> str:
    if not oai:
        raise RuntimeError("OPENAI_API_KEY missing")
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
def call_claude_text(prompt: str, temperature: float = 0.2) -> str:
    if not ac:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
def call_deepseek_text(prompt: str, model: str, temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        r = client.post("/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# =============================
# Minimal safety checks (stub)
# =============================
def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def safety_gate(note: str, row: pd.Series) -> tuple[bool, str]:
    red_flags = []
    lvef = _to_float(row.get("lvef"), None)
    if "trastuzumab" in (note or "").lower() and (lvef is None or lvef < 50):
        red_flags.append("HER2 therapy mentioned but LVEF <50% or missing.")
    if any(w in (note or "").lower() for w in ["double dose", "tripled dose"]):
        red_flags.append("Suspicious dose language.")
    return (len(red_flags) == 0), "; ".join(red_flags)

# =============================
# CADSS orchestration
# =============================
def cadss_flow(case_prompt: str) -> str:
    g_note   = call_gpt_text(case_prompt, temperature=GEN_TEMPERATURE)
    critique = call_deepseek_text(build_review_prompt(case_prompt, g_note),
                                  model=DEEPSEEK_R, temperature=REV_TEMPERATURE)
    final    = call_claude_text(build_synthesis_prompt(case_prompt, g_note, critique),
                                temperature=SYN_TEMPERATURE)
    return final

# =============================
# Data loading (robust sheet handling)
# =============================
def load_dataframe() -> pd.DataFrame:
    if INPUT_EXCEL.lower().endswith(".xlsx"):
        loaded = pd.read_excel(INPUT_EXCEL, sheet_name=SHEET_NAME if SHEET_NAME else None)
        if isinstance(loaded, dict):  # multiple sheets returned
            chosen = SHEET_NAME if (SHEET_NAME and SHEET_NAME in loaded) else next(iter(loaded))
            print(f"[info] Using sheet: {chosen}")
            df = loaded[chosen]
        else:
            df = loaded
    else:
        df = pd.read_csv(INPUT_EXCEL)

    if ROW_LIMIT and ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    return df

# =============================
# Main
# =============================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    print(f"[info] Loaded {len(df)} rows; columns: {list(df.columns)}")

    # --- HOT PATCH: normalize df if a dict somehow slipped through ---
    if isinstance(df, dict):
        key = SHEET_NAME if (SHEET_NAME and SHEET_NAME in df) else next(iter(df))
        print(f"[info] HOT-PATCH selected sheet: {key}")
        df = df[key]
    # -----------------------------------------------------------------

    records = []
    for i, row in df.iterrows():
        case_id = row.get("case_id", f"CASE-{i+1}")
        prompt  = build_case_prompt(row)

        # single-model runs
        if "gpt" in SOURCES:
            try:
                note = call_gpt_text(prompt, temperature=GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                if ok:
                    records.append({"Case ID": case_id, "Source": "ChatGPT-4o", "Recommendation": note})
            except Exception as e:
                print(f"[warn] GPT failed on {case_id}: {e}")

        if "claude" in SOURCES:
            try:
                note = call_claude_text(prompt, temperature=GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                if ok:
                    records.append({"Case ID": case_id, "Source": "Claude-3.5-Sonnet", "Recommendation": note})
            except Exception as e:
                print(f"[warn] Claude failed on {case_id}: {e}")

        if "deepseek" in SOURCES:
            try:
                note = call_deepseek_text(prompt, model=DEEPSEEK_MODEL, temperature=GEN_TEMPERATURE)
                ok, why = safety_gate(note, row)
                if ok:
                    records.append({"Case ID": case_id, "Source": "DeepSeek-V3", "Recommendation": note})
            except Exception as e:
                print(f"[warn] DeepSeek failed on {case_id}: {e}")

        # CADSS (collaborative) run
        if "cadss" in SOURCES:
            try:
                final_note = cadss_flow(prompt)
                ok, why = safety_gate(final_note, row)
                if ok:
                    records.append({"Case ID": case_id, "Source": "CADSS", "Recommendation": final_note})
            except Exception as e:
                print(f"[warn] CADSS failed on {case_id}: {e}")

        # gentle pacing to avoid rate limits
        if (i + 1) % 25 == 0:
            time.sleep(0.5)

    pd.DataFrame(records).to_csv(OUT_FILE, index=False)
    print(f"Saved {len(records)} recommendations to {OUT_FILE}")

if __name__ == "__main__":
    main()
