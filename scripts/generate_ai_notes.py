# scripts/generate_ai_notes.py
import os, time, json, math
from pathlib import Path
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()

INPUT_EXCEL = os.getenv("INPUT_EXCEL", "synthetic_breast_cancer_1000.xlsx")
SHEET_NAME  = os.getenv("INPUT_SHEET", None)  # or leave None
OUT_DIR     = Path(os.getenv("OUT_DIR", "outputs"))
OUT_FILE    = OUT_DIR / "AI_Recommendations.csv"

# choose which sources to run; change as needed
SOURCES = os.getenv("SOURCES", "gpt,claude,deepseek,cadss").split(",")

# modest token budgets via concise prompts; we keep temps low for determinism
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
REV_TEMPERATURE = float(os.getenv("REV_TEMPERATURE", "0.4"))
SYN_TEMPERATURE = float(os.getenv("SYN_TEMPERATURE", "0.2"))

# -----------------------------
# Model adapters
# -----------------------------
# OpenAI (GPT)
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY)

# Anthropic (Claude)
import anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# DeepSeek (simple HTTPX call to their OpenAI-compatible or REST endpoint)
import httpx
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com")  # keep default unless your account says otherwise

# Pick specific model ids you have access to
GPT_MODEL     = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
DEEPSEEK_MODEL= os.getenv("DEEPSEEK_MODEL", "deepseek-chat")      # for generator/synth
DEEPSEEK_R    = os.getenv("DEEPSEEK_R", "deepseek-reasoner")      # reviewer (reasoning)

# -----------------------------
# Prompt builders (short!)
# -----------------------------
def build_case_prompt(row: pd.Series) -> str:
    # Only the essentials — shorter context = cheaper + more stable
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

# -----------------------------
# API callers with retry
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
def call_gpt_text(prompt: str, temperature: float = 0.2) -> str:
    resp = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
def call_claude_text(prompt: str, temperature: float = 0.2) -> str:
    resp = ac.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        temperature=temperature,
        messages=[{"role":"user","content":prompt}],
    )
    return resp.content[0].text.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
def call_deepseek_text(prompt: str, model: str, temperature: float = 0.2) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "temperature": temperature,
    }
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=120) as client:
        r = client.post("/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# -----------------------------
# Safety post-process (stub)
# -----------------------------
def safety_gate(note: str, row: pd.Series) -> tuple[bool, str]:
    """
    Minimal deterministic checks (expand with your rules):
    - Ensure LVEF present for HER2 therapy mentions.
    - Flag missing ANC/platelets if your dataset includes them.
    - Soft cap extreme dose words (e.g., 'double', 'tripled').
    """
    red_flags = []
    if "trastuzumab" in note.lower() and (pd.isna(row.get("lvef")) or float(row.get("lvef", 0)) < 50):
        red_flags.append("HER2 therapy mentioned but LVEF <50% or missing.")
    if any(w in note.lower() for w in ["double dose", "tripled dose"]):
        red_flags.append("Suspicious dose language.")
    safe = len(red_flags) == 0
    return safe, "; ".join(red_flags)

# -----------------------------
# CADSS orchestration
# -----------------------------
def cadss_flow(case_prompt: str) -> str:
    # G = GPT (coherent generator)
    g_note = call_gpt_text(case_prompt, temperature=GEN_TEMPERATURE)

    # R = DeepSeek Reasoner (cheap reasoning passes) OR Claude if you prefer
    critique = call_deepseek_text(build_review_prompt(case_prompt, g_note),
                                  model=DEEPSEEK_R, temperature=REV_TEMPERATURE)

    # S = Claude (strong instruction following) OR GPT if you prefer
    final_note = call_claude_text(
        build_synthesis_prompt(case_prompt, g_note, critique),
        temperature=SYN_TEMPERATURE
    )
    return final_note

# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_EXCEL, sheet_name=SHEET_NAME) if INPUT_EXCEL.endswith(".xlsx") else pd.read_csv(INPUT_EXCEL)

    records = []
    for i, row in df.iterrows():
        case_id = row.get("case_id", f"CASE-{i+1}")
        prompt = build_case_prompt(row)

        # single-model runs
        if "gpt" in SOURCES:
            note = call_gpt_text(prompt, temperature=GEN_TEMPERATURE)
            ok, why = safety_gate(note, row)
            if ok:
                records.append({"Case ID":case_id, "Source":"ChatGPT-4o", "Recommendation":note})
        if "claude" in SOURCES:
            note = call_claude_text(prompt, temperature=GEN_TEMPERATURE)
            ok, why = safety_gate(note, row)
            if ok:
                records.append({"Case ID":case_id, "Source":"Claude-3.5-Sonnet", "Recommendation":note})
        if "deepseek" in SOURCES:
            note = call_deepseek_text(prompt, model=DEEPSEEK_MODEL, temperature=GEN_TEMPERATURE)
            ok, why = safety_gate(note, row)
            if ok:
                records.append({"Case ID":case_id, "Source":"DeepSeek-V3", "Recommendation":note})

        # CADSS run
        if "cadss" in SOURCES:
            final_note = cadss_flow(prompt)
            ok, why = safety_gate(final_note, row)
            if ok:
                records.append({"Case ID":case_id, "Source":"CADSS", "Recommendation":final_note})

        # gentle pacing to avoid rate limits
        if (i+1) % 25 == 0:
            time.sleep(0.5)

    pd.DataFrame(records).to_csv(OUT_FILE, index=False)
    print(f"Saved {len(records)} recommendations to {OUT_FILE}")

if __name__ == "__main__":
    main()
