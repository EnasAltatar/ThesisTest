"""
generate_ai_notes.py — FIXED KHCC VERSION

Produces AI clinical pharmacist recommendations for each KHCC case.

KEY FEATURES:
- Robust de-identification of PHI before sending to any LLM
- AI-generated patient summary from human notes using GPT
- Multi-model note generation (GPT, Claude, DeepSeek)
- CADSS pipeline with fallback handling

INPUT (env / defaults)
----------------------
- INPUT_EXCEL: Excel file with KHCC cases (default: "khcc_cases_200.xlsx")
- INPUT_SHEET: Sheet index or name (default: 0)

Required columns in the Excel file (case-insensitive):
- Case_ID            (becomes "case_id")
- Note_Original      (becomes "note_original")

OUTPUT (env / defaults)
-----------------------
- OUT_DIR: directory for outputs (default: "outputs_khcc")
- OUT_FILE: full path to Excel file. If not provided, defaults to
  "<OUT_DIR>/KHCC_AI_Notes.xlsx"

Columns in the output:
- case_id
- patient_summary (AI-generated, de-identified)
- gpt_note
- claude_note
- deepseek_note
- cadss_note
- human_note (original, not scrubbed)
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

# FIXED: Use a valid current Claude model
# Options: "claude-3-5-sonnet-20241022", "claude-sonnet-4-20250514", "claude-3-5-sonnet-latest"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

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
# DE-IDENTIFICATION: Scrub PHI before sending to LLMs
# ---------------------------------------------------------------------

def scrub_identifiers(text: str) -> str:
    """
    Remove patient identifiers (names, IDs, MRNs, ages, dates) from text
    BEFORE sending to any LLM.
    
    This is the PRIMARY de-identification function for INPUT text.
    """
    if not isinstance(text, str):
        return ""
    
    scrubbed = text
    
    # 1) Remove explicit name/ID lines (common in clinical notes)
    # Pattern: "Patient Name: John Doe" or "Patient ID: 12345"
    scrubbed = re.sub(
        r'Patient\s+Name\s*:\s*[^\n]+',
        'Patient Name: [REDACTED_NAME]',
        scrubbed,
        flags=re.IGNORECASE
    )
    scrubbed = re.sub(
        r'Patient\s+ID\s*:\s*[^\n]+',
        'Patient ID: [REDACTED_ID]',
        scrubbed,
        flags=re.IGNORECASE
    )
    scrubbed = re.sub(
        r'MRN\s*:\s*[^\n]+',
        'MRN: [REDACTED_ID]',
        scrubbed,
        flags=re.IGNORECASE
    )
    
    # 2) Remove explicit ages
    # Pattern: "Patient Age: 54yr" or "Age: 54"
    scrubbed = re.sub(
        r'(Patient\s+)?Age\s*:\s*\d+\s*(yr|years?|y\.o\.|yo)?',
        'Age: [REDACTED_AGE]',
        scrubbed,
        flags=re.IGNORECASE
    )
    
    # 3) Remove dates in various formats
    # MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.
    scrubbed = re.sub(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        '[REDACTED_DATE]',
        scrubbed
    )
    scrubbed = re.sub(
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        '[REDACTED_DATE]',
        scrubbed
    )
    
    # 4) Remove timestamps (HH:MM format)
    scrubbed = re.sub(
        r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
        '[REDACTED_TIME]',
        scrubbed
    )
    
    # 5) Remove specific identifying numbers (MRN-like patterns)
    # This is conservative; adjust based on your ID formats
    scrubbed = re.sub(
        r'\b[A-Z]{2,}\d{5,}\b',
        '[REDACTED_ID]',
        scrubbed
    )
    
    # 6) Remove institution-specific identifiers if you know the pattern
    # Example: KHCC followed by numbers
    scrubbed = re.sub(
        r'\bKHCC[-\s]?\d+\b',
        '[REDACTED_INSTITUTION_ID]',
        scrubbed,
        flags=re.IGNORECASE
    )
    
    return scrubbed.strip()


def deidentify_model_output(text: str) -> str:
    """
    Additional light cleaning of model OUTPUT to catch any hallucinated
    identifiers that the model might have generated.
    
    This is SECONDARY - the primary defense is scrubbing the input.
    """
    if not isinstance(text, str):
        return text
    
    # Remove lines that contain explicit identifier fields the model might generate
    patterns = [
        r"patient\s*name\s*:\s*\S+",
        r"pt\s*name\s*:\s*\S+",
        r"patient\s*id\s*:\s*\S+",
        r"mrn\s*:\s*\S+",
        r"medical\s*record\s*(number|no\.?)\s*:\s*\S+",
    ]
    
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in patterns):
            continue
        cleaned_lines.append(line)
    
    cleaned = "\n".join(cleaned_lines)
    
    # Also strip any standalone dates that might have been hallucinated
    cleaned = re.sub(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        '[DATE_REMOVED]',
        cleaned
    )
    
    return cleaned.strip()


# ---------------------------------------------------------------------
# API wrappers with retry
# ---------------------------------------------------------------------

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_gpt(prompt: str, temp: float = 0.2) -> str:
    """Call OpenAI GPT model."""
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
    """
    Call Anthropic Claude model using the Messages API.
    
    FIXED: Now uses correct model name and proper API format.
    """
    if ac is None:
        raise RuntimeError("Anthropic client not configured (missing ANTHROPIC_API_KEY).")
    
    # Use the Messages API (not the old Completions API)
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=2048,  # Increased from 1600 for longer notes
        messages=[{"role": "user", "content": prompt}],
    )
    
    # Extract text from content blocks
    text_parts = []
    for block in r.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    
    return "".join(text_parts).strip()


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 3))
def call_deepseek(prompt: str, model: str, temp: float = 0.2) -> str:
    """Call DeepSeek model via HTTP API."""
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
# AI-GENERATED PATIENT SUMMARY (from human note)
# ---------------------------------------------------------------------

def build_ai_summary(human_note: str) -> str:
    """
    Generate a structured, de-identified patient summary from the human note
    using GPT.
    
    This is the KEY NEW FUNCTION that replaces the old placeholder summary.
    
    Args:
        human_note: The original KHCC clinical pharmacist note
        
    Returns:
        Structured summary string with key clinical information
    """
    if not human_note or pd.isna(human_note):
        return "No human note available for summary generation."
    
    # First, scrub identifiers from the human note
    scrubbed_note = scrub_identifiers(human_note)
    
    # Prompt GPT to extract and structure the key clinical information
    summary_prompt = f"""
You are a clinical oncology pharmacist. You will read a de-identified clinical note
and extract a structured summary with ONLY the key clinical information needed for
chemotherapy verification.

STRICT REQUIREMENTS:
- Do NOT include or mention any patient identifiers (names, IDs, MRNs, dates)
- Focus only on clinical facts
- Be concise but complete

INPUT NOTE (de-identified):
{scrubbed_note}

OUTPUT FORMAT:
Please provide a structured summary with these sections:

Diagnosis:
Regimen:
Key Clinical Problems:
Organ Function (renal, hepatic, cardiac):
Relevant Medications:
Comorbidities:
Other Relevant Information:

Return ONLY the structured summary, nothing else.
""".strip()
    
    try:
        summary = call_gpt(summary_prompt, temp=0.1)
        # Apply additional output cleaning
        summary = deidentify_model_output(summary)
        return summary
    except Exception as e:
        return f"[SUMMARY_ERROR] Could not generate summary: {type(e).__name__}"


# ---------------------------------------------------------------------
# PROMPTS for note generation
# ---------------------------------------------------------------------

def build_case_prompt(patient_summary: str) -> str:
    """
    Main generation prompt for creating clinical pharmacist notes.
    
    Uses the AI-generated summary (not individual fields).
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
- Refer to the person only as "the patient".
- Do NOT invent exact calendar dates. You may use phrases such as "baseline",
  "prior to cycle 1", "every cycle", "at follow-up", etc.

DE-IDENTIFIED PATIENT SUMMARY:
{patient_summary}

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
    """Prompt for reviewer (DeepSeek Reasoner) to critique the draft note."""
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
    """Prompt for synthesizer (Claude or GPT) to produce final note."""
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
# CADSS (Collaborative AI Development with Staged Synthesis)
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
    try:
        draft = call_gpt(case_prompt, GEN_TEMPERATURE)
    except Exception as e:
        return f"[CADSS_GEN_ERROR] Generator (GPT) failed: {type(e).__name__}"
    
    # 2) Reviewer (DeepSeek Reasoner)
    try:
        rev_json = call_deepseek(
            build_review_prompt(case_prompt, draft),
            model=DEEPSEEK_R,
            temp=REV_TEMPERATURE,
        )
    except Exception as e:
        # If reviewer fails, proceed with a placeholder critique
        rev_json = f'{{"issues": ["Reviewer unavailable: {type(e).__name__}"], "required_changes": ["Proceed with draft as-is"]}}'
    
    synth_prompt = build_synthesis_prompt(case_prompt, draft, rev_json)
    
    # 3) Synthesizer: prefer Claude, fallback to GPT if Claude fails
    try:
        final = call_claude(synth_prompt, SYN_TEMPERATURE)
        # Apply output de-identification
        final = deidentify_model_output(final)
    except Exception as e:
        # Fallback: GPT synthesizer
        try:
            fallback_note = call_gpt(
                f"The primary synthesizer is unavailable. Please act as the synthesizer "
                f"and complete this task.\n\n{synth_prompt}",
                SYN_TEMPERATURE,
            )
            final = f"[FALLBACK_FROM_CLAUDE] {deidentify_model_output(fallback_note)}"
        except Exception as e2:
            # Ultimate fallback: return the draft
            final = f"[CADSS_SYNTH_ERROR] Both synthesizers failed. Draft note:\n{draft}"
    
    return final


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def get_human_note(row: pd.Series) -> str:
    """
    Extract the human KHCC note from the Excel row.
    
    FIXED: Now correctly maps to 'note_original' after lowercasing.
    """
    # After lowercasing and stripping, the column is 'note_original'
    return str(row.get("note_original", ""))


def main():
    print(f"Loading KHCC cases from: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    
    # Normalize column names (lowercase and strip)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    print(f"Columns found: {df.columns.tolist()}")
    
    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)
        print(f"Processing only first {ROW_LIMIT} rows")
    
    # Ensure output directory exists
    out_dir_path = Path(OUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not OUT_FILE:
        out_path = out_dir_path / "KHCC_AI_Notes.xlsx"
    else:
        out_path = Path(OUT_FILE)
    
    records = []
    
    for idx, row in df.iterrows():
        # Validate required column
        if "case_id" not in row.index:
            raise KeyError(
                "Column 'case_id' not found after lowercasing headers. "
                "Make sure your Excel file has a 'Case_ID' column."
            )
        
        cid = row["case_id"]
        print(f"\n{'='*70}")
        print(f"Processing case_id={cid} (row {idx + 1}/{len(df)})")
        print(f"{'='*70}")
        
        # Get the human note (NOT de-identified for output)
        human_note = get_human_note(row)
        
        if not human_note or pd.isna(human_note):
            print(f"  WARNING: No human note found for case {cid}")
            human_note = ""
        
        # Generate AI summary from human note (THIS IS THE KEY FIX)
        print(f"  Step 1: Generating AI patient summary...")
        patient_summary = build_ai_summary(human_note)
        
        # Build the case prompt using the AI-generated summary
        case_prompt = build_case_prompt(patient_summary)
        
        # GPT note
        print(f"  Step 2: Generating GPT note...")
        try:
            gpt_raw = call_gpt(case_prompt, GEN_TEMPERATURE)
            gpt_note = deidentify_model_output(gpt_raw)
        except Exception as e:
            gpt_note = f"[GPT_ERROR] {type(e).__name__}: {e}"
            print(f"    ERROR: {gpt_note}")
        
        # Claude note (standalone)
        print(f"  Step 3: Generating Claude note...")
        try:
            claude_raw = call_claude(case_prompt, GEN_TEMPERATURE)
            claude_note = deidentify_model_output(claude_raw)
        except Exception as e:
            claude_note = f"[CLAUDE_ERROR] {type(e).__name__}: {e}"
            print(f"    ERROR: {claude_note}")
        
        # DeepSeek note
        print(f"  Step 4: Generating DeepSeek note...")
        try:
            ds_raw = call_deepseek(case_prompt, model=DEEPSEEK_MODEL, temp=GEN_TEMPERATURE)
            deepseek_note = deidentify_model_output(ds_raw)
        except Exception as e:
            deepseek_note = f"[DEEPSEEK_ERROR] {type(e).__name__}: {e}"
            print(f"    ERROR: {deepseek_note}")
        
        # CADSS consolidated note
        print(f"  Step 5: Running CADSS pipeline...")
        try:
            cadss_note = cadss_flow(case_prompt)
        except Exception as e:
            cadss_note = f"[CADSS_ERROR] {type(e).__name__}: {e}"
            print(f"    ERROR: {cadss_note}")
        
        # Assemble the record
        records.append(
            {
                "case_id": cid,
                "patient_summary": patient_summary,
                "gpt_note": gpt_note,
                "claude_note": claude_note,
                "deepseek_note": deepseek_note,
                "cadss_note": cadss_note,
                "human_note": human_note,  # Original, not scrubbed
            }
        )
        
        print(f"  ✓ Case {cid} complete")
        
        # Rate limiting
        time.sleep(PER_CASE_SLEEP)
    
    # Save to Excel
    out_df = pd.DataFrame(records)
    out_df.to_excel(out_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"SUCCESS! Saved AI notes to: {out_path.resolve()}")
    print(f"Total cases processed: {len(records)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
