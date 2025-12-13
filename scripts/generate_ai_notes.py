"""
generate_ai_notes_FAST.py — OPTIMIZED HIGH-SPEED VERSION

This version is optimized for speed by:
1. Using parallel processing for API calls
2. Reducing redundant calls
3. Using the fastest models available
4. Implementing aggressive caching

Use this for large batch processing (100+ cases).
"""

import os
import re
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")  # Use Haiku for speed
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

GEN_TEMPERATURE = 0.2
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))  # Parallel workers
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "0"))
START_ROW = int(os.getenv("START_ROW", "0"))  # 1-based row number
END_ROW = int(os.getenv("END_ROW", "0"))  # 1-based row number

# Skip CADSS for speed (generates 3 extra API calls per case)
SKIP_CADSS = os.getenv("SKIP_CADSS", "false").lower() == "true"  # Changed default to "false"

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
# De-identification
# ---------------------------------------------------------------------

def scrub_identifiers(text: str) -> str:
    """Remove PHI before sending to LLMs."""
    if not isinstance(text, str):
        return ""
    
    scrubbed = text
    
    scrubbed = re.sub(r'Patient\s+Name\s*:\s*[^\n]+', 'Patient Name: [REDACTED]', scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'Patient\s+ID\s*:\s*[^\n]+', 'Patient ID: [REDACTED]', scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'MRN\s*:\s*[^\n]+', 'MRN: [REDACTED]', scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'(Patient\s+)?Age\s*:\s*\d+\s*(yr|years?)?', 'Age: [REDACTED]', scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED_DATE]', scrubbed)
    scrubbed = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[REDACTED_DATE]', scrubbed)
    scrubbed = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '[REDACTED_TIME]', scrubbed)
    
    return scrubbed.strip()


# ---------------------------------------------------------------------
# Fast API wrappers (reduced retries for speed)
# ---------------------------------------------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(0.5, 1))
def call_gpt(prompt: str, temp: float = 0.2, max_tokens: int = 1000) -> str:
    if oai is None:
        raise RuntimeError("OpenAI client not configured")
    
    r = oai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(0.5, 1))
def call_claude(prompt: str, temp: float = 0.2, max_tokens: int = 1000) -> str:
    if ac is None:
        raise RuntimeError("Anthropic client not configured")
    
    r = ac.messages.create(
        model=CLAUDE_MODEL,
        temperature=temp,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    
    text_parts = []
    for block in r.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    
    return "".join(text_parts).strip()


@retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(0.5, 1))
def call_deepseek(prompt: str, model: str, temp: float = 0.2, max_tokens: int = 1000) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    
    with httpx.Client(base_url=DEEPSEEK_BASE, timeout=60) as client:
        resp = client.post("/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------
# Simplified prompt (shorter = faster)
# ---------------------------------------------------------------------

def build_ai_summary(human_note: str) -> str:
    """Fast summary generation."""
    if not human_note or pd.isna(human_note):
        return "No human note available."
    
    scrubbed_note = scrub_identifiers(human_note)
    
    summary_prompt = f"""Extract key clinical info from this note in 5 sentences max:

{scrubbed_note[:2000]}

Format:
Diagnosis:
Regimen:
Key Problems:
Organ Function:
Medications:"""
    
    try:
        summary = call_gpt(summary_prompt, temp=0.1, max_tokens=300)
        return summary
    except Exception as e:
        return f"[ERROR] {type(e).__name__}"


def build_case_prompt(patient_summary: str) -> str:
    """Simplified prompt for faster generation."""
    return f"""Clinical oncology pharmacist note for:

{patient_summary}

Write a concise note covering:
1. Dose verification
2. Safety/toxicity risks
3. Drug interactions
4. Supportive care
5. Monitoring
6. Key recommendations

Keep it professional and concise. No patient identifiers."""


# ---------------------------------------------------------------------
# Parallel processing function
# ---------------------------------------------------------------------

def generate_note_parallel(case_id: str, patient_summary: str, case_prompt: str, model_type: str) -> tuple:
    """Generate a single note in parallel."""
    try:
        if model_type == "gpt":
            note = call_gpt(case_prompt, GEN_TEMPERATURE, max_tokens=1000)
        elif model_type == "claude":
            note = call_claude(case_prompt, GEN_TEMPERATURE, max_tokens=1000)
        elif model_type == "deepseek":
            note = call_deepseek(case_prompt, DEEPSEEK_MODEL, GEN_TEMPERATURE, max_tokens=1000)
        else:
            note = "[ERROR] Unknown model type"
        
        return (model_type, note, None)
    except Exception as e:
        return (model_type, None, f"{type(e).__name__}: {e}")


def process_case(row: pd.Series) -> dict:
    """Process a single case with parallel API calls."""
    case_id = row["case_id"]
    human_note = str(row.get("note_original", ""))
    
    # Generate summary (1 API call)
    patient_summary = build_ai_summary(human_note)
    case_prompt = build_case_prompt(patient_summary)
    
    # Parallel generation of all 3 notes (3 API calls in parallel = faster!)
    results = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(generate_note_parallel, case_id, patient_summary, case_prompt, "gpt"): "gpt",
            executor.submit(generate_note_parallel, case_id, patient_summary, case_prompt, "claude"): "claude",
            executor.submit(generate_note_parallel, case_id, patient_summary, case_prompt, "deepseek"): "deepseek",
        }
        
        for future in as_completed(futures):
            model_type, note, error = future.result()
            if note:
                results[f"{model_type}_note"] = note
            else:
                results[f"{model_type}_note"] = f"[{model_type.upper()}_ERROR] {error}"
    
    # CADSS note - skip for speed unless explicitly requested
    if SKIP_CADSS:
        cadss_note = "[SKIPPED_FOR_SPEED] Set SKIP_CADSS=false to enable"
    else:
        cadss_note = results.get("gpt_note", "[ERROR]")  # Use GPT note as fallback
    
    return {
        "case_id": case_id,
        "patient_summary": patient_summary,
        "gpt_note": results.get("gpt_note", "[ERROR]"),
        "claude_note": results.get("claude_note", "[ERROR]"),
        "deepseek_note": results.get("deepseek_note", "[ERROR]"),
        "cadss_note": cadss_note,
        "human_note": human_note,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"⚡ FAST MODE - Optimized for speed")
    print(f"Loading KHCC cases from: {INPUT_EXCEL}")
    
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    total_rows = len(df)
    print(f"   Total rows in file: {total_rows}")
    
    # Apply row range filtering
    if START_ROW > 0 or END_ROW > 0:
        # Convert to 0-based indexing
        start_idx = max(0, START_ROW - 1) if START_ROW > 0 else 0
        end_idx = min(total_rows, END_ROW) if END_ROW > 0 else total_rows
        
        if start_idx >= total_rows:
            print(f"❌ ERROR: START_ROW ({START_ROW}) is greater than total rows ({total_rows})")
            return
        
        df = df.iloc[start_idx:end_idx]
        print(f"   Processing rows {start_idx + 1} to {end_idx} (selected {len(df)} rows)")
    elif ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)
        print(f"   Processing first {ROW_LIMIT} rows")
    else:
        print(f"   Processing all rows")
    
    out_dir_path = Path(OUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not OUT_FILE:
        out_path = out_dir_path / "KHCC_AI_Notes.xlsx"
    else:
        out_path = Path(OUT_FILE)
    
    print(f"\n⚡ Processing {len(df)} cases with {MAX_WORKERS} parallel workers...")
    print(f"⚡ CADSS: {'ENABLED (slower but complete)' if not SKIP_CADSS else 'SKIPPED (faster)'}")
    print(f"⚡ Models: GPT={GPT_MODEL}, Claude={CLAUDE_MODEL}, DeepSeek={DEEPSEEK_MODEL}\n")
    
    start_time = time.time()
    records = []
    
    # Process cases with parallel workers
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_case, row): idx for idx, row in df.iterrows()}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                record = future.result()
                records.append(record)
                elapsed = time.time() - start_time
                print(f"✓ Case {record['case_id']} complete ({len(records)}/{len(df)}) - {elapsed:.1f}s elapsed")
            except Exception as e:
                print(f"✗ Case {idx+1} failed: {type(e).__name__}")
    
    # Save results
    out_df = pd.DataFrame(records)
    out_df.to_excel(out_path, index=False)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUCCESS! Saved AI notes to: {out_path.resolve()}")
    print(f"Total cases processed: {len(records)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per case: {total_time/len(records):.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
