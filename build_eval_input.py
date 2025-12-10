"""
build_eval_input.py — FINAL VERSION
Converts AI notes + KHCC original case notes into long-format input
for the blinded evaluation app.

Input:
  - KHCC_AI_Notes.xlsx
Output:
  - khcc_eval_input.xlsx
"""

import pandas as pd

AI_NOTES_FILE = "KHCC_AI_Notes.xlsx"
OUT_FILE = "khcc_eval_input.xlsx"

def main():
    print(f"Loading AI file: {AI_NOTES_FILE}")
    df = pd.read_excel(AI_NOTES_FILE)

    long_rows = []

    for _, row in df.iterrows():
        cid = row["case_id"]
        summary = row["patient_summary"]

        sources = {
            "chatgpt": row.get("gpt_note", ""),
            "claude": row.get("claude_note", ""),
            "deepseek": row.get("deepseek_note", ""),
            "cadss": row.get("cadss_note", ""),
            "human": row.get("human_note", ""),
        }

        for src, note in sources.items():
            if str(note).strip() == "":
                continue

            long_rows.append({
                "case_id": cid,
                "patient_summary": summary,
                "note_source": src,
                "note_text": note,
            })

    out = pd.DataFrame(long_rows)
    out.to_excel(OUT_FILE, index=False)
    print(f"Saved long-format evaluation file: {OUT_FILE}")

if __name__ == "__main__":
    main()
