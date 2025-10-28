DEMO_CASES = pd.DataFrame([
    {
        "case_id": "KHCC-001", "phase": "1",
        "patient_summary": "54F, ER+/HER2-, AC→T planned; eGFR 48 ml/min; LVEF 60%; HTN on amlodipine.",
        "chatgpt_note": "Draft pharmacist recommendation…",
        "claude_note": "Draft pharmacist recommendation…",
        "cadss_note": "",
        "human_note": "Institutional reference pharmacist note…"
    },
    {
        "case_id": "KHCC-002", "phase": "2",
        "patient_summary": "61F, HER2+, docetaxel + trastuzumab + pertuzumab; DM2 (metformin); ALT 2× ULN.",
        "chatgpt_note": "",
        "claude_note": "",
        "cadss_note": "Draft pharmacist recommendation (composite)…",
        "human_note": "Reference pharmacist note…"
    },
    {
        "case_id": "KHCC-003", "phase": "3",
        "patient_summary": "48F, TNBC, neoadjuvant ddAC→paclitaxel; baseline ANC low-normal.",
        "chatgpt_note": "Draft pharmacist recommendation…",
        "claude_note": "Draft pharmacist recommendation…",
        "cadss_note": "",
        "human_note": "Final pharmacist note (ground truth)…"
    },
])
