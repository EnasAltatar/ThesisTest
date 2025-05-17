# AI Note Generation Script
# Private script to generate clinical pharmacist notes using ChatGPT, Claude, Gemini, and AI agent

import openai
import requests
import google.generativeai as genai
import gspread
import pandas as pd
import json
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Load secrets (set these in your Streamlit Secrets or local env)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("evaluation_log").worksheet("AI_Notes")

# Load case data
case_sheet = client.open("evaluation_log").worksheet("cases")
cases = pd.DataFrame(case_sheet.get_all_records())

# Load human pharmacist notes (should be in same sheet or separate tab)
human_notes_sheet = client.open("evaluation_log").worksheet("Human_Notes")
human_notes_df = pd.DataFrame(human_notes_sheet.get_all_records())

# Define prompt template
def build_prompt(row):
    return f"""
You are a clinical oncology pharmacist. Generate a full clinical note based on the following case:
- Diagnosis: {row['Diagnosis']}
- Regimen: {row['Regimen']}
- Labs: {row['Lab Tests']}
- Medications: {row['Medications']}
- Notes: {row['Notes']}
Include rationale, dosing, interaction checks, and supportive care.
"""

# ChatGPT (OpenAI)
def get_chatgpt_response(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Claude
def get_claude_response(prompt):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    return r.json()["content"][0]["text"].strip()

# Gemini
def get_gemini_response(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# AI Agent
def generate_agent_note(prompt):
    first = get_chatgpt_response(prompt)
    review = get_chatgpt_response(f"Please review and critique this note: \n{first}")
    final = get_chatgpt_response(f"Revise this clinical note based on critique: \n{first}\n\nCritique: {review}")
    return final

# Process all cases
for index, row in cases.iterrows():
    case_id = row["Case ID"]
    prompt = build_prompt(row)
    chatgpt_note = get_chatgpt_response(prompt)
    claude_note = get_claude_response(prompt)
    gemini_note = get_gemini_response(prompt)
    agent_note = generate_agent_note(prompt)

    human_note = human_notes_df[human_notes_df["Case ID"] == case_id]["Note"].values[0] if not human_notes_df[human_notes_df["Case ID"] == case_id].empty else ""

    # Append to sheet
    sheet.append_row([
        datetime.now().isoformat(),
        case_id,
        chatgpt_note,
        claude_note,
        gemini_note,
        agent_note,
        human_note
    ])
