import os
import json
import pandas as pd
import openai
import google.generativeai as genai
import anthropic
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load secrets from environment (set in Streamlit or GitHub)
openai.api_key = os.getenv("OPENAI_API_KEY")
claude_key = os.getenv("CLAUDE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
creds_dict = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# Authorize Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet_url = "https://docs.google.com/spreadsheets/d/1MsXao47OMA715hwvpDhHoAimcEYJtjrqbK7Ed9wXw0c"
sheet = client.open_by_url(sheet_url)
cases = pd.DataFrame(sheet.worksheet("cases").get_all_records())

# Filter out cases that already have AI notes
ai_notes = pd.DataFrame(sheet.worksheet("ai_outputs").get_all_records())
evaluated_ids = ai_notes["Case ID"].tolist() if not ai_notes.empty else []
unevaluated = cases[~cases["Case ID"].isin(evaluated_ids)]

# Load Claude and Gemini clients
claude_client = anthropic.Anthropic(api_key=claude_key)
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel("gemini-pro")

def generate_prompt(row):
    return f"""
You are a clinical pharmacist. Given the medications, dosages, lab tests, and prior pharmacist note, generate a structured recommendation.

Case ID: {row['Case ID']}
Medications: {row['Medications']}
Dosages: {row['Dosages']}
Lab Tests: {row['Lab Tests']}
Clinical Pharmacist Note: {row['Notes']}

Provide a recommendation as if you are validating chemotherapy: include safety checks, dose appropriateness, and supportive care.
"""

def generate_chatgpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ChatGPT Error: {e}"

def generate_claude(prompt):
    try:
        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"Claude Error: {e}"

def generate_agent(chat_note, critique):
    prompt = f"""
The following recommendation was written by an AI:
{chat_note}

The following critique was provided:
{critique}

Now synthesize a final recommendation that integrates both.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Agent Error: {e}"

# Begin generating notes
results = []

for _, row in unevaluated.iterrows():
    prompt = generate_prompt(row)
    chat_note = generate_chatgpt(prompt)
    critique = generate_chatgpt(f"Critique this recommendation:\n{chat_note}")
    agent_note = generate_agent(chat_note, critique)
    claude_note = generate_claude(prompt)
    
    results.append([
        row["Case ID"],
        chat_note,
        claude_note,
        agent_note
    ])

# Append to Google Sheet
output_sheet = sheet.worksheet("ai_outputs")
for row in results:
    output_sheet.append_row(row)

print("✅ AI note generation completed.")
