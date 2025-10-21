import os
from dotenv import load_dotenv
load_dotenv()

print("Has OPENAI_API_KEY?", bool(os.getenv("OPENAI_API_KEY")))
print("Has CLAUDE_API_KEY?", bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")))
print("Has DEEPSEEK_API_KEY?", bool(os.getenv("DEEPSEEK_API_KEY")))

# --- OpenAI (new SDK) ---
from openai import OpenAI
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
r = oai.chat.completions.create(
    model=os.getenv("GPT_MODEL","gpt-4o-mini"),
    messages=[{"role":"user","content":"Say: OPENAI OK"}],
)
print("OpenAI:", r.choices[0].message.content)

# --- Anthropic (Claude) ---
import anthropic
ac = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"))
r = ac.messages.create(
    model=os.getenv("CLAUDE_MODEL","claude-3-5-sonnet-20240620"),
    max_tokens=50,
    messages=[{"role":"user","content":"Say: CLAUDE OK"}],
)
print("Anthropic:", r.content[0].text)

# --- DeepSeek ---
import httpx
headers = {
    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
    "Content-Type": "application/json",
}
payload = {
    "model": os.getenv("DEEPSEEK_MODEL","deepseek-chat"),
    "messages":[{"role":"user","content":"Say: DEEPSEEK OK"}],
}
with httpx.Client(base_url="https://api.deepseek.com", timeout=60) as client:
    resp = client.post("/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print("DeepSeek:", data["choices"][0]["message"]["content"])
