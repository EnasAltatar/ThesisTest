name: Run AI Note Generation

on:
  workflow_dispatch:  # Manual trigger from GitHub

jobs:
  run-script:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install openai requests google-generativeai gspread oauth2client pandas

      - name: Run AI generation script
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CLAUDE_API_KEY: ${{ secrets.CLAUDE_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GOOGLE_APPLICATION_CREDENTIALS }}
        run: python scripts/generate_ai_notes.py
