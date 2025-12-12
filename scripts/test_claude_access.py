"""
Test script to check Claude API access and determine which models work.
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    print("❌ ERROR: ANTHROPIC_API_KEY not found in environment variables!")
    print("Please set it in your .env file or export it:")
    print("  export ANTHROPIC_API_KEY=sk-ant-api03-...")
    exit(1)

print(f"✓ API Key found: {ANTHROPIC_API_KEY[:20]}...{ANTHROPIC_API_KEY[-4:]}")
print()

# Initialize client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# List of Claude models to test (from most recent to older)
models_to_test = [
    "claude-sonnet-4-20250514",           # Latest Sonnet 4
    "claude-3-5-sonnet-20241022",         # Sonnet 3.5 (Oct 2024)
    "claude-3-5-sonnet-latest",           # Latest Sonnet 3.5
    "claude-3-5-sonnet-20240620",         # Sonnet 3.5 (June 2024)
    "claude-3-sonnet-20240229",           # Sonnet 3 (Feb 2024)
    "claude-3-haiku-20240307",            # Haiku 3
]

print("Testing Claude models with your API key...")
print("=" * 70)

working_models = []
failed_models = []

for model in models_to_test:
    print(f"\nTesting: {model}")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'hello' if you can read this."}
            ]
        )
        
        # Extract response text
        response_text = "".join([block.text for block in response.content if hasattr(block, "text")])
        
        print(f"  ✅ SUCCESS! Response: {response_text[:50]}")
        working_models.append(model)
        
    except anthropic.NotFoundError as e:
        print(f"  ❌ Not Found - Model not available or name incorrect")
        failed_models.append((model, "NotFoundError"))
        
    except anthropic.PermissionDeniedError as e:
        print(f"  ❌ Permission Denied - You don't have access to this model")
        failed_models.append((model, "PermissionDeniedError"))
        
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {str(e)[:100]}")
        failed_models.append((model, type(e).__name__))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if working_models:
    print(f"\n✅ WORKING MODELS ({len(working_models)}):")
    for i, model in enumerate(working_models, 1):
        print(f"  {i}. {model}")
    
    print(f"\n🎯 RECOMMENDED: Use this model in your script:")
    print(f"   export CLAUDE_MODEL=\"{working_models[0]}\"")
    print(f"\n   Or add to your .env file:")
    print(f"   CLAUDE_MODEL={working_models[0]}")
else:
    print("\n❌ NO WORKING MODELS FOUND!")
    print("   This could mean:")
    print("   1. Your API key doesn't have access to Claude models yet")
    print("   2. There's an issue with your API key")
    print("   3. You need to enable Claude access in your Anthropic account")

if failed_models:
    print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
    for model, error in failed_models:
        print(f"  • {model} - {error}")

print("\n" + "=" * 70)
print("For more info, visit: https://docs.anthropic.com/en/docs/models-overview")
print("=" * 70)
