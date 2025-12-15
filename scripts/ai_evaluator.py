"""
AI-based Evaluation System for Clinical Pharmacist Notes
Evaluates notes using PCNE V9.1 classification and holistic rubric (1-5 scale)
"""

import anthropic
import pandas as pd
import json
import os
from typing import Dict, List
from datetime import datetime
import time

# Load PCNE codes
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
pcne_path = os.path.join(script_dir, 'pcne_codes.json')

# If not found in script directory, try current directory
if not os.path.exists(pcne_path):
    pcne_path = 'pcne_codes.json'

# If still not found, try going up one level
if not os.path.exists(pcne_path):
    pcne_path = os.path.join(os.path.dirname(script_dir), 'scripts', 'pcne_codes.json')

print(f"Loading PCNE codes from: {pcne_path}")
with open(pcne_path, 'r') as f:
    PCNE_CODES = json.load(f)

EVALUATION_PROMPT = """You are an expert clinical pharmacist evaluator specializing in oncology pharmacy. Your task is to evaluate clinical pharmacist notes for breast cancer patients receiving chemotherapy.

You will evaluate the note using two frameworks:

1. **PCNE V9.1 Classification System**: Identify drug-related problems (DRPs) and code them
2. **Holistic Rubric**: Rate the note on six domains using a 1-5 scale

## PCNE V9.1 CODES AVAILABLE:

### Problems (P):
{problems}

### Causes (C):
{causes}

### Interventions (I):
{interventions}

### Outcomes (O):
{outcomes}

## HOLISTIC RUBRIC (1-5 Scale):

Rate each domain where:
- **1 = Poor**: Major deficiencies, unsafe, or inappropriate
- **2 = Below Acceptable**: Significant issues that need correction
- **3 = Acceptable/Borderline**: Meets minimum standards, room for improvement
- **4 = Good**: Solid performance with minor areas for enhancement
- **5 = Excellent**: Exemplary, comprehensive, and highly appropriate

### Domains to Rate:
1. **Clinical Reasoning Accuracy**: Logical soundness, appropriate clinical judgment, correct interpretation of patient data
2. **Safety and Risk Sensitivity**: Identification of safety concerns, risk mitigation, contraindication awareness
3. **Completeness and Relevance**: Thoroughness while staying focused on pertinent issues, addresses key clinical questions
4. **Guideline and Protocol Adherence**: Alignment with evidence-based oncology guidelines (NCCN, ASCO, etc.), standard protocols
5. **Clinical Communication Clarity**: Professional tone, clear recommendations, actionable for healthcare team
6. **Overall Clinical Value**: Utility for patient care, impact on clinical decision-making

---

## PATIENT CLINICAL CONTEXT:
{patient_summary}

## PHARMACIST NOTE TO EVALUATE:
{note_text}

---

## YOUR EVALUATION TASK:

Provide a comprehensive evaluation in the following JSON format:

```json
{{
  "pcne_classification": {{
    "problems": ["P1.2", "P1.3"],  // List all applicable problem codes
    "causes": ["C3.1", "C1.6"],     // List all applicable cause codes
    "interventions": ["I1.3", "I3.5"], // List all applicable intervention codes
    "outcomes": ["O1.1"],            // List all applicable outcome codes
    "drp_description": "Brief description of key drug-related problems identified"
  }},
  "holistic_scores": {{
    "clinical_reasoning_accuracy": {{
      "score": 4,
      "rationale": "Explain your rating"
    }},
    "safety_and_risk_sensitivity": {{
      "score": 5,
      "rationale": "Explain your rating"
    }},
    "completeness_and_relevance": {{
      "score": 3,
      "rationale": "Explain your rating"
    }},
    "guideline_and_protocol_adherence": {{
      "score": 4,
      "rationale": "Explain your rating"
    }},
    "clinical_communication_clarity": {{
      "score": 4,
      "rationale": "Explain your rating"
    }},
    "overall_clinical_value": {{
      "score": 4,
      "rationale": "Explain your rating"
    }}
  }},
  "overall_comments": "Provide comprehensive commentary on the note's strengths and weaknesses",
  "confidence_level": "high/medium/low - your confidence in this evaluation"
}}
```

**Important Instructions:**
- Only select PCNE codes that are clearly applicable based on the note content
- If no DRP is identified or addressed in the note, use empty arrays for PCNE codes
- Be objective and evidence-based in your scoring
- Provide specific rationales tied to the note content
- Consider oncology-specific guidelines and breast cancer treatment standards
- Rate based on what is present in the note, not on missing information you might wish was included (unless completeness is severely lacking)

Return ONLY the JSON object, no additional text.
"""

def format_pcne_codes_for_prompt():
    """Format PCNE codes in a readable way for the prompt"""
    problems = "\n".join([f"- {code}: {desc}" for code, desc in PCNE_CODES["Problems"].items()])
    causes = "\n".join([f"- {code}: {desc}" for code, desc in PCNE_CODES["Causes"].items()])
    interventions = "\n".join([f"- {code}: {desc}" for code, desc in PCNE_CODES["Interventions"].items()])
    outcomes = "\n".join([f"- {code}: {desc}" for code, desc in PCNE_CODES["Outcomes"].items()])
    
    return problems, causes, interventions, outcomes

def evaluate_note_with_ai(client, patient_summary: str, note_text: str, case_id: str, note_label: str) -> Dict:
    """
    Evaluate a single clinical pharmacist note using Claude API
    """
    problems, causes, interventions, outcomes = format_pcne_codes_for_prompt()
    
    prompt = EVALUATION_PROMPT.format(
        problems=problems,
        causes=causes,
        interventions=interventions,
        outcomes=outcomes,
        patient_summary=patient_summary,
        note_text=note_text
    )
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0,  # Use 0 for more consistent evaluations
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        response_text = response.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        evaluation = json.loads(response_text.strip())
        
        # Add metadata
        evaluation['case_id'] = case_id
        evaluation['note_label'] = note_label
        evaluation['evaluation_timestamp'] = datetime.now().isoformat()
        
        return evaluation
        
    except Exception as e:
        print(f"Error evaluating {case_id}-{note_label}: {str(e)}")
        return {
            'case_id': case_id,
            'note_label': note_label,
            'error': str(e),
            'evaluation_timestamp': datetime.now().isoformat()
        }

def flatten_evaluation_for_csv(evaluation: Dict) -> Dict:
    """
    Flatten the nested evaluation structure for CSV export
    """
    flat = {
        'case_id': evaluation.get('case_id'),
        'note_label': evaluation.get('note_label'),
        'evaluation_timestamp': evaluation.get('evaluation_timestamp'),
        'error': evaluation.get('error', '')
    }
    
    if 'pcne_classification' in evaluation:
        pcne = evaluation['pcne_classification']
        flat['pcne_problems'] = '|'.join(pcne.get('problems', []))
        flat['pcne_causes'] = '|'.join(pcne.get('causes', []))
        flat['pcne_interventions'] = '|'.join(pcne.get('interventions', []))
        flat['pcne_outcomes'] = '|'.join(pcne.get('outcomes', []))
        flat['pcne_drp_description'] = pcne.get('drp_description', '')
    
    if 'holistic_scores' in evaluation:
        scores = evaluation['holistic_scores']
        for domain, data in scores.items():
            flat[f'{domain}_score'] = data.get('score')
            flat[f'{domain}_rationale'] = data.get('rationale', '')
    
    flat['overall_comments'] = evaluation.get('overall_comments', '')
    flat['confidence_level'] = evaluation.get('confidence_level', '')
    
    return flat

def main():
    """
    Main evaluation pipeline
    """
    # Initialize Claude client
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load input data
    print("Loading evaluation input data...")
    
    # Try multiple possible paths
    possible_paths = [
        '../data/khcc_eval_input.xlsx',  # GitHub structure
        'data/khcc_eval_input.xlsx',     # From repo root
        'khcc_eval_input.xlsx',          # Current directory
        '/mnt/user-data/uploads/khcc_eval_input.xlsx'  # Claude.ai
    ]
    
    input_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path = path
            break
    
    if input_path is None:
        raise FileNotFoundError("Could not find khcc_eval_input.xlsx in any expected location")
    
    print(f"Found input file at: {input_path}")
    df = pd.read_excel(input_path)
    print(f"Loaded {len(df)} notes to evaluate")
    
    # Optional: Limit for testing
    TEST_MODE = True
    if TEST_MODE:
        print("\n🧪 TEST MODE: Evaluating first 5 notes only")
        df = df.head(5)
    
    # Evaluate each note
    evaluations = []
    total = len(df)
    
    for idx, row in df.iterrows():
        print(f"\nEvaluating {idx+1}/{total}: {row['case_id']} - Note {row['note_label']}")
        
        evaluation = evaluate_note_with_ai(
            client=client,
            patient_summary=row['patient_summary'],
            note_text=row['note_text'],
            case_id=row['case_id'],
            note_label=row['note_label']
        )
        
        evaluations.append(evaluation)
        
        # Rate limiting: slight delay between requests
        if idx < total - 1:
            time.sleep(1)
    
    # Save results
    print("\n" + "="*50)
    print("Saving evaluation results...")
    
    # Create outputs directory if it doesn't exist
    output_dir = '../outputs' if os.path.exists('../outputs') else 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save full JSON
    output_json = os.path.join(output_dir, 'ai_evaluations_full.json')
    with open(output_json, 'w') as f:
        json.dump(evaluations, f, indent=2)
    print(f"✓ Full evaluations saved to: {output_json}")
    
    # Save flattened CSV
    flat_evaluations = [flatten_evaluation_for_csv(e) for e in evaluations]
    df_results = pd.DataFrame(flat_evaluations)
    
    output_csv = os.path.join(output_dir, 'ai_evaluations.csv')
    df_results.to_csv(output_csv, index=False)
    print(f"✓ Flattened results saved to: {output_csv}")
    
    # Save summary statistics
    if not df_results.empty and 'error' not in df_results.columns or df_results['error'].isna().all():
        score_columns = [col for col in df_results.columns if col.endswith('_score')]
        summary = df_results[score_columns].describe()
        
        output_summary = os.path.join(output_dir, 'ai_evaluations_summary.csv')
        summary.to_csv(output_summary)
        print(f"✓ Summary statistics saved to: {output_summary}")
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(summary)
    
    print("\n✅ AI evaluation complete!")
    print(f"Total notes evaluated: {len(evaluations)}")
    print(f"Successful evaluations: {sum(1 for e in evaluations if 'error' not in e)}")
    print(f"Failed evaluations: {sum(1 for e in evaluations if 'error' in e)}")

if __name__ == "__main__":
    main()
