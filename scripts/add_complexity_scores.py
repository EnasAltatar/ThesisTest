#!/usr/bin/env python3
"""
Add Case Complexity Scores to KHCC Dataset
===========================================

This script calculates complexity scores for all cases and adds them as new columns.
Run this to add complexity information to your existing dataset.

Usage:
    python add_complexity_scores.py

The script will:
1. Load your data
2. Calculate complexity scores for each case
3. Add new columns with complexity information
4. Save the enhanced dataset
"""

import pandas as pd
import numpy as np
from typing import Dict

def calculate_complexity_score(row: pd.Series) -> Dict[str, any]:
    """
    Calculate case complexity score based on clinical and laboratory parameters.
    
    Returns dictionary with all complexity components and final classification.
    """
    
    complexity_data = {
        'comorbidity_count': 0,
        'renal_impairment': False,
        'renal_severity': None,
        'hepatic_impairment': False,
        'hepatic_severity': None,
        'cardiac_dysfunction': False,
        'hematologic_abnormality': False,
        'hematologic_issues_count': 0,
        'complexity_score': 0,
        'complexity_level': 'Simple'
    }
    
    # -------------------------------------------------------------------------
    # 1. COMORBIDITIES (1 point each)
    # -------------------------------------------------------------------------
    comorbidity_cols = [
        'Comorbidity_DM', 
        'Comorbidity_HTN', 
        'Comorbidity_CAD',
        'Comorbidity_CKD', 
        'Comorbidity_Asthma', 
        'Comorbidity_Depression'
    ]
    
    for col in comorbidity_cols:
        val = str(row.get(col, '')).lower().strip()
        if val in ['yes', '1', 'true', 'present', 'prediabetis', 'pre-diabetic', 
                   'history of dm', 'hypertension', 'off treatment', 
                   'mitral stenosis', 'single kidney']:
            complexity_data['comorbidity_count'] += 1
    
    complexity_data['complexity_score'] += complexity_data['comorbidity_count']
    
    # -------------------------------------------------------------------------
    # 2. RENAL FUNCTION
    # -------------------------------------------------------------------------
    egfr = row.get('eGFR_mL_min_1.73m2', np.nan)
    creat = row.get('Creatinine_mg_dL', np.nan)
    
    if not pd.isna(egfr):
        try:
            egfr_val = float(egfr)
            if egfr_val < 30:  # Severe
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Severe'
                complexity_data['complexity_score'] += 2
            elif egfr_val < 60:  # Moderate
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Moderate'
                complexity_data['complexity_score'] += 1
        except (ValueError, TypeError):
            pass
    
    if not complexity_data['renal_impairment'] and not pd.isna(creat):
        try:
            creat_val = float(creat)
            if creat_val > 2.0:
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Severe'
                complexity_data['complexity_score'] += 2
            elif creat_val > 1.5:
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Moderate'
                complexity_data['complexity_score'] += 1
        except (ValueError, TypeError):
            pass
    
    # -------------------------------------------------------------------------
    # 3. HEPATIC FUNCTION
    # -------------------------------------------------------------------------
    alt = row.get('ALT_U_L', np.nan)
    ast = row.get('AST_U_L', np.nan)
    bili = row.get('TotalBilirubin_mg_dL', np.nan)
    
    hepatic_abnormalities = 0
    
    if not pd.isna(alt):
        try:
            if float(alt) > 80:
                hepatic_abnormalities += 2
            elif float(alt) > 40:
                hepatic_abnormalities += 1
        except (ValueError, TypeError):
            pass
    
    if not pd.isna(ast):
        try:
            if float(ast) > 80:
                hepatic_abnormalities += 2
            elif float(ast) > 40:
                hepatic_abnormalities += 1
        except (ValueError, TypeError):
            pass
    
    if not pd.isna(bili):
        try:
            if float(bili) > 2.0:
                hepatic_abnormalities += 2
            elif float(bili) > 1.5:
                hepatic_abnormalities += 1
        except (ValueError, TypeError):
            pass
    
    if hepatic_abnormalities >= 3:
        complexity_data['hepatic_impairment'] = True
        complexity_data['hepatic_severity'] = 'Severe'
        complexity_data['complexity_score'] += 2
    elif hepatic_abnormalities >= 1:
        complexity_data['hepatic_impairment'] = True
        complexity_data['hepatic_severity'] = 'Mild-Moderate'
        complexity_data['complexity_score'] += 1
    
    # -------------------------------------------------------------------------
    # 4. CARDIAC FUNCTION
    # -------------------------------------------------------------------------
    lvef = row.get('LVEF_percent', np.nan)
    
    if not pd.isna(lvef):
        try:
            lvef_val = float(lvef)
            if lvef_val < 40:
                complexity_data['cardiac_dysfunction'] = True
                complexity_data['complexity_score'] += 3
            elif lvef_val < 50:
                complexity_data['cardiac_dysfunction'] = True
                complexity_data['complexity_score'] += 2
        except (ValueError, TypeError):
            pass
    
    # -------------------------------------------------------------------------
    # 5. HEMATOLOGIC ABNORMALITIES
    # -------------------------------------------------------------------------
    
    # WBC
    wbc = row.get('WBC_x10^9_L', np.nan)
    if not pd.isna(wbc):
        try:
            wbc_val = float(wbc) if not isinstance(wbc, str) else float(wbc)
            if wbc_val < 2.0 or wbc_val > 20.0:
                complexity_data['hematologic_issues_count'] += 2
            elif wbc_val < 3.5 or wbc_val > 11.0:
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    # ANC
    anc = row.get('ANC_x10^9_L', np.nan)
    if not pd.isna(anc):
        try:
            anc_val = float(anc)
            if anc_val < 1.0:
                complexity_data['hematologic_issues_count'] += 2
            elif anc_val < 1.5:
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    # Hemoglobin
    hgb = row.get('Hemoglobin_g_dL', np.nan)
    if not pd.isna(hgb):
        try:
            hgb_val = float(hgb)
            if hgb_val < 8.0:
                complexity_data['hematologic_issues_count'] += 2
            elif hgb_val < 10.0:
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    # Platelets
    plt = row.get('Platelets_x10^9_L', np.nan)
    if not pd.isna(plt):
        try:
            plt_val = float(plt)
            if plt_val < 75:
                complexity_data['hematologic_issues_count'] += 2
            elif plt_val < 100:
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    if complexity_data['hematologic_issues_count'] > 0:
        complexity_data['hematologic_abnormality'] = True
        complexity_data['complexity_score'] += min(complexity_data['hematologic_issues_count'], 3)
    
    # -------------------------------------------------------------------------
    # 6. FINAL CLASSIFICATION
    # -------------------------------------------------------------------------
    total_score = complexity_data['complexity_score']
    
    if total_score >= 5:
        complexity_data['complexity_level'] = 'Complex'
    elif total_score >= 2:
        complexity_data['complexity_level'] = 'Intermediate'
    else:
        complexity_data['complexity_level'] = 'Simple'
    
    return complexity_data


def add_complexity_to_dataset(input_file: str, output_file: str):
    """
    Add complexity scores to dataset and save enhanced version.
    
    Parameters:
    -----------
    input_file : str
        Path to input Excel file
    output_file : str
        Path for output Excel file with complexity columns
    """
    
    print("="*70)
    print("ADDING CASE COMPLEXITY SCORES TO DATASET")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_excel(input_file)
    print(f"✓ Loaded {len(df)} cases")
    
    # Calculate complexity for all cases
    print("\nCalculating complexity scores...")
    
    complexity_results = []
    for idx, row in df.iterrows():
        complexity_data = calculate_complexity_score(row)
        complexity_results.append(complexity_data)
    
    # Add complexity columns to dataframe
    complexity_df = pd.DataFrame(complexity_results)
    
    df['Complexity_Score'] = complexity_df['complexity_score']
    df['Complexity_Level'] = complexity_df['complexity_level']
    df['Comorbidity_Count'] = complexity_df['comorbidity_count']
    df['Has_Renal_Impairment'] = complexity_df['renal_impairment']
    df['Renal_Severity'] = complexity_df['renal_severity']
    df['Has_Hepatic_Impairment'] = complexity_df['hepatic_impairment']
    df['Hepatic_Severity'] = complexity_df['hepatic_severity']
    df['Has_Cardiac_Dysfunction'] = complexity_df['cardiac_dysfunction']
    df['Has_Hematologic_Abnormality'] = complexity_df['hematologic_abnormality']
    df['Hematologic_Issues_Count'] = complexity_df['hematologic_issues_count']
    
    print("✓ Complexity calculation complete")
    
    # Display distribution
    print("\n" + "="*70)
    print("COMPLEXITY DISTRIBUTION")
    print("="*70)
    
    complexity_counts = df['Complexity_Level'].value_counts()
    print(f"\nTotal cases: {len(df)}")
    for level in ['Simple', 'Intermediate', 'Complex']:
        count = complexity_counts.get(level, 0)
        pct = 100 * count / len(df)
        print(f"  {level:15s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nComplexity Score Statistics:")
    print(f"  Mean: {df['Complexity_Score'].mean():.2f}")
    print(f"  Median: {df['Complexity_Score'].median():.2f}")
    print(f"  Range: [{df['Complexity_Score'].min():.0f}, {df['Complexity_Score'].max():.0f}]")
    
    print(f"\nComplexity Components:")
    print(f"  Cases with renal impairment: {df['Has_Renal_Impairment'].sum()}")
    print(f"  Cases with hepatic impairment: {df['Has_Hepatic_Impairment'].sum()}")
    print(f"  Cases with cardiac dysfunction: {df['Has_Cardiac_Dysfunction'].sum()}")
    print(f"  Cases with hematologic abnormalities: {df['Has_Hematologic_Abnormality'].sum()}")
    print(f"  Mean comorbidity count: {df['Comorbidity_Count'].mean():.2f}")
    
    # Save enhanced dataset
    print(f"\n" + "="*70)
    print("SAVING ENHANCED DATASET")
    print("="*70)
    
    df.to_excel(output_file, index=False)
    print(f"✓ Enhanced dataset saved: {output_file}")
    print(f"  Total cases: {len(df)}")
    print(f"  New columns added: 10 complexity-related columns")
    
    # Create summary report
    report_file = output_file.replace('.xlsx', '_complexity_summary.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CASE COMPLEXITY ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Total cases: {len(df)}\n\n")
        
        f.write("COMPLEXITY DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        for level in ['Simple', 'Intermediate', 'Complex']:
            count = complexity_counts.get(level, 0)
            pct = 100 * count / len(df)
            f.write(f"  {level:15s}: {count:3d} ({pct:5.1f}%)\n")
        
        f.write(f"\nComplexity Score Statistics:\n")
        f.write(f"  Mean: {df['Complexity_Score'].mean():.2f}\n")
        f.write(f"  Median: {df['Complexity_Score'].median():.2f}\n")
        f.write(f"  Range: [{df['Complexity_Score'].min():.0f}, {df['Complexity_Score'].max():.0f}]\n\n")
        
        f.write("Complexity Components:\n")
        f.write(f"  Renal impairment: {df['Has_Renal_Impairment'].sum()} cases\n")
        f.write(f"  Hepatic impairment: {df['Has_Hepatic_Impairment'].sum()} cases\n")
        f.write(f"  Cardiac dysfunction: {df['Has_Cardiac_Dysfunction'].sum()} cases\n")
        f.write(f"  Hematologic abnormalities: {df['Has_Hematologic_Abnormality'].sum()} cases\n")
        f.write(f"  Mean comorbidity count: {df['Comorbidity_Count'].mean():.2f}\n\n")
        
        f.write("NEW COLUMNS ADDED:\n")
        f.write("-"*70 + "\n")
        f.write("  1. Complexity_Score (0-10+)\n")
        f.write("  2. Complexity_Level (Simple/Intermediate/Complex)\n")
        f.write("  3. Comorbidity_Count\n")
        f.write("  4. Has_Renal_Impairment (True/False)\n")
        f.write("  5. Renal_Severity (None/Moderate/Severe)\n")
        f.write("  6. Has_Hepatic_Impairment (True/False)\n")
        f.write("  7. Hepatic_Severity (None/Mild-Moderate/Severe)\n")
        f.write("  8. Has_Cardiac_Dysfunction (True/False)\n")
        f.write("  9. Has_Hematologic_Abnormality (True/False)\n")
        f.write("  10. Hematologic_Issues_Count\n")
    
    print(f"✓ Summary report saved: {report_file}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nYour dataset now includes case complexity scores.")
    print(f"You can now use these columns for:")
    print(f"  - Filtering cases by complexity")
    print(f"  - Stratified sampling")
    print(f"  - Subgroup analysis")
    print(f"  - Reporting case distribution")
    
    return df


if __name__ == "__main__":
    # Configuration - EDIT THESE PATHS FOR YOUR SETUP
    
    # For GitHub Actions (the workflow does 'cd scripts' first)
    INPUT_FILE = "khcc_cases_200.xlsx"  # File is in scripts/ folder
    OUTPUT_FILE = "khcc_cases_200_with_complexity.xlsx"  # Output in scripts/ folder
    
    # Run the function
    enhanced_df = add_complexity_to_dataset(INPUT_FILE, OUTPUT_FILE)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print("="*70)
    print(f"1. Open '{OUTPUT_FILE}' to see the new complexity columns")
    print(f"2. Use 'Complexity_Level' column to filter/sort cases")
    print(f"3. Use 'Complexity_Score' for more granular analysis")
    print(f"4. Check '{OUTPUT_FILE.replace('.xlsx', '_complexity_summary.txt')}' for detailed report")
