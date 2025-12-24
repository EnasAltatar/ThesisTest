#!/usr/bin/env python3
"""
KHCC Breast Cancer Case Complexity-Based Stratification
========================================================

Stratifies cases based solely on case complexity (Simple, Intermediate, Complex)
using clinical and laboratory parameters.

Author: Enas
Date: December 24, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

# ==============================================================================
# COMPLEXITY SCORING ALGORITHM
# ==============================================================================

def calculate_complexity_score(row: pd.Series) -> Dict[str, any]:
    """
    Calculate case complexity score based on:
    1. Number of comorbidities
    2. Renal function impairment
    3. Hepatic function impairment
    4. Cardiac function impairment
    5. Hematologic abnormalities
    
    Returns:
    --------
    Dictionary with complexity components and final classification
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
            if egfr_val < 30:  # Severe renal impairment (CKD Stage 4-5)
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Severe'
                complexity_data['complexity_score'] += 2
            elif egfr_val < 60:  # Moderate renal impairment (CKD Stage 3)
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Moderate'
                complexity_data['complexity_score'] += 1
        except (ValueError, TypeError):
            pass
    
    # If eGFR missing, check creatinine
    if not complexity_data['renal_impairment'] and not pd.isna(creat):
        try:
            creat_val = float(creat)
            if creat_val > 2.0:  # Severe elevation
                complexity_data['renal_impairment'] = True
                complexity_data['renal_severity'] = 'Severe'
                complexity_data['complexity_score'] += 2
            elif creat_val > 1.5:  # Moderate elevation
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
    
    # ALT (normal upper limit ~35-40 U/L)
    if not pd.isna(alt):
        try:
            if float(alt) > 80:  # >2x ULN
                hepatic_abnormalities += 2
            elif float(alt) > 40:  # >1x ULN
                hepatic_abnormalities += 1
        except (ValueError, TypeError):
            pass
    
    # AST (normal upper limit ~35-40 U/L)
    if not pd.isna(ast):
        try:
            if float(ast) > 80:  # >2x ULN
                hepatic_abnormalities += 2
            elif float(ast) > 40:  # >1x ULN
                hepatic_abnormalities += 1
        except (ValueError, TypeError):
            pass
    
    # Total Bilirubin (normal <1.2 mg/dL)
    if not pd.isna(bili):
        try:
            if float(bili) > 2.0:  # Moderate-severe elevation
                hepatic_abnormalities += 2
            elif float(bili) > 1.5:  # Mild elevation
                hepatic_abnormalities += 1
        except (ValueError, TypeError):
            pass
    
    # Classify hepatic impairment
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
            if lvef_val < 40:  # Severe cardiac dysfunction
                complexity_data['cardiac_dysfunction'] = True
                complexity_data['complexity_score'] += 3
            elif lvef_val < 50:  # Mild-moderate cardiac dysfunction
                complexity_data['cardiac_dysfunction'] = True
                complexity_data['complexity_score'] += 2
        except (ValueError, TypeError):
            pass
    
    # -------------------------------------------------------------------------
    # 5. HEMATOLOGIC ABNORMALITIES
    # -------------------------------------------------------------------------
    
    # White Blood Cell Count
    wbc = row.get('WBC_x10^9_L', np.nan)
    if not pd.isna(wbc):
        try:
            wbc_val = float(wbc) if not isinstance(wbc, str) else float(wbc)
            if wbc_val < 2.0 or wbc_val > 20.0:  # Severe abnormality
                complexity_data['hematologic_issues_count'] += 2
            elif wbc_val < 3.5 or wbc_val > 11.0:  # Mild abnormality
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    # Absolute Neutrophil Count
    anc = row.get('ANC_x10^9_L', np.nan)
    if not pd.isna(anc):
        try:
            anc_val = float(anc)
            if anc_val < 1.0:  # Severe neutropenia
                complexity_data['hematologic_issues_count'] += 2
            elif anc_val < 1.5:  # Mild neutropenia
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    # Hemoglobin
    hgb = row.get('Hemoglobin_g_dL', np.nan)
    if not pd.isna(hgb):
        try:
            hgb_val = float(hgb)
            if hgb_val < 8.0:  # Severe anemia
                complexity_data['hematologic_issues_count'] += 2
            elif hgb_val < 10.0:  # Mild-moderate anemia
                complexity_data['hematologic_issues_count'] += 1
        except (ValueError, TypeError):
            pass
    
    # Platelet Count
    plt = row.get('Platelets_x10^9_L', np.nan)
    if not pd.isna(plt):
        try:
            plt_val = float(plt)
            if plt_val < 75:  # Severe thrombocytopenia
                complexity_data['hematologic_issues_count'] += 2
            elif plt_val < 100:  # Mild thrombocytopenia
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


# ==============================================================================
# STRATIFIED SAMPLING BASED ON COMPLEXITY
# ==============================================================================

def stratified_sample_by_complexity(
    df: pd.DataFrame,
    target_simple: int = 34,
    target_intermediate: int = 26,
    target_complex: int = 4,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate stratified sample based on case complexity only.
    
    Default targets yield n=64 total (Approach B)
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset with complexity classifications
    target_simple : int
        Number of simple cases to sample
    target_intermediate : int
        Number of intermediate cases to sample
    target_complex : int
        Number of complex cases to sample
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame with sampled cases
    """
    np.random.seed(seed)
    
    samples = []
    
    # Sample Simple cases
    simple_cases = df[df['Complexity_Level'] == 'Simple']
    n_simple = min(target_simple, len(simple_cases))
    if n_simple > 0:
        sample_simple = simple_cases.sample(n=n_simple, random_state=seed)
        samples.append(sample_simple)
    
    # Sample Intermediate cases
    intermediate_cases = df[df['Complexity_Level'] == 'Intermediate']
    n_intermediate = min(target_intermediate, len(intermediate_cases))
    if n_intermediate > 0:
        sample_intermediate = intermediate_cases.sample(n=n_intermediate, random_state=seed)
        samples.append(sample_intermediate)
    
    # Sample Complex cases
    complex_cases = df[df['Complexity_Level'] == 'Complex']
    n_complex = min(target_complex, len(complex_cases))
    if n_complex > 0:
        sample_complex = complex_cases.sample(n=n_complex, random_state=seed)
        samples.append(sample_complex)
    
    # Combine
    final_sample = pd.concat(samples, ignore_index=True)
    
    print(f"\nSampling Summary:")
    print(f"  Simple: Sampled {n_simple} of {len(simple_cases)} available (target: {target_simple})")
    print(f"  Intermediate: Sampled {n_intermediate} of {len(intermediate_cases)} available (target: {target_intermediate})")
    print(f"  Complex: Sampled {n_complex} of {len(complex_cases)} available (target: {target_complex})")
    print(f"  Total: {len(final_sample)} cases")
    
    return final_sample


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main(input_file: str, output_file: str, sample_size: int = 64):
    """
    Main stratification and sampling workflow
    
    Parameters:
    -----------
    input_file : str
        Path to input Excel file
    output_file : str
        Path for output Excel file
    sample_size : int
        Total desired sample size (default 64 for Approach B)
    """
    
    print("="*70)
    print("KHCC CASE COMPLEXITY-BASED STRATIFICATION")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_excel(input_file)
    print(f"✓ Loaded {len(df)} cases")
    
    # Calculate complexity for all cases
    print("\nCalculating case complexity scores...")
    
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
    print("COMPLEXITY DISTRIBUTION (Full Dataset)")
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
    
    # Calculate stratified sample targets
    simple_pct = complexity_counts.get('Simple', 0) / len(df)
    intermediate_pct = complexity_counts.get('Intermediate', 0) / len(df)
    complex_pct = complexity_counts.get('Complex', 0) / len(df)
    
    target_simple = int(sample_size * simple_pct)
    target_intermediate = int(sample_size * intermediate_pct)
    target_complex = sample_size - target_simple - target_intermediate
    
    print(f"\n" + "="*70)
    print(f"STRATIFIED SAMPLING (n={sample_size})")
    print("="*70)
    
    # Generate stratified sample
    sample_df = stratified_sample_by_complexity(
        df=df,
        target_simple=target_simple,
        target_intermediate=target_intermediate,
        target_complex=target_complex,
        seed=42
    )
    
    # Save full stratified dataset
    print(f"\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Prepare columns for output
    output_cols = [
        'Case_ID', 'MRN',
        'Complexity_Level', 'Complexity_Score',
        'Comorbidity_Count', 
        'Has_Renal_Impairment', 'Renal_Severity',
        'Has_Hepatic_Impairment', 'Hepatic_Severity',
        'Has_Cardiac_Dysfunction',
        'Has_Hematologic_Abnormality', 'Hematologic_Issues_Count',
        'Regimen', 'CycleNumber',
        'BSA_m2', 'Height_cm', 'Weight_kg',
        'eGFR_mL_min_1.73m2', 'Creatinine_mg_dL',
        'AST_U_L', 'ALT_U_L', 'TotalBilirubin_mg_dL',
        'WBC_x10^9_L', 'ANC_x10^9_L', 'Hemoglobin_g_dL', 'Platelets_x10^9_L',
        'LVEF_percent',
        'Comorbidity_DM', 'Comorbidity_HTN', 'Comorbidity_CAD',
        'Comorbidity_CKD', 'Comorbidity_Asthma', 'Comorbidity_Depression',
        'Note_Original', 'Recommendations_Only'
    ]
    
    # Include only existing columns
    available_cols = [col for col in output_cols if col in sample_df.columns]
    
    # Save sample
    sample_df[available_cols].to_excel(output_file, index=False)
    print(f"✓ Stratified sample saved: {output_file}")
    print(f"  Total cases: {len(sample_df)}")
    
    # Save full dataset with complexity scores
    full_output = output_file.replace('.xlsx', '_FULL_DATASET.xlsx')
    df[available_cols].to_excel(full_output, index=False)
    print(f"✓ Full dataset with complexity scores saved: {full_output}")
    
    # Create summary report
    report_file = output_file.replace('.xlsx', '_REPORT.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("KHCC BREAST CANCER CASE COMPLEXITY STRATIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Random seed: 42\n\n")
        
        f.write("STRATIFICATION METHODOLOGY\n")
        f.write("-"*70 + "\n")
        f.write("Cases stratified based on clinical complexity score calculated from:\n")
        f.write("  1. Comorbidity count (1 point each)\n")
        f.write("  2. Renal impairment (1-2 points based on severity)\n")
        f.write("  3. Hepatic impairment (1-2 points based on severity)\n")
        f.write("  4. Cardiac dysfunction (2-3 points based on LVEF)\n")
        f.write("  5. Hematologic abnormalities (1-3 points total)\n\n")
        
        f.write("Classification thresholds:\n")
        f.write("  Simple: Score 0-1\n")
        f.write("  Intermediate: Score 2-4\n")
        f.write("  Complex: Score ≥5\n\n")
        
        f.write("="*70 + "\n")
        f.write("FULL DATASET RESULTS (n=200)\n")
        f.write("="*70 + "\n\n")
        
        f.write("Complexity Distribution:\n")
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
        
        f.write("="*70 + "\n")
        f.write(f"STRATIFIED SAMPLE (n={len(sample_df)})\n")
        f.write("="*70 + "\n\n")
        
        sample_complexity_counts = sample_df['Complexity_Level'].value_counts()
        f.write("Sample Complexity Distribution:\n")
        for level in ['Simple', 'Intermediate', 'Complex']:
            count = sample_complexity_counts.get(level, 0)
            pct = 100 * count / len(sample_df)
            f.write(f"  {level:15s}: {count:3d} ({pct:5.1f}%)\n")
        
        f.write(f"\nSampling Proportions:\n")
        f.write(f"  Target: Maintain population proportions\n")
        f.write(f"  Simple: {target_simple}/{complexity_counts.get('Simple', 0)} ({100*target_simple/complexity_counts.get('Simple', 1):.1f}% of available)\n")
        f.write(f"  Intermediate: {target_intermediate}/{complexity_counts.get('Intermediate', 0)} ({100*target_intermediate/complexity_counts.get('Intermediate', 1):.1f}% of available)\n")
        f.write(f"  Complex: {target_complex}/{complexity_counts.get('Complex', 0)} ({100*target_complex/max(complexity_counts.get('Complex', 1), 1):.1f}% of available)\n")
    
    print(f"✓ Summary report saved: {report_file}")
    
    print("\n" + "="*70)
    print("STRATIFICATION COMPLETE!")
    print("="*70)
    print(f"\nFiles generated:")
    print(f"  1. {output_file} - Stratified sample (n={len(sample_df)})")
    print(f"  2. {full_output} - Full dataset with complexity scores (n={len(df)})")
    print(f"  3. {report_file} - Detailed report")
    
    return df, sample_df


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "/mnt/user-data/uploads/khcc_cases_200__2_.xlsx"
    OUTPUT_FILE = "/mnt/user-data/outputs/KHCC_Stratified_Sample_n64.xlsx"
    SAMPLE_SIZE = 64
    
    # Run stratification
    full_df, sample_df = main(INPUT_FILE, OUTPUT_FILE, SAMPLE_SIZE)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review sampled cases for data quality")
    print("2. Generate AI notes (GPT, Claude, DeepSeek, CADSS) for 64 cases")
    print("3. Prepare blinded evaluation forms")
    print("4. Conduct human pharmacist evaluation")
    print("5. Perform statistical analysis using Data_Analysis_Guide.md")
