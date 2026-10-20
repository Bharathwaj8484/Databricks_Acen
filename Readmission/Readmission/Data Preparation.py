# Databricks notebook source
import pandas as pd
import numpy as np

# input_path = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data.csv"
input_path = "diabetic_data.csv"
# output_path = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"
output_path = "diabetic_data_cleaned.csv"
print("STARTING DATA PREPARATION...")
print(f"Reading from: {input_path}")
df = pd.read_csv(input_path, na_values=['?'])
print(f"Initial Shape: {df.shape}")
cols_to_drop = ['weight', 'payer_code', 'encounter_id', 'examide', 'citoglipton']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# 'medical_specialty': Missing often means General Practice or No Specific Specialist
df['medical_specialty'] = df['medical_specialty'].fillna('GeneralPractice')
# 'race': Missing -> 'Other'
df['race'] = df['race'].fillna('Other')
# 'diag_1,2,3': If missing, assume no diagnosis (use a placeholder code or 0)
for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('0')

#  Lab Results "None" -> "Not Tested"
#  'None' isn't missing; it means "Doctor didn't order test".
lab_cols = ['max_glu_serum', 'A1Cresult']
for col in lab_cols:
    if col in df.columns:
        df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')





# 5. FINAL SAFETY CHECK
# Drop any row that still has critical missing info (should be none/very few)
df = df.dropna(subset=['gender', 'age', 'admission_type_id'])

# 6. SAVE
print(f"Saving Cleaned Dataset to: {output_path}")
df.to_csv(output_path, index=False)
print("DATA PREPARATION COMPLETE.")
print(f"Final Shape: {df.shape}")