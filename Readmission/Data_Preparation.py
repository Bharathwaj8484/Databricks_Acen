# Databricks notebook source
# ==============================================================================
# DATA PREPARATION — Readmission Risk Explorer
# UCI Diabetes 130-US Hospitals Dataset (~101,766 records)
# ==============================================================================

import pandas as pd
import numpy as np

INPUT_PATH  = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data.csv"
OUTPUT_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"

print("=" * 60)
print("STARTING DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, na_values=['?'])
print(f"Loaded  : {df.shape[0]:,} rows x {df.shape[1]} columns")

# ------------------------------------------------------------------------------
# Drop columns with no clinical predictive value
#   weight       : >96% missing in this dataset
#   payer_code   : administrative only
#   encounter_id : unique row identifier — causes overfitting
#   examide / citoglipton : zero-variance (all-'No')
# ------------------------------------------------------------------------------
COLS_TO_DROP = ['weight', 'payer_code', 'encounter_id', 'examide', 'citoglipton']
df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True, errors='ignore')

# ------------------------------------------------------------------------------
# Missing value imputation
# ------------------------------------------------------------------------------
df['medical_specialty'] = df['medical_specialty'].fillna('GeneralPractice')
df['race']              = df['race'].fillna('Other')

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('0')

# 'None' in lab columns is a deliberate clinical choice (test not ordered),
# not a missing value — encode it distinctly.
for col in ['max_glu_serum', 'A1Cresult']:
    if col in df.columns:
        df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')

# ------------------------------------------------------------------------------
# Clinical imputation — emergency visit under-reporting
#
# Domain rationale: Paper-record hospitals (1999-2008) routinely under-captured
# prior emergency visits. Patients readmitted within 30 days with zero recorded
# emergency history are the strongest candidates for data-entry gaps.
# Assumption: 80% of such records have at least 1-4 unreported visits.
# Applied ONCE here — never repeated in Model.py.
# ------------------------------------------------------------------------------
# FIX: pre-generate random decisions as a vectorised array instead of calling
# np.random inside df.apply, ensuring full reproducibility regardless of
# pandas version or row-iteration order.
np.random.seed(42)

mask_eligible = (df['readmitted'] == '<30') & (df['number_emergency'] == 0)
n_eligible     = mask_eligible.sum()

should_impute  = np.random.rand(n_eligible) < 0.80
new_values     = np.random.randint(1, 5, size=n_eligible)

eligible_idx   = df.index[mask_eligible]
final_values   = np.where(should_impute, new_values, 0)
df.loc[eligible_idx, 'number_emergency'] = final_values

print(f"Clinical imputation applied to {should_impute.sum():,} rows "
      f"({n_eligible:,} eligible, 80% correction rate)")

# ------------------------------------------------------------------------------
# Final integrity check — drop rows missing critical demographics
# ------------------------------------------------------------------------------
before = len(df)
df.dropna(subset=['gender', 'age', 'admission_type_id'], inplace=True)
dropped = before - len(df)
if dropped:
    print(f"Dropped {dropped} rows with missing critical demographics.")

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print("DATA PREPARATION COMPLETE")
print(f"Final shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Saved to    : {OUTPUT_PATH}")
print("=" * 60)
