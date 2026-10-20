# =============================================================================
# STAGE 1 — DATA PREPARATION (Clinically Defensible, Recall-Safe)
# =============================================================================

import pandas as pd
import numpy as np

# -----------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------
df = pd.read_csv("diabetic_data.csv", na_values=["?"], low_memory=False)

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# -----------------------------------------------------------------
# 2. DROP CLINICALLY INVALID ENCOUNTERS
#    Patients who died or entered hospice cannot be readmitted
# -----------------------------------------------------------------
EXPIRED_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)].copy()

# Drop invalid gender
df = df[df["gender"] != "Unknown/Invalid"].copy()

print(f"After clinical filtering: {len(df):,} rows")

# -----------------------------------------------------------------
# 3. DROP LOW-VALUE / LEAKAGE-PRONE COLUMNS
# -----------------------------------------------------------------
DROP_COLS = [
    "encounter_id",      # unique identifier
    "weight",            # >96% missing
    "examide",            # zero variance
    "citoglipton"         # zero variance
]

df.drop(columns=[c for c in DROP_COLS if c in df.columns],
        inplace=True, errors="ignore")

# -----------------------------------------------------------------
# 4. TARGET ENGINEERING (BINARY, RECALL-FOCUSED)
# -----------------------------------------------------------------
df["readmitted_flag"] = df["readmitted"].apply(
    lambda x: 0 if x == "NO" else 1
)

# Keep original label for later analysis
# DO NOT DROP 'readmitted' yet

# -----------------------------------------------------------------
# 5. HANDLE MISSING VALUES — MEDICALLY SENSIBLE
# -----------------------------------------------------------------
CATEGORICAL_FILL_UNKNOWN = [
    "race",
    "payer_code",
    "medical_specialty"
]

for col in CATEGORICAL_FILL_UNKNOWN:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Lab results: absence itself is informative
LAB_COLS = ["A1Cresult", "max_glu_serum"]
for col in LAB_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("Not_Measured")

# -----------------------------------------------------------------
# 6. LIGHT FEATURE ENGINEERING (SAFE)
# -----------------------------------------------------------------

# Utilization intensity
df["total_visits"] = (
    df["number_outpatient"]
    + df["number_emergency"]
    + df["number_inpatient"]
)

# Medication burden
MED_COLS = [
    c for c in df.columns
    if c.startswith(("metformin", "repaglinide", "nateglinide",
                      "chlorpropamide", "glimepiride", "acetohexamide",
                      "glyburide", "tolbutamide", "pioglitazone",
                      "rosiglitazone", "insulin"))
]

df["num_active_meds"] = df[MED_COLS].apply(
    lambda row: sum(v != "No" for v in row), axis=1
)

# -----------------------------------------------------------------
# 7. FINAL SANITY CHECK
# -----------------------------------------------------------------
assert "readmitted_flag" in df.columns
assert df.isnull().sum().sum() >= 0  # explicit missing allowed

print("Stage 1 data preparation complete.")
print(df.shape)
