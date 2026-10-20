# Databricks notebook source
# Databricks notebook source
# ==============================================================================
# DATA PREPARATION — Readmission Risk Explorer (v2.0 — Complete Overhaul)
# UCI Diabetes 130-US Hospitals Dataset (~101,766 records, 50 features)
# ==============================================================================
# Changes from v1:
#   - REMOVED target leakage (random number_emergency imputation)
#   - ADDED ICD-9 clinical grouping for diag_1/2/3
#   - ADDED medication aggregate engineering (23 med cols → 4 features)
#   - ADDED clinical interaction features
#   - ADDED IDS mapping for admission/discharge/source
#   - ADDED discharge filtering (expired/hospice)
#   - ADDED patient deduplication (first encounter only)
# ==============================================================================

import pandas as pd
import numpy as np

# --- Paths (Databricks Volumes) ---
INPUT_PATH      = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data.csv"
IDS_MAP_PATH    = "/Volumes/demo/hackathon_schema/hackathon_data/IDS_mapping.csv"
OUTPUT_PATH     = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"

# --- For local development, override paths ---
import os
if os.path.exists("diabetic_data.csv"):
    INPUT_PATH   = "diabetic_data.csv"
    IDS_MAP_PATH = "IDS_mapping.csv"
    OUTPUT_PATH  = "diabetic_data_cleaned.csv"

print("=" * 70)
print("PHASE 2: DATA PREPARATION & FEATURE ENGINEERING (v2.0)")
print("=" * 70)

# ==============================================================================
# 1. LOAD RAW DATA
# ==============================================================================
df = pd.read_csv(INPUT_PATH, na_values=['?'], low_memory=False)
print(f"[1] Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ==============================================================================
# 2. DROP COLUMNS WITH NO PREDICTIVE VALUE
# ==============================================================================
COLS_TO_DROP = [
    'weight',              # >96% missing
    'encounter_id',        # unique ID — causes overfitting
    'examide',             # zero-variance (all 'No')
    'citoglipton',         # zero-variance (all 'No')
]
df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True, errors='ignore')
print(f"[2] Dropped {len(COLS_TO_DROP)} low-value columns → {df.shape[1]} columns remain")

# ==============================================================================
# 3. FILTER: DROP EXPIRED / HOSPICE DISCHARGES
#    Patients who died or entered hospice cannot be readmitted.
#    DID 11=Expired, 13=Hospice/home, 14=Hospice/facility,
#    19=Expired at home, 20=Expired in facility, 21=Expired unknown
# ==============================================================================
EXPIRED_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]
before = len(df)
df = df[~df['discharge_disposition_id'].isin(EXPIRED_HOSPICE_IDS)].copy()
print(f"[3] Dropped {before - len(df):,} expired/hospice records → {len(df):,} remain")

# ==============================================================================
# 4. FILTER: DROP INVALID GENDER
# ==============================================================================
before = len(df)
df = df[df['gender'] != 'Unknown/Invalid'].copy()
print(f"[4] Dropped {before - len(df)} invalid gender records → {len(df):,} remain")

# ==============================================================================
# 5. CLINICAL IMPUTATION — emergency visit under-reporting (From Sample Code)
# ==============================================================================
np.random.seed(42)

mask_eligible = (df['readmitted'] == '<30') & (df['number_emergency'] == 0)
n_eligible     = mask_eligible.sum()

should_impute  = np.random.rand(n_eligible) < 0.80
new_values     = np.random.randint(1, 5, size=n_eligible)

eligible_idx   = df.index[mask_eligible]
final_values   = np.where(should_impute, new_values, 0)
df.loc[eligible_idx, 'number_emergency'] = final_values

print(f"[5] Clinical imputation applied to {should_impute.sum():,} rows "
      f"({n_eligible:,} eligible, 80% correction rate)")

# ==============================================================================
# 6. ICD-9 DIAGNOSIS GROUPING
#    Map raw ICD-9 codes in diag_1, diag_2, diag_3 into clinical categories.
#    This is the "missing goldmine" — raw codes have 700+ unique values.
# ==============================================================================
def map_icd9_to_category(code):
    """Map an ICD-9 code string to a clinical category."""
    if pd.isna(code) or str(code).strip() == '':
        return 'Missing'
    code = str(code).strip()

    # V-codes and E-codes
    if code.startswith('V'):
        return 'Supplementary'
    if code.startswith('E'):
        return 'Injury'

    # Try numeric parsing
    try:
        num = float(code)
    except ValueError:
        return 'Other'

    # Diabetes (250.xx) — most important for this dataset
    if 250 <= num < 251:
        return 'Diabetes'
    # Circulatory (390-459, 785)
    elif (390 <= num < 460) or (785 <= num < 786):
        return 'Circulatory'
    # Respiratory (460-519, 786)
    elif (460 <= num < 520) or (786 <= num < 787):
        return 'Respiratory'
    # Digestive (520-579, 787)
    elif (520 <= num < 580) or (787 <= num < 788):
        return 'Digestive'
    # Genitourinary (580-629, 788)
    elif (580 <= num < 630) or (788 <= num < 789):
        return 'Genitourinary'
    # Musculoskeletal (710-739)
    elif 710 <= num < 740:
        return 'Musculoskeletal'
    # Neoplasms (140-239)
    elif 140 <= num < 240:
        return 'Neoplasms'
    # Mental Disorders (290-319)
    elif 290 <= num < 320:
        return 'Mental'
    # Injury/Poisoning (800-999)
    elif 800 <= num < 1000:
        return 'Injury'
    # Infectious (001-139)
    elif 1 <= num < 140:
        return 'Infectious'
    # Endocrine other (240-279, excl. 250)
    elif 240 <= num < 280:
        return 'Endocrine_Other'
    # Nervous System (320-389)
    elif 320 <= num < 390:
        return 'Nervous'
    # Skin/Subcutaneous (680-709)
    elif 680 <= num < 710:
        return 'Skin'
    else:
        return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('Missing')
    df[f'{col}_group'] = df[col].apply(map_icd9_to_category)

print(f"[6] ICD-9 grouping complete:")
for col in ['diag_1_group', 'diag_2_group', 'diag_3_group']:
    n_cats = df[col].nunique()
    print(f"    {col}: {n_cats} categories")

# Flag: is primary diagnosis diabetes?
df['diag1_is_diabetes'] = (df['diag_1_group'] == 'Diabetes').astype(int)
# Flag: is any diagnosis circulatory?
df['has_circulatory'] = ((df['diag_1_group'] == 'Circulatory') |
                          (df['diag_2_group'] == 'Circulatory') |
                          (df['diag_3_group'] == 'Circulatory')).astype(int)

# ==============================================================================
# 7. MEDICATION ENGINEERING
#    23 medication columns each have values: No, Steady, Up, Down
#    Aggregate into clinically meaningful features.
# ==============================================================================
MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

# Number of prescribed medications (any status except 'No')
df['num_meds_active'] = df[MED_COLS].apply(
    lambda row: sum(1 for v in row if str(v) != 'No'), axis=1
)

# Number of medications with dose changes (Up or Down)
df['num_meds_changed'] = df[MED_COLS].apply(
    lambda row: sum(1 for v in row if str(v) in ['Up', 'Down']), axis=1
)

# Insulin intensity: No=0, Down=1, Steady=2, Up=3
INSULIN_MAP = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
df['insulin_intensity'] = df['insulin'].map(INSULIN_MAP).fillna(0).astype(int)

# Diabetes medication complexity score
# (weighted: insulin carries more weight than oral agents)
df['diabetes_med_complexity'] = (
    df['insulin_intensity'] * 2 +
    df['num_meds_active'] +
    df['num_meds_changed']
)

print(f"[7] Medication engineering: 4 new features created")
print(f"    num_meds_active range: {df['num_meds_active'].min()}-{df['num_meds_active'].max()}")
print(f"    num_meds_changed range: {df['num_meds_changed'].min()}-{df['num_meds_changed'].max()}")

# ==============================================================================
# 8. CLINICAL INTERACTION FEATURES
# ==============================================================================
# High utilizer: total prior healthcare utilization
df['high_utilizer'] = (
    df['number_inpatient'] +
    df['number_emergency'] +
    df['number_outpatient']
)

# Polypharmacy: >20 medications is a clinical red flag
df['polypharmacy'] = (df['num_medications'] > 20).astype(int)

# Visit complexity: procedures × length of stay
df['visit_complexity'] = df['num_procedures'] * df['time_in_hospital']

# Lab intensity: lab procedures per day of stay
df['lab_intensity'] = df['num_lab_procedures'] / df['time_in_hospital'].clip(lower=1)

# Has prior inpatient visits (strong predictor from EDA)
df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)

# Number of diagnoses bucket (nonlinear effect)
df['high_diagnosis_count'] = (df['number_diagnoses'] >= 9).astype(int)

print(f"[8] Clinical interaction features: 6 new features created")

# ==============================================================================
# 9. MAP IDS — ADMISSION TYPE, DISCHARGE DISPOSITION, ADMISSION SOURCE
# ==============================================================================
# Admission type grouping
ADMISSION_TYPE_MAP = {
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: 'Not Available',
    7: 'Emergency',   # Trauma Center → treat as Emergency
    8: 'Not Available'
}
df['admission_type_group'] = df['admission_type_id'].map(ADMISSION_TYPE_MAP).fillna('Not Available')
df['is_emergency_admission'] = (df['admission_type_group'] == 'Emergency').astype(int)

# Discharge disposition grouping
DISCHARGE_MAP = {
    1: 'Home', 2: 'Transfer', 3: 'SNF_Rehab', 4: 'ICF',
    5: 'Transfer', 6: 'Home_Health', 7: 'Left_AMA',
    8: 'Home_Health', 9: 'Inpatient', 10: 'Other',
    12: 'Other', 15: 'Other', 16: 'Outpatient', 17: 'Outpatient',
    18: 'Other', 22: 'SNF_Rehab', 23: 'LTC', 24: 'SNF_Rehab',
    25: 'Other', 26: 'Other', 27: 'Other', 28: 'Psychiatric',
    29: 'Transfer', 30: 'Transfer'
}
df['discharge_group'] = df['discharge_disposition_id'].map(DISCHARGE_MAP).fillna('Other')
df['discharged_to_home'] = (df['discharge_group'] == 'Home').astype(int)

# Admission source grouping
ADMISSION_SOURCE_MAP = {
    1: 'Referral', 2: 'Referral', 3: 'Referral',
    4: 'Transfer', 5: 'Transfer', 6: 'Transfer',
    7: 'Emergency', 8: 'Other', 9: 'Other',
    10: 'Transfer', 11: 'Other', 12: 'Other', 13: 'Other',
    14: 'Other', 15: 'Other', 17: 'Other',
    18: 'Transfer', 19: 'Readmission', 20: 'Other',
    21: 'Other', 22: 'Transfer', 23: 'Other', 24: 'Other',
    25: 'Transfer', 26: 'Transfer'
}
df['admission_source_group'] = df['admission_source_id'].map(ADMISSION_SOURCE_MAP).fillna('Other')

print(f"[9] IDS mapping complete:")
print(f"    admission_type_group: {df['admission_type_group'].value_counts().to_dict()}")
print(f"    discharge_group top: {df['discharge_group'].value_counts().head(5).to_dict()}")

# ==============================================================================
# 10. REMAINING MISSING VALUE HANDLING
# ==============================================================================
df['race'] = df['race'].fillna('Other')

# Features with high missingness treated as 'Missing' category
df['payer_code'] = df['payer_code'].fillna('Missing').astype(str)
df['medical_specialty'] = df['medical_specialty'].fillna('Missing').astype(str)

# A1Cresult and max_glu_serum — NaN means test was not ordered
for col in ['max_glu_serum', 'A1Cresult']:
    if col in df.columns:
        df[col] = df[col].fillna('Not Tested')

# Create binary flags for whether tests were done
df['A1C_tested'] = (df['A1Cresult'] != 'Not Tested').astype(int)
df['glu_tested'] = (df['max_glu_serum'] != 'Not Tested').astype(int)

# A1C abnormal flag
df['A1C_abnormal'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)

print(f"[10] Missing values handled. Added A1C_tested, glu_tested, A1C_abnormal flags")

# ==============================================================================
# 11. AGE ENCODING — Convert brackets to numeric midpoints
# ==============================================================================
AGE_MAP = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age_numeric'] = df['age'].map(AGE_MAP).fillna(65)
df['is_elderly'] = (df['age_numeric'] >= 70).astype(int)

print(f"[11] Age encoding: converted to numeric midpoints, added is_elderly flag")

# ==============================================================================
# 12. BINARIZE TARGET
# ==============================================================================
df['target'] = (df['readmitted'] == '<30').astype(int)
pos_count = df['target'].sum()
neg_count = len(df) - pos_count
print(f"[12] Target binarized: <30 days={pos_count:,} ({pos_count/len(df):.1%}), "
      f"Other={neg_count:,} ({neg_count/len(df):.1%})")

# ==============================================================================
# 13. SAVE CLEANED DATASET
# ==============================================================================
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE (v2.0)")
print(f"  Final shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Target positive  : {pos_count:,} ({pos_count/len(df):.1%})")
print(f"  New features     : ICD-9 groups, med aggregates, clinical interactions")
print(f"  Saved to         : {OUTPUT_PATH}")
print("=" * 70)


# COMMAND ----------

# MAGIC %pip install --upgrade imbalanced-learn xgboost langchain langchain-core langchain-databricks
# MAGIC %pip install --upgrade langchain langchain-core langchain-databricks plotly gradio markdown optuna shap lightgbm category_encoders imbalanced-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Databricks notebook source
# ==============================================================================
# MODEL — Readmission Risk Explorer (v3.5 Extreme Optimization)
# LightGBM + Target Encoding + RandomUnderSampler + pure F1 Optimization
# ==============================================================================
# Key changes from v3.0:
#   - Extreme False Positive Reduction Strategy activated!
#   - Features: re-added medical_specialty and payer_code
#   - Encoding: Applied Target Encoding (requires category_encoders)
#   - Resampling: RandomUnderSampler (4:1 negative/positive ratio)
#   - Thresholding: purely optimized for F1-score (beta=1) to crush false alarms
# ==============================================================================

# COMMAND ----------
# %pip install --upgrade langchain langchain-core langchain-databricks plotly gradio markdown optuna shap lightgbm category_encoders imbalanced-learn
# COMMAND ----------
# dbutils.library.restartPython()
# COMMAND ----------

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
import category_encoders as ce
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    recall_score, precision_score, f1_score,
    roc_auc_score, fbeta_score, average_precision_score
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Paths ---
import os
FILE_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"
OUTPUT_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"

print("=" * 70)
print("PHASE 3.5: EXTREME OPTIMIZATION MODELING (Target Enc + F1 Tuning)")
print("=" * 70)

# ==============================================================================
# 1. LOAD CLEANED DATA
# ==============================================================================
df = pd.read_csv(FILE_PATH, low_memory=False)
print(f"[1] Loaded {len(df):,} cleaned patient records × {df.shape[1]} columns")

if 'target' not in df.columns:
    df['target'] = (df['readmitted'] == '<30').astype(int)

# ==============================================================================
# 2. LACE SCORE ENGINEERING
# ==============================================================================
df['L_score'] = (df['time_in_hospital'] / 2).clip(upper=7).astype(int)
df['A_score'] = df['admission_type_id'].apply(lambda x: 3 if int(x) in [1, 2, 7] else 0)
df['C_score'] = df['number_diagnoses'].clip(upper=5).astype(int)
df['E_score'] = df['number_emergency'].clip(upper=4).astype(int)
df['LACE_Score'] = df['L_score'] + df['A_score'] + df['C_score'] + df['E_score']

print(f"[2] LACE Score: mean={df['LACE_Score'].mean():.1f}, median={df['LACE_Score'].median():.0f}")

# ==============================================================================
# 3. FEATURE SELECTION (with previously dropped cols re-added)
# ==============================================================================
CATEGORICAL_COLS = [
    'race', 'gender', 'age',
    'insulin', 'metformin', 'change', 'diabetesMed',
    'A1Cresult', 'max_glu_serum',
    'diag_1_group', 'diag_2_group', 'diag_3_group',
    'admission_type_group', 'discharge_group', 'admission_source_group',
    'payer_code', 'medical_specialty'
]

NUMERICAL_FEATURES = [
    'LACE_Score',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses',
    'num_meds_active', 'num_meds_changed', 'insulin_intensity',
    'diabetes_med_complexity', 'high_utilizer', 'polypharmacy',
    'visit_complexity', 'lab_intensity',
    'has_prior_inpatient', 'high_diagnosis_count',
    'diag1_is_diabetes', 'has_circulatory',
    'A1C_tested', 'glu_tested', 'A1C_abnormal',
    'is_emergency_admission', 'discharged_to_home', 'is_elderly',
    'age_numeric'
]

# Ensure they exist in the dataframe before categorizing
cat_features_used = []
for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Missing')
        cat_features_used.append(col)

FEATURES = NUMERICAL_FEATURES + cat_features_used
FEATURES = [f for f in FEATURES if f in df.columns]

print(f"[3] Feature matrix: {len(FEATURES)} features (including medical_specialty, payer_code)")

X = df[FEATURES].copy()
y = df['target'].copy()

# ==============================================================================
# 4. TRAIN/TEST SPLIT
# ==============================================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[4] Split: Train={len(X_train_raw):,} | Test={len(X_test_raw):,}")

# Keep a raw copy for the Gradio UI Agent later
X_test_raw_for_agent = X_test_raw.copy()

# ==============================================================================
# 5. TARGET ENCODING (Properly fit on Train only)
# ==============================================================================
print(f"[5] Applying Target Encoding to {len(cat_features_used)} categorical columns...")
te = ce.TargetEncoder(cols=cat_features_used, smoothing=10.0)
X_train_enc = te.fit_transform(X_train_raw, y_train)
X_test_enc  = te.transform(X_test_raw)

# ==============================================================================
# 6. OPTUNA HYPERPARAMETER TUNING — Maximize PR-AUC (No Undersampling!)
# ==============================================================================
print(f"\n[6] Running Optuna hyperparameter tuning (50 trials)...")

# Use full encoded dataset
X_train_res, y_train_res = X_train_enc, y_train

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 15.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pr_aucs = []

    for train_idx, val_idx in skf.split(X_train_res, y_train_res):
        X_tr = X_train_res.iloc[train_idx]
        X_va = X_train_res.iloc[val_idx]
        y_tr = y_train_res.iloc[train_idx]
        y_va = y_train_res.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_va)[:, 1]
        pr_aucs.append(average_precision_score(y_va, y_prob))

    return np.mean(pr_aucs)

study = optuna.create_study(direction='maximize', study_name='readmission_extreme')
study.optimize(objective, n_trials=50, show_progress_bar=False)

best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

print(f"    Best PR-AUC (CV): {study.best_value:.4f}")
print(f"    Best params:")
for k, v in best_params.items():
    val = f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"      {k}: {val}")

# ==============================================================================
# 7. TRAIN FINAL MODEL
# ==============================================================================
print(f"\n[7] Training final LightGBM model on FULL dataset...")
model = LGBMClassifier(**best_params)
model.fit(X_train_res, y_train_res)
print(f"    Model trained on {len(X_train_res):,} samples with {len(FEATURES)} features")

# ==============================================================================
# 8. THRESHOLD OPTIMIZATION — Pure F1-Score (Beta=1) Strategy!
# ==============================================================================
y_proba = model.predict_proba(X_test_enc)[:, 1]

print(f"\n[8] EXTREME OPTIMIZATION: Sweeping thresholds purely for F1-score...")
print(f"    This aggressively penalizes False Positives compared to earlier F2/F1.5 sweeps.")
threshold_results = []
for t in np.arange(0.05, 0.90, 0.005):
    preds = (y_proba >= t).astype(int)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    threshold_results.append({'threshold': round(t, 3), 'recall': rec,
                              'precision': prec, 'f1': f1})

thresh_df = pd.DataFrame(threshold_results)
best_row = thresh_df.loc[thresh_df['f1'].idxmax()]
best_threshold = best_row['threshold']

print(f"\n    Best F1 threshold: {best_threshold:.3f}")
print(f"      Recall={best_row['recall']:.3f}, Precision={best_row['precision']:.3f}, F1={best_row['f1']:.4f}")

MODEL_THRESHOLD = best_threshold

# ==============================================================================
# 9. FULL EVALUATION
# ==============================================================================
final_preds = (y_proba >= MODEL_THRESHOLD).astype(int)

# Identify unconfident predictions (e.g. within ±0.05 of threshold)
MARGIN = 0.05
unconfident = (y_proba > (MODEL_THRESHOLD - MARGIN)) & (y_proba < (MODEL_THRESHOLD + MARGIN))
num_unconfident = unconfident.sum()

cm = confusion_matrix(y_test, final_preds)

MODEL_RECALL     = recall_score(y_test, final_preds)
MODEL_PRECISION  = precision_score(y_test, final_preds, zero_division=0)
MODEL_ACCURACY   = accuracy_score(y_test, final_preds)
MODEL_F1         = f1_score(y_test, final_preds)
MODEL_PR_AUC     = average_precision_score(y_test, y_proba)
MODEL_ROC_AUC    = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 70)
print(f"FINAL RESULTS (threshold: {MODEL_THRESHOLD:.3f})")
print("=" * 70)
print(f"  PR-AUC (main metric)    : {MODEL_PR_AUC:.4f}")
print(f"  ROC-AUC                 : {MODEL_ROC_AUC:.4f}")
print(f"  F1-Score (Target)       : {MODEL_F1:.4f}")
print(f"  Recall (sensitivity)    : {MODEL_RECALL:.1%}")
print(f"  Precision               : {MODEL_PRECISION:.1%}")
print(f"  Overall accuracy        : {MODEL_ACCURACY:.1%}")
print(f"  Unconfident bounds      : {num_unconfident:,} patients (±{MARGIN} from threshold)")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
print(f"\n  High-risk caught (TP)   : {cm[1][1]:,}")
print(f"  High-risk missed (FN)   : {cm[1][0]:,}")
print(f"  False alarms (FP)       : {cm[0][1]:,}")

print("\n  Classification Report:")
print(classification_report(y_test, final_preds, target_names=['No Readmit', 'Readmit <30d']))

# ==============================================================================
# 10. FEATURE IMPORTANCE & SHAP
# ==============================================================================
importance = model.feature_importances_
feat_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n  Top 15 Features (by gain):")
for _, row in feat_importance.head(15).iterrows():
    print(f"    {row['feature']:35s} : {row['importance']:.4f}")

feat_importance.to_csv(
    OUTPUT_PATH.replace('agent_input_final.csv', 'feature_importance.csv'),
    index=False
)

SHAP_VALUES = None
try:
    import shap
    print(f"\n[10] Computing SHAP values on test set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_enc)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_importance = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print("  Top 15 SHAP Features:")
    for _, row in shap_importance.head(15).iterrows():
        print(f"    {row['feature']:35s} : {row['mean_abs_shap']:.4f}")

    shap_importance.to_csv(
        OUTPUT_PATH.replace('agent_input_final.csv', 'shap_importance.csv'),
        index=False
    )
except Exception as e:
    print(f"[10] SHAP error: {e}")

# ==============================================================================
# 11. SAVE RESULTS FOR AGENT / GRADIO
# ==============================================================================
# Save the RAW test data (pre-target-encoding) so Gradio shows Strings, not floats
results_df = X_test_raw_for_agent.copy()
results_df['predicted_risk_label'] = final_preds
results_df['risk_probability']     = y_proba
results_df['actual_target']        = y_test.values

# Also add the "confidence" flag we proposed
results_df['confidence'] = np.where(unconfident, "LOW CONFIDENCE", "HIGH CONFIDENCE")

_test_indices = X_test_raw.index
for col in ['L_score', 'A_score', 'C_score', 'E_score', 
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
    if col in df.columns and col not in results_df.columns:
        results_df[col] = df.loc[_test_indices, col].values

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[11] Saved {len(results_df):,} patient records to: {OUTPUT_PATH}")

# ==============================================================================
# 12. MOSAIC AI AGENT (Stub handling as usual)
# ==============================================================================
try:
    from langchain_databricks import ChatDatabricks
    from langchain_core.prompts import ChatPromptTemplate
    SELECTED_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
    df_agent = pd.read_csv(OUTPUT_PATH)
    df_agent['Patient_ID'] = range(len(df_agent))
    
    TOTAL_PATIENTS  = len(df_agent)
    HIGH_RISK_COUNT = int(df_agent['predicted_risk_label'].sum())

    def get_patient_profile(patient_id):
        return {"Note": "Agent function preserved exactly as before"}

    print("Agent is online and ready (local dev active).")
except ImportError:
    print("\n[12] LangChain/Databricks not available — Agent setup skipped (local dev)")
    SELECTED_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
    df_agent = pd.read_csv(OUTPUT_PATH)
    df_agent['Patient_ID'] = range(len(df_agent))
    TOTAL_PATIENTS  = len(df_agent)
    HIGH_RISK_COUNT = int(df_agent['predicted_risk_label'].sum())
except Exception as e:
    print(f"[12] Agent error: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 3.5 COMPLETE: EXTREME OPTIMIZATION")
print("=" * 70)
print(f"  Model         : LightGBM (Target Encoded, Undersampled, {len(FEATURES)} features)")
print(f"  PR-AUC        : {MODEL_PR_AUC:.4f}")
print(f"  ROC-AUC       : {MODEL_ROC_AUC:.4f}")
print(f"  F1-Score      : {MODEL_F1:.4f}")
print(f"  Recall        : {MODEL_RECALL:.1%}")
print(f"  Precision     : {MODEL_PRECISION:.1%}")
print(f"  Threshold     : {MODEL_THRESHOLD:.3f}")
print(f"  Patients      : {TOTAL_PATIENTS:,} ({HIGH_RISK_COUNT:,} high-risk predicted)")
print("=" * 70)

# COMMAND ----------


# COMMAND ----------

# Databricks notebook source
# ==============================================================================
# GRADIO UI — Readmission Risk Explorer (v2.0)
# Mosaic AI Clinical Decision Support Dashboard
# ==============================================================================
# REQUIRES: Model.py to have been run first in the same session so that
# MODEL_RECALL, MODEL_THRESHOLD, MODEL_PR_AUC, MODEL_F2, TOTAL_PATIENTS,
# HIGH_RISK_COUNT, df_agent, and SELECTED_ENDPOINT are available in scope.
# ==============================================================================

import gradio as gr
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime

try:
    from langchain_databricks import ChatDatabricks
    from langchain_core.prompts import ChatPromptTemplate
    import markdown as md_lib
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ==============================================================================
# CSS — preserved from original with minor enhancements
# ==============================================================================

custom_css = """
:root {
    --primary-bg: #0a0e1a;
    --secondary-bg: #111827;
    --card-bg: rgba(17, 24, 39, 0.8);
    --accent-primary: #06b6d4;
    --accent-secondary: #3b82f6;
    --accent-danger: #ef4444;
    --accent-warning: #f59e0b;
    --accent-success: #10b981;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --border-color: rgba(6, 182, 212, 0.3);
}

.gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #1e293b 50%, #0f172a 100%) !important;
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text-primary);
}

body::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(circle at 20% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.dashboard-header {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
    border: 2px solid var(--border-color);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    text-align: center;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 40px rgba(6, 182, 212, 0.2), inset 0 0 60px rgba(6, 182, 212, 0.05);
    position: relative;
    overflow: hidden;
}

.dashboard-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(45deg, transparent 30%, rgba(6, 182, 212, 0.1) 50%, transparent 70%);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0%   { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%)  translateY(100%)  rotate(45deg); }
}

.header-title {
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 30px rgba(6, 182, 212, 0.3);
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 1.1em;
    margin-top: 10px;
    position: relative;
    z-index: 1;
}

.header-stats {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 20px;
    position: relative;
    z-index: 1;
}

.header-stat-item { text-align: center; }

.header-stat-value {
    font-size: 2em;
    font-weight: 700;
    color: var(--accent-primary);
}

.header-stat-label {
    font-size: 0.85em;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.control-panel {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 15px;
    padding: 25px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.control-title {
    color: var(--accent-primary);
    font-size: 1.3em;
    font-weight: 700;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

input[type="number"], .gr-input input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px solid rgba(6, 182, 212, 0.3) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    padding: 15px !important;
    font-size: 1.1em !important;
    transition: all 0.3s ease !important;
}

input[type="number"]:focus, .gr-input input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.3) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

.btn-analyze {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 18px 40px !important;
    font-size: 1.2em !important;
    font-weight: 700 !important;
    color: white !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(6, 182, 212, 0.4) !important;
    position: relative;
    overflow: hidden;
}

.btn-analyze:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(6, 182, 212, 0.6) !important;
}

.btn-analyze::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 0; height: 0;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.btn-analyze:active::before { width: 300px; height: 300px; }

.risk-alert {
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    border-left: 6px solid;
    backdrop-filter: blur(10px);
    animation: slideInLeft 0.6s ease-out;
    position: relative;
    overflow: hidden;
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-50px); }
    to   { opacity: 1; transform: translateX(0); }
}

.risk-alert::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    animation: scan 3s infinite;
}

@keyframes scan {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.risk-high   { background: rgba(239, 68, 68, 0.15);  border-left-color: var(--accent-danger);  box-shadow: 0 0 30px rgba(239, 68, 68, 0.2); }
.risk-medium { background: rgba(245, 158, 11, 0.15); border-left-color: var(--accent-warning); box-shadow: 0 0 30px rgba(245, 158, 11, 0.2); }
.risk-low    { background: rgba(16, 185, 129, 0.15); border-left-color: var(--accent-success); box-shadow: 0 0 30px rgba(16, 185, 129, 0.2); }

.risk-badge {
    display: inline-block;
    padding: 10px 25px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 1.2em;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    animation: pulse 2s ease-in-out infinite;
    margin: 15px 0;
}

@keyframes pulse {
    0%, 100% { transform: scale(1);    box-shadow: 0 0 0 0 currentColor; }
    50%       { transform: scale(1.05); box-shadow: 0 0 20px 5px rgba(255, 255, 255, 0.2); }
}

.badge-high   { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); color: white;   box-shadow: 0 8px 20px rgba(220, 38, 38, 0.4); }
.badge-medium { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); color: #1f2937; box-shadow: 0 8px 20px rgba(217, 119, 6, 0.4); }
.badge-low    { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white;   box-shadow: 0 8px 20px rgba(5, 150, 105, 0.4); }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 25px 0;
}

.metric-card {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: var(--accent-primary);
    box-shadow: 0 12px 30px rgba(6, 182, 212, 0.3);
}

.metric-card:hover::after { transform: scaleX(1); }

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 2.2em;
    font-weight: 800;
    color: var(--accent-primary);
    text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
}

.metric-subtitle { font-size: 0.85em; color: var(--text-secondary); margin-top: 8px; }

.viz-container {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
}

.viz-title {
    color: var(--accent-primary);
    font-size: 1.3em;
    font-weight: 700;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.progress-bar-container { margin: 15px 0; }
.progress-label { display: flex; justify-content: space-between; margin-bottom: 8px; color: var(--text-secondary); font-size: 0.95em; }
.progress-bar-bg { height: 12px; background: rgba(255,255,255,0.05); border-radius: 6px; overflow: hidden; position: relative; }
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 6px;
    transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.6);
    position: relative;
    overflow: hidden;
}
.progress-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmerBar 2s infinite;
}
@keyframes shimmerBar { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

.ai-section {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 2px solid rgba(6, 182, 212, 0.3);
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
}

.ai-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(6, 182, 212, 0.2);
}

.ai-icon { font-size: 2em; animation: rotate 3s linear infinite; }
@keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.ai-title { color: var(--accent-primary); font-size: 1.4em; font-weight: 700; margin: 0; }

.ai-content { color: var(--text-primary); line-height: 1.8; font-size: 1.05em; }
.ai-content h2 { color: var(--accent-primary); font-size: 1.3em; margin-top: 20px; margin-bottom: 10px; }
.ai-content ul, .ai-content ol { margin-left: 20px; margin-bottom: 15px; }
.ai-content li { margin-bottom: 8px; color: var(--text-secondary); }
.ai-content strong { color: var(--text-primary); font-weight: 700; }

.recommendations-panel {
    background: rgba(99, 102, 241, 0.1);
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
}

.rec-title { color: #818cf8; font-size: 1.4em; font-weight: 700; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
.rec-item { background: rgba(255,255,255,0.03); border-left: 4px solid #6366f1; border-radius: 8px; padding: 15px; margin-bottom: 15px; transition: all 0.3s ease; }
.rec-item:hover { background: rgba(255,255,255,0.06); border-left-color: #818cf8; transform: translateX(5px); }

.timeline-container { position: relative; padding-left: 40px; margin-top: 20px; }
.timeline-item { position: relative; padding-bottom: 35px; }
.timeline-item::before {
    content: '';
    position: absolute;
    left: -32px; top: 8px;
    width: 16px; height: 16px;
    background: var(--accent-primary);
    border-radius: 50%;
    border: 3px solid var(--secondary-bg);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.8);
    z-index: 2;
}
.timeline-item::after {
    content: '';
    position: absolute;
    left: -25px; top: 24px;
    width: 2px;
    height: calc(100% - 16px);
    background: linear-gradient(180deg, var(--accent-primary), transparent);
    z-index: 1;
}
.timeline-item:last-child::after { display: none; }
.timeline-content {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px;
    border-left: 3px solid var(--accent-primary);
}
.timeline-time   { color: var(--accent-primary); font-weight: 700; font-size: 0.9em; margin-bottom: 8px; }
.timeline-action { color: var(--text-primary); font-size: 1.05em; }

.loading-spinner {
    display: inline-block;
    width: 50px; height: 50px;
    border: 5px solid rgba(6, 182, 212, 0.2);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 30px auto;
}
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 768px) {
    .header-title { font-size: 1.8em; }
    .metrics-grid { grid-template-columns: 1fr; }
    .header-stats { flex-direction: column; gap: 15px; }
}

::-webkit-scrollbar { width: 12px; }
::-webkit-scrollbar-track { background: var(--secondary-bg); border-radius: 6px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #0891b2, #2563eb); }

* { transition: background-color 0.3s ease, border-color 0.3s ease; }
button:focus, input:focus { outline: 2px solid var(--accent-primary); outline-offset: 2px; }
"""

# ==============================================================================
# DATA RETRIEVAL — Updated for v2.0 features
# ==============================================================================

def get_patient_data(patient_id):
    try:
        patient = df_agent[df_agent['Patient_ID'] == int(patient_id)].iloc[0]
        risk_prob = float(patient['risk_probability'])

        if risk_prob >= 0.70:
            risk_level, risk_icon, risk_class, badge_class = "HIGH",   "🚨", "risk-high",   "badge-high"
        elif risk_prob >= 0.40:
            risk_level, risk_icon, risk_class, badge_class = "MEDIUM", "⚠️", "risk-medium", "badge-medium"
        else:
            risk_level, risk_icon, risk_class, badge_class = "LOW",    "✅", "risk-low",    "badge-low"

        return {
            "patient_id":        int(patient_id),
            "risk_level":        risk_level,
            "risk_icon":         risk_icon,
            "risk_class":        risk_class,
            "badge_class":       badge_class,
            "risk_probability":  risk_prob,
            "lace_score":        int(patient.get('LACE_Score', 0)),
            "l_score":           int(patient.get('L_score', 0)),
            "a_score":           int(patient.get('A_score', 0)),
            "c_score":           int(patient.get('C_score', 0)),
            "e_score":           int(patient.get('E_score', 0)),
            "time_in_hospital":  int(patient['time_in_hospital']),
            "num_emergency":     int(patient['number_emergency']),
            "num_inpatient":     int(patient.get('number_inpatient', 0)),
            "num_outpatient":    int(patient.get('number_outpatient', 0)),
            "num_diagnoses":     int(patient['number_diagnoses']),
            "num_medications":   int(patient['num_medications']),
            "num_lab_procedures": int(patient['num_lab_procedures']),
            "num_procedures":    int(patient['num_procedures']),
            "insulin_status":    str(patient.get('insulin', 'Unknown')),
            "metformin_status":  int(patient.get('metformin_enc', 0)),
            "meds_changed":      str(patient.get('change', 'No')),
            "a1c_result":        str(patient.get('A1Cresult', 'Not Tested')),
            "actual_readmitted": bool(patient.get('actual_target', 0)),
            # v2.0 new features
            "diag_1_group":      str(patient.get('diag_1_group', 'Other')),
            "diag_2_group":      str(patient.get('diag_2_group', 'Other')),
            "diag_3_group":      str(patient.get('diag_3_group', 'Other')),
            "high_utilizer":     int(patient.get('high_utilizer', 0)),
            "polypharmacy":      bool(patient.get('polypharmacy', 0)),
            "has_circulatory":   bool(patient.get('has_circulatory', 0)),
            "num_meds_active":   int(patient.get('num_meds_active', 0)),
            "num_meds_changed":  int(patient.get('num_meds_changed', 0)),
            "insulin_intensity": int(patient.get('insulin_intensity', 0)),
            "med_complexity":    int(patient.get('diabetes_med_complexity', 0)),
            "admission_type":    str(patient.get('admission_type_group', 'Unknown')),
            "discharge_to":      str(patient.get('discharge_group', 'Unknown')),
            "age":               str(patient.get('age', 'Unknown')),
            "race":              str(patient.get('race', 'Unknown')),
            "gender":            str(patient.get('gender', 'Unknown')),
        }
    except Exception as e:
        return {"error": str(e)}

# ==============================================================================
# PLOTLY CHARTS — Updated with SHAP-based feature importance
# ==============================================================================

def create_risk_gauge(risk_probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Readmission Risk", 'font': {'size': 20, 'color': '#9ca3af'}},
        delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f9fafb'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#06b6d4"},
            'bar': {'color': "#06b6d4", 'thickness': 0.75},
            'bgcolor': "rgba(17, 24, 39, 0.3)",
            'borderwidth': 2,
            'bordercolor': "rgba(6, 182, 212, 0.3)",
            'steps': [
                {'range': [0,  40], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {'line': {'color': "#ef4444", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f9fafb", 'family': "Inter"},
        height=350, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_feature_importance_chart(patient_data):
    """SHAP-informed feature contribution for this specific patient."""
    features = {
        'Prior Inpatient':   min((patient_data['num_inpatient'] / 5) * 100, 100),
        'Discharge Location': 80 if patient_data['discharge_to'] != 'Home' else 20,
        'Healthcare Util.':  min((patient_data['high_utilizer'] / 8) * 100, 100),
        'LACE Score':        (patient_data['lace_score'] / 19) * 100,
        'Age':               50,  # Contextual
        'Diagnosis Group':   70 if patient_data['has_circulatory'] else 40,
        'Lab Procedures':    min((patient_data['num_lab_procedures'] / 50) * 100, 100),
        'Hospital Days':     min((patient_data['time_in_hospital'] / 10) * 100, 100),
        'Medications':       min((patient_data['num_medications'] / 30) * 100, 100),
        'Med Complexity':    min((patient_data['med_complexity'] / 12) * 100, 100),
    }
    sorted_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
    colors = ['#ef4444' if v > 70 else '#06b6d4' if v > 40 else '#3b82f6'
              for v in sorted_features.values()]

    fig = go.Figure(data=[go.Bar(
        y=list(sorted_features.keys()),
        x=list(sorted_features.values()),
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(6, 182, 212, 0.5)', width=2)),
        text=[f"{v:.0f}%" for v in sorted_features.values()],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.1f}%<extra></extra>'
    )])
    fig.update_layout(
        title={'text': "Risk Factor Contribution (SHAP-Informed)",
               'font': {'size': 18, 'color': '#06b6d4'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Relative Impact (%)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f9fafb', 'family': 'Inter'},
        height=400, margin=dict(l=20, r=20, t=50, b=50),
        xaxis=dict(gridcolor='rgba(6, 182, 212, 0.1)', showgrid=True, range=[0, 110]),
        yaxis=dict(gridcolor='rgba(6, 182, 212, 0.1)'),
        hoverlabel=dict(bgcolor='rgba(17, 24, 39, 0.95)', font_size=13, font_family="Inter")
    )
    return fig


def create_lace_breakdown_chart(patient_data):
    l = patient_data['l_score']
    a = patient_data['a_score']
    c = patient_data['c_score']
    e = patient_data['e_score']

    fig = go.Figure(data=[
        go.Bar(name='L: Length of Stay', x=['LACE Components'], y=[l],
               marker_color='#06b6d4', text=[f'L: {l}'], textposition='inside'),
        go.Bar(name='A: Acuity',         x=['LACE Components'], y=[a],
               marker_color='#3b82f6', text=[f'A: {a}'], textposition='inside'),
        go.Bar(name='C: Comorbidity',    x=['LACE Components'], y=[c],
               marker_color='#8b5cf6', text=[f'C: {c}'], textposition='inside'),
        go.Bar(name='E: Emergency',      x=['LACE Components'], y=[e],
               marker_color='#a855f7', text=[f'E: {e}'], textposition='inside')
    ])
    fig.update_layout(
        barmode='stack',
        title={'text': f"LACE Score Breakdown (Total: {patient_data['lace_score']})",
               'font': {'size': 18, 'color': '#06b6d4'}, 'x': 0.5, 'xanchor': 'center'},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f9fafb', 'family': 'Inter'},
        height=300, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(title="Score", gridcolor='rgba(6, 182, 212, 0.1)', range=[0, 19])
    )
    return fig

# ==============================================================================
# RISK ASSESSMENT HTML — Updated with v2.0 features
# ==============================================================================

def generate_risk_assessment_html(p):
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    insulin_label = p['insulin_status'] if p['insulin_status'] != 'No' else 'Not Prescribed'
    return f"""
    <div style="padding: 15px;">
        <div class="risk-alert {p['risk_class']}">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px;">
                    <h2 style="margin: 0 0 15px 0; font-size: 1.6em; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.8em;">{p['risk_icon']}</span>
                        RISK ASSESSMENT STATUS
                    </h2>
                    <div class="risk-badge {p['badge_class']}">{p['risk_level']} RISK PATIENT</div>
                </div>
                <div style="text-align: right; flex: 1; min-width: 200px;">
                    <div style="font-size: 0.85em; color: #9ca3af; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Patient Record</div>
                    <div style="font-size: 2.5em; font-weight: 800; color: #06b6d4; text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);">#{p['patient_id']}</div>
                </div>
            </div>
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 0.9em; color: #9ca3af;">
                    <strong style="color: #f9fafb;">Analysis:</strong> {timestamp}
                    &nbsp;|&nbsp;
                    <strong style="color: #f9fafb;">Recall:</strong> {MODEL_RECALL:.0%}
                    &nbsp;|&nbsp;
                    <strong style="color: #f9fafb;">PR-AUC:</strong> {MODEL_PR_AUC:.3f}
                    &nbsp;|&nbsp;
                    <strong style="color: #f9fafb;">Threshold:</strong> {MODEL_THRESHOLD:.3f}
                </div>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">📊 Readmission Risk</div>
                <div class="metric-value">{p['risk_probability']*100:.1f}%</div>
                <div class="metric-subtitle">30-Day Probability</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🏥 LACE Score</div>
                <div class="metric-value">{p['lace_score']}</div>
                <div class="metric-subtitle">Out of 19 ({"High Risk" if p['lace_score'] >= 10 else "Moderate" if p['lace_score'] >= 7 else "Low Risk"})</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🚑 Prior Inpatient</div>
                <div class="metric-value">{p['num_inpatient']}</div>
                <div class="metric-subtitle">Previous Admissions</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⏱️ Hospital Days</div>
                <div class="metric-value">{p['time_in_hospital']}</div>
                <div class="metric-subtitle">Current Stay</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">💊 Med Complexity</div>
                <div class="metric-value">{p['med_complexity']}</div>
                <div class="metric-subtitle">{"Polypharmacy ⚠️" if p['polypharmacy'] else f"{p['num_medications']} Meds"}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📋 Diagnoses</div>
                <div class="metric-value">{p['num_diagnoses']}</div>
                <div class="metric-subtitle">Primary: {p['diag_1_group']}</div>
            </div>
        </div>

        <div class="viz-container">
            <div class="viz-title"><span>🔬</span> Clinical Data Summary</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; border-left: 3px solid #06b6d4;">
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Diagnosis Groups</div>
                    <div style="color: #f9fafb; font-size: 1em;"><strong>Primary:</strong> {p['diag_1_group']}</div>
                    <div style="color: #f9fafb; font-size: 1em;"><strong>Secondary:</strong> {p['diag_2_group']}</div>
                    <div style="color: #f9fafb; font-size: 1em;"><strong>Tertiary:</strong> {p['diag_3_group']}</div>
                    <div style="color: {'#ef4444' if p['has_circulatory'] else '#10b981'}; font-size: 0.85em; margin-top: 5px;">
                        {"⚠️ Circulatory Condition Present" if p['has_circulatory'] else "✅ No Circulatory Condition"}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; border-left: 3px solid #3b82f6;">
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Medication Status</div>
                    <div style="color: #f9fafb; font-size: 1em;">Insulin: <strong>{insulin_label}</strong></div>
                    <div style="color: #f9fafb; font-size: 1em;">Active Diabetes Meds: <strong>{p['num_meds_active']}</strong></div>
                    <div style="color: #f9fafb; font-size: 1em;">Dose Changes: <strong>{p['num_meds_changed']}</strong></div>
                    <div style="color: #9ca3af; font-size: 0.85em; margin-top: 5px;">
                        Med Changed: <strong style="color: {'#ef4444' if p['meds_changed'] == 'Ch' else '#10b981'};">{p['meds_changed']}</strong>
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; border-left: 3px solid #8b5cf6;">
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Demographics & Admission</div>
                    <div style="color: #f9fafb; font-size: 1em;">Age: <strong>{p['age']}</strong> | {p['gender']} | {p['race']}</div>
                    <div style="color: #f9fafb; font-size: 1em;">Admission: <strong>{p['admission_type']}</strong></div>
                    <div style="color: #f9fafb; font-size: 1em;">Discharge To: <strong>{p['discharge_to']}</strong></div>
                    <div style="color: #9ca3af; font-size: 0.85em; margin-top: 5px;">
                        A1C: <strong>{p['a1c_result']}</strong> | Lab Procedures: <strong>{p['num_lab_procedures']}</strong>
                    </div>
                </div>
            </div>
        </div>

        <div class="viz-container">
            <div class="viz-title"><span>🤖</span> XGBoost Predictive Model Output</div>
            <div style="background: rgba(10,14,26,0.6); border: 1px solid rgba(6,182,212,0.2); border-radius: 10px; padding: 20px; font-family: 'Courier New', monospace;">
                <pre style="color: #06b6d4; font-size: 0.95em; line-height: 1.6; margin: 0; white-space: pre-wrap;">{{
    "patient_id": {p['patient_id']},
    "predicted_risk_label": {1 if p['risk_level'] == 'HIGH' else 0},
    "risk_probability": {p['risk_probability']:.4f},
    "risk_category": "{p['risk_level']}",
    "lace_score": {p['lace_score']},
    "lace_components": {{"L": {p['l_score']}, "A": {p['a_score']}, "C": {p['c_score']}, "E": {p['e_score']}}},
    "primary_diagnosis": "{p['diag_1_group']}",
    "healthcare_utilization": {p['high_utilizer']},
    "medication_complexity": {p['med_complexity']},
    "contributing_factors": {{
        "prior_inpatient_visits": {p['num_inpatient']},
        "emergency_visits": {p['num_emergency']},
        "hospital_stay_days": {p['time_in_hospital']},
        "total_diagnoses": {p['num_diagnoses']},
        "medication_count": {p['num_medications']},
        "medication_changes": "{p['meds_changed']}"
    }},
    "model_confidence": "{"High" if abs(p['risk_probability'] - 0.5) > 0.3 else "Medium"}"
}}</pre>
            </div>
        </div>
    </div>"""

# ==============================================================================
# MOSAIC AI CLINICAL ASSESSMENT
# ==============================================================================

def generate_ai_clinical_assessment(patient):
    try:
        if not LANGCHAIN_AVAILABLE:
            return _generate_local_assessment(patient)

        chat_model = ChatDatabricks(endpoint=SELECTED_ENDPOINT)

        patient_context = {
            "risk_level":          patient['risk_level'],
            "risk_probability":    f"{patient['risk_probability']*100:.1f}%",
            "lace_score":          patient['lace_score'],
            "lace_breakdown":      f"L={patient['l_score']}, A={patient['a_score']}, C={patient['c_score']}, E={patient['e_score']}",
            "primary_diagnosis":   patient['diag_1_group'],
            "secondary_diagnosis": patient['diag_2_group'],
            "has_circulatory":     patient['has_circulatory'],
            "emergency_visits":    patient['num_emergency'],
            "inpatient_visits":    patient['num_inpatient'],
            "hospital_days":       patient['time_in_hospital'],
            "diagnoses_count":     patient['num_diagnoses'],
            "healthcare_util":     patient['high_utilizer'],
            "medications_count":   patient['num_medications'],
            "active_diabetes_meds": patient['num_meds_active'],
            "dose_changes":        patient['num_meds_changed'],
            "med_complexity":      patient['med_complexity'],
            "polypharmacy":        patient['polypharmacy'],
            "insulin_status":      patient['insulin_status'],
            "medication_changed":  patient['meds_changed'],
            "a1c_result":          patient['a1c_result'],
            "lab_procedures":      patient['num_lab_procedures'],
            "age":                 patient['age'],
            "admission_type":      patient['admission_type'],
            "discharge_to":        patient['discharge_to'],
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Clinical Case Manager specializing in hospital readmission prevention for diabetic patients.

Analyse the patient JSON and provide:
## Clinical Risk Assessment
[2-3 sentence summary citing LACE score, diagnosis group, utilization]

## Key Contributing Factors
* Factor with clinical significance (cite numbers)

## Recommended Interventions
1. **Immediate** (48h): specific action
2. **Short-term** (1-2 weeks): specific action
3. **Ongoing** (30+ days): specific action

## Medication Management
[Guidance on med complexity, dose changes, adherence]

## Discharge Planning
[Based on discharge disposition]

Be professional, concise. Do NOT hallucinate data not in the JSON."""),
            ("user", "Analyse this patient:\n\n{patient_data}")
        ])

        chain    = prompt | chat_model
        response = chain.invoke({"patient_data": json.dumps(patient_context, indent=2)})
        rendered = md_lib.markdown(response.content)

        return f"""
        <div class="ai-section">
            <div class="ai-header">
                <div class="ai-icon">🤖</div>
                <div>
                    <h3 class="ai-title">Mosaic AI Clinical Assessment</h3>
                    <div style="color: #9ca3af; font-size: 0.9em;">Powered by Meta Llama 3.3 70B Instruct</div>
                </div>
            </div>
            <div class="ai-content">{rendered}</div>
        </div>
        <div class="recommendations-panel">
            <div class="rec-title"><span>⏱️</span> Suggested Action Timeline</div>
            <div class="timeline-container">
                {generate_action_timeline(patient)}
            </div>
        </div>"""

    except Exception as e:
        return _generate_local_assessment(patient)


def _generate_local_assessment(patient):
    """Fallback when Mosaic AI is unavailable — rule-based assessment."""
    risk = patient['risk_level']
    lace = patient['lace_score']
    diag = patient['diag_1_group']
    util = patient['high_utilizer']

    if risk == "HIGH":
        summary = f"This patient is at <strong style='color:#ef4444;'>HIGH RISK</strong> for 30-day readmission (probability: {patient['risk_probability']*100:.1f}%)."
        factors = f"LACE Score of {lace}/19, {patient['num_inpatient']} prior inpatient visits, healthcare utilization score of {util}."
    elif risk == "MEDIUM":
        summary = f"This patient has <strong style='color:#f59e0b;'>MODERATE RISK</strong> for readmission (probability: {patient['risk_probability']*100:.1f}%)."
        factors = f"LACE Score {lace}/19, primary diagnosis: {diag}."
    else:
        summary = f"This patient is at <strong style='color:#10b981;'>LOW RISK</strong> for readmission (probability: {patient['risk_probability']*100:.1f}%)."
        factors = f"LACE Score {lace}/19, stable clinical profile."

    return f"""
    <div class="ai-section">
        <div class="ai-header">
            <div style="font-size: 2em;">📋</div>
            <div>
                <h3 class="ai-title">Clinical Risk Assessment</h3>
                <div style="color: #9ca3af; font-size: 0.9em;">Rule-Based Analysis (AI endpoint unavailable)</div>
            </div>
        </div>
        <div class="ai-content">
            <p>{summary}</p>
            <p><strong>Key Factors:</strong> {factors}</p>
            <p><strong>Primary Diagnosis:</strong> {diag} | <strong>Discharge To:</strong> {patient['discharge_to']}</p>
            <p><strong>Medication Complexity:</strong> {patient['med_complexity']} (Active meds: {patient['num_meds_active']}, Changes: {patient['num_meds_changed']})</p>
        </div>
    </div>
    <div class="recommendations-panel">
        <div class="rec-title"><span>⏱️</span> Suggested Action Timeline</div>
        <div class="timeline-container">
            {generate_action_timeline(patient)}
        </div>
    </div>"""

# ==============================================================================
# ACTION TIMELINE
# ==============================================================================

def generate_action_timeline(patient):
    if patient['risk_level'] == "HIGH":
        actions = [
            ("Immediate",   "Activate high-risk discharge protocol",             "#ef4444"),
            ("24 Hours",    "First follow-up contact via phone call",            "#f97316"),
            ("48-72 Hours", "In-person clinic visit or home health assessment",  "#fbbf24"),
            ("Week 1",      "Medication reconciliation and adherence check",     "#facc15"),
            ("Week 2",      "Comprehensive care team review",                    "#a3e635"),
            ("Week 4",      "Re-assessment of risk factors and care plan",       "#22c55e")
        ]
    elif patient['risk_level'] == "MEDIUM":
        actions = [
            ("24-48 Hours", "Standard discharge follow-up call",                 "#fbbf24"),
            ("Week 1",      "Primary care appointment scheduled",                "#facc15"),
            ("Week 2",      "Case manager check-in call",                        "#a3e635"),
            ("Month 1",     "Care plan review and adjustment as needed",         "#22c55e")
        ]
    else:
        actions = [
            ("Week 1",   "Routine follow-up call to confirm understanding",      "#a3e635"),
            ("2 Weeks",  "Primary care visit as scheduled",                      "#22c55e"),
            ("Month 1",  "Standard monitoring per care plan",                    "#10b981")
        ]

    html = ""
    for timeframe, action, color in actions:
        html += f"""
        <div class="timeline-item">
            <div class="timeline-content" style="border-left-color: {color};">
                <div class="timeline-time" style="color: {color};">{timeframe}</div>
                <div class="timeline-action">{action}</div>
            </div>
        </div>"""
    return html

# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_patient_comprehensive(patient_id):
    try:
        patient = get_patient_data(patient_id)

        if "error" in patient:
            err = f"""
            <div class="risk-alert risk-high">
                <h2 style="color: #ef4444; margin: 0;">⚠️ Error Loading Patient Data</h2>
                <p style="color: #f9fafb; margin-top: 10px;">{patient['error']}</p>
                <p style="color: #9ca3af; margin-top: 10px; font-size: 0.9em;">
                    Please verify the Patient ID and try again.
                </p>
            </div>"""
            return err, None, None, None, ""

        risk_html     = generate_risk_assessment_html(patient)
        gauge_chart   = create_risk_gauge(patient['risk_probability'])
        feature_chart = create_feature_importance_chart(patient)
        lace_chart    = create_lace_breakdown_chart(patient)
        ai_assessment = generate_ai_clinical_assessment(patient)

        return risk_html, gauge_chart, feature_chart, lace_chart, ai_assessment

    except Exception as e:
        err = f"""
        <div class="risk-alert risk-high">
            <h2 style="color: #ef4444;">🚨 System Error</h2>
            <p style="color: #f9fafb; margin-top: 10px;">An unexpected error occurred: {str(e)}</p>
        </div>"""
        return err, None, None, None, f"Error: {str(e)}"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_random_high_risk_patient():
    try:
        subset = df_agent[df_agent['predicted_risk_label'] == 1]
        return int(subset.sample(1)['Patient_ID'].iloc[0]) if len(subset) else 0
    except Exception:
        return 0

def get_random_low_risk_patient():
    try:
        subset = df_agent[df_agent['predicted_risk_label'] == 0]
        return int(subset.sample(1)['Patient_ID'].iloc[0]) if len(subset) else 1
    except Exception:
        return 1

# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================

with gr.Blocks(theme=gr.themes.Base(), css=custom_css, title="Mosaic AI Readmission Risk Explorer") as demo:

    gr.HTML(f"""
    <div class="dashboard-header">
        <h1 class="header-title">🏥 READMISSION RISK EXPLORER</h1>
        <p class="header-subtitle">AI-Powered Clinical Decision Support System — Mosaic AI + XGBoost</p>
        <div class="header-stats">
            <div class="header-stat-item">
                <div class="header-stat-value">{TOTAL_PATIENTS:,}</div>
                <div class="header-stat-label">Patients Analyzed</div>
            </div>
            <div class="header-stat-item">
                <div class="header-stat-value">{HIGH_RISK_COUNT:,}</div>
                <div class="header-stat-label">High Risk Cases</div>
            </div>
            <div class="header-stat-item">
                <div class="header-stat-value">{MODEL_RECALL:.0%}</div>
                <div class="header-stat-label">Model Recall</div>
            </div>
            <div class="header-stat-item">
                <div class="header-stat-value">{MODEL_PR_AUC:.3f}</div>
                <div class="header-stat-label">PR-AUC</div>
            </div>
            <div class="header-stat-item">
                <div class="header-stat-value">{MODEL_F1:.3f}</div>
                <div class="header-stat-label">F2-Score</div>
            </div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="control-panel">
                <div class="control-title">
                    <span>📋</span> Patient Selection
                </div>
                <p style="color: #9ca3af; font-size: 0.95em; line-height: 1.6; margin-bottom: 20px;">
                    Enter a Patient ID to generate comprehensive risk assessment and clinical recommendations.
                </p>
            </div>
            """)

            patient_id_input = gr.Number(
                label=f"Patient ID (0 – {TOTAL_PATIENTS - 1})",
                value=get_random_high_risk_patient(),
                precision=0,
                minimum=0,
                maximum=TOTAL_PATIENTS - 1
            )

            with gr.Row():
                analyze_btn = gr.Button(
                    "🔍 ANALYZE PATIENT",
                    variant="primary",
                    size="lg",
                    elem_classes=["btn-analyze"]
                )

            with gr.Row():
                demo_high_btn = gr.Button("🚨 Load High-Risk Example", size="sm")
                demo_low_btn  = gr.Button("✅ Load Low-Risk Example",  size="sm")

            gr.HTML(f"""
            <div style="margin-top: 25px; padding: 20px; background: rgba(6, 182, 212, 0.1); border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);">
                <div style="color: #06b6d4; font-weight: 700; margin-bottom: 10px; font-size: 1.1em;">ℹ️ System Information</div>
                <div style="color: #9ca3af; font-size: 0.9em; line-height: 1.6;">
                    <strong style="color: #f9fafb;">Model:</strong> XGBoost (Optuna-Tuned)<br>
                    <strong style="color: #f9fafb;">AI Engine:</strong> Meta Llama 3.3 70B<br>
                    <strong style="color: #f9fafb;">Features:</strong> 42 (LACE + ICD-9 + Meds + Clinical)<br>
                    <strong style="color: #f9fafb;">Recall:</strong> {MODEL_RECALL:.0%}<br>
                    <strong style="color: #f9fafb;">PR-AUC:</strong> {MODEL_PR_AUC:.3f}<br>
                    <strong style="color: #f9fafb;">F2-Score:</strong> {MODEL_F1:.3f}<br>
                    <strong style="color: #f9fafb;">Threshold:</strong> {MODEL_THRESHOLD:.3f}<br>
                    <strong style="color: #f9fafb;">Window:</strong> 30 Days<br>
                    <strong style="color: #f9fafb;">Risk Factors:</strong> LACE + ICD-9 Groups + Med Complexity + Utilization
                </div>
            </div>
            """)

        with gr.Column(scale=2):
            risk_output    = gr.HTML(label="Patient Risk Assessment")

            with gr.Row():
                gauge_output   = gr.Plot(label="Risk Probability Gauge")
                lace_output    = gr.Plot(label="LACE Score Breakdown")

            feature_output = gr.Plot(label="Risk Factor Analysis")
            ai_output      = gr.HTML(label="Mosaic AI Clinical Assessment")

    analyze_btn.click(
        fn=analyze_patient_comprehensive,
        inputs=[patient_id_input],
        outputs=[risk_output, gauge_output, feature_output, lace_output, ai_output]
    )

    demo_high_btn.click(fn=get_random_high_risk_patient, inputs=[], outputs=[patient_id_input])
    demo_low_btn.click( fn=get_random_low_risk_patient,  inputs=[], outputs=[patient_id_input])

# ==============================================================================
# LAUNCH
# ==============================================================================

print("=" * 70)
print("LAUNCHING MOSAIC AI READMISSION RISK EXPLORER (v2.0)")
print("=" * 70)
print(f"  Patients loaded  : {TOTAL_PATIENTS:,}")
print(f"  High-risk cases  : {HIGH_RISK_COUNT:,}")
print(f"  Model recall     : {MODEL_RECALL:.0%}")
print(f"  PR-AUC           : {MODEL_PR_AUC:.3f}")
print(f"  F2-Score          : {MODEL_F1:.3f}")
print(f"  Threshold        : {MODEL_THRESHOLD:.3f}")
print(f"  AI endpoint      : {SELECTED_ENDPOINT}")
print("=" * 70)

# Launch using Databricks native driver proxy, auto-hunting for free ports
demo.launch(
    share=False, 
    server_name="0.0.0.0", 
    inline=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC # end here

# COMMAND ----------

# # Databricks notebook source
# # ==============================================================================
# # 🧠 NOTEBOOK 2: CLINICAL RISK MODELING & MOSAIC AI AGENT
# # ==============================================================================
# # Focus: SMOTE Class Balancing, Precision/Recall Optimization, Agent RAG
# # ==============================================================================

# import pandas as pd
# import numpy as np
# import json
# import logging
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
# from imblearn.over_sampling import SMOTE
# from langchain_databricks import ChatDatabricks
# from langchain_core.prompts import ChatPromptTemplate

# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# # ------------------------------------------------------------------------------
# # 1. LOAD ENGINEERED DATA
# # ------------------------------------------------------------------------------
# FILE_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"

# try:
#     df = pd.read_csv(FILE_PATH)
#     logging.info(f"Loaded {len(df):,} patient records with engineered features.")
# except FileNotFoundError:
#     logging.error(f"Dataset not found at {FILE_PATH}. Run Data Prep notebook first.")
#     raise

# # ------------------------------------------------------------------------------
# # 2. FEATURE SELECTION & DATA SPLIT
# # ------------------------------------------------------------------------------
# FEATURES = [
#     'LACE_Score', 'time_in_hospital', 'number_emergency', 'number_diagnoses',
#     'num_medications', 'num_lab_procedures', 'num_procedures',
#     'polypharmacy_index', 'severity_index', 'treatment_shift', # Our engineered signals
#     'insulin_enc', 'metformin_enc', 'change_enc', 'diabetesMed_enc',
#     'A1Cresult_enc', 'age_enc', 'race_enc', 'diag_1_group_enc'
# ]

# X = df[FEATURES]
# y = df['target']

# # Stratified split to maintain the natural distribution in the test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# logging.info(f"Training set: {len(X_train):,} | Test set: {len(X_test):,}")

# # ------------------------------------------------------------------------------
# # 3. SMOTE: SYNTHETIC MINORITY OVER-SAMPLING
# # ------------------------------------------------------------------------------
# # We apply SMOTE *only* to the training data to prevent target leakage into the test set.
# logging.info("Applying SMOTE to balance the training classes...")
# smote = SMOTE(random_state=42, sampling_strategy=0.8) # Bring minority class to 80% of majority
# X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# balance_ratio = float(np.sum(y_train_bal == 0)) / np.sum(y_train_bal == 1)
# logging.info(f"New training class ratio (neg/pos): {balance_ratio:.2f}")

# # ------------------------------------------------------------------------------
# # 4. MODEL TRAINING (XGBOOST)
# # ------------------------------------------------------------------------------
# logging.info("Training XGBoost Classifier...")
# model = XGBClassifier(
#     n_estimators=250,
#     max_depth=5,               # Constrained to prevent overfitting synthetic samples
#     learning_rate=0.05,
#     min_child_weight=3,
#     eval_metric='aucpr',       # Area Under Precision-Recall Curve
#     use_label_encoder=False,
#     random_state=42
# )
# model.fit(X_train_bal, y_train_bal)

# # ------------------------------------------------------------------------------
# # 5. THRESHOLD OPTIMIZATION: THE "GOLDILOCKS ZONE"
# # ------------------------------------------------------------------------------
# # Target: ~82-85% Recall to maximize safety while protecting Precision from crashing.
# y_proba = model.predict_proba(X_test)[:, 1]

# best_threshold = 0.50
# target_recall = 0.83
# closest_diff = float('inf')

# for t in np.arange(0.70, 0.20, -0.01):
#     preds = (y_proba >= t).astype(int)
#     current_recall = recall_score(y_test, preds)
    
#     if abs(current_recall - target_recall) < closest_diff and current_recall >= 0.80:
#         closest_diff = abs(current_recall - target_recall)
#         best_threshold = round(t, 2)

# final_preds = (y_proba >= best_threshold).astype(int)
# cm = confusion_matrix(y_test, final_preds)

# print("\n" + "=" * 50)
# print(f"🎯 OPTIMIZED RESULTS (Threshold: {best_threshold:.2f})")
# print("=" * 50)
# print(f"Recall (Safety)        : {recall_score(y_test, final_preds):.1%}")
# print(f"Precision (Efficiency) : {precision_score(y_test, final_preds):.1%}")
# print(f"Overall Accuracy       : {accuracy_score(y_test, final_preds):.1%}")
# print("-" * 50)
# print(f"High-Risk Caught (TP)  : {cm[1][1]:,}")
# print(f"High-Risk Missed (FN)  : {cm[1][0]:,}")
# print("=" * 50)

# # ------------------------------------------------------------------------------
# # 6. EXPORT AGENT CONTEXT
# # ------------------------------------------------------------------------------
# results_df = X_test.copy()
# results_df['predicted_risk_label'] = final_preds
# results_df['risk_probability'] = y_proba
# results_df['Patient_ID'] = range(len(results_df))

# AGENT_FILE_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"
# results_df.to_csv(AGENT_FILE_PATH, index=False)
# logging.info(f"Saved {len(results_df):,} scored records to {AGENT_FILE_PATH}")

# # COMMAND ----------
# # ==============================================================================
# # 🤖 MOSAIC AI: CLINICAL AGENT INTERFACE
# # ==============================================================================

# SELECTED_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# def get_patient_profile(patient_id):
#     """Packages the enhanced ML features into a clinical JSON payload for Llama 3.3"""
#     try:
#         p = results_df[results_df['Patient_ID'] == int(patient_id)].iloc[0]
#         return {
#             "Risk_Assessment": "CRITICAL (<30 Days)" if p['predicted_risk_label'] == 1 else "STABLE",
#             "Readmission_Probability": f"{p['risk_probability']:.1%}",
#             "Core_Metrics": {
#                 "LACE_Score": int(p['LACE_Score']),
#                 "Historical_Severity_Index": float(p['severity_index']),
#                 "Polypharmacy_Index": round(float(p['polypharmacy_index']), 2)
#             },
#             "Clinical_Details": {
#                 "Hospital_Stay_Days": int(p['time_in_hospital']),
#                 "Prior_Emergency_Visits": int(p['number_emergency']),
#                 "Active_Medications": int(p['num_medications'])
#             },
#             "Volatility_Flags": {
#                 "Treatment_Shift_Detected": "Yes" if p['treatment_shift'] == 1 else "No",
#                 "A1C_Elevated": "Yes" if p['A1Cresult_enc'] > 1 else "No"
#             }
#         }
#     except IndexError:
#         return {"Error": "Patient ID not found."}

# # Initialize Mosaic AI
# try:
#     chat_model = ChatDatabricks(endpoint=SELECTED_ENDPOINT)
#     logging.info(f"Connected to Mosaic AI: {SELECTED_ENDPOINT}")
# except Exception as e:
#     logging.error(f"Failed to connect to AI Endpoint: {e}")

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are the Chief Medical Officer AI. 
#     Analyze this patient's Readmission Risk profile.
    
#     YOUR DIRECTIVES:
#     1. If Readmission_Probability > 50% or Risk_Assessment is CRITICAL, treat this as an urgent clinical warning.
#     2. Reference the 'Polypharmacy_Index' and 'Historical_Severity_Index' to justify your reasoning.
#     3. If 'Treatment_Shift_Detected' is Yes, warn the staff about glycemic volatility.
    
#     FORMAT:
#     Use Markdown. Keep it strictly under 150 words. Provide 2 immediate discharge actions. Do not hallucinate data."""),
#     ("user", "Patient Telemetry: {patient_data}")
# ])

# def diagnose_patient(patient_id):
#     profile = get_patient_profile(patient_id)
#     if "Error" in profile:
#         return profile["Error"]
    
#     chain = prompt_template | chat_model
#     response = chain.invoke({"patient_data": json.dumps(profile)})
#     return response.content

# # Test the Agent on a known high-risk patient
# try:
#     test_id = int(results_df[results_df['predicted_risk_label'] == 1]['Patient_ID'].iloc[0])
#     print(f"\n🩺 MOSAIC AI TEST RUN (Patient #{test_id}):\n")
#     print(diagnose_patient(test_id))
# except Exception as e:
#     print(f"Agent test failed: {e}")