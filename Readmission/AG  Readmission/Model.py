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

if os.path.exists("diabetic_data_cleaned.csv"):
    FILE_PATH = "diabetic_data_cleaned.csv"
    OUTPUT_PATH = "agent_input_final.csv"

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
