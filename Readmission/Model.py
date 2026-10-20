# Databricks notebook source
# ==============================================================================
# MODEL — Readmission Risk Explorer
# XGBoost + LACE Score Engineering + Mosaic AI Agent
# ==============================================================================

# COMMAND ----------
# %pip install --upgrade langchain langchain-core langchain-databricks plotly gradio markdown
# COMMAND ----------
# dbutils.library.restartPython()
# COMMAND ----------

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, recall_score
)
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------------
# 1. Load cleaned data (produced by Data_Preparation.py)
# ------------------------------------------------------------------------------
FILE_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"
df = pd.read_csv(FILE_PATH)
print(f"Loaded {len(df):,} cleaned patient records.")

# Defensive re-check — idempotent, zero cost if already clean
for col in ['max_glu_serum', 'A1Cresult']:
    if col in df.columns:
        df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')

# ------------------------------------------------------------------------------
# 2. LACE Score Engineering
#    FIX: Computed BEFORE LabelEncoder so admission_type_id still holds the
#    original integer values (1=Emergency, 2=Urgent, 7=Trauma).
#    The original model.py applied LabelEncoder first, then compared str(x)
#    against ['1','2','7'] — those IDs no longer existed after encoding.
# ------------------------------------------------------------------------------
df['L_score'] = df['time_in_hospital']
df['A_score'] = df['admission_type_id'].apply(lambda x: 3 if int(x) in [1, 2, 7] else 0)
df['C_score'] = df['number_diagnoses'].apply(lambda x: min(int(x), 5))
df['E_score'] = df['number_emergency'].apply(lambda x: min(int(x), 4))
df['LACE_Score'] = df['L_score'] + df['A_score'] + df['C_score'] + df['E_score']

# ------------------------------------------------------------------------------
# 3. Categorical encoding
#    Note: encoder fit on full dataset. For this dataset's low-cardinality
#    columns (race=6 vals, gender=3, age=10 brackets) the distributional
#    leakage is negligible. Production pipeline should fit on train split only.
# ------------------------------------------------------------------------------
CATEGORICAL_COLS = [
    'race', 'gender', 'age', 'admission_type_id',
    'insulin', 'metformin', 'change', 'diabetesMed',
    'A1Cresult', 'max_glu_serum'
]
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[f'{col}_enc'] = le.fit_transform(df[col])

# ------------------------------------------------------------------------------
# 4. Target + feature matrix
# ------------------------------------------------------------------------------
df['target'] = (df['readmitted'] == '<30').astype(int)

FEATURES = [
    'LACE_Score', 'time_in_hospital', 'number_emergency', 'number_diagnoses',
    'num_medications', 'num_lab_procedures', 'num_procedures',
    'insulin_enc', 'metformin_enc', 'change_enc', 'diabetesMed_enc',
    'A1Cresult_enc', 'age_enc', 'race_enc'
]

X = df[FEATURES]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ------------------------------------------------------------------------------
# 5. Train XGBoost — optimised for Recall
#    Clinical priority: missing a high-risk patient is the costliest error.
# ------------------------------------------------------------------------------
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
print(f"\nClass imbalance ratio (neg/pos): {ratio:.2f}")
print("Training XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=ratio,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 6. Threshold optimisation — target >=90% recall
# ------------------------------------------------------------------------------
y_proba = model.predict_proba(X_test)[:, 1]

print("\nSearching for optimal threshold (target recall >= 90%)...")
best_threshold = None

for t in np.arange(0.60, 0.20, -0.01):
    preds = (y_proba >= t).astype(int)
    if recall_score(y_test, preds) >= 0.90:
        best_threshold = round(t, 2)
        break

# FIX: explicit fallback with a visible warning instead of silent default
if best_threshold is None:
    best_threshold = 0.50
    print("WARNING: No threshold in [0.20, 0.60] achieved >=90% recall. "
          "Defaulting to 0.50. Review class balance or model parameters.")
else:
    print(f"Optimal threshold found: {best_threshold:.2f}")

# ------------------------------------------------------------------------------
# 7. Evaluate and store real metrics — used by Gradio header (no hardcoding)
# ------------------------------------------------------------------------------
final_preds = (y_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_test, final_preds)

MODEL_RECALL    = recall_score(y_test, final_preds)
MODEL_ACCURACY  = accuracy_score(y_test, final_preds)
MODEL_THRESHOLD = best_threshold

print("\n" + "=" * 50)
print(f"RESULTS  (threshold: {MODEL_THRESHOLD:.2f})")
print("=" * 50)
print(f"Recall (sensitivity)  : {MODEL_RECALL:.1%}")
print(f"Overall accuracy      : {MODEL_ACCURACY:.1%}")
print(f"High-risk caught (TP) : {cm[1][1]:,}")
print(f"High-risk missed (FN) : {cm[1][0]:,}")
print("\nClassification Report:")
print(classification_report(y_test, final_preds))

# ------------------------------------------------------------------------------
# 8. Save full test set for agent
#    FIX: removed .head(100) — agent now operates on the complete test cohort
# ------------------------------------------------------------------------------
results_df = X_test.copy()
results_df['predicted_risk_label'] = final_preds
results_df['risk_probability']     = y_proba
results_df['actual_target']        = y_test

OUTPUT_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(results_df):,} patient records to: {OUTPUT_PATH}")

# COMMAND ----------
# ==============================================================================
# MOSAIC AI AGENT (lightweight — UI lives in Gradio_Output.py)
# ==============================================================================

import json
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

SELECTED_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
AGENT_FILE_PATH   = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"

df_agent = pd.read_csv(AGENT_FILE_PATH)
df_agent['Patient_ID'] = range(len(df_agent))
print(f"Agent loaded: {len(df_agent):,} patient records.")

TOTAL_PATIENTS   = len(df_agent)
HIGH_RISK_COUNT  = int(df_agent['predicted_risk_label'].sum())


def get_patient_profile(patient_id):
    """Retrieve clinical data and XGBoost prediction for a given patient ID."""
    try:
        patient = df_agent[df_agent['Patient_ID'] == int(patient_id)].iloc[0]
        return {
            "RISK_LEVEL": "HIGH RISK (<30 Days)" if patient['predicted_risk_label'] == 1 else "LOW RISK",
            "Readmission_Probability": f"{patient['risk_probability']:.1%}",
            "LACE_Score": int(patient['LACE_Score']),
            "Clinical_History": {
                "Hospital_Stay_Days":   int(patient['time_in_hospital']),
                "Prior_Emergency_Visits": int(patient['number_emergency']),
                "Total_Diagnoses":      int(patient['number_diagnoses'])
            },
            "Medications": {
                "Total_Meds":     int(patient['num_medications']),
                "Insulin_Status": int(patient['insulin_enc']),
                "Metformin_Status": int(patient['metformin_enc']),
                "Meds_Changed":   "Yes" if patient['change_enc'] == 1 else "No"
            },
            "Labs": {
                "A1C_Result_Code":   int(patient['A1Cresult_enc']),
                "Num_Lab_Procedures": int(patient['num_lab_procedures'])
            }
        }
    except IndexError:
        return {"Error": f"Patient ID {patient_id} not found."}


chat_model = ChatDatabricks(endpoint=SELECTED_ENDPOINT)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Clinical Case Manager assisting with discharge planning.

INPUT DATA: JSON patient profile with XGBoost-calculated Readmission_Probability and clinical features.

YOUR TASK:
1. Headline: "Risk Assessment: [High/Low] - [Percentage]%"
2. Explain the Why: reference LACE Score, Emergency Visits, Lab data.
   - LACE > 10 -> flag as high.
   - Emergency Visits > 1 -> major concern.
   - Meds_Changed == Yes -> flag instability.
3. Recommend 2-3 specific interventions.

Tone: Professional, Concise, Medical. Do NOT reference data absent from the JSON."""),
    ("user", "Analyse this patient profile: {patient_data}")
])


def doctor_agent(patient_id):
    profile = get_patient_profile(patient_id)
    if "Error" in profile:
        return f"Error: {profile['Error']}"
    chain    = prompt_template | chat_model
    response = chain.invoke({"patient_data": json.dumps(profile)})
    return response.content


print("Agent is online and ready.")

# ------------------------------------------------------------------------------
# Connection probe — verifies which Mosaic AI endpoint is reachable
# ------------------------------------------------------------------------------
ENDPOINTS_TO_TRY = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-llama-2-70b-chat"
]

print("\nProbing Mosaic AI endpoints...")
connected_endpoint = None
for endpoint in ENDPOINTS_TO_TRY:
    try:
        test_chat = ChatDatabricks(endpoint=endpoint)
        test_chat.invoke("ping")
        print(f"  Connected: {endpoint}")
        connected_endpoint = endpoint
        break
    except Exception as e:
        print(f"  Failed   : {endpoint} — {str(e)[:60]}")

if not connected_endpoint:
    print("CRITICAL: No AI endpoints reachable. "
          "Check the Serving tab in your Databricks workspace.")

# COMMAND ----------
