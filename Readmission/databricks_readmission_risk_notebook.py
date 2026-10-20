# Databricks notebook source
# =============================================================================
# CELL 1: Install Required Dependencies
# =============================================================================
%pip install --upgrade langchain langchain-core langchain-databricks gradio plotly markdown kaleido scikit-learn pyngrok==6.0.0 psutil --quiet

print("Restarting Python to load the new notebook dependencies...")
dbutils.library.restartPython()

# COMMAND ----------

# =============================================================================
# CELL 2: Notebook Configuration
# =============================================================================
import json
import os
import socket
import time
import urllib.request
from datetime import datetime

import gradio as gr
import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.prompts import ChatPromptTemplate
from langchain_databricks import ChatDatabricks
from pyngrok import conf, ngrok
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_text_widget(name: str, default_value: str) -> None:
    try:
        dbutils.widgets.get(name)
        return
    except Exception:
        pass
    try:
        dbutils.widgets.text(name, default_value)
    except Exception:
        pass


ensure_text_widget("input_csv_path", "dbfs:/FileStore/readmission_risk/diabetic_data.csv")
ensure_text_widget(
    "results_output_path",
    "dbfs:/FileStore/readmission_risk/prediction_results.csv",
)
ensure_text_widget(
    "mosaic_endpoint",
    "databricks-meta-llama-3-3-70b-instruct",
)
ensure_text_widget("ngrok_auth_token", "")

INPUT_CSV_PATH = dbutils.widgets.get("input_csv_path").strip()
RESULTS_OUTPUT_PATH = dbutils.widgets.get("results_output_path").strip()
MOSAIC_ENDPOINT = dbutils.widgets.get("mosaic_endpoint").strip()
NGROK_AUTH_TOKEN = dbutils.widgets.get("ngrok_auth_token").strip()
FIXED_THRESHOLD = 0.53


def normalize_local_path(path: str) -> str:
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path[len("dbfs:/") :]
    return path


LOCAL_INPUT_CSV_PATH = normalize_local_path(INPUT_CSV_PATH)
LOCAL_RESULTS_OUTPUT_PATH = normalize_local_path(RESULTS_OUTPUT_PATH)
os.makedirs(os.path.dirname(LOCAL_RESULTS_OUTPUT_PATH), exist_ok=True)

results_df = None
df = None
pipeline = None
demo = None
model_metrics = {}

print("Notebook configuration loaded:")
print(f"  Input CSV path      : {INPUT_CSV_PATH}")
print(f"  Results output path : {RESULTS_OUTPUT_PATH}")
print(f"  Mosaic endpoint     : {MOSAIC_ENDPOINT}")
print(f"  Threshold           : {FIXED_THRESHOLD}")
print("  Ngrok token source  : widget" if NGROK_AUTH_TOKEN else "  Ngrok token source  : env/secret/widget at launch time")

# COMMAND ----------

# =============================================================================
# CELL 3: Clinical Data Preparation (Preserved from ngrok.py)
# =============================================================================
# The preparation logic below intentionally keeps the same core steps used in ngrok.py.

print(f"Loading data from: {INPUT_CSV_PATH}")
df = pd.read_csv(LOCAL_INPUT_CSV_PATH, na_values=["?"], low_memory=False)
print(f"Initial Shape: {df.shape}")

EXPIRED_IDS = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(EXPIRED_IDS)].copy()

df = df[df["gender"] != "Unknown/Invalid"].copy()

DROP_COLS = ["encounter_id", "weight", "examide", "citoglipton", "patient_nbr"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")

CATEGORICAL_FILL = ["race", "payer_code", "medical_specialty"]
for col in CATEGORICAL_FILL:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

LAB_COLS = ["A1Cresult", "max_glu_serum"]
for col in LAB_COLS:
    if col in df.columns:
        df[col] = df[col].replace("None", "Not Tested").fillna("Not Tested")

df["readmitted_flag"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

df["total_visits"] = (
    df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
)

MED_COLS = [
    c for c in df.columns if any(med in c for med in ["metformin", "insulin", "glimepiride"])
]
df["num_active_meds"] = df[MED_COLS].apply(lambda row: sum(v != "No" for v in row), axis=1)

df["L_score"] = df["time_in_hospital"]
df["A_score"] = df["admission_type_id"].apply(lambda x: 3 if str(x) in ["1", "2", "7"] else 0)
df["C_score"] = df["number_diagnoses"].apply(lambda x: min(x, 5))
df["E_score"] = df["number_emergency"].apply(lambda x: min(x, 4))
df["LACE_Score"] = df["L_score"] + df["A_score"] + df["C_score"] + df["E_score"]

print(f"After Clinical Filtering: {df.shape}")
print("Stage 1 Data Preparation Complete! ✅")

# COMMAND ----------

# =============================================================================
# CELL 4: Model Training and Results Persistence (Preserved from ngrok.py)
# =============================================================================
X = df.drop(columns=["readmitted", "readmitted_flag"], errors="ignore")
y = df["readmitted_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 2},
    solver="lbfgs",
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", model),
    ]
)

print("Training Logistic Regression model...")
pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= FIXED_THRESHOLD).astype(int)

print(f"\n=== Evaluation at Threshold {FIXED_THRESHOLD} ===")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}\n")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

actual_positives = tp + fn
actual_negatives = tn + fp

tp_pct = (tp / actual_positives) * 100 if actual_positives > 0 else 0
fn_pct = (fn / actual_positives) * 100 if actual_positives > 0 else 0
tn_pct = (tn / actual_negatives) * 100 if actual_negatives > 0 else 0
fp_pct = (fp / actual_negatives) * 100 if actual_negatives > 0 else 0

print("=== Quick Metrics Summary ===")
print(f"Overall Accuracy   : {acc:.3f}")
print(f"Precision (Class 1): {prec:.3f}")
print(f"Recall (Class 1)   : {rec:.3f}\n")

print("=== Patient Outcome Breakdown ===")
print(f"True Positives (TP) : {tp}  -> ({tp_pct:.0f}%) Correctly found at-risk patients")
print(f"False Negatives (FN): {fn}  -> ({fn_pct:.0f}%) Missed at-risk patients")
print(f"True Negatives (TN) : {tn}  -> ({tn_pct:.0f}%) Correctly cleared patients")
print(f"False Positives (FP): {fp}  -> ({fp_pct:.0f}%) False alarms")

results_df = X_test.copy()
results_df["readmitted_flag"] = y_test.values
results_df["probability"] = y_prob
results_df["predicted_label"] = y_pred
results_df["threshold_used"] = FIXED_THRESHOLD

for col in [
    "time_in_hospital",
    "number_emergency",
    "number_diagnoses",
    "num_medications",
    "LACE_Score",
]:
    if col in df.columns:
        results_df[col] = df.loc[X_test.index, col].values

# Compatibility aliases so the richer UI from Best_Gradio_Design.txt can be reused cleanly.
results_df = results_df.reset_index(drop=True)
results_df.insert(0, "Patient_ID", range(len(results_df)))
results_df["risk_probability"] = results_df["probability"]
results_df["predicted_risk_label"] = results_df["predicted_label"]
results_df["actual_target"] = results_df["readmitted_flag"]

results_df.to_csv(LOCAL_RESULTS_OUTPUT_PATH, index=False)

model_metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "tp": int(tp),
    "fp": int(fp),
    "tn": int(tn),
    "fn": int(fn),
}

print(f"\n✅ Results saved to: {RESULTS_OUTPUT_PATH}")
print(f"✅ Local file path : {LOCAL_RESULTS_OUTPUT_PATH}")
print(f"✅ Test Set Size   : {len(results_df)}")

# COMMAND ----------

# =============================================================================
# CELL 5: Mosaic AI Clinical Agent (Using ngrok.py approach because it matches the trained output schema)
# =============================================================================
chat_model = ChatDatabricks(endpoint=MOSAIC_ENDPOINT)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Senior Clinical Case Manager assisting with discharge planning.
Your goal is to explain Readmission Risk to doctors clearly.

INPUT: Patient JSON profile with risk probability and clinical features.

TASK:
1. Start with a clear risk summary headline
2. Explain why using LACE Score, emergency visits, medications, and labs
3. If LACE > 10, flag it as high concern
4. If emergency visits > 1, highlight it as a major risk factor
5. If medications changed, mention care-plan instability
6. Recommend 2-3 specific interventions

FORMAT: Markdown with headers, bullet points, and bold key terms
TONE: Professional, concise, medical
RULE: Do not hallucinate facts that are not present in the JSON""",
        ),
        ("user", "Analyze this patient profile: {patient_data}"),
    ]
)


def medication_changed(patient_row: pd.Series) -> str:
    if str(patient_row.get("change", "No")) == "Ch":
        return "Yes"
    for med_col in ["insulin", "metformin", "glimepiride"]:
        if str(patient_row.get(med_col, "No")) in ["Up", "Down"]:
            return "Yes"
    return "No"


def get_patient_profile(patient_row: pd.Series) -> dict:
    probability = float(patient_row["probability"])
    threshold = float(patient_row.get("threshold_used", FIXED_THRESHOLD))
    risk_level = "HIGH" if probability >= threshold else "LOW"

    return {
        "RISK_LEVEL": f"{'🔴' if risk_level == 'HIGH' else '🟢'} {risk_level} RISK",
        "Readmission_Probability": f"{probability:.1%}",
        "Threshold_Used": f"{threshold:.2f}",
        "LACE_Score": int(patient_row.get("LACE_Score", 0)),
        "Clinical_History": {
            "Hospital_Stay_Days": int(patient_row.get("time_in_hospital", 0)),
            "Prior_Emergency_Visits": int(patient_row.get("number_emergency", 0)),
            "Total_Diagnoses": int(patient_row.get("number_diagnoses", 0)),
        },
        "Medications": {
            "Total_Meds": int(patient_row.get("num_medications", 0)),
            "Insulin_Status": str(patient_row.get("insulin", "No")),
            "Metformin_Status": str(patient_row.get("metformin", "No")),
            "Meds_Changed": medication_changed(patient_row),
        },
        "Labs": {
            "A1C_Result_Code": str(patient_row.get("A1Cresult", "Not Tested")),
            "Num_Lab_Procedures": int(patient_row.get("num_lab_procedures", 0)),
        },
    }


def generate_rule_based_assessment(patient_row: pd.Series) -> str:
    profile = get_patient_profile(patient_row)
    lace_score = profile["LACE_Score"]
    emergency = profile["Clinical_History"]["Prior_Emergency_Visits"]
    probability = profile["Readmission_Probability"]
    meds_changed = profile["Medications"]["Meds_Changed"]

    recommendations = []
    if lace_score > 10:
        recommendations.append("Schedule follow-up contact within 48 hours of discharge.")
    if emergency > 1:
        recommendations.append("Arrange close outpatient monitoring because prior ED utilization is elevated.")
    if meds_changed == "Yes":
        recommendations.append("Complete medication reconciliation before discharge and re-check adherence.")
    if not recommendations:
        recommendations.append("Maintain standard discharge follow-up and reinforce return precautions.")

    markdown_text = f"""
## Clinical Risk Assessment
**Readmission risk:** {probability}

## Key Contributing Factors
- **LACE Score:** {lace_score}
- **Emergency Visits:** {emergency}
- **Medication Changes:** {meds_changed}
- **A1C Result:** {profile['Labs']['A1C_Result_Code']}

## Recommended Interventions
""" + "\n".join([f"- {item}" for item in recommendations[:3]])

    return markdown.markdown(markdown_text)


def generate_ai_assessment_from_row(patient_row: pd.Series) -> str:
    try:
        profile = get_patient_profile(patient_row)
        chain = prompt_template | chat_model
        response = chain.invoke({"patient_data": json.dumps(profile)})
        return markdown.markdown(response.content)
    except Exception:
        return generate_rule_based_assessment(patient_row)


print(f"✅ Mosaic AI agent configured for endpoint: {MOSAIC_ENDPOINT}")

# COMMAND ----------

# =============================================================================
# CELL 6: Gradio Dashboard UI (Best_Gradio_Design.txt styling + compatible data wiring)
# =============================================================================
custom_css = """
/* ============================================
   GLOBAL THEME: Medical Command Center
   ============================================ */
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

/* Main Container - Dark Medical Theme */
.gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #1e293b 50%, #0f172a 100%) !important;
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text-primary);
}

/* Animated Background Pattern */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image:
        radial-gradient(circle at 20% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ============================================
   HEADER SECTION
   ============================================ */
.dashboard-header {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
    border: 2px solid var(--border-color);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    text-align: center;
    backdrop-filter: blur(20px);
    box-shadow:
        0 0 40px rgba(6, 182, 212, 0.2),
        inset 0 0 60px rgba(6, 182, 212, 0.05);
    position: relative;
    overflow: hidden;
}

.dashboard-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent 30%, rgba(6, 182, 212, 0.1) 50%, transparent 70%);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
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

.header-stat-item {
    text-align: center;
}

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

/* ============================================
   CONTROL PANEL
   ============================================ */
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

/* Input Fields */
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

/* Analyze Button */
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
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.btn-analyze:active::before {
    width: 300px;
    height: 300px;
}

/* ============================================
   RISK ALERT CARDS
   ============================================ */
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
    from {
        opacity: 0;
        transform: translateX(-50px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.risk-alert::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    animation: scan 3s infinite;
}

@keyframes scan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.risk-high {
    background: rgba(239, 68, 68, 0.15);
    border-left-color: var(--accent-danger);
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
}

.risk-medium {
    background: rgba(245, 158, 11, 0.15);
    border-left-color: var(--accent-warning);
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
}

.risk-low {
    background: rgba(16, 185, 129, 0.15);
    border-left-color: var(--accent-success);
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
}

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
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 currentColor;
    }
    50% {
        transform: scale(1.05);
        box-shadow: 0 0 20px 5px rgba(255, 255, 255, 0.2);
    }
}

.badge-high {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    color: white;
    box-shadow: 0 8px 20px rgba(220, 38, 38, 0.4);
}

.badge-medium {
    background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    color: #1f2937;
    box-shadow: 0 8px 20px rgba(217, 119, 6, 0.4);
}

.badge-low {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    box-shadow: 0 8px 20px rgba(5, 150, 105, 0.4);
}

/* ============================================
   METRIC CARDS GRID
   ============================================ */
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
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: var(--accent-primary);
    box-shadow: 0 12px 30px rgba(6, 182, 212, 0.3);
}

.metric-card:hover::after {
    transform: scaleX(1);
}

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

.metric-subtitle {
    font-size: 0.85em;
    color: var(--text-secondary);
    margin-top: 8px;
}

/* ============================================
   DATA VISUALIZATION SECTION
   ============================================ */
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

/* Progress Bars for Feature Importance */
.progress-bar-container {
    margin: 15px 0;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    color: var(--text-secondary);
    font-size: 0.95em;
}

.progress-bar-bg {
    height: 12px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}

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
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmerBar 2s infinite;
}

@keyframes shimmerBar {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ============================================
   AI ASSESSMENT SECTION
   ============================================ */
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

.ai-icon {
    font-size: 2em;
    animation: rotate 3s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.ai-title {
    color: var(--accent-primary);
    font-size: 1.4em;
    font-weight: 700;
    margin: 0;
}

.ai-content {
    color: var(--text-primary);
    line-height: 1.8;
    font-size: 1.05em;
}

.ai-content h2 {
    color: var(--accent-primary);
    font-size: 1.3em;
    margin-top: 20px;
    margin-bottom: 10px;
}

.ai-content ul, .ai-content ol {
    margin-left: 20px;
    margin-bottom: 15px;
}

.ai-content li {
    margin-bottom: 8px;
    color: var(--text-secondary);
}

.ai-content strong {
    color: var(--text-primary);
    font-weight: 700;
}

/* ============================================
   RECOMMENDATIONS PANEL
   ============================================ */
.recommendations-panel {
    background: rgba(99, 102, 241, 0.1);
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
}

.rec-title {
    color: #818cf8;
    font-size: 1.4em;
    font-weight: 700;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.rec-item {
    background: rgba(255, 255, 255, 0.03);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    transition: all 0.3s ease;
}

.rec-item:hover {
    background: rgba(255, 255, 255, 0.06);
    border-left-color: #818cf8;
    transform: translateX(5px);
}

/* ============================================
   TIMELINE VISUALIZATION
   ============================================ */
.timeline-container {
    position: relative;
    padding-left: 40px;
    margin-top: 20px;
}

.timeline-item {
    position: relative;
    padding-bottom: 35px;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -32px;
    top: 8px;
    width: 16px;
    height: 16px;
    background: var(--accent-primary);
    border-radius: 50%;
    border: 3px solid var(--secondary-bg);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.8);
    z-index: 2;
}

.timeline-item::after {
    content: '';
    position: absolute;
    left: -25px;
    top: 24px;
    width: 2px;
    height: calc(100% - 16px);
    background: linear-gradient(180deg, var(--accent-primary), transparent);
    z-index: 1;
}

.timeline-item:last-child::after {
    display: none;
}

.timeline-content {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px;
    border-left: 3px solid var(--accent-primary);
}

.timeline-time {
    color: var(--accent-primary);
    font-weight: 700;
    font-size: 0.9em;
    margin-bottom: 8px;
}

.timeline-action {
    color: var(--text-primary);
    font-size: 1.05em;
}

/* ============================================
   LOADING STATES
   ============================================ */
.loading-spinner {
    display: inline-block;
    width: 50px;
    height: 50px;
    border: 5px solid rgba(6, 182, 212, 0.2);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 30px auto;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ============================================
   RESPONSIVE DESIGN
   ============================================ */
@media (max-width: 768px) {
    .header-title {
        font-size: 1.8em;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .header-stats {
        flex-direction: column;
        gap: 15px;
    }
}

/* ============================================
   ACCESSIBILITY & POLISH
   ============================================ */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: var(--secondary-bg);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #0891b2, #2563eb);
}

/* Smooth Transitions for All Elements */
* {
    transition: background-color 0.3s ease, border-color 0.3s ease;
}

/* Focus States for Accessibility */
button:focus, input:focus {
    outline: 2px solid var(--accent-primary);
    outline-offset: 2px;
}
"""


def ensure_results_loaded() -> pd.DataFrame:
    global results_df
    if results_df is None:
        results_df = pd.read_csv(LOCAL_RESULTS_OUTPUT_PATH)
    return results_df


def get_patient_data(patient_id):
    try:
        patient_id = int(patient_id)
        current_results = ensure_results_loaded()
        patient_row = current_results[current_results["Patient_ID"] == patient_id].iloc[0]

        risk_prob = float(patient_row["risk_probability"])
        if risk_prob >= 0.70:
            risk_level = "HIGH"
            risk_icon = "🚨"
            risk_class = "risk-high"
            badge_class = "badge-high"
        elif risk_prob >= 0.40:
            risk_level = "MEDIUM"
            risk_icon = "⚠️"
            risk_class = "risk-medium"
            badge_class = "badge-medium"
        else:
            risk_level = "LOW"
            risk_icon = "✅"
            risk_class = "risk-low"
            badge_class = "badge-low"

        patient_data = {
            "patient_id": patient_id,
            "risk_level": risk_level,
            "risk_icon": risk_icon,
            "risk_class": risk_class,
            "badge_class": badge_class,
            "risk_probability": risk_prob,
            "predicted_risk_label": int(patient_row["predicted_risk_label"]),
            "threshold_used": float(patient_row["threshold_used"]),
            "lace_score": int(patient_row.get("LACE_Score", 0)),
            "time_in_hospital": int(patient_row.get("time_in_hospital", 0)),
            "num_emergency": int(patient_row.get("number_emergency", 0)),
            "num_diagnoses": int(patient_row.get("number_diagnoses", 0)),
            "num_medications": int(patient_row.get("num_medications", 0)),
            "num_lab_procedures": int(patient_row.get("num_lab_procedures", 0)),
            "num_procedures": int(patient_row.get("num_procedures", 0)),
            "admission_type": int(patient_row.get("admission_type_id", 0)),
            "insulin_status": str(patient_row.get("insulin", "No")),
            "metformin_status": str(patient_row.get("metformin", "No")),
            "meds_changed": medication_changed(patient_row),
            "a1c_result": str(patient_row.get("A1Cresult", "Not Tested")),
            "actual_readmitted": bool(patient_row.get("actual_target", 0)),
            "raw_row": patient_row,
        }
        return patient_data
    except Exception as exc:
        return {"error": str(exc)}


def create_risk_gauge(risk_probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Readmission Risk", "font": {"size": 20, "color": "#9ca3af"}},
            delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            number={"suffix": "%", "font": {"size": 40, "color": "#f9fafb"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 2, "tickcolor": "#06b6d4"},
                "bar": {"color": "#06b6d4", "thickness": 0.75},
                "bgcolor": "rgba(17, 24, 39, 0.3)",
                "borderwidth": 2,
                "bordercolor": "rgba(6, 182, 212, 0.3)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.3)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb", "family": "Inter"},
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_feature_importance_chart(patient_data):
    features = {
        "LACE Score": (patient_data["lace_score"] / 19) * 100,
        "Emergency Visits": min((patient_data["num_emergency"] / 4) * 100, 100),
        "Hospital Days": min((patient_data["time_in_hospital"] / 10) * 100, 100),
        "Medications": min((patient_data["num_medications"] / 30) * 100, 100),
        "Diagnoses": min((patient_data["num_diagnoses"] / 10) * 100, 100),
        "Lab Procedures": min((patient_data["num_lab_procedures"] / 50) * 100, 100),
    }

    sorted_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
    colors = [
        "#06b6d4" if value > 60 else "#3b82f6" if value > 30 else "#6366f1"
        for value in sorted_features.values()
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=list(sorted_features.keys()),
                x=list(sorted_features.values()),
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(6, 182, 212, 0.5)", width=2),
                ),
                text=[f"{value:.1f}%" for value in sorted_features.values()],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Impact: %{x:.1f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Risk Factor Contribution Analysis",
            "font": {"size": 18, "color": "#06b6d4"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Relative Impact (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb", "family": "Inter"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=50),
        xaxis=dict(gridcolor="rgba(6, 182, 212, 0.1)", showgrid=True, range=[0, 110]),
        yaxis=dict(gridcolor="rgba(6, 182, 212, 0.1)"),
        hoverlabel=dict(bgcolor="rgba(17, 24, 39, 0.95)", font_size=13, font_family="Inter"),
    )
    return fig


def create_lace_breakdown_chart(patient_data):
    l_score = patient_data["time_in_hospital"]
    a_score = 3 if patient_data.get("admission_type", 0) in [1, 2, 7] else 0
    c_score = min(patient_data["num_diagnoses"], 5)
    e_score = min(patient_data["num_emergency"], 4)

    fig = go.Figure(
        data=[
            go.Bar(
                name="Length of Stay",
                x=["LACE Components"],
                y=[l_score],
                marker_color="#06b6d4",
                text=[f"L: {l_score}"],
                textposition="inside",
            ),
            go.Bar(
                name="Acuity",
                x=["LACE Components"],
                y=[a_score],
                marker_color="#3b82f6",
                text=[f"A: {a_score}"],
                textposition="inside",
            ),
            go.Bar(
                name="Comorbidity",
                x=["LACE Components"],
                y=[c_score],
                marker_color="#8b5cf6",
                text=[f"C: {c_score}"],
                textposition="inside",
            ),
            go.Bar(
                name="Emergency",
                x=["LACE Components"],
                y=[e_score],
                marker_color="#a855f7",
                text=[f"E: {e_score}"],
                textposition="inside",
            ),
        ]
    )

    fig.update_layout(
        barmode="stack",
        title={
            "text": f"LACE Score Breakdown (Total: {patient_data['lace_score']})",
            "font": {"size": 18, "color": "#06b6d4"},
            "x": 0.5,
            "xanchor": "center",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f9fafb", "family": "Inter"},
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(title="Score", gridcolor="rgba(6, 182, 212, 0.1)", range=[0, 19]),
    )
    return fig


def generate_action_timeline(patient):
    if patient["risk_level"] == "HIGH":
        actions = [
            ("Immediate", "Activate high-risk discharge protocol", "#ef4444"),
            ("24 Hours", "First follow-up contact via phone call", "#f97316"),
            ("48-72 Hours", "In-person clinic visit or home health assessment", "#fbbf24"),
            ("Week 1", "Medication reconciliation and adherence check", "#facc15"),
            ("Week 2", "Comprehensive care team review", "#a3e635"),
            ("Week 4", "Re-assessment of risk factors and care plan adjustment", "#22c55e"),
        ]
    elif patient["risk_level"] == "MEDIUM":
        actions = [
            ("24-48 Hours", "Standard discharge follow-up call", "#fbbf24"),
            ("Week 1", "Primary care appointment scheduled", "#facc15"),
            ("Week 2", "Case manager check-in call", "#a3e635"),
            ("Month 1", "Care plan review and adjustment as needed", "#22c55e"),
        ]
    else:
        actions = [
            ("Week 1", "Routine follow-up call to confirm understanding", "#a3e635"),
            ("2 Weeks", "Primary care visit as scheduled", "#22c55e"),
            ("Month 1", "Standard monitoring per care plan", "#10b981"),
        ]

    timeline_html = ""
    for timeframe, action, color in actions:
        timeline_html += f"""
        <div class="timeline-item">
            <div class="timeline-content" style="border-left-color: {color};">
                <div class="timeline-time" style="color: {color};">{timeframe}</div>
                <div class="timeline-action">{action}</div>
            </div>
        </div>
        """
    return timeline_html


def generate_risk_assessment_html(patient):
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    html = f"""
    <div style="padding: 15px;">

        <div class="risk-alert {patient['risk_class']}">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px;">
                    <h2 style="margin: 0 0 15px 0; font-size: 1.6em; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.8em;">{patient['risk_icon']}</span>
                        RISK ASSESSMENT STATUS
                    </h2>
                    <div class="risk-badge {patient['badge_class']}">
                        {patient['risk_level']} RISK PATIENT
                    </div>
                </div>
                <div style="text-align: right; flex: 1; min-width: 200px;">
                    <div style="font-size: 0.85em; color: #9ca3af; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Patient Record</div>
                    <div style="font-size: 2.5em; font-weight: 800; color: #06b6d4; text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);">#{patient['patient_id']}</div>
                </div>
            </div>

            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 0.9em; color: #9ca3af;">
                    <strong style="color: #f9fafb;">Analysis Generated:</strong> {timestamp}
                </div>
            </div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">📊 Readmission Risk</div>
                <div class="metric-value">{patient['risk_probability']*100:.1f}%</div>
                <div class="metric-subtitle">30-Day Probability</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">🏥 LACE Score</div>
                <div class="metric-value">{patient['lace_score']}</div>
                <div class="metric-subtitle">Out of 19 ({'High Risk' if patient['lace_score'] >= 10 else 'Moderate Risk'})</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">🚑 Emergency Visits</div>
                <div class="metric-value">{patient['num_emergency']}</div>
                <div class="metric-subtitle">Past 6 Months</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">⏱️ Hospital Days</div>
                <div class="metric-value">{patient['time_in_hospital']}</div>
                <div class="metric-subtitle">Current Admission</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">💊 Medications</div>
                <div class="metric-value">{patient['num_medications']}</div>
                <div class="metric-subtitle">Active Prescriptions</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">📋 Diagnoses</div>
                <div class="metric-value">{patient['num_diagnoses']}</div>
                <div class="metric-subtitle">Total Conditions</div>
            </div>
        </div>

        <div class="viz-container">
            <div class="viz-title">
                <span>🔬</span> Clinical Data Summary
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div style="background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 10px; border-left: 3px solid #06b6d4;">
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Laboratory Work</div>
                    <div style="color: #f9fafb; font-size: 1.1em;"><strong>{patient['num_lab_procedures']}</strong> Lab Procedures</div>
                    <div style="color: #9ca3af; font-size: 0.85em; margin-top: 5px;">A1C Result: {patient['a1c_result']}</div>
                </div>

                <div style="background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 10px; border-left: 3px solid #3b82f6;">
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Medication Status</div>
                    <div style="color: #f9fafb; font-size: 1.1em;">Insulin: <strong>{patient['insulin_status']}</strong></div>
                    <div style="color: #f9fafb; font-size: 1.1em;">Metformin: <strong>{patient['metformin_status']}</strong></div>
                    <div style="color: #9ca3af; font-size: 0.85em; margin-top: 5px;">Recent Changes: <strong style="color: {'#ef4444' if patient['meds_changed'] == 'Yes' else '#10b981'};">{patient['meds_changed']}</strong></div>
                </div>

                <div style="background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 10px; border-left: 3px solid #8b5cf6;">
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Procedures</div>
                    <div style="color: #f9fafb; font-size: 1.1em;"><strong>{patient['num_procedures']}</strong> Total Procedures</div>
                    <div style="color: #9ca3af; font-size: 0.85em; margin-top: 5px;">During Current Stay</div>
                </div>
            </div>
        </div>

        <div class="viz-container">
            <div class="viz-title">
                <span>🤖</span> Logistic Regression Predictive Model Output
            </div>
            <div style="background: rgba(10, 14, 26, 0.6); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 10px; padding: 20px; font-family: 'Courier New', monospace;">
                <pre style="color: #06b6d4; font-size: 0.95em; line-height: 1.6; margin: 0; white-space: pre-wrap;">{{
    "patient_id": {patient['patient_id']},
    "predicted_risk_label": {patient['predicted_risk_label']},
    "risk_probability": {patient['risk_probability']:.4f},
    "risk_category": "{patient['risk_level']}",
    "lace_score": {patient['lace_score']},
    "threshold_used": {patient['threshold_used']:.2f},
    "contributing_factors": {{
        "emergency_visits": {patient['num_emergency']},
        "hospital_length_of_stay": {patient['time_in_hospital']},
        "total_diagnoses": {patient['num_diagnoses']},
        "medication_count": {patient['num_medications']},
        "medication_changes": "{patient['meds_changed']}"
    }},
    "model_confidence": "{('High' if abs(patient['risk_probability'] - 0.5) > 0.3 else 'Medium')}"
}}</pre>
            </div>
        </div>

    </div>
    """
    return html


def generate_ai_clinical_assessment(patient):
    ai_body = generate_ai_assessment_from_row(patient["raw_row"])
    return f"""
    <div class="ai-section">
        <div class="ai-header">
            <div class="ai-icon">🤖</div>
            <div>
                <h3 class="ai-title">Mosaic AI Clinical Assessment</h3>
                <div style="color: #9ca3af; font-size: 0.9em;">Powered by {MOSAIC_ENDPOINT}</div>
            </div>
        </div>
        <div class="ai-content">
            {ai_body}
        </div>
    </div>

    <div class="recommendations-panel">
        <div class="rec-title">
            <span>⏱️</span> Suggested Action Timeline
        </div>
        <div class="timeline-container">
            {generate_action_timeline(patient)}
        </div>
    </div>
    """


def analyze_patient_comprehensive(patient_id):
    try:
        patient = get_patient_data(patient_id)
        if "error" in patient:
            error_html = f"""
            <div class="risk-alert risk-high">
                <h2 style="color: #ef4444; margin: 0;">⚠️ Error Loading Patient Data</h2>
                <p style="color: #f9fafb; margin-top: 10px;">{patient['error']}</p>
                <p style="color: #9ca3af; margin-top: 10px; font-size: 0.9em;">Please verify the Patient ID and try again.</p>
            </div>
            """
            return error_html, None, None, None, ""

        risk_html = generate_risk_assessment_html(patient)
        gauge_chart = create_risk_gauge(patient["risk_probability"])
        feature_chart = create_feature_importance_chart(patient)
        lace_chart = create_lace_breakdown_chart(patient)
        ai_assessment = generate_ai_clinical_assessment(patient)
        return risk_html, gauge_chart, feature_chart, lace_chart, ai_assessment
    except Exception as exc:
        error_html = f"""
        <div class="risk-alert risk-high">
            <h2 style="color: #ef4444;">🚨 System Error</h2>
            <p style="color: #f9fafb; margin-top: 10px;">An unexpected error occurred: {str(exc)}</p>
        </div>
        """
        return error_html, None, None, None, f"Error: {str(exc)}"


def get_random_high_risk_patient():
    current_results = ensure_results_loaded()
    high_risk_patients = current_results[current_results["predicted_risk_label"] == 1]
    if len(high_risk_patients) > 0:
        return int(high_risk_patients.sample(1)["Patient_ID"].iloc[0])
    return 0


def get_random_low_risk_patient():
    current_results = ensure_results_loaded()
    low_risk_patients = current_results[current_results["predicted_risk_label"] == 0]
    if len(low_risk_patients) > 0:
        return int(low_risk_patients.sample(1)["Patient_ID"].iloc[0])
    return 0


def build_demo():
    current_results = ensure_results_loaded()
    max_patient_id = int(current_results["Patient_ID"].max())
    high_risk_count = int(current_results["predicted_risk_label"].sum())
    accuracy_pct = model_metrics.get("accuracy", 0.0) * 100

    with gr.Blocks(
        theme=gr.themes.Base(),
        css=custom_css,
        title="Mosaic AI Readmission Risk Explorer",
    ) as app:
        gr.HTML(
            f"""
            <div class="dashboard-header">
                <h1 class="header-title">🏥 READMISSION RISK EXPLORER</h1>
                <p class="header-subtitle">AI-Powered Clinical Decision Support System</p>
                <div class="header-stats">
                    <div class="header-stat-item">
                        <div class="header-stat-value">{len(current_results)}</div>
                        <div class="header-stat-label">Patients Analyzed</div>
                    </div>
                    <div class="header-stat-item">
                        <div class="header-stat-value">{high_risk_count}</div>
                        <div class="header-stat-label">High Risk Cases</div>
                    </div>
                    <div class="header-stat-item">
                        <div class="header-stat-value">{accuracy_pct:.1f}%</div>
                        <div class="header-stat-label">Model Accuracy</div>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div class="control-panel">
                        <div class="control-title">
                            <span>📋</span> Patient Selection
                        </div>
                        <p style="color: #9ca3af; font-size: 0.95em; line-height: 1.6; margin-bottom: 20px;">
                            Enter a Patient ID to generate comprehensive risk assessment and clinical recommendations.
                        </p>
                    </div>
                    """
                )

                patient_id_input = gr.Number(
                    label=f"Patient ID (0-{max_patient_id})",
                    value=get_random_high_risk_patient(),
                    precision=0,
                    minimum=0,
                    maximum=max_patient_id,
                )

                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 ANALYZE PATIENT",
                        variant="primary",
                        size="lg",
                        elem_classes=["btn-analyze"],
                    )

                with gr.Row():
                    demo_high_btn = gr.Button("🚨 Load High-Risk Example", size="sm")
                    demo_low_btn = gr.Button("✅ Load Low-Risk Example", size="sm")

                gr.HTML(
                    f"""
                    <div style="margin-top: 25px; padding: 20px; background: rgba(6, 182, 212, 0.1); border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);">
                        <div style="color: #06b6d4; font-weight: 700; margin-bottom: 10px; font-size: 1.1em;">ℹ️ System Information</div>
                        <div style="color: #9ca3af; font-size: 0.9em; line-height: 1.6;">
                            <strong style="color: #f9fafb;">Model:</strong> Logistic Regression<br>
                            <strong style="color: #f9fafb;">AI Engine:</strong> {MOSAIC_ENDPOINT}<br>
                            <strong style="color: #f9fafb;">Prediction Window:</strong> 30 Days<br>
                            <strong style="color: #f9fafb;">Risk Factors:</strong> LACE + Clinical History
                        </div>
                    </div>
                    """
                )

            with gr.Column(scale=2):
                risk_output = gr.HTML(label="Patient Risk Assessment")

                with gr.Row():
                    gauge_output = gr.Plot(label="Risk Probability Gauge")
                    lace_output = gr.Plot(label="LACE Score Breakdown")

                feature_output = gr.Plot(label="Risk Factor Analysis")
                ai_output = gr.HTML(label="Mosaic AI Clinical Assessment")

        analyze_btn.click(
            fn=analyze_patient_comprehensive,
            inputs=[patient_id_input],
            outputs=[risk_output, gauge_output, feature_output, lace_output, ai_output],
        )

        demo_high_btn.click(
            fn=lambda: get_random_high_risk_patient(),
            inputs=[],
            outputs=[patient_id_input],
        )

        demo_low_btn.click(
            fn=lambda: get_random_low_risk_patient(),
            inputs=[],
            outputs=[patient_id_input],
        )

    app.queue(default_concurrency_limit=4)
    return app


demo = build_demo()
print("✅ Gradio dashboard assembled successfully.")

# COMMAND ----------

# =============================================================================
# CELL 7: Notebook Sanity Check
# =============================================================================
sample_patient_id = get_random_high_risk_patient()
risk_html, gauge_chart, feature_chart, lace_chart, ai_html = analyze_patient_comprehensive(sample_patient_id)

print(f"Sanity check patient ID: {sample_patient_id}")
print(f"Risk HTML generated : {bool(risk_html)}")
print(f"Gauge generated     : {gauge_chart is not None}")
print(f"Feature chart ready : {feature_chart is not None}")
print(f"LACE chart ready    : {lace_chart is not None}")
print(f"AI section generated: {bool(ai_html)}")

# COMMAND ----------

# =============================================================================
# CELL 8: Launch Gradio with pyngrok for a Public Share Link
# =============================================================================
def find_open_port(start_port=7865, max_attempts=25):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found between {start_port} and {start_port + max_attempts - 1}")


def resolve_ngrok_auth_token():
    if NGROK_AUTH_TOKEN:
        print("✅ Loaded ngrok auth token from notebook widget")
        return NGROK_AUTH_TOKEN

    env_token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if env_token:
        print("✅ Loaded ngrok auth token from environment variable NGROK_AUTHTOKEN")
        return env_token

    try:
        secret_token = dbutils.secrets.get("hackathon_scope", "ngrok_auth_token").strip()
        if secret_token:
            print("✅ Loaded ngrok auth token from secret scope hackathon_scope/ngrok_auth_token")
            return secret_token
    except Exception:
        pass

    raise RuntimeError(
        "Ngrok auth token not found. Paste it into the 'ngrok_auth_token' widget, "
        "or set NGROK_AUTHTOKEN in the cluster environment."
    )


def is_local_port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_gradio(host, port, timeout_seconds=45):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        if is_local_port_open(host, port):
            try:
                with urllib.request.urlopen(f"http://{host}:{port}", timeout=2) as response:
                    print(f"✅ Local Gradio responded with HTTP {response.getcode()}")
                    return True
            except Exception as request_error:
                last_error = request_error
        time.sleep(1)
    raise RuntimeError(
        f"Gradio did not become reachable on http://{host}:{port} within {timeout_seconds}s. "
        f"Last error: {last_error}"
    )


def clear_existing_ngrok():
    try:
        for tunnel in ngrok.get_tunnels():
            print(f"ℹ️ Closing stale tunnel: {tunnel.public_url}")
            ngrok.disconnect(tunnel.public_url)
    except Exception:
        pass

    try:
        ngrok.kill()
        print("✅ Reset ngrok background agent")
    except Exception:
        pass


def launch_public_demo():
    token = resolve_ngrok_auth_token()
    conf.get_default().auth_token = token
    clear_existing_ngrok()

    public_port = find_open_port(7865)
    print(f"🚀 Launching Gradio on port {public_port}")

    launch_result = demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(public_port),
        prevent_thread_lock=True,
        show_error=True,
        quiet=False,
        inline=False,
    )

    if isinstance(launch_result, tuple):
        local_url = launch_result[1] if len(launch_result) > 1 else f"http://127.0.0.1:{public_port}"
    else:
        local_url = f"http://127.0.0.1:{public_port}"

    wait_for_gradio("127.0.0.1", int(public_port), timeout_seconds=45)

    tunnel = ngrok.connect(
        addr=int(public_port),
        proto="http",
        bind_tls=True,
    )

    print("\n" + "=" * 70)
    print("PUBLIC DEMO READY")
    print("=" * 70)
    print(f"Local URL : {local_url}")
    print(f"Ngrok URL : {tunnel.public_url}")
    print("Keep this cell running while the shared demo is being used.")
    print("=" * 70)
    return tunnel.public_url


public_url = launch_public_demo()
print(f"✅ Share this URL: {public_url}")
