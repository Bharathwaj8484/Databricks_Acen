# Databricks notebook source
# CELL 1: Install Required Dependencies
%pip install --upgrade langchain langchain-core langchain-databricks fastapi "uvicorn[standard]" plotly markdown kaleido scikit-learn ngrok nest_asyncio psutil --quiet
%pip install mlflow>=3.0 --upgrade

print("Restarting Python to load the new notebook dependencies...")
dbutils.library.restartPython()

# COMMAND ----------

# CELL 2: Notebook Configuration
import json
import os
import socket
import time
import urllib.request
from datetime import datetime

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.prompts import ChatPromptTemplate
from langchain_databricks import ChatDatabricks
import ngrok
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


ensure_text_widget("input_csv_path", "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data.csv")
ensure_text_widget(
    "results_output_path",
    "/Volumes/demo/hackathon_schema/hackathon_data/prediction_results.csv",
)
ensure_text_widget(
    "mosaic_endpoint",
    "databricks-meta-llama-3-3-70b-instruct",
)

INPUT_CSV_PATH = dbutils.widgets.get("input_csv_path").strip()
RESULTS_OUTPUT_PATH = dbutils.widgets.get("results_output_path").strip()
MOSAIC_ENDPOINT = dbutils.widgets.get("mosaic_endpoint").strip()
NGROK_AUTH_TOKEN = (os.environ.get("NGORK", "") or os.environ.get("NGROK_AUTHTOKEN", "")).strip()
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
print("  Ngrok token source  : cluster environment variable" if NGROK_AUTH_TOKEN else "  Ngrok token source  : cluster environment variable missing")

# COMMAND ----------

# CELL 3: Clinical Data Preparation 

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


# CELL 4: Model Training and Results Persistence (Preserved from ngrok.py)

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


# CELL 5: Mosaic AI Clinical Agent

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


# CELL 6: FastAPI Storytelling Frontend (Custom HTML/CSS/JS + GSAP + Plotly)

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

body::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(56, 189, 248, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56, 189, 248, 0.04) 1px, transparent 1px);
    background-size: 120px 120px;
    mask-image: radial-gradient(circle at center, black 25%, transparent 85%);
    animation: gridDrift 22s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 0.7;
}

@keyframes gridDrift {
    0% { transform: translate3d(0, 0, 0); }
    50% { transform: translate3d(-24px, -18px, 0); }
    100% { transform: translate3d(0, 0, 0); }
}

.gradio-container > * {
    position: relative;
    z-index: 1;
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

.scene-banner {
    position: relative;
    overflow: hidden;
    margin: 0 0 28px 0;
    padding: 24px 28px;
    border-radius: 22px;
    border: 1px solid rgba(125, 211, 252, 0.18);
    background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.16), transparent 34%),
        radial-gradient(circle at bottom right, rgba(99, 102, 241, 0.18), transparent 38%),
        rgba(15, 23, 42, 0.72);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 24px 60px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(18px);
}

.scene-banner::before {
    content: '';
    position: absolute;
    inset: -20% auto auto -10%;
    width: 180px;
    height: 180px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.28), transparent 70%);
    animation: floatOrb 9s ease-in-out infinite;
}

.scene-banner::after {
    content: '';
    position: absolute;
    right: -40px;
    bottom: -50px;
    width: 220px;
    height: 220px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(129, 140, 248, 0.22), transparent 72%);
    animation: floatOrbAlt 11s ease-in-out infinite;
}

@keyframes floatOrb {
    0%, 100% { transform: translate3d(0, 0, 0); }
    50% { transform: translate3d(16px, 14px, 0); }
}

@keyframes floatOrbAlt {
    0%, 100% { transform: translate3d(0, 0, 0); }
    50% { transform: translate3d(-18px, -12px, 0); }
}

.scene-kicker {
    position: relative;
    z-index: 1;
    display: inline-block;
    margin-bottom: 12px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(15, 118, 110, 0.18);
    border: 1px solid rgba(34, 211, 238, 0.25);
    color: #a5f3fc;
    font-size: 0.82em;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 700;
}

.scene-copy {
    position: relative;
    z-index: 1;
    max-width: 920px;
    color: #d7ebff;
    line-height: 1.75;
    font-size: 1.02em;
}

.scene-pill-row {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 18px;
}

.scene-pill {
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(8, 15, 28, 0.56);
    border: 1px solid rgba(96, 165, 250, 0.22);
    color: #bfdbfe;
    font-size: 0.9em;
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

.gradio-plot,
.gr-plot,
.plot-container,
.js-plotly-plot {
    background: linear-gradient(180deg, rgba(8, 15, 28, 0.82) 0%, rgba(15, 23, 42, 0.72) 100%) !important;
    border: 1px solid rgba(96, 165, 250, 0.16) !important;
    border-radius: 18px !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.03),
        0 20px 38px rgba(2, 6, 23, 0.28) !important;
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

        risk_prob = float(patient_row.get("risk_probability", patient_row.get("probability", 0.0)))
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
            "predicted_risk_label": int(patient_row.get("predicted_risk_label", patient_row.get("predicted_label", 0))),
            "threshold_used": float(patient_row.get("threshold_used", FIXED_THRESHOLD)),
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
            "actual_readmitted": bool(patient_row.get("actual_target", patient_row.get("readmitted_flag", 0))),
            "raw_row": patient_row,
        }
        return patient_data
    except Exception as exc:
        return {"error": str(exc)}

def apply_plot_theme(fig, height, is_3d=False):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2f3ff", "family": "Inter"},
        margin=dict(l=45, r=45, t=85, b=55),
        height=height,
        hoverlabel=dict(
            bgcolor="rgba(7, 15, 30, 0.95)",
            bordercolor="rgba(56, 189, 248, 0.35)",
            font_size=13,
            font_family="Inter",
        ),
    )

    if is_3d:
        fig.update_layout(
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    gridcolor="rgba(56, 189, 248, 0.14)",
                    zerolinecolor="rgba(56, 189, 248, 0.18)",
                    showbackground=False,
                    color="#b8d8ff",
                ),
                yaxis=dict(
                    gridcolor="rgba(56, 189, 248, 0.14)",
                    zerolinecolor="rgba(56, 189, 248, 0.18)",
                    showbackground=False,
                    color="#b8d8ff",
                ),
                zaxis=dict(
                    gridcolor="rgba(56, 189, 248, 0.14)",
                    zerolinecolor="rgba(56, 189, 248, 0.18)",
                    showbackground=False,
                    color="#b8d8ff",
                ),
                camera=dict(eye=dict(x=1.55, y=1.38, z=0.9)),
            )
        )
    return fig

def create_risk_gauge(risk_probability):
    risk_value = risk_probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_value,
            domain={"x": [0.05, 0.95], "y": [0.1, 0.9]},
            title={"text": "Readmission Risk Signal", "font": {"size": 22, "color": "#8be9fd"}},
            delta={"reference": 50, "increasing": {"color": "#fb7185"}, "decreasing": {"color": "#34d399"}},
            number={"suffix": "%", "font": {"size": 30, "color": "#f8fbff"}},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#7dd3fc",
                    "tickfont": {"color": "#dff6ff"},
                },
                "bar": {"color": "#22d3ee", "thickness": 0.18},
                "bgcolor": "rgba(11, 18, 37, 0.25)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.24)"},
                ],
                "threshold": {
                    "line": {"color": "#fb7185", "width": 6},
                    "thickness": 0.9,
                    "value": 70,
                },
            },
        )
    )

    fig.add_annotation(
        x=0.5, y=0.08, xref="paper", yref="paper",
        text="High-concern zone begins at 70%",
        showarrow=False,
        font=dict(size=12, color="#9bc3e7"),
    )
    return apply_plot_theme(fig, height=390)

def create_feature_importance_chart(patient_data):
    # Top 3 most clinically meaningful risk drivers only
    features = {
        "LACE Score": (patient_data["lace_score"] / 19) * 100,
        "Prior ED Visits": min((patient_data["num_emergency"] / 4) * 100, 100),
        "Hospital Stay": min((patient_data["time_in_hospital"] / 10) * 100, 100),
    }

    sorted_items = sorted(features.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#22d3ee", "#60a5fa", "#a78bfa"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(
                    color=colors[: len(values)],
                    line=dict(color="rgba(224, 242, 254, 0.1)", width=1.2),
                ),
                text=[f"{value:.0f}%" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Relative impact: %{x:.1f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Top Risk Drivers",
            "font": {"size": 17, "color": "#8be9fd"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Relative Impact (%)",
        xaxis=dict(
            range=[0, 120],
            dtick=30,
            gridcolor="rgba(56, 189, 248, 0.08)",
            tickcolor="rgba(56, 189, 248, 0.15)",
            tickfont=dict(size=11),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(56, 189, 248, 0.06)",
            autorange="reversed",
            zeroline=False,
            tickfont=dict(size=12),
        ),
    )
    fig = apply_plot_theme(fig, height=330)
    fig.update_layout(margin=dict(l=90, r=55, t=60, b=40))
    return fig

def create_lace_breakdown_chart(patient_data):
    l_score = patient_data["time_in_hospital"]
    a_score = 3 if patient_data.get("admission_type", 0) in [1, 2, 7] else 0
    c_score = min(patient_data["num_diagnoses"], 5)
    e_score = min(patient_data["num_emergency"], 4)

    component_labels = ["Length (L)", "Acuity (A)", "Comorb (C)", "ED (E)"]
    component_values = [l_score, a_score, c_score, e_score]
    component_colors = ["#22d3ee", "#60a5fa", "#8b5cf6", "#f97316"]

    max_val = max(component_values) if max(component_values) > 0 else 1
    y_max = max(max_val + 2, 9)

    fig = go.Figure(
        data=[
            go.Bar(
                x=component_labels,
                y=component_values,
                marker=dict(color=component_colors, line=dict(color="rgba(224,242,254,0.1)", width=1.2)),
                text=component_values,
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": f"LACE Breakdown (Total: {patient_data['lace_score']})",
            "font": {"size": 17, "color": "#8be9fd"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(gridcolor="rgba(56, 189, 248, 0.06)", tickangle=0, tickfont=dict(size=11)),
        yaxis=dict(
            title="Score", range=[0, y_max], dtick=3,
            gridcolor="rgba(56, 189, 248, 0.08)", zeroline=False,
            tickfont=dict(size=11),
        ),
        showlegend=False,
    )
    fig = apply_plot_theme(fig, height=330)
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig

def create_risk_terrain_chart(patient_data):
    # Clean donut chart — top 3 clinical risk contributors
    raw = {
        "LACE Score": patient_data["lace_score"],
        "Hospital Stay": patient_data["time_in_hospital"],
        "ED Visits": patient_data["num_emergency"],
    }
    # Filter out zeros to keep it clean
    filtered = {k: v for k, v in raw.items() if v > 0}
    if not filtered:
        filtered = {"No Signals": 1}

    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = ["#22d3ee", "#8b5cf6", "#f97316", "#fb7185"][:len(labels)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(
            colors=colors,
            line=dict(color="rgba(3,8,20,0.9)", width=3)
        ),
        textinfo="label+percent",
        textfont=dict(size=13, color="#f0f8ff"),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Share: %{percent}<extra></extra>",
        pull=[0.04] * len(labels),
    )])

    fig.add_annotation(
        text=f"<b>{sum(values)}</b><br><span style='font-size:10px'>Total</span>",
        x=0.5, y=0.5, font_size=18, font_color="#e2f3ff",
        showarrow=False, xref="paper", yref="paper",
        align="center",
    )

    fig.update_layout(
        title={
            "text": "Risk Factor Share",
            "font": {"size": 17, "color": "#8be9fd"},
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend=dict(
            orientation="v", x=1.0, y=0.5,
            font=dict(color="#c8dff0", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig = apply_plot_theme(fig, height=330)
    fig.update_layout(margin=dict(l=10, r=90, t=60, b=20))
    return fig

def generate_action_timeline(patient):
    if patient["risk_level"] == "HIGH":
        actions = [
            ("Immediate", "Activate high-risk discharge protocol", "#ef4444"),
            ("24 Hours", "First follow-up contact via phone call", "#f97316"),
            ("48 Hours", "In-person clinic visit or home health assessment", "#fbbf24"),
            ("Week 1", "Medication reconciliation and adherence check", "#facc15"),
        ]
    elif patient["risk_level"] == "MEDIUM":
        actions = [
            ("24-48 Hours", "Standard discharge follow-up call", "#fbbf24"),
            ("Week 1", "Primary care appointment scheduled", "#facc15"),
            ("Week 2", "Case manager check-in call", "#a3e635"),
            ("Month 1", "Care plan review & adjustment", "#22c55e"),
        ]
    else:
        actions = [
            ("Week 1", "Routine follow-up call", "#a3e635"),
            ("2 Weeks", "Primary care visit", "#22c55e"),
            ("Month 1", "Standard monitoring per plan", "#10b981"),
        ]

    timeline_html = "<div style='display: flex; flex-direction: column; gap: 15px; position: relative; padding-left: 20px;'>"
    timeline_html += "<div style='position: absolute; left: 6px; top: 10px; bottom: 10px; width: 2px; background: rgba(255,255,255,0.1);'></div>"
    for i, (timeframe, action, color) in enumerate(actions):
        timeline_html += f"""
        <div style="position: relative; display: flex; align-items: center; gap: 15px; background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); transition: transform 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="position: absolute; left: -21px; top: 50%; transform: translateY(-50%); width: 14px; height: 14px; border-radius: 50%; background: {color}; box-shadow: 0 0 10px {color}; border: 2px solid #000;"></div>
            <div style="min-width: 100px; font-weight: bold; color: {color}; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.05em;">{timeframe}</div>
            <div style="color: #f9fafb; font-size: 1.05em;">{action}</div>
        </div>
        """
    timeline_html += "</div>"
    return timeline_html

def generate_risk_assessment_html(patient):
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    risk_color_map = {"HIGH": "#ff4d6d", "MEDIUM": "#f4a261", "LOW": "#43aa8b"}
    risk_glow_map  = {"HIGH": "rgba(255,77,109,0.32)", "MEDIUM": "rgba(244,162,97,0.28)", "LOW": "rgba(67,170,139,0.28)"}
    risk_bg_map    = {"HIGH": "rgba(255,77,109,0.07)", "MEDIUM": "rgba(244,162,97,0.06)", "LOW": "rgba(67,170,139,0.06)"}
    rlevel = patient["risk_level"]
    rc     = risk_color_map.get(rlevel, "#66e3ff")
    rg     = risk_glow_map.get(rlevel, "rgba(102,227,255,0.28)")
    rb     = risk_bg_map.get(rlevel, "rgba(102,227,255,0.06)")
    conf_val = "High" if abs(patient["risk_probability"] - 0.5) > 0.3 else "Medium"
    conf_c   = "#43aa8b" if conf_val == "High" else "#f4a261"

    # 6 metric tiles
    metrics = [
        ("Readmission Risk",   f"{patient['risk_probability']*100:.1f}%",  "30-Day Probability",      "#5ce6ff", "📊"),
        ("LACE Score",         str(patient['lace_score']),                  f"{'High Risk' if patient['lace_score']>=10 else 'Moderate'} · /26", "#a78bfa", "⚡"),
        ("Emergency Visits",   str(patient['num_emergency']),               "Prior 6 Months",          "#f87171", "🚨"),
        ("Hospital Days",      str(patient['time_in_hospital']),            "Current Admission",        "#4ba3ff", "🏥"),
        ("Medications",        str(patient['num_medications']),             "Active Prescriptions",     "#34d399", "💊"),
        ("Diagnoses",          str(patient['num_diagnoses']),               "Recorded Conditions",      "#fb923c", "📋"),
    ]

    metric_cards = ""
    for label, val, sub, col, icon in metrics:
        metric_cards += f"""
        <div style="
            position:relative; overflow:hidden;
            background:linear-gradient(145deg,rgba(8,18,36,0.95),rgba(4,10,22,0.85));
            border:1px solid {col}30; border-top:2px solid {col};
            border-radius:16px; padding:20px 18px;
            transition:transform 0.25s ease, box-shadow 0.25s ease;
        " onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 36px {col}28'"
           onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none'">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                <span style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{col}bb;">{label}</span>
                <span style="font-size:1.05em;opacity:0.8;">{icon}</span>
            </div>
            <div style="font-size:2.1rem;font-weight:800;color:#f0f8ff;line-height:1;font-family:'Space Grotesk',sans-serif;letter-spacing:-0.03em;margin-bottom:6px;">{val}</div>
            <div style="font-size:0.75rem;color:#7a9bb8;">{sub}</div>
        </div>"""

    # Syntax-highlighted JSON rows
    json_rows = [
        ('{',                                                                           '#9eb4cb'),
        (f'  "patient_id": {patient["patient_id"]},',                                  '#5ce6ff'),
        (f'  "predicted_risk_label": {patient["predicted_risk_label"]},',              '#a78bfa'),
        (f'  "risk_probability": {patient["risk_probability"]:.4f},',                  rc),
        (f'  "risk_category": "{rlevel}",',                                             rc),
        (f'  "lace_score": {patient["lace_score"]},',                                  '#5ce6ff'),
        (f'  "threshold_used": {patient["threshold_used"]:.2f},',                      '#9eb4cb'),
        ('  "contributing_factors": {',                                                '#9eb4cb'),
        (f'    "emergency_visits": {patient["num_emergency"]},',                       '#f87171'),
        (f'    "hospital_length_of_stay": {patient["time_in_hospital"]},',             '#4ba3ff'),
        (f'    "total_diagnoses": {patient["num_diagnoses"]},',                        '#fb923c'),
        (f'    "medication_count": {patient["num_medications"]},',                     '#34d399'),
        (f'    "medication_changes": "{patient["meds_changed"]}"',                     '#f4a261'),
        ('  },',                                                                        '#9eb4cb'),
        (f'  "model_confidence": "{conf_val}"',                                        conf_c),
        ('}',                                                                           '#9eb4cb'),
    ]
    json_html = "".join(f'<div style="color:{c};line-height:1.85;">{l}</div>' for l, c in json_rows)

    html = f"""
    <div style="padding:8px 4px;">

        <!-- ── RISK BANNER ── -->
        <div style="
            position:relative; overflow:hidden;
            background:{rb}; border:1px solid {rc}3a; border-left:4px solid {rc};
            border-radius:18px; padding:24px 28px; margin-bottom:24px;
            box-shadow:0 0 40px {rg}, inset 0 1px 0 rgba(255,255,255,0.03);
        ">
            <div style="position:absolute;top:0;right:0;width:280px;height:100%;
                background:radial-gradient(ellipse at top right,{rc}14,transparent 65%);
                pointer-events:none;"></div>

            <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;position:relative;z-index:1;">
                <div style="
                    width:46px;height:46px;border-radius:13px;
                    background:{rc}1e;border:1px solid {rc}50;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.4em;box-shadow:0 0 16px {rc}28;flex-shrink:0;
                ">{patient['risk_icon']}</div>
                <div>
                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:{rc}aa;margin-bottom:3px;">Clinical Risk Assessment</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#f0f8ff;font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em;">Patient Discharge Risk Profile</div>
                </div>
                <div style="margin-left:auto;text-align:right;flex-shrink:0;">
                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#7a9bb8;margin-bottom:4px;">Patient Record</div>
                    <div style="font-size:2.6rem;font-weight:900;font-family:'Space Grotesk',sans-serif;letter-spacing:-0.04em;
                        background:linear-gradient(135deg,{rc},{rc}88);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
                    ">#{patient['patient_id']}</div>
                </div>
            </div>

            <!-- Risk badge + timestamp row -->
            <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:relative;z-index:1;">
                <div style="
                    display:inline-flex;align-items:center;gap:8px;
                    padding:7px 18px;border-radius:999px;font-weight:800;
                    font-size:0.88rem;letter-spacing:0.08em;text-transform:uppercase;
                    background:{rc};color:#fff;
                    box-shadow:0 4px 18px {rg};
                ">
                    <span style="width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,0.55);display:inline-block;
                        animation:riskDot 1.5s ease-in-out infinite;"></span>
                    {rlevel} RISK PATIENT
                </div>
                <div style="font-size:0.82rem;color:#7a9bb8;display:flex;align-items:center;gap:6px;">
                    <span>⏱️</span> Analysis: {timestamp}
                </div>
            </div>
        </div>
        <style>
        @keyframes riskDot {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.4;transform:scale(1.5)}} }}
        </style>

        <!-- ── METRIC TILES ── -->
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:24px;">
            {metric_cards}
        </div>

        <!-- ── CLINICAL DATA SUMMARY ── -->
        <div style="
            background:linear-gradient(145deg,rgba(8,18,36,0.88),rgba(4,10,22,0.78));
            border:1px solid rgba(100,180,255,0.12); border-radius:18px;
            padding:22px 24px; margin-bottom:20px;
        ">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
                <div style="width:3px;height:20px;background:linear-gradient(180deg,#5ce6ff,#4ba3ff);border-radius:2px;"></div>
                <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#5ce6ff;">Clinical Data Snapshot</span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;">
                <div style="background:rgba(92,230,255,0.04);border:1px solid rgba(92,230,255,0.12);border-radius:14px;padding:16px 18px;">
                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#5ce6ffaa;margin-bottom:10px;">Laboratory Work</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#f0f8ff;margin-bottom:4px;"><span style="color:#5ce6ff;">{patient['num_lab_procedures']}</span> Lab Procedures</div>
                    <div style="font-size:0.8rem;color:#7a9bb8;">A1C Result: <span style="color:#c8dff0;font-weight:600;">{patient['a1c_result']}</span></div>
                </div>
                <div style="background:rgba(167,139,250,0.04);border:1px solid rgba(167,139,250,0.12);border-radius:14px;padding:16px 18px;">
                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#a78bfaaa;margin-bottom:10px;">Medication Status</div>
                    <div style="font-size:0.92rem;color:#f0f8ff;margin-bottom:3px;">Insulin: <span style="font-weight:700;color:#a78bfa;">{patient['insulin_status']}</span></div>
                    <div style="font-size:0.92rem;color:#f0f8ff;margin-bottom:6px;">Metformin: <span style="font-weight:700;color:#a78bfa;">{patient['metformin_status']}</span></div>
                    <div style="font-size:0.8rem;color:#7a9bb8;">Changes: <span style="font-weight:700;color:{'#ff4d6d' if patient['meds_changed']=='Yes' else '#43aa8b'};">{patient['meds_changed']}</span></div>
                </div>
                <div style="background:rgba(251,146,60,0.04);border:1px solid rgba(251,146,60,0.12);border-radius:14px;padding:16px 18px;">
                    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#fb923caa;margin-bottom:10px;">Procedures</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#f0f8ff;margin-bottom:4px;"><span style="color:#fb923c;">{patient['num_procedures']}</span> Total Procedures</div>
                    <div style="font-size:0.8rem;color:#7a9bb8;">During Current Admission</div>
                </div>
            </div>
        </div>

        <!-- ── MODEL OUTPUT TERMINAL ── -->
        <div style="
            background:linear-gradient(145deg,rgba(4,8,20,0.98),rgba(3,6,16,0.94));
            border:1px solid rgba(92,230,255,0.15); border-radius:18px;
            overflow:hidden; margin-bottom:8px;
        ">
            <div style="
                display:flex;align-items:center;justify-content:space-between;
                padding:13px 20px; border-bottom:1px solid rgba(92,230,255,0.09);
                background:rgba(92,230,255,0.03);
            ">
                <div style="display:flex;align-items:center;gap:9px;">
                    <div style="width:3px;height:18px;background:linear-gradient(180deg,#5ce6ff,#4ba3ff);border-radius:2px;"></div>
                    <span style="font-size:0.7rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#5ce6ff;">Logistic Regression · Model Output</span>
                </div>
                <div style="display:flex;gap:6px;">
                    <div style="width:10px;height:10px;border-radius:50%;background:#ff5f57;"></div>
                    <div style="width:10px;height:10px;border-radius:50%;background:#febc2e;"></div>
                    <div style="width:10px;height:10px;border-radius:50%;background:#28c840;"></div>
                </div>
            </div>
            <div style="padding:20px 24px;font-family:'Courier New',monospace;font-size:0.86rem;overflow-x:auto;">
                {json_html}
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
                <h3 class="ai-title">Clinical Assessment (Rule-Based Fallback)</h3>
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

def get_random_high_risk_patient():
    current_results = ensure_results_loaded()
    if current_results.empty: return 0
    # Use "predicted_label" or "predicted_risk_label" depending on exact column name in CSV
    label_col = "predicted_risk_label" if "predicted_risk_label" in current_results.columns else "predicted_label"
    high_risk_patients = current_results[current_results[label_col] == 1]
    if len(high_risk_patients) > 0:
        return int(high_risk_patients.sample(1)["Patient_ID"].iloc[0])
    return 0

def get_random_low_risk_patient():
    current_results = ensure_results_loaded()
    if current_results.empty: return 0
    label_col = "predicted_risk_label" if "predicted_risk_label" in current_results.columns else "predicted_label"
    low_risk_patients = current_results[current_results[label_col] == 0]
    if len(low_risk_patients) > 0:
        return int(low_risk_patients.sample(1)["Patient_ID"].iloc[0])
    return 0

def figure_to_payload(fig):
    return json.loads(fig.to_json()) if fig is not None else None

def build_model_output(patient):
    return {
        "patient_id": patient["patient_id"],
        "predicted_risk_label": patient["predicted_risk_label"],
        "risk_probability": round(patient["risk_probability"], 4),
        "risk_category": patient["risk_level"],
        "lace_score": patient["lace_score"],
        "threshold_used": round(patient["threshold_used"], 2),
        "contributing_factors": {
            "emergency_visits": patient["num_emergency"],
            "hospital_length_of_stay": patient["time_in_hospital"],
            "total_diagnoses": patient["num_diagnoses"],
            "medication_count": patient["num_medications"],
            "medication_changes": patient["meds_changed"],
        },
        "model_confidence": "High" if abs(patient["risk_probability"] - 0.5) > 0.3 else "Medium",
    }

def build_story_steps(patient):
    return [
        {
            "eyebrow": "Scene 01",
            "title": "Risk Signal",
            "body": (
                f"Patient {patient['patient_id']} is currently classified as {patient['risk_level']} risk "
                f"with a {patient['risk_probability'] * 100:.1f}% predicted probability of 30-day readmission."
            ),
            "chart": "gauge",
        },
        {
            "eyebrow": "Scene 02",
            "title": "Drivers Of Risk",
            "body": (
                f"LACE score {patient['lace_score']}, {patient['num_emergency']} recent emergency visits, "
                f"{patient['num_medications']} medications, and {patient['num_diagnoses']} diagnoses are the main drivers."
            ),
            "chart": "features",
        },
        {
            "eyebrow": "Scene 03",
            "title": "LACE Transparency",
            "body": (
                f"The LACE components combine stay length {patient['time_in_hospital']}, admission acuity, "
                f"comorbidity burden, and emergency utilization to explain how the logistic model is being grounded clinically."
            ),
            "chart": "lace",
        },
        {
            "eyebrow": "Scene 04",
            "title": "Risk Breakdown",
            "body": (
                "The 3D pie chart simplifies the terrain into an easy-to-read distribution "
                "of LACE, Stay, and ED variables—preventing visual fatigue while maintaining precision."
            ),
            "chart": "terrain",
        },
    ]

def build_dashboard_summary():
    current_results = ensure_results_loaded()
    if current_results.empty:
        return {}
    
    label_col = "predicted_risk_label" if "predicted_risk_label" in current_results.columns else "predicted_label"
    high_risk_count = int((current_results[label_col] == 1).sum())
    low_risk_count = int((current_results[label_col] == 0).sum())
    high_example = get_random_high_risk_patient()
    low_example = get_random_low_risk_patient()

    return {
        "title": "Readmission Risk Explorer",
        "subtitle": "Scroll-driven clinical storytelling with transparent model evidence and Mosaic AI recommendations.",
        "patients_analyzed": int(len(current_results)),
        "high_risk_cases": high_risk_count,
        "low_risk_cases": low_risk_count,
        "recall_pct": round(model_metrics.get("recall", 0.0) * 100, 1),
        "roc_auc": round(model_metrics.get("roc_auc", 0.0), 3),
        "threshold": FIXED_THRESHOLD,
        "mosaic_endpoint": MOSAIC_ENDPOINT,
        "prediction_window": "30 Days",
        "risk_factors": "LACE + Clinical History",
        "default_patient_id": int(high_example if high_example is not None else 0),
        "high_risk_example": int(high_example if high_example is not None else 0),
        "low_risk_example": int(low_example if low_example is not None else 0),
        "max_patient_id": int(current_results["Patient_ID"].max()),
    }

def build_patient_payload(patient_id):
    patient = get_patient_data(patient_id)
    if "error" in patient:
        return patient

    return {
        "patient": {
            "patient_id": patient["patient_id"],
            "risk_level": patient["risk_level"],
            "risk_probability": round(patient["risk_probability"], 4),
            "risk_icon": patient["risk_icon"],
            "lace_score": patient["lace_score"],
            "num_emergency": patient["num_emergency"],
            "time_in_hospital": patient["time_in_hospital"],
            "num_medications": patient["num_medications"],
            "num_diagnoses": patient["num_diagnoses"],
            "a1c_result": patient["a1c_result"],
            "meds_changed": patient["meds_changed"],
        },
        "risk_html": generate_risk_assessment_html(patient),
        "ai_html": generate_ai_clinical_assessment(patient),
        "charts": {
            "gauge": figure_to_payload(create_risk_gauge(patient["risk_probability"])),
            "features": figure_to_payload(create_feature_importance_chart(patient)),
            "lace": figure_to_payload(create_lace_breakdown_chart(patient)),
            "terrain": figure_to_payload(create_risk_terrain_chart(patient)),
        },
        "story_steps": build_story_steps(patient),
        "model_output": build_model_output(patient),
    }

# --- FRONTEND HTML (EXACTLY as you provided, with basePath fix injected for HF Spaces) ---
FRONTEND_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Readmission Risk Explorer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --bg: #040814;
            --panel: rgba(8, 17, 32, 0.72);
            --line: rgba(120, 196, 255, 0.18);
            --text: #f5fbff;
            --muted: #9eb4cb;
            --cyan: #66e3ff;
            --blue: #4ba3ff;
            --indigo: #7c8cff;
            --pink: #ff6e9a;
            --green: #2ee6a6;
            --amber: #ffb347;
            --shadow: 0 30px 80px rgba(0, 0, 0, 0.42);
        }

        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; background: var(--bg); }
        body {
            margin: 0;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 20%, rgba(76, 163, 255, 0.18), transparent 28%),
                radial-gradient(circle at 82% 14%, rgba(255, 110, 154, 0.15), transparent 25%),
                radial-gradient(circle at 50% 88%, rgba(46, 230, 166, 0.12), transparent 28%),
                linear-gradient(180deg, #030611 0%, #06101f 35%, #050b16 100%);
            overflow-x: hidden;
        }

        body::before,
        body::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
        }

        body::before {
            background-image:
                linear-gradient(rgba(115, 177, 255, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(115, 177, 255, 0.05) 1px, transparent 1px);
            background-size: 120px 120px;
            mask-image: radial-gradient(circle at center, rgba(0, 0, 0, 1), transparent 85%);
        }

        body::after {
            background: linear-gradient(180deg, rgba(3, 7, 16, 0.08), rgba(3, 7, 16, 0.54));
        }

        .aurora { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
        .aurora-orb {
            position: absolute;
            border-radius: 999px;
            filter: blur(24px);
            opacity: 0.42;
            will-change: transform;
        }
        .orb-1 { width: 340px; height: 340px; left: -100px; top: 12vh; background: rgba(102, 227, 255, 0.32); }
        .orb-2 { width: 420px; height: 420px; right: -120px; top: 24vh; background: rgba(124, 140, 255, 0.28); }
        .orb-3 { width: 360px; height: 360px; left: 36vw; bottom: -140px; background: rgba(255, 110, 154, 0.24); }

        .page-shell {
            position: relative;
            z-index: 1;
            width: min(1380px, calc(100vw - 32px));
            margin: 0 auto;
            padding-bottom: 80px;
        }

        .hero { min-height: 100vh; display: grid; align-items: center; padding: 48px 0 32px; }
        .hero-panel {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 36px;
            padding: 36px;
            background:
                linear-gradient(145deg, rgba(10, 20, 37, 0.92), rgba(6, 14, 28, 0.74)),
                radial-gradient(circle at top right, rgba(102, 227, 255, 0.12), transparent 30%);
            box-shadow: var(--shadow);
            backdrop-filter: blur(22px);
        }

        .hero-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent 20%, rgba(255,255,255,0.08) 48%, transparent 76%);
            transform: translateX(-110%);
            animation: sheen 8s linear infinite;
        }

        @keyframes sheen { to { transform: translateX(110%); } }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid rgba(102, 227, 255, 0.26);
            background: rgba(8, 21, 40, 0.68);
            color: #b8ecff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }

        .hero-grid { display: grid; grid-template-columns: 1.25fr 0.95fr; gap: 28px; margin-top: 22px; }
        .hero h1,
        .panel-title,
        .story-step h3,
        .control-card h2 { font-family: "Space Grotesk", "IBM Plex Sans", sans-serif; letter-spacing: -0.03em; }
        .hero h1 { font-size: clamp(3rem, 7vw, 6rem); line-height: 0.94; margin: 16px 0 18px; max-width: 760px; }
        .hero-copy { color: #d8e8f7; max-width: 720px; font-size: 1.06rem; line-height: 1.9; margin-bottom: 24px; }
        .hero-copy strong { color: var(--cyan); }
        .hero-badges { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
        .hero-badge {
            padding: 11px 15px;
            border-radius: 999px;
            background: rgba(10, 20, 36, 0.72);
            border: 1px solid rgba(142, 196, 255, 0.18);
            color: #d7e8ff;
            font-size: 0.92rem;
        }
        .hero-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
        .stat-card {
            padding: 18px;
            border-radius: 24px;
            background: rgba(7, 16, 30, 0.76);
            border: 1px solid rgba(120, 196, 255, 0.12);
            min-height: 126px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .stat-label {
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 10px;
        }
        .stat-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(1.8rem, 4vw, 3rem);
            line-height: 1;
            color: var(--text);
        }
        .stat-note { margin-top: 10px; color: var(--muted); font-size: 0.92rem; }
        .control-ribbon { display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 22px; margin-top: -20px; z-index: 10; position: relative; margin-bottom: 52px; }
        .panel,
        .control-card,
        .story-stage,
        .story-step,
        .insight-card {
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }
        .control-card { border-radius: 30px; padding: 26px; }
        .control-card h2 { margin: 0 0 10px; font-size: 1.6rem; }
        .control-copy { color: var(--muted); line-height: 1.8; margin-bottom: 18px; }
        .control-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
        .patient-input {
            flex: 1 1 220px;
            min-width: 0;
            border: 1px solid rgba(120, 196, 255, 0.22);
            border-radius: 18px;
            background: rgba(4, 10, 20, 0.92);
            color: var(--text);
            padding: 16px 18px;
            font-size: 1rem;
        }
        .btn {
            appearance: none;
            border: none;
            border-radius: 18px;
            padding: 15px 18px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.25s ease, box-shadow 0.25s ease, opacity 0.25s ease;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-primary {
            background: linear-gradient(135deg, #66e3ff, #4ba3ff 60%, #7c8cff);
            color: #04101d;
            box-shadow: 0 18px 40px rgba(75, 163, 255, 0.32);
        }
        .btn-secondary {
            background: rgba(9, 19, 35, 0.88);
            color: var(--text);
            border: 1px solid rgba(120, 196, 255, 0.18);
        }
        .system-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; height: 100%; }
        .insight-card { border-radius: 24px; padding: 18px; }
        .insight-label {
            color: var(--muted);
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            margin-bottom: 8px;
        }
        .insight-value { font-size: 1.1rem; line-height: 1.5; }
        .story-section { position: relative; padding: 30px 0 80px; }
        .story-grid { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(420px, 1.1fr); gap: 28px; align-items: start; }
        .story-copy-col { padding-bottom: 48px; }
        .story-step {
            position: relative;
            padding: 26px 24px;
            border-radius: 28px;
            margin-bottom: 18px;
            transition: transform 0.35s ease, border-color 0.35s ease, background 0.35s ease;
        }
        .story-step.active {
            transform: translateX(10px) scale(1.01);
            border-color: rgba(102, 227, 255, 0.32);
            background: rgba(10, 26, 46, 0.9);
        }
        .story-step-label {
            color: #a3dfff;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            margin-bottom: 10px;
        }
        .story-step h3 { margin: 0 0 10px; font-size: 1.4rem; }
        .story-step p { margin: 0; color: var(--muted); line-height: 1.85; }
        .story-pin-shell { position: relative; min-height: 100vh; }
        .story-stage {
            position: sticky;
            top: 22px;
            border-radius: 34px;
            padding: 24px;
            overflow: hidden;
        }
        .stage-topbar {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .patient-chip {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(10, 22, 38, 0.88);
            border: 1px solid rgba(120, 196, 255, 0.18);
            color: #cfe8ff;
        }
        .risk-pill { padding: 8px 12px; border-radius: 999px; font-weight: 700; font-size: 0.86rem; }
        .risk-pill.high { background: rgba(255, 107, 129, 0.16); color: #ffafbf; }
        .risk-pill.medium { background: rgba(255, 179, 71, 0.16); color: #ffd39a; }
        .risk-pill.low { background: rgba(46, 230, 166, 0.16); color: #97ffd8; }
        .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
        .chart-card {
            min-height: 360px;
            border-radius: 24px;
            padding: 12px;
            background: rgba(4, 10, 20, 0.72);
            border: 1px solid rgba(120, 196, 255, 0.12);
            transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease, opacity 0.35s ease;
            opacity: 0.68;
            transform: translateY(14px) scale(0.985);
        }
        .chart-card.active {
            opacity: 1;
            transform: translateY(0) scale(1);
            border-color: rgba(102, 227, 255, 0.34);
            box-shadow: 0 28px 70px rgba(11, 30, 58, 0.48);
        }
        .chart-title { font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); padding: 6px 8px 0; }
        .chart-host { width: 100%; height: 310px; }
        .deep-section { padding: 34px 0; }
        .panel { border-radius: 34px; padding: 28px; overflow: hidden; }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 18px;
        }
        .panel-title { margin: 0; font-size: clamp(2rem, 4vw, 3rem); }
        .panel-copy { color: var(--muted); max-width: 760px; line-height: 1.8; }
        .html-panel {
            background: rgba(4, 10, 20, 0.62);
            border-radius: 28px;
            border: 1px solid rgba(120, 196, 255, 0.12);
            padding: 8px 14px 18px;
        }
        .footer-note { margin-top: 26px; color: var(--muted); text-align: center; font-size: 0.95rem; }
        .status-line { color: var(--muted); min-height: 22px; margin-top: 12px; }

        @media (max-width: 1180px) {
            .hero-grid,
            .control-ribbon,
            .story-grid { grid-template-columns: 1fr; }
            .chart-grid,
            .system-grid,
            .hero-stats { grid-template-columns: 1fr; }
            .control-ribbon { margin-top: 0; }
            .story-stage { position: relative; top: 0; }
        }

        @media (max-width: 768px) {
            .page-shell { width: min(100vw - 18px, 100%); }
            .hero-panel, .control-card, .story-stage, .panel { padding: 22px; border-radius: 24px; }
            .hero h1 { font-size: 2.7rem; }
            .chart-card { min-height: 320px; }
            .chart-host { height: 270px; }
        }
    
        .badge-high {
            background: linear-gradient(135deg, #ff0044 0%, #ff4b2b 100%);
            color: #ffffff !important;
            font-weight: 900;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            box-shadow: 0 0 20px rgba(255, 0, 68, 0.8), 0 0 40px rgba(255, 75, 43, 0.6);
            border: 2px solid rgba(255, 179, 179, 0.5);
            text-shadow: 0 0 10px rgba(255,255,255,0.8);
            animation: pulse-high-risk 1.5s infinite;
        }
        .badge-medium {
            background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
            color: #1f2937;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            font-weight: 900;
            box-shadow: 0 8px 20px rgba(217, 119, 6, 0.4);
        }
        .badge-low {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: #ffffff;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            font-weight: 900;
            box-shadow: 0 8px 20px rgba(5, 150, 105, 0.4);
        }
        @keyframes pulse-high-risk {
            0% { box-shadow: 0 0 20px rgba(255, 0, 68, 0.8), 0 0 40px rgba(255, 75, 43, 0.6); }
            50% { box-shadow: 0 0 30px rgba(255, 0, 68, 1), 0 0 60px rgba(255, 75, 43, 0.8); transform: scale(1.02); }
            100% { box-shadow: 0 0 20px rgba(255, 0, 68, 0.8), 0 0 40px rgba(255, 75, 43, 0.6); }
        }
        </style>
</head>
<body>
    <canvas id="bg-canvas" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0; opacity: 0.88;"></canvas>
    <div class="aurora">
        <div class="aurora-orb orb-1"></div>
        <div class="aurora-orb orb-2"></div>
        <div class="aurora-orb orb-3"></div>
    </div>

    <main class="page-shell">
        <section class="hero">
            <div class="hero-panel">
                <div class="eyebrow">Clinical Signal Deck</div>
                <div class="hero-grid">
                    <div>
                        <h1>Readmission Risk, told as a scroll story.</h1>
                        <div class="hero-copy" id="hero-copy">
                            This enterprise-grade predictive analytics engine harnesses advanced logistic regression and Mosaic AI reasoning.
                            It transforms raw clinical history into an interactive, 3D command center, surfacing precise readmission risks with full model transparency and actionable insights.
                        </div>
                        <div class="hero-badges">
                            <div class="hero-badge">FastAPI + Custom Frontend</div>
                            <div class="hero-badge">GSAP ScrollTrigger Storytelling</div>
                            <div class="hero-badge">Plotly Analytics + 3D Terrain</div>
                        </div>
                    </div>
                    <div class="hero-stats" id="hero-stats"></div>
                </div>
            </div>
        </section>

        <section class="control-ribbon">
            <div class="control-card">
                <h2>Choose a patient and drive the demo.</h2>
                <div class="control-copy">
                    Instantly query any patient ID from the evaluation cohort, or initialize a pre-configured high-risk/low-risk clinical scenario.
                    Interactive charts, logistic model evidence, and AI directives synchronize in real-time.
                </div>
                <div class="control-row">
                    <input id="patient-id-input" class="patient-input" type="number" min="0" step="1" placeholder="Enter Patient ID" />
                    <button id="analyze-btn" class="btn btn-primary">Analyze Patient</button>
                </div>
                <div class="control-row" style="margin-top: 12px;">
                    <button id="high-risk-btn" class="btn btn-secondary">Load High-Risk Example</button>
                    <button id="low-risk-btn" class="btn btn-secondary">Load Low-Risk Example</button>
                </div>
                <div class="status-line" id="status-line"></div>
            </div>

            <div class="control-card">
                <h2>System context</h2>
                <div class="system-grid" id="system-grid"></div>
            </div>
        </section>

        <section class="story-section">
            <div class="panel" style="margin-bottom: 22px;">
                <div class="panel-header">
                    <div>
                        <div class="eyebrow">Scroll Story</div>
                        <h2 class="panel-title">Move through the patient signal layer by layer.</h2>
                    </div>
                    <div class="panel-copy">
                        Scroll seamlessly through the patient\'s diagnostic profile. The synchronized narrative and advanced visual stage ensure that clinical nuance and model transparency are conveyed with maximum precision.
                    </div>
                </div>
            </div>

            <div class="story-grid">
                <div class="story-copy-col" id="story-steps"></div>
                <div class="story-pin-shell">
                    <div class="story-stage" id="story-stage">
                        <div class="stage-topbar">
                            <div class="patient-chip" id="patient-chip">Patient loading...</div>
                            <div class="risk-pill low" id="risk-pill">LOW RISK</div>
                        </div>
                        <div class="chart-grid">
                            <div class="chart-card active" data-chart="gauge">
                                <div class="chart-title">Risk Signal Gauge</div>
                                <div id="gauge-chart" class="chart-host"></div>
                            </div>
                            <div class="chart-card" data-chart="features">
                                <div class="chart-title">Risk Contribution Deck</div>
                                <div id="features-chart" class="chart-host"></div>
                            </div>
                            <div class="chart-card" data-chart="lace">
                                <div class="chart-title">LACE Breakdown</div>
                                <div id="lace-chart" class="chart-host"></div>
                            </div>
                            <div class="chart-card" data-chart="terrain">
                                <div class="chart-title">Risk Factor Breakdown</div>
                                <div id="terrain-chart" class="chart-host"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section class="deep-section">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <div class="eyebrow">Clinical Summary</div>
                        <h2 class="panel-title">Full patient evidence and model output.</h2>
                    </div>
                    <div class="panel-copy">
                        Nothing has been removed from the original dashboard logic. This panel preserves the patient metrics,
                        clinical snapshot, and logistic regression output in a richer presentation frame.
                    </div>
                </div>
                <div class="html-panel" id="clinical-summary"></div>
            </div>
        </section>

        <section class="deep-section">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <div class="eyebrow">Mosaic AI</div>
                        <h2 class="panel-title">AI assessment and intervention timeline.</h2>
                    </div>
                    <div class="panel-copy">
                        Mosaic AI remains part of the experience. If the endpoint is unavailable, the backend still falls back to the same
                        rule-based assessment path so the demo keeps working.
                    </div>
                </div>
                <div class="html-panel" id="ai-summary"></div>
            </div>
        </section>

        <div class="footer-note">
            Built for presentation-first clinical storytelling. Hosted seamlessly via Hugging Face Spaces.
        </div>
    </main>

    <script>
        const state = { summary: null, currentPatientId: null };
        const chartHosts = {
            gauge: "gauge-chart",
            features: "features-chart",
            lace: "lace-chart",
            terrain: "terrain-chart"
        };
        
        // This makes sure the API calls work perfectly inside the Hugging Face iframe
        const basePath = window.location.pathname.endsWith('/') 
            ? window.location.pathname.slice(0, -1) 
            : window.location.pathname;

        function setStatus(message) {
            document.getElementById("status-line").textContent = message || "";
        }

        function formatPct(value) {
            return `${Number(value).toFixed(1)}%`;
        }

        async function fetchJson(url) {
            const response = await fetch(url);
            if (!response.ok) {
                let detail = `HTTP ${response.status}`;
                try {
                    const errorBody = await response.json();
                    detail = errorBody.detail || detail;
                } catch (error) {
                    detail = response.statusText || detail;
                }
                throw new Error(detail);
            }
            return response.json();
        }

        function renderSummary(summary) {
            state.summary = summary;
            document.getElementById("hero-copy").innerHTML =
                `This enterprise-grade predictive analytics engine harnesses advanced logistic regression and <strong>Mosaic AI reasoning</strong>. ` + 
            `It transforms raw clinical history into an interactive, 3D command center, surfacing precise readmission risks with full model transparency and actionable insights. ` +
                `The model currently surfaces <strong>${summary.high_risk_cases}</strong> high-risk cases across <strong>${summary.patients_analyzed}</strong> evaluated patients.`;

            const heroStats = [
                { label: "Patients Analyzed", value: summary.patients_analyzed, note: "Evaluation cohort ready for live lookup" },
                { label: "High-Risk Cases", value: summary.high_risk_cases, note: "Positive predictions at the fixed threshold" },
                { label: "Recall", value: `${summary.recall_pct}%`, note: "Share of at-risk patients detected" },
                { label: "ROC AUC", value: summary.roc_auc, note: "Discrimination quality of the logistic model" }
            ];

            document.getElementById("hero-stats").innerHTML = heroStats.map((item) => `
                <div class="stat-card">
                    <div class="stat-label">${item.label}</div>
                    <div class="stat-value">${item.value}</div>
                    <div class="stat-note">${item.note}</div>
                </div>
            `).join("");

            document.getElementById("system-grid").innerHTML = [
                { label: "Model", value: "Logistic Regression" },
                { label: "AI Engine", value: summary.mosaic_endpoint },
                { label: "Prediction Window", value: summary.prediction_window },
                { label: "Risk Factors", value: summary.risk_factors },
                { label: "Threshold", value: summary.threshold },
                { label: "Patient Range", value: `0 - ${summary.max_patient_id}` }
            ].map((item) => `
                <div class="insight-card">
                    <div class="insight-label">${item.label}</div>
                    <div class="insight-value">${item.value}</div>
                </div>
            `).join("");

            const patientInput = document.getElementById("patient-id-input");
            patientInput.max = summary.max_patient_id;
            patientInput.value = summary.default_patient_id;
        }

        function renderStorySteps(steps) {
            document.getElementById("story-steps").innerHTML = steps.map((step, index) => `
                <article class="story-step ${index === 0 ? "active" : ""}" data-step-index="${index}" data-chart="${step.chart}">
                    <div class="story-step-label">${step.eyebrow}</div>
                    <h3>${step.title}</h3>
                    <p>${step.body}</p>
                </article>
            `).join("");
            activateStep(0);
            initScrollAnimations();
        }

        function activateStep(index) {
            const steps = document.querySelectorAll(".story-step");
            const cards = document.querySelectorAll(".chart-card");
            steps.forEach((step, currentIndex) => step.classList.toggle("active", currentIndex === index));

            const targetChart = steps[index]?.dataset.chart;
            cards.forEach((card) => card.classList.toggle("active", card.dataset.chart === targetChart));
        }

        function renderFigure(hostId, spec) {
            if (!spec) {
                return;
            }
            Plotly.react(hostId, spec.data, spec.layout, {
                responsive: true,
                displayModeBar: false,
                scrollZoom: false
            });
        }

        function applyRiskPill(level) {
            const pill = document.getElementById("risk-pill");
            pill.textContent = `${level} RISK`;
            pill.className = `risk-pill ${String(level || "").toLowerCase()}`;
        }

        function renderPatient(payload) {
            state.currentPatientId = payload.patient.patient_id;
            document.getElementById("patient-chip").textContent =
                `Patient #${payload.patient.patient_id} | ${formatPct(payload.patient.risk_probability * 100)} probability | LACE ${payload.patient.lace_score}`;
            applyRiskPill(payload.patient.risk_level);
            document.getElementById("clinical-summary").innerHTML = payload.risk_html;
            document.getElementById("ai-summary").innerHTML = payload.ai_html;
            renderFigure(chartHosts.gauge, payload.charts.gauge);
            renderFigure(chartHosts.features, payload.charts.features);
            renderFigure(chartHosts.lace, payload.charts.lace);
            renderFigure(chartHosts.terrain, payload.charts.terrain);
            renderStorySteps(payload.story_steps);
        }

        async function loadPatient(patientId) {
            setStatus(`Loading patient ${patientId}...`);
            try {
                const payload = await fetchJson(`${basePath}/api/patient/${patientId}`);
                renderPatient(payload);
                setStatus(`Loaded patient ${patientId}.`);
            } catch (error) {
                setStatus(`Unable to load patient ${patientId}: ${error.message}`);
            }
        }

        function initControls() {
            const patientInput = document.getElementById("patient-id-input");

            document.getElementById("analyze-btn").addEventListener("click", () => {
                const requestedId = Number(patientInput.value);
                if (Number.isFinite(requestedId)) {
                    loadPatient(requestedId);
                }
            });

            patientInput.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    event.preventDefault();
                    document.getElementById("analyze-btn").click();
                }
            });

            document.getElementById("high-risk-btn").addEventListener("click", () => {
                if (!state.summary) {
                    return;
                }
                patientInput.value = state.summary.high_risk_example;
                loadPatient(state.summary.high_risk_example);
            });

            document.getElementById("low-risk-btn").addEventListener("click", () => {
                if (!state.summary) {
                    return;
                }
                patientInput.value = state.summary.low_risk_example;
                loadPatient(state.summary.low_risk_example);
            });
        }

        
        function initThreeJS() {
            const canvas = document.getElementById('bg-canvas');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

            const group = new THREE.Group();
            
            const createCross = () => {
                const shape = new THREE.Shape();
                const d = 0.5, w = 1.5;
                shape.moveTo(-d, d); shape.lineTo(-w, d); shape.lineTo(-w, -d); shape.lineTo(-d, -d);
                shape.lineTo(-d, -w); shape.lineTo(d, -w); shape.lineTo(d, -d); shape.lineTo(w, -d);
                shape.lineTo(w, d); shape.lineTo(d, d); shape.lineTo(d, w); shape.lineTo(-d, w);
                
                const extrudeSettings = { depth: 0.5, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.1, bevelThickness: 0.1 };
                const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                geometry.center();
                
                const material = new THREE.MeshPhongMaterial({
                    color: 0x22d3ee,
                    emissive: 0x06b6d4,
                    emissiveIntensity: 1.4,
                    specular: 0xffffff,
                    shininess: 120,
                    transparent: true,
                    opacity: 0.92,
                });
                return new THREE.Mesh(geometry, material);
            };

            for(let i=0; i<15; i++) {
                const mesh = createCross();
                mesh.position.set((Math.random() - 0.5) * 40, (Math.random() - 0.5) * 40, (Math.random() - 0.5) * 20 - 10);
                mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                const scale = Math.random() * 0.8 + 0.4;
                mesh.scale.set(scale, scale, scale);
                group.add(mesh);
            }
            scene.add(group);

            // Strong multi-source lighting
            const ambientLight = new THREE.AmbientLight(0xb0e0ff, 3.5);
            scene.add(ambientLight);
            
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 5.0);
            directionalLight1.position.set(2, 3, 5);
            scene.add(directionalLight1);
            
            const directionalLight2 = new THREE.DirectionalLight(0x66e3ff, 3.5);
            directionalLight2.position.set(-2, -2, 2);
            scene.add(directionalLight2);

            const pointLight = new THREE.PointLight(0x7c8cff, 4.0, 60);
            pointLight.position.set(0, 0, 8);
            scene.add(pointLight);

            camera.position.z = 10;

            function animate() {
                requestAnimationFrame(animate);
                group.rotation.x += 0.0005;
                group.rotation.y += 0.001;
                renderer.render(scene, camera);
            }
            animate();

            if (window.gsap && window.ScrollTrigger) {
                gsap.to(group.rotation, {
                    y: Math.PI * 2, x: Math.PI,
                    scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: 1 }
                });
                gsap.to(group.position, {
                    y: 5, z: 2,
                    scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: true }
                });
            }

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        }

        function initScrollAnimations() {
            if (!(window.gsap && window.ScrollTrigger)) {
                return;
            }

            gsap.registerPlugin(ScrollTrigger);
            ScrollTrigger.getAll().forEach((trigger) => trigger.kill());

            gsap.to(".orb-1", {
                y: -180,
                x: 120,
                scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: true }
            });
            gsap.to(".orb-2", {
                y: 120,
                x: -140,
                scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: true }
            });
            gsap.to(".orb-3", {
                y: -100,
                x: 80,
                scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: true }
            });

            document.querySelectorAll(".story-step").forEach((step, index) => {
                ScrollTrigger.create({
                    trigger: step,
                    start: "top center+=40",
                    end: "bottom center",
                    onEnter: () => activateStep(index),
                    onEnterBack: () => activateStep(index)
                });
            });

            gsap.fromTo(
                ".story-stage",
                { y: 50, opacity: 0.7 },
                {
                    y: 0,
                    opacity: 1,
                    ease: "power2.out",
                    scrollTrigger: {
                        trigger: ".story-section",
                        start: "top 72%",
                        end: "top 20%",
                        scrub: 0.8
                    }
                }
            );
        }

        async function boot() {
            initControls();
            initThreeJS();
            const summary = await fetchJson(`${basePath}/api/summary`);
            renderSummary(summary);
            await loadPatient(summary.default_patient_id);
        }

        window.addEventListener("resize", () => {
            if (!window.Plotly) {
                return;
            }
            Object.values(chartHosts).forEach((hostId) => {
                const element = document.getElementById(hostId);
                if (element) {
                    Plotly.Plots.resize(element);
                }
            });
        });

        boot().catch((error) => setStatus(`Application failed to initialize: ${error.message}`));
    </script>
</body>
</html>
'''


def create_frontend_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Readmission Risk Explorer API")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FRONTEND_HTML

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/api/summary")
    def api_summary():
        return build_dashboard_summary()

    @app.get("/api/patient/{patient_id}")
    def api_patient(patient_id: int):
        payload = build_patient_payload(patient_id)
        if "error" in payload:
            raise HTTPException(status_code=404, detail=payload["error"])
        return payload


web_app = create_frontend_app()
demo = web_app
print("✅ FastAPI storytelling frontend assembled successfully.")

# COMMAND ----------


# CELL 7: Notebook Sanity Check

summary_payload = build_dashboard_summary()
sample_patient_id = summary_payload["default_patient_id"]
sample_payload = build_patient_payload(sample_patient_id)

print(f"Sanity check patient ID : {sample_patient_id}")
print(f"Summary generated       : {bool(summary_payload)}")
print(f"Risk HTML generated     : {bool(sample_payload.get('risk_html'))}")
print(f"AI HTML generated       : {bool(sample_payload.get('ai_html'))}")
print(f"Gauge chart ready       : {sample_payload.get('charts', {}).get('gauge') is not None}")
print(f"Feature chart ready     : {sample_payload.get('charts', {}).get('features') is not None}")
print(f"LACE chart ready        : {sample_payload.get('charts', {}).get('lace') is not None}")
print(f"3D chart ready          : {sample_payload.get('charts', {}).get('terrain') is not None}")
print(f"Story steps generated   : {len(sample_payload.get('story_steps', []))}")
print("✅ API payloads are ready for FastAPI serving.")

# COMMAND ----------

# # =============================================================================
# # CELL 8: Launch Gradio with pyngrok for a Public Share Link
# # =============================================================================
# import os
# import socket
# import tempfile
# import time
# import urllib.request
# from urllib.parse import urlparse

# from pyngrok import conf, ngrok


# NGROK_DEV_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip()


# def find_open_port(start_port=7865, max_attempts=25):
#     for port in range(start_port, start_port + max_attempts):
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#                 sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#                 sock.bind(("", port))
#                 return port
#         except OSError:
#             continue
#     raise RuntimeError(
#         f"No free port found between {start_port} and {start_port + max_attempts - 1}"
#     )


# def resolve_ngrok_auth_token():
#     if NGROK_AUTH_TOKEN:
#         print("✅ Loaded ngrok auth token from cluster environment variable")
#         return NGROK_AUTH_TOKEN

#     env_token = (os.getenv("NGORK", "") or os.getenv("NGROK_AUTHTOKEN", "")).strip()
#     if env_token:
#         print("✅ Loaded ngrok auth token from cluster environment variable")
#         return env_token

#     raise RuntimeError(
#         "Ngrok auth token not found in cluster environment variables. "
#         "Set NGORK or NGROK_AUTHTOKEN in the cluster environment."
#     )


# def build_isolated_pyngrok_config(token):
#     temp_dir = tempfile.mkdtemp(prefix="pyngrok_databricks_")
#     config_path = os.path.join(temp_dir, "ngrok.yml")

#     with open(config_path, "w", encoding="utf-8") as f:
#         f.write("version: 3\n")

#     pyngrok_config = conf.PyngrokConfig(
#         config_path=config_path,
#         auth_token=token,
#         start_new_session=True,
#     )
#     conf.set_default(pyngrok_config)
#     print(f"✅ Using isolated ngrok config: {config_path}")
#     return pyngrok_config


# def is_local_port_open(host, port, timeout=1.0):
#     try:
#         with socket.create_connection((host, port), timeout=timeout):
#             return True
#     except OSError:
#         return False


# def wait_for_gradio(host, port, timeout_seconds=60):
#     deadline = time.time() + timeout_seconds
#     last_error = None

#     while time.time() < deadline:
#         if is_local_port_open(host, port):
#             try:
#                 with urllib.request.urlopen(f"http://{host}:{port}", timeout=2) as response:
#                     print(f"✅ Local Gradio responded with HTTP {response.getcode()}")
#                     return True
#             except Exception as request_error:
#                 last_error = request_error
#         time.sleep(1)

#     raise RuntimeError(
#         f"Gradio did not become reachable on http://{host}:{port} within {timeout_seconds}s. "
#         f"Last error: {last_error}"
#     )


# def clear_existing_ngrok(pyngrok_config):
#     try:
#         for tunnel in ngrok.get_tunnels(pyngrok_config=pyngrok_config):
#             try:
#                 print(f"ℹ️ Closing stale tunnel: {tunnel.public_url}")
#                 ngrok.disconnect(tunnel.public_url, pyngrok_config=pyngrok_config)
#             except Exception:
#                 pass
#     except Exception:
#         pass

#     try:
#         ngrok.kill(pyngrok_config=pyngrok_config)
#         print("✅ Reset ngrok background agent")
#     except Exception:
#         pass


# def close_existing_gradio():
#     try:
#         demo.close()
#         print("✅ Closed previous Gradio server instance")
#         time.sleep(2)
#     except Exception:
#         pass


# def launch_public_demo():
#     token = resolve_ngrok_auth_token()
#     pyngrok_config = build_isolated_pyngrok_config(token)

#     clear_existing_ngrok(pyngrok_config)
#     close_existing_gradio()

#     requested_port = find_open_port(7865)
#     print(f"🚀 Launching Gradio on port {requested_port}")

#     launch_result = demo.launch(
#         share=False,
#         server_name="0.0.0.0",
#         server_port=int(requested_port),
#         prevent_thread_lock=True,
#         show_error=True,
#         quiet=False,
#         inline=False,
#     )

#     if isinstance(launch_result, tuple):
#         local_url = launch_result[1] if len(launch_result) > 1 else f"http://127.0.0.1:{requested_port}"
#     else:
#         local_url = f"http://127.0.0.1:{requested_port}"

#     parsed = urlparse(local_url)
#     actual_port = parsed.port or requested_port

#     wait_for_gradio("127.0.0.1", int(actual_port), timeout_seconds=60)

#     connect_kwargs = {
#         "addr": int(actual_port),
#         "proto": "http",
#         "pyngrok_config": pyngrok_config,
#     }
#     if NGROK_DEV_DOMAIN:
#         print(f"🌐 Opening ngrok tunnel on https://{NGROK_DEV_DOMAIN}")
#         connect_kwargs["domain"] = NGROK_DEV_DOMAIN
#         connect_kwargs["pooling_enabled"] = True
#     else:
#         print("🌐 Opening ngrok tunnel")

#     tunnel = ngrok.connect(**connect_kwargs)

#     print("\n" + "=" * 70)
#     print("PUBLIC DEMO READY")
#     print("=" * 70)
#     print(f"Local URL : {local_url}")
#     print(f"Ngrok URL : {tunnel.public_url}")
#     print("Keep this cell running while the shared demo is being used.")
#     print("=" * 70)
#     return tunnel.public_url


# public_url = launch_public_demo()
# print(f"✅ Share this URL: {public_url}")

# COMMAND ----------

# Fix Cell 6 - Properly define create_frontend_app() and web_app

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI(title="Readmission Risk Explorer API")

@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND_HTML

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/api/summary")
def api_summary():
    return build_dashboard_summary()

@app.get("/api/patient/{patient_id}")
def api_patient(patient_id: int):
    payload = build_patient_payload(patient_id)
    if "error" in payload:
        raise HTTPException(status_code=404, detail=payload["error"])
    return payload

web_app = app  # FIX: Directly use the app object
print("✅ FastAPI app created successfully")
print(f"   web_app type: {type(web_app)}")
print(f"   web_app value: {web_app}")
# Verify web_app before running Cell 8
assert web_app is not None, "❌ web_app is None!"
assert callable(web_app), "❌ web_app is not callable!"
print("✅ FastAPI app is ready for Uvicorn")

# COMMAND ----------

import os
import socket
import threading
import time
import urllib.request
import asyncio

import nest_asyncio
import uvicorn


NGROK_DEV_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip()
_uvicorn_server = None
_uvicorn_thread = None
_ngrok_listener = None

nest_asyncio.apply()


def find_open_port(start_port=7865, max_attempts=25):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No free port found between {start_port} and {start_port + max_attempts - 1}"
    )


def resolve_ngrok_auth_token():
    env_token = (os.getenv("NGORK", "") or os.getenv("NGROK_AUTHTOKEN", "")).strip()
    if env_token:
        print("✅ Loaded ngrok auth token from cluster environment variable")
        return env_token

    raise RuntimeError(
        "Ngrok auth token not found in cluster environment variables. "
        "Set NGORK or NGROK_AUTHTOKEN in the cluster environment."
    )


def wait_for_http_service(host, port, timeout_seconds=90):
    deadline = time.time() + timeout_seconds
    last_error = None
    url = f"http://{host}:{port}/healthz"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.getcode() == 200:
                    print(f"✅ Local FastAPI responded with HTTP {response.getcode()}")
                    return True
        except Exception as request_error:
            last_error = request_error
        time.sleep(1)

    raise RuntimeError(
        f"FastAPI did not become reachable on {url} within {timeout_seconds}s. "
        f"Last error: {last_error}"
    )


def clear_existing_ngrok():
    global _ngrok_listener

    try:
        ngrok.disconnect()
        print("✅ Reset ngrok listeners")
    except Exception:
        pass

    _ngrok_listener = None


def stop_existing_web_server():
    global _uvicorn_server, _uvicorn_thread

    if _uvicorn_server is not None:
        try:
            _uvicorn_server.should_exit = True
            print("✅ Signaled previous FastAPI server to stop")
        except Exception:
            pass

    if _uvicorn_thread is not None and _uvicorn_thread.is_alive():
        _uvicorn_thread.join(timeout=5)

    _uvicorn_server = None
    _uvicorn_thread = None


def start_fastapi_server(port):
    global _uvicorn_server, _uvicorn_thread

    config = uvicorn.Config(
        web_app,
        host="0.0.0.0",
        port=int(port),
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _uvicorn_server = server
    _uvicorn_thread = thread
    return f"http://127.0.0.1:{port}"


def resolve_ngrok_listener(listener_or_task):
    if hasattr(listener_or_task, "url"):
        return listener_or_task

    if asyncio.isfuture(listener_or_task) or asyncio.iscoroutine(listener_or_task):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(listener_or_task)

    raise TypeError(f"Unexpected ngrok listener type: {type(listener_or_task)}")


def launch_public_demo():
    global _ngrok_listener

    token = resolve_ngrok_auth_token()
    clear_existing_ngrok()
    stop_existing_web_server()

    requested_port = find_open_port(7865)
    print(f"🚀 Launching FastAPI frontend on port {requested_port}")
    local_url = start_fastapi_server(requested_port)

    wait_for_http_service("127.0.0.1", int(requested_port), timeout_seconds=90)

    connect_kwargs = {
        "addr": int(requested_port),
        "authtoken": token,
    }

    if NGROK_DEV_DOMAIN:
        print(f"🌐 Opening ngrok tunnel on https://{NGROK_DEV_DOMAIN}")
        connect_kwargs["domain"] = NGROK_DEV_DOMAIN
    else:
        print("🌐 Opening ngrok tunnel")

    _ngrok_listener = resolve_ngrok_listener(ngrok.forward(**connect_kwargs))
    public_url = _ngrok_listener.url()

    print("\n" + "=" * 70)
    print("PUBLIC DEMO READY")
    print("=" * 70)
    print(f"Local URL : {local_url}")
    print(f"Ngrok URL : {public_url}")
    print("Keep this cell running while the shared demo is being used.")
    print("=" * 70)
    return public_url


public_url = launch_public_demo()
print(f"✅ Share this URL: {public_url}")


# COMMAND ----------

# from pyngrok import ngrok

# ngrok.kill()