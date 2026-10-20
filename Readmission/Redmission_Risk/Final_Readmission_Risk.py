# Databricks notebook source
# =============================================================================
# CELL 1: Install Required Dependencies
# =============================================================================
%pip install --upgrade langchain langchain-core langchain-databricks fastapi "uvicorn[standard]" plotly markdown kaleido scikit-learn pyngrok==6.0.0 psutil --quiet

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
# CELL 6: FastAPI Storytelling Frontend (Custom HTML/CSS/JS + GSAP + Plotly)
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


def apply_plot_theme(fig, height, is_3d=False):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2f3ff", "family": "Inter"},
        margin=dict(l=28, r=28, t=70, b=38),
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
            domain={"x": [0.02, 0.98], "y": [0.0, 1.0]},
            title={"text": "Readmission Risk Signal", "font": {"size": 22, "color": "#8be9fd"}},
            delta={"reference": 50, "increasing": {"color": "#fb7185"}, "decreasing": {"color": "#34d399"}},
            number={"suffix": "%", "font": {"size": 48, "color": "#f8fbff"}},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#7dd3fc",
                    "tickfont": {"color": "#dff6ff"},
                },
                "bar": {"color": "#22d3ee", "thickness": 0.42},
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
        x=0.5,
        y=0.08,
        xref="paper",
        yref="paper",
        text="High-concern zone begins at 70%",
        showarrow=False,
        font=dict(size=12, color="#9bc3e7"),
    )

    return apply_plot_theme(fig, height=390)


def create_feature_importance_chart(patient_data):
    features = {
        "LACE Score": (patient_data["lace_score"] / 19) * 100,
        "Emergency Visits": min((patient_data["num_emergency"] / 4) * 100, 100),
        "Hospital Days": min((patient_data["time_in_hospital"] / 10) * 100, 100),
        "Medications": min((patient_data["num_medications"] / 30) * 100, 100),
        "Diagnoses": min((patient_data["num_diagnoses"] / 10) * 100, 100),
        "Lab Procedures": min((patient_data["num_lab_procedures"] / 50) * 100, 100),
    }

    sorted_items = sorted(features.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#22d3ee", "#38bdf8", "#3b82f6", "#6366f1", "#8b5cf6", "#f97316"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(
                    color=colors[: len(values)],
                    line=dict(color="rgba(224, 242, 254, 0.14)", width=1.6),
                ),
                text=[f"{value:.1f}%" for value in values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Relative impact: %{x:.1f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Risk Factor Contribution Analysis",
            "font": {"size": 20, "color": "#8be9fd"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Relative Impact (%)",
        xaxis=dict(
            range=[0, 110],
            gridcolor="rgba(56, 189, 248, 0.10)",
            tickcolor="rgba(56, 189, 248, 0.2)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(56, 189, 248, 0.08)",
            autorange="reversed",
            zeroline=False,
        ),
    )

    return apply_plot_theme(fig, height=430)


def create_lace_breakdown_chart(patient_data):
    l_score = patient_data["time_in_hospital"]
    a_score = 3 if patient_data.get("admission_type", 0) in [1, 2, 7] else 0
    c_score = min(patient_data["num_diagnoses"], 5)
    e_score = min(patient_data["num_emergency"], 4)

    component_labels = ["Length of Stay", "Acuity", "Comorbidity", "Emergency"]
    component_values = [l_score, a_score, c_score, e_score]
    component_colors = ["#22d3ee", "#60a5fa", "#8b5cf6", "#f97316"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=component_labels,
                y=component_values,
                marker=dict(color=component_colors, line=dict(color="rgba(224,242,254,0.14)", width=1.5)),
                text=component_values,
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": f"LACE Score Breakdown (Total: {patient_data['lace_score']})",
            "font": {"size": 20, "color": "#8be9fd"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(gridcolor="rgba(56, 189, 248, 0.08)", tickangle=0),
        yaxis=dict(title="Score", range=[0, 8], gridcolor="rgba(56, 189, 248, 0.1)", zeroline=False),
        showlegend=False,
    )

    return apply_plot_theme(fig, height=390)


def create_risk_terrain_chart(patient_data):
    factors = ["LACE", "ED Visits", "Stay", "Meds", "Dx", "Labs"]
    normalized = np.array(
        [
            (patient_data["lace_score"] / 19) * 100,
            min((patient_data["num_emergency"] / 4) * 100, 100),
            min((patient_data["time_in_hospital"] / 10) * 100, 100),
            min((patient_data["num_medications"] / 30) * 100, 100),
            min((patient_data["num_diagnoses"] / 10) * 100, 100),
            min((patient_data["num_lab_procedures"] / 50) * 100, 100),
        ]
    )

    z_surface = np.vstack(
        [
            normalized * 0.62,
            np.clip(normalized * 0.84 + 8, 0, 100),
            np.clip(np.maximum(normalized, np.array([55, 45, 50, 60, 55, 50])), 0, 100),
        ]
    )

    x_positions = np.arange(len(factors))
    y_positions = np.array([0, 1, 2])

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            z=z_surface,
            x=x_positions,
            y=y_positions,
            colorscale=[
                [0.0, "#0f172a"],
                [0.2, "#164e63"],
                [0.45, "#0ea5e9"],
                [0.7, "#8b5cf6"],
                [1.0, "#fb7185"],
            ],
            opacity=0.95,
            showscale=False,
            contours={
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "#e0f2fe",
                    "project_z": True,
                }
            },
            hovertemplate="Factor index: %{x}<br>Layer: %{y}<br>Risk signal: %{z:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_positions,
            y=np.zeros_like(x_positions),
            z=z_surface[0],
            mode="markers+text",
            text=[f"{value:.0f}" for value in normalized],
            textposition="top center",
            marker=dict(size=6, color="#f8fafc", line=dict(color="#22d3ee", width=2)),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": "3D Clinical Risk Terrain",
            "font": {"size": 20, "color": "#8be9fd"},
            "x": 0.5,
            "xanchor": "center",
        },
        scene=dict(
            xaxis=dict(
                tickmode="array",
                tickvals=x_positions.tolist(),
                ticktext=factors,
                title="Clinical Factors",
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=y_positions.tolist(),
                ticktext=["Observed", "Amplified", "Alert"],
                title="Signal Layer",
            ),
            zaxis=dict(title="Risk Signal", range=[0, 100]),
        ),
    )

    return apply_plot_theme(fig, height=460, is_3d=True)


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
            return error_html, None, None, None, None, ""

        risk_html = generate_risk_assessment_html(patient)
        gauge_chart = create_risk_gauge(patient["risk_probability"])
        feature_chart = create_feature_importance_chart(patient)
        lace_chart = create_lace_breakdown_chart(patient)
        terrain_chart = create_risk_terrain_chart(patient)
        ai_assessment = generate_ai_clinical_assessment(patient)
        return risk_html, gauge_chart, feature_chart, lace_chart, terrain_chart, ai_assessment
    except Exception as exc:
        error_html = f"""
        <div class="risk-alert risk-high">
            <h2 style="color: #ef4444;">🚨 System Error</h2>
            <p style="color: #f9fafb; margin-top: 10px;">An unexpected error occurred: {str(exc)}</p>
        </div>
        """
        return error_html, None, None, None, None, f"Error: {str(exc)}"


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
            "title": "Escalation Terrain",
            "body": (
                "The 3D terrain view amplifies the observed signals into a presentation-first risk landscape, "
                "making the case easier to explain to judges without losing clinical nuance."
            ),
            "chart": "terrain",
        },
    ]


def build_dashboard_summary():
    current_results = ensure_results_loaded()
    high_risk_count = int((current_results["predicted_risk_label"] == 1).sum())
    low_risk_count = int((current_results["predicted_risk_label"] == 0).sum())
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
        .control-ribbon { display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 22px; margin-top: -72px; margin-bottom: 52px; }
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
    </style>
</head>
<body>
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
                            This experience keeps the original data preparation, logistic regression model,
                            and Mosaic AI reasoning intact, while upgrading the presentation layer into a cinematic,
                            judge-friendly risk command center.
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
                    Search any patient ID from the test results, or jump straight into a prepared high-risk or low-risk example.
                    The charts, model evidence, and Mosaic AI recommendations update together.
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
                        As we scroll, the narrative steps and chart stage stay in sync, making the case easy to explain live
                        without losing clinical detail or model transparency.
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
                                <div class="chart-title">3D Clinical Terrain</div>
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
            Built for Databricks notebook execution with ngrok sharing, non-admin friendly dependency install, and presentation-first clinical storytelling.
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
                `This experience keeps the original <strong>data preparation</strong>, <strong>logistic regression model</strong>, and ` +
                `<strong>Mosaic AI reasoning</strong> intact, while upgrading the presentation layer into a cinematic, judge-friendly risk command center. ` +
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
                const payload = await fetchJson(`/api/patient/${patientId}`);
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
            const summary = await fetchJson("/api/summary");
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

    return app


web_app = create_frontend_app()
demo = web_app
print("✅ FastAPI storytelling frontend assembled successfully.")

# COMMAND ----------

# =============================================================================
# CELL 7: Notebook Sanity Check
# =============================================================================
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

# =============================================================================
# CELL 8: Launch FastAPI with pyngrok for a Public Share Link
# =============================================================================
import os
import socket
import tempfile
import threading
import time
import urllib.request

import uvicorn


NGROK_DEV_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip()
_uvicorn_server = None
_uvicorn_thread = None


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


def build_isolated_pyngrok_config(token):
    temp_dir = tempfile.mkdtemp(prefix="pyngrok_databricks_")
    config_path = os.path.join(temp_dir, "ngrok.yml")

    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write("version: 3\n")

    pyngrok_config = conf.PyngrokConfig(
        config_path=config_path,
        auth_token=token,
        start_new_session=True,
    )
    conf.set_default(pyngrok_config)
    print(f"✅ Using isolated ngrok config: {config_path}")
    return pyngrok_config


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


def clear_existing_ngrok(pyngrok_config):
    try:
        for tunnel in ngrok.get_tunnels(pyngrok_config=pyngrok_config):
            try:
                print(f"ℹ️ Closing stale tunnel: {tunnel.public_url}")
                ngrok.disconnect(tunnel.public_url, pyngrok_config=pyngrok_config)
            except Exception:
                pass
    except Exception:
        pass

    try:
        ngrok.kill(pyngrok_config=pyngrok_config)
        print("✅ Reset ngrok background agent")
    except Exception:
        pass


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


def launch_public_demo():
    token = resolve_ngrok_auth_token()
    pyngrok_config = build_isolated_pyngrok_config(token)

    clear_existing_ngrok(pyngrok_config)
    stop_existing_web_server()

    requested_port = find_open_port(7865)
    print(f"🚀 Launching FastAPI frontend on port {requested_port}")
    local_url = start_fastapi_server(requested_port)

    wait_for_http_service("127.0.0.1", int(requested_port), timeout_seconds=90)

    connect_kwargs = {
        "addr": int(requested_port),
        "proto": "http",
        "pyngrok_config": pyngrok_config,
    }

    if NGROK_DEV_DOMAIN:
        print(f"🌐 Opening ngrok tunnel on https://{NGROK_DEV_DOMAIN}")
        connect_kwargs["domain"] = NGROK_DEV_DOMAIN
        connect_kwargs["pooling_enabled"] = True
    else:
        print("🌐 Opening ngrok tunnel")

    tunnel = ngrok.connect(**connect_kwargs)

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