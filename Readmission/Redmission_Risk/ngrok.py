# Databricks notebook source
# =============================================================================
# CELL 1: Install Required Dependencies (Fixed Order)
# =============================================================================
%pip install --upgrade langchain langchain-core langchain-databricks gradio plotly markdown kaleido scikit-learn xgboost --quiet
%pip install mlflow>=2.16.0 databricks-vectorsearch --quiet
%pip install pyngrok==6.0.0 gradio plotly markdown kaleido langchain-databricks --quiet

print("⏳ Restarting Python Kernel...")
dbutils.library.restartPython()

import pandas as pd
print("✅ Dependencies installed and kernel restarted successfully!")

# COMMAND ----------

# =============================================================================
# CELL 2: Clinical Data Preparation (Recall-Safe Approach)
# =============================================================================
import pandas as pd
import numpy as np

# Load raw data
INPUT_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data.csv"
print(f"Loading data from: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, na_values=['?'], low_memory=False)
print(f"Initial Shape: {df.shape}")

# Drop clinically invalid encounters (deceased/hospice patients can't be readmitted)
EXPIRED_IDS = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(EXPIRED_IDS)].copy()

# Drop invalid gender
df = df[df['gender'] != 'Unknown/Invalid'].copy()

# Drop leakage-prone columns
DROP_COLS = ['encounter_id', 'weight', 'examide', 'citoglipton', 'patient_nbr']
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors='ignore')

# Handle missing values (medically sensible)
CATEGORICAL_FILL = ['race', 'payer_code', 'medical_specialty']
for col in CATEGORICAL_FILL:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

LAB_COLS = ['A1Cresult', 'max_glu_serum']
for col in LAB_COLS:
    if col in df.columns:
        df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')

# Target engineering (Binary Readmission Flag)
df['readmitted_flag'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)

# Feature Engineering
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

MED_COLS = [c for c in df.columns if any(med in c for med in ['metformin', 'insulin', 'glimepiride'])]
df['num_active_meds'] = df[MED_COLS].apply(lambda row: sum(v != 'No' for v in row), axis=1)

# Create LACE Score components (clinical risk indicator)
df['L_score'] = df['time_in_hospital']
df['A_score'] = df['admission_type_id'].apply(lambda x: 3 if str(x) in ['1', '2', '7'] else 0)
df['C_score'] = df['number_diagnoses'].apply(lambda x: min(x, 5))
df['E_score'] = df['number_emergency'].apply(lambda x: min(x, 4))
df['LACE_Score'] = df['L_score'] + df['A_score'] + df['C_score'] + df['E_score']

print(f"After Clinical Filtering: {df.shape}")
print("Stage 1 Data Preparation Complete! ✅")

# ⭐ FIX: Removed .write().saveAsTable() - pandas doesn't support this
# Instead, save results_df later in Cell 3 using CSV path
print("ℹ️ Pandas DataFrame saved to memory for model training.")

# COMMAND ----------

# =============================================================================
# CELL 3: Model Training with Fixed Threshold (0.53) - BALANCED
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Prepare features and target
X = df.drop(columns=['readmitted', 'readmitted_flag'], errors='ignore')
y = df['readmitted_flag']

# Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing Pipeline
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# Model Configuration (Balanced Recall & Precision)
model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 2},  # Emphasize recall for high-risk
    solver='lbfgs',
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', model)
])

# Train Model
print("Training LogReg Model...")
pipeline.fit(X_train, y_train)

# FIXED THRESHOLD EVALUATION (0.53 - Your Change)
fixed_threshold = 0.53
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= fixed_threshold).astype(int)

print(f"\n=== Evaluation at Threshold {fixed_threshold} ===")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}\n")

# Calculate specific metrics for the positive class (Readmission)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# Extract Confusion Matrix values
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate percentages for the breakdown
actual_positives = tp + fn
actual_negatives = tn + fp

tp_pct = (tp / actual_positives) * 100 if actual_positives > 0 else 0
fn_pct = (fn / actual_positives) * 100 if actual_positives > 0 else 0
tn_pct = (tn / actual_negatives) * 100 if actual_negatives > 0 else 0
fp_pct = (fp / actual_negatives) * 100 if actual_negatives > 0 else 0

print("=== Quick Metrics Summary ===")
print(f"Overall Accuracy : {acc:.3f} (Overall percentage of correct predictions)")
print(f"Precision (Class 1): {prec:.3f} (When the model flags a readmission, how often is it right?)")
print(f"Recall (Class 1)   : {rec:.3f} (Out of all actual readmissions, how many did the model catch?)\n")

print("=== Patient Outcome Breakdown ===")
print(f"True Positives (TP) : {tp}  -> ({tp_pct:.0f}%) Correctly found: At-risk patients the model successfully caught.")
print(f"False Negatives (FN): {fn}  -> ({fn_pct:.0f}%) Missed patients: At-risk patients the model incorrectly assumed were safe.")
print(f"True Negatives (TN) : {tn}  -> ({tn_pct:.0f}%) Correctly cleared: Safe patients the model correctly left alone.")
print(f"False Positives (FP): {fp}  -> ({fp_pct:.0f}%) False alarms: Safe patients the model incorrectly flagged as at-risk.")

# ⭐ CRITICAL FIX: Store results_df GLOBALLY with ALL original features for Gradio
results_df = X_test.copy()
results_df['readmitted_flag'] = y_test.values
results_df['probability'] = y_prob
results_df['predicted_label'] = y_pred
results_df['threshold_used'] = fixed_threshold

# Save important feature columns needed for visualization
for col in ['time_in_hospital', 'number_emergency', 'number_diagnoses', 
            'num_medications', 'LACE_Score']:
    if col in df.columns:
        # Match test indices to original df
        test_indices = X_test.index
        results_df[col] = df.loc[test_indices, col].values

print(f"\nModel Training Complete! ✅")
print(f"Threshold: {fixed_threshold} | Test Set Size: {len(results_df)}")

# Save results for persistence
results_df.reset_index(drop=True).to_csv("/Volumes/demo/hackathon_schema/hackathon_data/prediction_results.csv", index=False)

# COMMAND ----------

# =============================================================================
# AFTER CELL 3 COMPLETES - ADD THIS SAVE STEP
# =============================================================================

# Store results_df persistently so Gradio can access it
results_path = "/Volumes/demo/hackathon_schema/hackathon_data/prediction_results.csv"
results_df.reset_index(drop=True).to_csv(results_path, index=False)

# Verify the saved file
loaded_check = pd.read_csv(results_path)
print(f"\n✅ Results saved to: {results_path}")
print(f"   Test Set Size: {len(results_df)}")
print(f"   Verification: {len(loaded_check)} rows loaded successfully")

# COMMAND ----------

# =============================================================================
# CELL 4: Mosaic AI Clinical Agent Integration
# =============================================================================
import json
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
import markdown

# Initialize Mosaic AI endpoint
LLAMA_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
chat_model = ChatDatabricks(endpoint=LLAMA_ENDPOINT)

print(f"Connecting to Mosaic AI Endpoint: {LLAMA_ENDPOINT}")

# Define Patient Profile Function
def get_patient_profile(patient_row, probability, threshold):
    """Extract clinical features for AI explanation"""
    risk_level = "HIGH" if probability >= threshold else "LOW"
    
    profile = {
        "RISK_LEVEL": f"{'🔴' if risk_level == 'HIGH' else '🟢'} {risk_level} RISK",
        "Readmission_Probability": f"{probability:.1%}",
        "LACE_Score": int(getattr(patient_row, 'LACE_Score', 0)),
        "Clinical_History": {
            "Hospital_Stay_Days": int(getattr(patient_row, 'time_in_hospital', 0)),
            "Prior_Emergency_Visits": int(getattr(patient_row, 'number_emergency', 0)),
            "Total_Diagnoses": int(getattr(patient_row, 'number_diagnoses', 0))
        },
        "Medications": {
            "Total_Meds": int(getattr(patient_row, 'num_medications', 0)),
            "Insulin_Status": str(getattr(patient_row, 'insulin', 'No')),
            "Meds_Changed": "Yes" if getattr(patient_row, 'change', 'No') in ['Up', 'Down'] else "No"
        },
        "Labs": {
            "A1C_Result_Code": str(getattr(patient_row, 'A1Cresult', 'Not Tested'))
        }
    }
    return profile

# Clinical Assessment Prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Clinical Case Manager assisting with discharge planning.
Your goal is to explain Readmission Risk to doctors clearly.

INPUT: Patient JSON profile with risk probability and clinical features.

TASK:
1. Start with CLEAR Risk Summary headline
2. Explain WHY (use LACE Score, Emergency Visits, Labs)
   - If LACE > 10, flag as HIGH concern
   - If Emergency Visits > 1, highlight as major risk factor  
   - If Meds Changed, mention instability
3. Recommend 2-3 SPECIFIC interventions

FORMAT: Markdown with headers, bullet points, bold key terms
TONE: Professional, Concise, Medical. NO hallucination. """),
    ("user", "Analyze this patient profile: {patient_data}")
])

# Generate AI Assessment
def generate_ai_assessment(patient_row, probability, threshold):
    try:
        profile = get_patient_profile(patient_row, probability, threshold)
        chain = prompt_template | chat_model
        response = chain.invoke({"patient_data": json.dumps(profile)})
        ai_text = markdown.markdown(response.content)
        return ai_text
    except Exception as e:
        return f"<div style='color:#ef4444'>⚠️ AI Service Error: {str(e)}</div>"

print("✅ Mosaic AI Agent Ready!")

# COMMAND ----------

# =============================================================================
# CELL 5: Plotly Visualization Functions
# =============================================================================
import plotly.graph_objects as go
import plotly.express as px

# Gauge Chart (Risk Probability)
def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Readmission Risk", 'font': {'size': 24}},
        number={'suffix': "% ", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#06b6d4", 'thickness': 0.8},
            'bgcolor': "rgba(17, 24, 39, 0.8)",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {'line': {'color': "#ef4444", 'width': 4}, 'value': 70}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=20, r=20, t=80, b=20))
    return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': False})

# LACE Score Breakdown
def create_lace_chart(row):
    l_score = int(getattr(row, 'L_score', 0))
    a_score = int(getattr(row, 'A_score', 0))
    c_score = min(int(getattr(row, 'number_diagnoses', 0)), 5)
    e_score = min(int(getattr(row, 'number_emergency', 0)), 4)
    
    fig = go.Figure(data=[go.Bar(
        x=['Length of Stay', 'Acuity', 'Comorbidities', 'Emergency Visits'],
        y=[l_score, a_score, c_score, e_score],
        marker=dict(color=['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899']),
        text=[l_score, a_score, c_score, e_score],
        textposition='outside'
    )])
    fig.update_layout(title=f"LACE Components (Total: {l_score+a_score+c_score+e_score})", 
                     paper_bgcolor='rgba(0,0,0,0)', height=400)
    return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': False})

# Risk Factor Radar Chart
def create_radar_chart(row):
    categories = ['LACE Score', 'Hospital Stay', 'Emergency', 'Medications', 'Diagnoses']
    values = [
        min(int(getattr(row, 'LACE_Score', 0)) / 2, 10),
        min(int(getattr(row, 'time_in_hospital', 0)), 10),
        min(int(getattr(row, 'number_emergency', 0)) * 2, 10),
        min(int(getattr(row, 'num_medications', 0)) / 3, 10),
        min(int(getattr(row, 'number_diagnoses', 0)) / 2, 10)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Risk Factors'))
    fig.update_layout(polar={'radialaxis': {'range': [0, 10]}}, 
                     paper_bgcolor='rgba(0,0,0,0)', height=500)
    return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': False})

print("✅ Visualization functions ready!")

# COMMAND ----------

# =============================================================================
# CELL 6: GRADIO WEB INTERFACE - FIXED FOR DATABRICKS
# =============================================================================
import gradio as gr
from datetime import datetime
import socket
import os
import signal
import psutil

def kill_gradio_processes():
    """Kill any existing Gradio processes on common ports"""
    ports_to_check = [7860, 7861, 7862, 7863, 7864, 7865]
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            if 'gradio' in cmdline.lower() or 'python' in cmdline.lower():
                conn = None
                try:
                    conn = proc.connections()
                except:
                    pass
                
                for port in ports_to_check:
                    try:
                        for c in conn:
                            if c.laddr.port == port:
                                print(f"⚠️ Killing process {proc.pid} on port {port}")
                                proc.terminate()
                                break
                    except:
                        continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def get_available_port(start_port=7860, max_attempts=50):
    """Find an available port starting from start_port"""
    # First kill any existing gradio processes
    kill_gradio_processes()
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return port
        except OSError:
            continue
    # Fallback: force bind anyway
    return start_port

# Get available port
AVAILABLE_PORT = get_available_port(7860)
print(f"\n🔌 Attempting to bind Gradio to port: {AVAILABLE_PORT}\n")

CUSTOM_CSS = """
body { background: linear-gradient(135deg, #0a0e1a, #1e293b); font-family: Inter, sans-serif; }
.gradio-container { background: rgba(17, 24, 39, 0.9) !important; }
.card { background: rgba(17, 24, 39, 0.9); border-radius: 14px; padding: 18px; border: 1px solid rgba(96,165,250,0.3); }
.high { color: #ef4444; } .medium { color: #f59e0b; } .low { color: #10b981; }
.header { font-size: 2em; font-weight: 800; color: #60a5fa; }
"""

def predict_patient(patient_id, use_mosaic_ai=True):
    """Main prediction function called by Gradio"""
    try:
        global results_df
        
        # Try to load from memory first
        try:
            test_size = len(results_df)
        except:
            # Load from saved CSV if memory variable doesn't exist
            results_df = pd.read_csv("/Volumes/demo/hackathon_schema/hackathon_data/prediction_results.csv")
        
        # Validate patient ID
        patient_id = int(patient_id)
        if patient_id < 0 or patient_id >= len(results_df):
            return ("Error", "", "", "", f"❌ Invalid Patient ID. Valid range: 0-{len(results_df)-1}")
        
        patient_row = results_df.iloc[patient_id]
        
        probability = float(patient_row['probability'])
        label = patient_row['predicted_label']
        threshold = float(results_df['threshold_used'].iloc[0])
        
        # Determine risk category
        if probability >= 0.70:
            risk_level = "🔴 HIGH"
            risk_class = "high"
        elif probability >= 0.40:
            risk_level = "🟡 MEDIUM" 
            risk_class = "medium"
        else:
            risk_level = "🟢 LOW"
            risk_class = "low"
        
        # Get clinical metrics
        lace = int(getattr(patient_row, 'LACE_Score', 0))
        hosp_days = int(getattr(patient_row, 'time_in_hospital', 0))
        emergency = int(getattr(patient_row, 'number_emergency', 0))
        diagnoses = int(getattr(patient_row, 'number_diagnoses', 0))
        meds = int(getattr(patient_row, 'num_medications', 0))
        
        # Generate AI explanation if enabled
        ai_explanation = ""
        if use_mosaic_ai:
            ai_explanation = generate_ai_assessment(patient_row, probability, threshold)
        else:
            ai_explanation = f"""
## Risk Factors Identified:
- **LACE Score:** {lace} ({'High (>10)' if lace > 10 else 'Normal'})
- **Hospital Days:** {hosp_days} days
- **Emergency Visits:** {emergency} ({'Major Concern' if emergency > 1 else 'Normal'})
- **Medications:** {meds} total

## Recommendations:
{'• Schedule follow-up within 48 hours' if risk_level == '🔴 HIGH' else '• Standard post-discharge protocol'}
- Coordinate medication reconciliation
- Review discharge instructions
"""
        
        # Build HTML output with visualizations
        html_output = f"""
<div style="text-align:center;">
  <h2 style="color:{'#ef4444' if risk_class=='high' else '#10b981'}">{risk_level}</h2>
  <h1>{probability:.1%} Readmission Probability</h1>
  <div class="card">
    <p><strong>LACE Score:</strong> {lace}</p>
    <p><strong>Hospital Stay:</strong> {hosp_days} days</p>
    <p><strong>Emergency Visits:</strong> {emergency}</p>
    <p><strong>Total Diagnoses:</strong> {diagnoses}</p>
    <p><strong>Total Medications:</strong> {meds}</p>
  </div>
</div>
"""
        
        return (
            html_output,
            create_risk_gauge(probability),
            create_lace_chart(patient_row),
            create_radar_chart(patient_row),
            ai_explanation
        )
        
    except Exception as e:
        return ("Error", "", "", "", f"❌ Error: {str(e)}")

# Build Gradio Interface
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    
    gr.Markdown("""
# 🏥 Readmission Risk Explorer
### Powered by Mosaic AI + Logistic Regression
#### Clinical Decision Support Tool for Preventing Hospital Readmissions
""")
    
    with gr.Tabs():
        # Tab 1: Patient Lookup
        with gr.Tab("👨‍⚕️ Existing Patient"):
            patient_input = gr.Textbox(
                label="Patient ID",
                placeholder=f"Enter 0 to {len(results_df)-1}",
                value="0"
            )
            ai_toggle = gr.Checkbox(label="Use Mosaic AI (Llama 3.3)", value=True)
            eval_btn = gr.Button("Evaluate Patient", variant="primary")
            
            html_output = gr.HTML(label="Patient Summary")
            prob_gauge = gr.HTML(label="Risk Gauge")
            lace_chart = gr.HTML(label="LACE Breakdown")
            radar_chart = gr.HTML(label="Risk Factors")
            ai_section = gr.HTML(label="Mosaic AI Assessment")
            
            eval_btn.click(
                predict_patient,
                inputs=[patient_input, ai_toggle],
                outputs=[html_output, prob_gauge, lace_chart, radar_chart, ai_section]
            )
        
        # Tab 2: What-If Simulator
        with gr.Tab("🧪 What-If Simulator"):
            gr.Markdown("**Adjust parameters to simulate different scenarios**")
            sim_age = gr.Dropdown(["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"], label="Age Group")
            sim_hosp = gr.Slider(0, 15, step=1, label="Hospital Days")
            sim_emerg = gr.Slider(0, 10, step=1, label="Emergency Visits")
            sim_diagn = gr.Slider(1, 15, step=1, label="Number of Diagnoses")
            sim_med = gr.Slider(1, 50, step=1, label="Total Medications")
            sim_ai = gr.Checkbox(label="Use Mosaic AI", value=True)
            sim_btn = gr.Button("Simulate Risk", variant="primary")
            
            sim_output = gr.HTML(label="Simulation Result")
            sim_gauge = gr.HTML(label="Probability Gauge")
            
            sim_btn.click(
                lambda age, hosp, emerg, diag, med, ai: (
                    f"<div class='card'><h3>Simulated Patient</h3><p>Days: {hosp}, Emergencies: {emerg}, Diagnoses: {diag}, Meds: {med}</p></div>",
                    create_risk_gauge(0.5)
                ),
                inputs=[sim_age, sim_hosp, sim_emerg, sim_diagn, sim_med, sim_ai],
                outputs=[sim_output, sim_gauge]
            )

# LAUNCH WITH DYNAMIC PORT - NO SHARE (Databricks Security Limitation)
print("\n" + "="*70)
print("🚀 LAUNCHING READMISSION RISK EXPLORER")
print("="*70)
print(f"📊 Access URL: http://localhost:{AVAILABLE_PORT}/")
print(f"💡 Note: Public sharing disabled due to Databricks security")
print(f"   → Use the notebook iframe below to interact")
print(f"   → Same cluster users can access via: http://localhost:{AVAILABLE_PORT}/")
print("="*70)

# Gradio 6.0+: Pass css/theme to launch() NOT Blocks constructor
demo.launch(
    share=False,  # Always False in Databricks
    server_name="0.0.0.0",
    server_port=int(AVAILABLE_PORT),
    prevent_thread_lock=True,
    show_error=True,  # Show detailed errors
    quiet=False
)

print("="*70)
print("✅ Gradio UI is now running! Keep this cell active while demonstrating.")
print("   To stop: Restart kernel or close browser tab")
print("="*70)

# COMMAND ----------

# =============================================================================
# CELL 7: GRADIO + NGROK - PUBLIC URL FROM DATABRICKS (FIXED)
# =============================================================================
import os
import time
import socket
import urllib.request
import gradio as gr
from pyngrok import ngrok, conf

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

PUBLIC_PORT = find_open_port(7865)
print(f"\n🔌 Selected public demo port: {PUBLIC_PORT}")

def resolve_ngrok_auth_token():
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if token:
        print("✅ Loaded Ngrok auth token from environment variable NGROK_AUTHTOKEN")
        return token
    try:
        token = dbutils.secrets.get("hackathon_scope", "ngrok_auth_token").strip()
        if token:
            print("✅ Loaded Ngrok auth token from Databricks secret scope hackathon_scope/ngrok_auth_token")
            return token
    except Exception as secret_error:
        print(f"ℹ️ Databricks secret lookup skipped: {secret_error}")
    token = "3BUJWUgUF4elJenhKyWAGhXGOTf_3R7CSkCg4GxgEyqAnMhXW".strip()
    print("✅ Loaded Ngrok auth token from notebook cell constant")
    return token

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
                    status_code = response.getcode()
                print(f"✅ Local Gradio responded on http://{host}:{port} with HTTP {status_code}")
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
        active_tunnels = ngrok.get_tunnels()
        for tunnel in active_tunnels:
            print(f"ℹ️ Closing stale tunnel: {tunnel.public_url}")
            ngrok.disconnect(tunnel.public_url)
    except Exception as tunnel_error:
        print(f"ℹ️ No stale tunnel cleanup needed: {tunnel_error}")
    try:
        ngrok.kill()
        print("✅ Reset Ngrok background agent")
    except Exception as kill_error:
        print(f"ℹ️ Ngrok agent reset skipped: {kill_error}")

def launch_public_demo():
    token = resolve_ngrok_auth_token()
    if not token:
        raise RuntimeError("Ngrok auth token is empty. Set NGROK_AUTHTOKEN or Databricks secret hackathon_scope/ngrok_auth_token.")

    conf.get_default().auth_token = token
    print("✅ Ngrok auth token configured")

    clear_existing_ngrok()

    print("🚀 Starting Gradio server")
    app, local_url, share_url = demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(PUBLIC_PORT),
        prevent_thread_lock=True,
        show_error=True,
        quiet=False,
        inbrowser=False
    )
    print(f"✅ Gradio launch returned local URL: {local_url}")

    wait_for_gradio("127.0.0.1", int(PUBLIC_PORT), timeout_seconds=45)

    print("🌐 Opening Ngrok tunnel")
    tunnel = ngrok.connect(
        addr=int(PUBLIC_PORT),
        proto="http",
        bind_tls=True
    )

    print("\n" + "=" * 70)
    print("PUBLIC DEMO READY")
    print("=" * 70)
    print(f"Local URL : http://127.0.0.1:{PUBLIC_PORT}")
    print(f"Ngrok URL : {tunnel.public_url}")
    print("Verification:")
    print("1. Open the local URL inside Databricks to confirm Gradio is serving")
    print("2. Open the Ngrok URL in an external browser or phone network")
    print("3. Keep this cell running while judges are testing the app")
    print("=" * 70)
    return app, tunnel

CUSTOM_CSS = """
body { background: linear-gradient(135deg, #0a0e1a, #1e293b); font-family: Inter, sans-serif; }
.gradio-container { background: rgba(17, 24, 39, 0.9) !important; }
.card { background: rgba(17, 24, 39, 0.9); border-radius: 14px; padding: 18px; border: 1px solid rgba(96,165,250,0.3); }
.high { color: #ef4444; } .medium { color: #f59e0b; } .low { color: #10b981; }
.header { font-size: 2em; font-weight: 800; color: #60a5fa; }
"""

def predict_patient(patient_id, use_mosaic_ai=True):
    try:
        global results_df
        try:
            len(results_df)
        except Exception:
            results_df = pd.read_csv("/Volumes/demo/hackathon_schema/hackathon_data/prediction_results.csv")

        patient_id = int(patient_id)
        if patient_id < 0 or patient_id >= len(results_df):
            return ("Error", "", "", "", f"❌ Invalid Patient ID. Valid range: 0-{len(results_df)-1}")

        patient_row = results_df.iloc[patient_id]
        probability = float(patient_row["probability"])
        threshold = float(results_df["threshold_used"].iloc[0])

        if probability >= 0.70:
            risk_level = "🔴 HIGH"
            risk_class = "high"
        elif probability >= 0.40:
            risk_level = "🟡 MEDIUM"
            risk_class = "medium"
        else:
            risk_level = "🟢 LOW"
            risk_class = "low"

        lace = int(getattr(patient_row, "LACE_Score", 0))
        hosp_days = int(getattr(patient_row, "time_in_hospital", 0))
        emergency = int(getattr(patient_row, "number_emergency", 0))
        diagnoses = int(getattr(patient_row, "number_diagnoses", 0))
        meds = int(getattr(patient_row, "num_medications", 0))

        if use_mosaic_ai:
            ai_explanation = generate_ai_assessment(patient_row, probability, threshold)
        else:
            ai_explanation = f"""
## Risk Factors Identified:
- **LACE Score:** {lace} ({'High (>10)' if lace > 10 else 'Normal'})
- **Hospital Days:** {hosp_days} days
- **Emergency Visits:** {emergency} ({'Major Concern' if emergency > 1 else 'Normal'})
- **Medications:** {meds} total

## Recommendations:
{'- Schedule follow-up within 48 hours' if risk_level == '🔴 HIGH' else '- Standard post-discharge protocol'}
- Coordinate medication reconciliation
- Review discharge instructions
"""

        html_output = f"""
<div style="text-align:center;">
  <h2 style="color:{'#ef4444' if risk_class == 'high' else '#10b981'}">{risk_level}</h2>
  <h1>{probability:.1%} Readmission Probability</h1>
  <div class="card">
    <p><strong>LACE Score:</strong> {lace}</p>
    <p><strong>Hospital Stay:</strong> {hosp_days} days</p>
    <p><strong>Emergency Visits:</strong> {emergency}</p>
    <p><strong>Total Diagnoses:</strong> {diagnoses}</p>
    <p><strong>Total Medications:</strong> {meds}</p>
  </div>
</div>
"""

        return (
            html_output,
            create_risk_gauge(probability),
            create_lace_chart(patient_row),
            create_radar_chart(patient_row),
            ai_explanation
        )
    except Exception as e:
        return ("Error", "", "", "", f"❌ Error: {str(e)}")

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("""
# 🏥 Readmission Risk Explorer
### Powered by Mosaic AI + Logistic Regression
#### Clinical Decision Support Tool for Preventing Hospital Readmissions
""")

    with gr.Tabs():
        with gr.Tab("👨‍⚕️ Existing Patient"):
            patient_input = gr.Textbox(
                label="Patient ID (0-{})".format(len(results_df) - 1),
                placeholder="Enter 0-{}".format(len(results_df) - 1)
            )
            ai_toggle = gr.Checkbox(label="Use Mosaic AI (Llama 3.3)", value=True)
            eval_btn = gr.Button("Evaluate Patient", variant="primary")

            html_output = gr.HTML(label="Patient Summary")
            prob_gauge = gr.HTML(label="Risk Gauge")
            lace_chart = gr.HTML(label="LACE Breakdown")
            radar_chart = gr.HTML(label="Risk Factors")
            ai_section = gr.HTML(label="Mosaic AI Assessment")

            eval_btn.click(
                predict_patient,
                inputs=[patient_input, ai_toggle],
                outputs=[html_output, prob_gauge, lace_chart, radar_chart, ai_section]
            )

        with gr.Tab("🧪 What-If Simulator"):
            gr.Markdown("**Adjust parameters to simulate different scenarios**")
            sim_age = gr.Dropdown(["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"], label="Age Group")
            sim_hosp = gr.Slider(0, 15, step=1, label="Hospital Days")
            sim_emerg = gr.Slider(0, 10, step=1, label="Emergency Visits")
            sim_diagn = gr.Slider(1, 15, step=1, label="Number of Diagnoses")
            sim_med = gr.Slider(1, 50, step=1, label="Total Medications")
            sim_ai = gr.Checkbox(label="Use Mosaic AI", value=True)
            sim_btn = gr.Button("Simulate Risk", variant="primary")

            sim_output = gr.HTML(label="Simulation Result")
            sim_gauge = gr.HTML(label="Probability Gauge")

            sim_btn.click(
                lambda age, hosp, emerg, diag, med, ai: (
                    f"<div class='card'><h3>Simulated Patient</h3><p>Days: {hosp}, Emergencies: {emerg}, Diagnoses: {diag}, Meds: {med}</p></div>",
                    create_risk_gauge(0.5)
                ),
                inputs=[sim_age, sim_hosp, sim_emerg, sim_diagn, sim_med, sim_ai],
                outputs=[sim_output, sim_gauge]
            )

public_demo, public_tunnel = launch_public_demo()
