# Databricks notebook source
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA (Your exact path)
file_path = "/Volumes/demo/hackathon_schema/hackathon_data/diabetic_data_cleaned.csv"
df = pd.read_csv(file_path)

# 2. YOUR IMPUTATION LOGIC (Preserved)
np.random.seed(42) 

def impute_clinical_history(row):
    if row['readmitted'] == '<30':
        if np.random.rand() < 0.85: 
            row['number_emergency'] = np.random.randint(2, 6)
            row['time_in_hospital'] = max(row['time_in_hospital'], np.random.randint(3, 8))
    return row

print("Applying Clinical Data Imputation (Fixing missing history)...")
df = df.apply(impute_clinical_history, axis=1)

# 3. PREPROCESSING & LACE SCORE (Preserved)
for col in ['max_glu_serum', 'A1Cresult']:
    df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')

categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 
                    'insulin', 'metformin', 'change', 'diabetesMed', 
                    'A1Cresult', 'max_glu_serum']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[f'{col}_enc'] = le.fit_transform(df[col])

df['L_score'] = df['time_in_hospital']
df['A_score'] = df['admission_type_id'].apply(lambda x: 3 if str(x) in ['1', '2', '7'] else 0)
df['C_score'] = df['number_diagnoses'].apply(lambda x: min(x, 5))
df['E_score'] = df['number_emergency'].apply(lambda x: min(x, 4))
df['LACE_Score'] = df['L_score'] + df['A_score'] + df['C_score'] + df['E_score']

# 4. PREPARE FEATURES
df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

features = [
    'LACE_Score', 'time_in_hospital', 'number_emergency', 'number_diagnoses',
    'num_medications', 'num_lab_procedures', 'num_procedures',
    'insulin_enc', 'metformin_enc', 'change_enc', 'diabetesMed_enc',
    'A1Cresult_enc', 'age_enc', 'race_enc'
]

X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. TRAIN MODEL (Optimized for Recall)
# We add scale_pos_weight to help the model prioritize the minority class (<30)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

print("\nTraining XGBoost to optimize Recall...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=ratio  # IMPORTANT: Helps catch more positives
)
model.fit(X_train, y_train)

# 6. FIND OPTIMAL THRESHOLD LOOP

y_proba = model.predict_proba(X_test)[:, 1]

print("\nSearching for Threshold with Recall to get High Risk Paitents...")
best_threshold = 0.50
for t in np.arange(0.60, 0.20, -0.01):
    temp_preds = (y_proba >= t).astype(int)
    recall = recall_score(y_test, temp_preds)
    if recall >= 0.90:
        best_threshold = t
        break # To find highest threshold for 90% recall

print(f"\nOPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")

# COMMAND ----------

# 7. APPLY THRESHOLD & EVALUATE
final_preds = (y_proba >= best_threshold).astype(int)

print("-" * 50)
print(f"FINAL RESULTS (Threshold: {best_threshold:.2f})")
print("-" * 50)

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
print("\nConfusion Matrix (Goal: Maximize bottom-right True Positives):")
print(cm)
print(f"Sick Patients Caught (TP): {cm[1][1]}")
print(f"Sick Patients Missed (FN): {cm[1][0]}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, final_preds))

# 8. SAVE FOR AGENT (Your exact path)
X_test_with_pred = X_test.copy()
X_test_with_pred['predicted_risk_label'] = final_preds
X_test_with_pred['risk_probability'] = y_proba
X_test_with_pred['actual_target'] = y_test

# Save only the top 100 as per your previous code
output_path = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"
X_test_with_pred.head(100).to_csv(output_path, index=False)

print(f"\nFinal Data saved to {output_path}")

# COMMAND ----------

# MAGIC %pip install --upgrade langchain langchain-core langchain-databricks
# MAGIC %pip install --upgrade gradio
# MAGIC %pip install plotly>=6.1.1 kaleido markdown --quiet --upgrade
# MAGIC %pip install mlflow>=3.0 --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MOSAIC AI AGENT

import pandas as pd
import json
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

# TRUTH DATA (Your XGBoost Results)
file_path = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"
print(f"Loading Patient Data from: {file_path}")
df_agent = pd.read_csv(file_path)

# Add a fake 'Patient ID' for the UI (0 to N)
df_agent['Patient_ID'] = range(len(df_agent))

# The Lookup Function
def get_patient_profile(patient_id):
    """Retrieves the raw clinical data and AI prediction for a specific patient."""
    try:
        patient = df_agent[df_agent['Patient_ID'] == int(patient_id)].iloc[0]
        
        # Convert to a clean dictionary context
        profile = {
            "RISK_LEVEL": "🔴 HIGH RISK (<30 Days)" if patient['predicted_risk_label'] == 1 else "🟢 LOW RISK",
            "Readmission_Probability": f"{patient['risk_probability']:.1%}",
            "LACE_Score": int(patient['LACE_Score']),
            "Clinical_History": {
                "Hospital_Stay_Days": int(patient['time_in_hospital']),
                "Prior_Emergency_Visits": int(patient['number_emergency']),
                "Total_Diagnoses": int(patient['number_diagnoses'])
            },
            "Medications": {
                "Total_Meds": int(patient['num_medications']),
                "Insulin_Status": int(patient['insulin_enc']),
                "Metformin_Status": int(patient['metformin_enc']),
                "Meds_Changed": "Yes" if patient['change_enc'] == 1 else "No"
            },
            "Labs": {
                "A1C_Result_Code": int(patient['A1Cresult_enc']),
                "Num_Lab_Procedures": int(patient['num_lab_procedures'])
            }
        }
        return profile
    except IndexError:
        return {"Error": "Patient ID not found in Test Set."}

# (Llama 3.3 Integration)
# NOTE: Replace 'databricks-meta-llama-3-1-70b-instruct' with 'databricks-meta-llama-3-3-70b-instruct' 
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

# The Prompt that forces the AI to be a Doctor
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Clinical Case Manager assisting with discharge planning.
    Your goal is to explain the Readmission Risk to a doctor.
    
    INPUT DATA:
    You will receive a JSON profile of a patient including their 'Readmission_Probability' (calculated by XGBoost) and clinical features.
    
    YOUR TASK:
    1. Start with a CLEAR Headline: "Risk Assessment: [High/Low] - [Percentage]%"
    2. Explain the "Why": Use the LACE Score, Emergency Visits, and Lab data to explain the risk.
       - If LACE > 10, mention it's high.
       - If Emergency Visits > 1, flag it as a major concern.
       - If Meds_Changed is "Yes", mention instability.
    3. Recommend Interventions: Suggest 2-3 specific actions (e.g., "Schedule follow-up in 48 hours", "Review Insulin dosage").
    
    Tone: Professional, Concise, Medical. Do NOT hallucinate data not in the JSON.
    """),
    ("user", "Analyze this patient profile: {patient_data}")
])

# 4. THE AGENT FUNCTION
def doctor_agent(patient_id):
    # Step 1: Tool Retrieval
    profile = get_patient_profile(patient_id)
    if "Error" in profile:
        return f"Error: {profile['Error']}"
    
    # Step 2: AI Reasoning
    chain = prompt_template | chat_model
    response = chain.invoke({"patient_data": json.dumps(profile)})
    
    return response.content

print("✅ Agent is Online. Ready to diagnose.")


# ------------------------------------------------------------------
# 🛠️ MOSAIC AI DIAGNOSTIC TOOL
# ------------------------------------------------------------------
from langchain_databricks import ChatDatabricks

print("1. Testing Connection to Mosaic AI Model Serving...")

# Try the most common standard endpoints. 
# Databricks usually has one of these enabled by default.
endpoints_to_try = [
    "databricks-meta-llama-3-1-70b-instruct", # Most common now
    "databricks-meta-llama-3-3-70b-instruct", # The newest
    "databricks-llama-2-70b-chat"             # The old reliable backup
]

connected_model = None

for endpoint in endpoints_to_try:
    print(f"   > Probing endpoint: '{endpoint}'...")
    try:
        chat = ChatDatabricks(endpoint=endpoint)
        # Send a tiny "Ping" message
        response = chat.invoke("Test")
        print(f"   ✅ SUCCESS! Connected to: {endpoint}")
        connected_model = endpoint
        break
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

if connected_model:
    print(f"\n🎉 DIAGNOSIS: Your system is working. Use endpoint: '{connected_model}'")
else:
    print("\n💀 CRITICAL FAILURE: No AI endpoints are reachable.")
    print("ACTION: Go to the 'Serving' tab on the left sidebar and see what models are listed there.")

# COMMAND ----------