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
file_path = "agent_input_final.csv"
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