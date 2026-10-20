# MOSAIC AI: READMISSION RISK EXPLORER - SERVERLESS GPU COMPATIBLE
# Fixed version with proper Plotly/Kaleido compatibility

import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
import markdown

# CONFIGURATION & DATA LOADING

SELECTED_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
FILE_PATH = "/Volumes/demo/hackathon_schema/hackathon_data/agent_input_final.csv"

# Load the patient data
try:
    df_agent = pd.read_csv(FILE_PATH)
    df_agent['Patient_ID'] = range(len(df_agent))
    print(f"✅ Successfully loaded {len(df_agent)} patient records")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df_agent = pd.DataFrame()


# CORE FUNCTIONS

def get_patient_profile(patient_id):
    """Retrieves patient data and calculates risk metrics"""
    try:
        patient = df_agent[df_agent['Patient_ID'] == int(patient_id)].iloc[0]
        
        # Determine risk level
        risk_prob = patient['risk_probability']
        if risk_prob >= 0.70:
            risk_level = "HIGH"
            risk_color = "#ef4444"
            risk_icon = "🔴"
        elif risk_prob >= 0.40:
            risk_level = "MEDIUM"
            risk_color = "#f59e0b"
            risk_icon = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "#10b981"
            risk_icon = "🟢"
        
        profile = {
            "Patient_ID": int(patient_id),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_icon": risk_icon,
            "risk_probability": float(risk_prob),
            "predicted_risk_label": int(patient['predicted_risk_label']),
            "LACE_Score": int(patient['LACE_Score']),
            "time_in_hospital": int(patient['time_in_hospital']),
            "number_emergency": int(patient['number_emergency']),
            "number_diagnoses": int(patient['number_diagnoses']),
            "num_medications": int(patient['num_medications']),
            "num_lab_procedures": int(patient['num_lab_procedures']),
            "num_procedures": int(patient['num_procedures']),
            "insulin_enc": int(patient['insulin_enc']),
            "metformin_enc": int(patient['metformin_enc']),
            "change_enc": int(patient['change_enc']),
            "diabetesMed_enc": int(patient['diabetesMed_enc']),
            "A1Cresult_enc": int(patient['A1Cresult_enc']),
            "age_enc": int(patient['age_enc'])
        }
        return profile
    except Exception as e:
        return None

def create_gauge_chart(probability):
    """Creates a risk probability gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Readmission Risk", 'font': {'size': 24, 'color': '#f9fafb'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#06b6d4'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#9ca3af"},
            'bar': {'color': "#06b6d4", 'thickness': 0.8},
            'bgcolor': "rgba(17, 24, 39, 0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(6, 182, 212, 0.3)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f9fafb", 'family': "Inter, sans-serif"},
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def create_lace_breakdown(patient):
    """Creates LACE score component breakdown"""
    
    l_score = patient['time_in_hospital']
    a_score = 3 if patient['time_in_hospital'] > 0 else 0
    c_score = min(patient['number_diagnoses'], 5)
    e_score = min(patient['number_emergency'], 4)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Length of Stay', 'Acuity', 'Comorbidities', 'Emergency Visits'],
            y=[l_score, a_score, c_score, e_score],
            marker=dict(
                color=['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899'],
                line=dict(color='rgba(6, 182, 212, 0.5)', width=2)
            ),
            text=[l_score, a_score, c_score, e_score],
            textposition='outside',
            textfont=dict(size=16, color='#f9fafb')
        )
    ])
    
    fig.update_layout(
        title={
            'text': f"LACE Score Components (Total: {patient['LACE_Score']})",
            'font': {'size': 20, 'color': '#f9fafb'},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17, 24, 39, 0.5)',
        font={'color': "#f9fafb", 'family': "Inter, sans-serif"},
        yaxis={'gridcolor': 'rgba(6, 182, 212, 0.1)', 'title': 'Score'},
        xaxis={'title': 'LACE Components'},
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def create_feature_importance(patient):
    """Creates feature importance radar chart"""
    
    categories = [
        'LACE Score',
        'Hospital Stay',
        'Emergency Visits',
        'Medications',
        'Diagnoses',
        'Lab Procedures'
    ]
    
    # Normalize values to 0-10 scale for visualization
    values = [
        min(patient['LACE_Score'] / 2, 10),
        min(patient['time_in_hospital'], 10),
        min(patient['number_emergency'] * 2, 10),
        min(patient['num_medications'] / 3, 10),
        min(patient['number_diagnoses'] / 2, 10),
        min(patient['num_lab_procedures'] / 5, 10)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(6, 182, 212, 0.3)',
        line=dict(color='#06b6d4', width=3),
        marker=dict(size=8, color='#06b6d4'),
        name='Risk Factors'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor='rgba(6, 182, 212, 0.2)',
                tickfont=dict(color='#9ca3af')
            ),
            angularaxis=dict(
                gridcolor='rgba(6, 182, 212, 0.2)',
                tickfont=dict(color='#f9fafb', size=12)
            ),
            bgcolor='rgba(17, 24, 39, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f9fafb", 'family': "Inter, sans-serif"},
        title={
            'text': "Clinical Risk Factor Profile",
            'font': {'size': 20, 'color': '#f9fafb'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def plot_to_html(fig):
    """Converts plotly figure to interactive HTML div"""
    return fig.to_html(
        include_plotlyjs='cdn',
        div_id=None,
        config={'displayModeBar': False, 'responsive': True}
    )

def generate_ai_assessment(patient):
    """Generates AI clinical assessment using Mosaic AI"""
    try:
        chat_model = ChatDatabricks(endpoint=SELECTED_ENDPOINT)
        
        profile_for_ai = {
            "RISK_LEVEL": f"{patient['risk_icon']} {patient['risk_level']} RISK",
            "Readmission_Probability": f"{patient['risk_probability']:.1%}",
            "LACE_Score": patient['LACE_Score'],
            "Clinical_History": {
                "Hospital_Stay_Days": patient['time_in_hospital'],
                "Prior_Emergency_Visits": patient['number_emergency'],
                "Total_Diagnoses": patient['number_diagnoses']
            },
            "Medications": {
                "Total_Meds": patient['num_medications'],
                "Insulin_Status": patient['insulin_enc'],
                "Metformin_Status": patient['metformin_enc'],
                "Meds_Changed": "Yes" if patient['change_enc'] == 1 else "No"
            },
            "Labs": {
                "A1C_Result_Code": patient['A1Cresult_enc'],
                "Num_Lab_Procedures": patient['num_lab_procedures']
            }
        }
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Clinical Case Manager assisting with discharge planning.
Your goal is to explain the Readmission Risk to a doctor in a clear, structured format.

INPUT DATA:
You will receive a JSON profile of a patient including their 'Readmission_Probability' (calculated by XGBoost) and clinical features.

YOUR TASK:
1. Start with a CLEAR Risk Summary
2. Explain the "Why": Use the LACE Score, Emergency Visits, and Lab data to explain the risk.
   - If LACE > 10, mention it's high.
   - If Emergency Visits > 1, flag it as a major concern.
   - If Meds_Changed is "Yes", mention instability.
3. Recommend Interventions: Suggest 2-3 specific actions (e.g., "Schedule follow-up in 48 hours", "Review Insulin dosage").

FORMAT YOUR RESPONSE IN MARKDOWN:
- Use ## for section headers
- Use **bold** for key terms
- Use bullet points for lists
- Keep it professional and concise

Tone: Professional, Concise, Medical. Do NOT hallucinate data not in the JSON.
"""),
            ("user", "Analyze this patient profile: {patient_data}")
        ])
        
        chain = prompt_template | chat_model
        response = chain.invoke({"patient_data": json.dumps(profile_for_ai)})
        
        # Convert markdown to HTML
        ai_response_html = markdown.markdown(response.content)
        
        return ai_response_html
        
    except Exception as e:
        return f"""
        <div style="color: #ef4444; padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 10px;">
            <h3>⚠️ AI Service Error</h3>
            <p>Unable to generate AI clinical assessment at this time.</p>
            <p style="color: #9ca3af; font-size: 0.9em;">Error: {str(e)}</p>
            <p style="color: #9ca3af; font-size: 0.9em;">Please verify that the Mosaic AI endpoint "{SELECTED_ENDPOINT}" is available in your workspace.</p>
        </div>
        """

def generate_action_timeline(patient):
    """Generates a visual timeline of recommended actions based on risk level"""
    
    if patient['risk_level'] == "HIGH":
        actions = [
            ("Immediate", "Activate high-risk discharge protocol", "#ef4444"),
            ("24 Hours", "First follow-up contact via phone call", "#f97316"),
            ("48-72 Hours", "In-person clinic visit or home health assessment", "#fbbf24"),
            ("Week 1", "Medication reconciliation and adherence check", "#facc15"),
            ("Week 2", "Comprehensive care team review", "#a3e635"),
            ("Week 4", "Re-assessment of risk factors and care plan adjustment", "#22c55e")
        ]
    elif patient['risk_level'] == "MEDIUM":
        actions = [
            ("24-48 Hours", "Standard discharge follow-up call", "#fbbf24"),
            ("Week 1", "Primary care appointment scheduled", "#facc15"),
            ("Week 2", "Case manager check-in call", "#a3e635"),
            ("Month 1", "Care plan review and adjustment as needed", "#22c55e")
        ]
    else:
        actions = [
            ("Week 1", "Routine follow-up call to confirm understanding", "#a3e635"),
            ("2 Weeks", "Primary care visit as scheduled", "#22c55e"),
            ("Month 1", "Standard monitoring per care plan", "#10b981")
        ]
    
    timeline_html = ""
    for timeframe, action, color in actions:
        timeline_html += f"""
        <div class="timeline-item">
            <div class="timeline-content" style="border-left: 4px solid {color}; padding-left: 20px; margin-bottom: 20px;">
                <div class="timeline-time" style="color: {color}; font-weight: 700; font-size: 1.1em;">{timeframe}</div>
                <div class="timeline-action" style="color: #f9fafb; margin-top: 5px;">{action}</div>
            </div>
        </div>
        """
    
    return timeline_html

# MAIN DASHBOARD HTML GENERATION

def generate_dashboard_html(patient_id):
    """Generates the complete dashboard HTML for a patient"""
    
    # Get patient data
    patient = get_patient_profile(patient_id)
    
    if patient is None:
        return """
        <html>
        <head><style>body { background: #0a0e1a; color: #f9fafb; font-family: Inter, sans-serif; padding: 40px; }</style></head>
        <body>
            <div style="text-align: center; padding: 60px; background: rgba(239, 68, 68, 0.1); border-radius: 20px;">
                <h1 style="color: #ef4444;">❌ Patient Not Found</h1>
                <p style="color: #9ca3af; font-size: 1.2em;">Patient ID not found in the test set. Please enter a valid ID (0-99).</p>
            </div>
        </body>
        </html>
        """
    
    # Generate visualizations as interactive HTML
    gauge_fig = create_gauge_chart(patient['risk_probability'])
    lace_fig = create_lace_breakdown(patient)
    feature_fig = create_feature_importance(patient)
    
    # Convert to HTML divs (interactive, not static images)
    gauge_html = plot_to_html(gauge_fig)
    lace_html = plot_to_html(lace_fig)
    feature_html = plot_to_html(feature_fig)
    
    # Generate AI assessment
    ai_assessment = generate_ai_assessment(patient)
    
    # Generate action timeline
    action_timeline = generate_action_timeline(patient)
    
    # Build complete HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Readmission Risk Explorer - Patient {patient_id}</title>
        <style>
            /* ============================================
               GLOBAL THEME: Medical Command Center
               ============================================ */
            :root {{
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
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                background: linear-gradient(135deg, #0a0e1a 0%, #1e293b 50%, #0f172a 100%);
                background-attachment: fixed;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                color: var(--text-primary);
                padding: 30px;
                min-height: 100vh;
            }}
            
            .dashboard-header {{
                background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
                border: 2px solid var(--border-color);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                text-align: center;
                backdrop-filter: blur(20px);
                box-shadow: 0 0 40px rgba(6, 182, 212, 0.2);
            }}
            
            .header-title {{
                font-size: 2.5em;
                font-weight: 800;
                background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            
            .header-subtitle {{
                color: var(--text-secondary);
                font-size: 1.1em;
            }}
            
            .patient-card {{
                background: var(--card-bg);
                border: 2px solid var(--border-color);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            .risk-banner {{
                text-align: center;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                border: 3px solid;
                background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            }}
            
            .risk-banner.high {{
                border-color: var(--accent-danger);
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
            }}
            
            .risk-banner.medium {{
                border-color: var(--accent-warning);
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%);
            }}
            
            .risk-banner.low {{
                border-color: var(--accent-success);
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
            }}
            
            .risk-icon {{
                font-size: 4em;
                margin-bottom: 10px;
            }}
            
            .risk-label {{
                font-size: 2em;
                font-weight: 800;
                margin-bottom: 5px;
            }}
            
            .risk-percentage {{
                font-size: 3em;
                font-weight: 700;
                color: var(--accent-primary);
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            
            .stat-card {{
                background: rgba(6, 182, 212, 0.1);
                border: 1px solid rgba(6, 182, 212, 0.3);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            }}
            
            .stat-value {{
                font-size: 2.5em;
                font-weight: 700;
                color: var(--accent-primary);
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                color: var(--text-secondary);
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .chart-container {{
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
            }}
            
            .ai-section {{
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
                border: 2px solid rgba(139, 92, 246, 0.3);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
            }}
            
            .ai-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
                padding-bottom: 20px;
                border-bottom: 2px solid rgba(139, 92, 246, 0.3);
            }}
            
            .ai-icon {{
                font-size: 2.5em;
            }}
            
            .ai-title {{
                font-size: 1.8em;
                font-weight: 700;
                color: #a78bfa;
            }}
            
            .ai-content {{
                color: var(--text-primary);
                line-height: 1.8;
            }}
            
            .ai-content h2 {{
                color: var(--accent-primary);
                font-size: 1.5em;
                margin: 20px 0 10px 0;
            }}
            
            .ai-content h3 {{
                color: var(--accent-secondary);
                font-size: 1.2em;
                margin: 15px 0 8px 0;
            }}
            
            .ai-content strong {{
                color: var(--accent-primary);
            }}
            
            .ai-content ul {{
                margin-left: 20px;
                margin-top: 10px;
            }}
            
            .ai-content li {{
                margin-bottom: 8px;
            }}
            
            .timeline-section {{
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
            }}
            
            .timeline-title {{
                font-size: 1.5em;
                font-weight: 700;
                color: var(--accent-primary);
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .timeline-item {{
                margin-bottom: 15px;
            }}
            
            .section-title {{
                font-size: 1.8em;
                font-weight: 700;
                color: var(--accent-primary);
                margin: 30px 0 20px 0;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--border-color);
            }}
            
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 30px;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <!-- Header -->
        <div class="dashboard-header">
            <h1 class="header-title">🏥 READMISSION RISK EXPLORER</h1>
            <p class="header-subtitle">AI-Powered Clinical Decision Support System</p>
        </div>
        
        <!-- Patient Risk Banner -->
        <div class="risk-banner {patient['risk_level'].lower()}">
            <div class="risk-icon">{patient['risk_icon']}</div>
            <div class="risk-label">PATIENT {patient_id} - {patient['risk_level']} RISK</div>
            <div class="risk-percentage">{patient['risk_probability']:.1%}</div>
            <div style="color: var(--text-secondary); margin-top: 10px;">30-Day Readmission Probability</div>
        </div>
        
        <!-- Clinical Metrics -->
        <div class="patient-card">
            <h2 class="section-title">📊 Clinical Metrics Dashboard</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{patient['LACE_Score']}</div>
                    <div class="stat-label">LACE Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{patient['time_in_hospital']}</div>
                    <div class="stat-label">Hospital Days</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{patient['number_emergency']}</div>
                    <div class="stat-label">Emergency Visits</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{patient['num_medications']}</div>
                    <div class="stat-label">Medications</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{patient['number_diagnoses']}</div>
                    <div class="stat-label">Diagnoses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{patient['num_lab_procedures']}</div>
                    <div class="stat-label">Lab Procedures</div>
                </div>
            </div>
        </div>
        
        <!-- Visualizations -->
        <h2 class="section-title">📈 Risk Analysis Visualizations</h2>
        
        <div class="charts-grid">
            <div class="chart-container">
                {gauge_html}
            </div>
            <div class="chart-container">
                {lace_html}
            </div>
        </div>
        
        <div class="chart-container">
            {feature_html}
        </div>
        
        <!-- AI Clinical Assessment -->
        <div class="ai-section">
            <div class="ai-header">
                <div class="ai-icon">🤖</div>
                <div>
                    <div class="ai-title">Mosaic AI Clinical Assessment</div>
                    <div style="color: var(--text-secondary); font-size: 0.9em;">Powered by Meta Llama 3.3 70B</div>
                </div>
            </div>
            <div class="ai-content">
                {ai_assessment}
            </div>
        </div>
        
        <!-- Action Timeline -->
        <div class="timeline-section">
            <div class="timeline-title">
                <span>⏱️</span> Suggested Action Timeline
            </div>
            <div class="timeline-container">
                {action_timeline}
            </div>
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; padding: 30px; color: var(--text-secondary); border-top: 1px solid var(--border-color); margin-top: 40px;">
            <p><strong>Model:</strong> XGBoost Classifier | <strong>AI Engine:</strong> Meta Llama 3.3 70B | <strong>Prediction Window:</strong> 30 Days</p>
            <p style="margin-top: 10px;">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </body>
    </html>
    """
    
    return html

# CELL 3: Interactive Patient Analysis Function

def analyze_patient(patient_id):
    html = generate_dashboard_html(patient_id)
    displayHTML(html)

# CELL 4: Quick Example Functions

def show_high_risk_example():
    """Shows a random high-risk patient"""
    try:
        high_risk_patients = df_agent[df_agent['predicted_risk_label'] == 1]
        if len(high_risk_patients) > 0:
            patient_id = int(high_risk_patients.sample(1)['Patient_ID'].iloc[0])
            print(f"🚨 Analyzing High-Risk Patient: {patient_id}")
            analyze_patient(patient_id)
        else:
            print("No high-risk patients found in dataset")
    except Exception as e:
        print(f"Error: {e}")

def show_low_risk_example():
    """Shows a random low-risk patient"""
    try:
        low_risk_patients = df_agent[df_agent['predicted_risk_label'] == 0]
        if len(low_risk_patients) > 0:
            patient_id = int(low_risk_patients.sample(1)['Patient_ID'].iloc[0])
            print(f"✅ Analyzing Low-Risk Patient: {patient_id}")
            analyze_patient(patient_id)
        else:
            print("No low-risk patients found in dataset")
    except Exception as e:
        print(f"Error: {e}")


# LAUNCH INFORMATION

print("=" * 70)
print("🚀 MOSAIC AI READMISSION RISK EXPLORER -  READY")
print("=" * 70)
print(f"✅ Loaded {len(df_agent)} patient records")
print(f"✅ XGBoost model predictions loaded")
print(f"✅ Mosaic AI endpoint: {SELECTED_ENDPOINT}")
print(f"✅ Interactive visualizations enabled (Plotly + Kaleido)")
print(f"✅ Works on Serverless GPU compute")
print("=" * 70)
print("\n📋 HOW TO USE:")
print("1. To analyze a specific patient: analyze_patient(patient_id)")
print("   Example: analyze_patient(0)")
print("\n2. To see a high-risk example: show_high_risk_example()")
print("3. To see a low-risk example: show_low_risk_example()")
print("\n4. Patient IDs range from 0 to", len(df_agent)-1)
print("=" * 70)

# ==============================================================================
# CELL 5: Example Usage - Uncomment to test
# ==============================================================================

# Analyze a specific patient
# analyze_patient(0)

# Or show a random high-risk patient
# show_high_risk_example()

# Or show a random low-risk patient
# show_low_risk_example()

# COMMAND ----------

analyze_patient(4)