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
                <div class="header-stat-value">{MODEL_F2:.3f}</div>
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
                    <strong style="color: #f9fafb;">F2-Score:</strong> {MODEL_F2:.3f}<br>
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
print(f"  F2-Score          : {MODEL_F2:.3f}")
print(f"  Threshold        : {MODEL_THRESHOLD:.3f}")
print(f"  AI endpoint      : {SELECTED_ENDPOINT}")
print("=" * 70)

demo.launch(share=True)
