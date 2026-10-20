# MISSION START: Readmission Risk Explorer — Databricks & Accenture Hackathon

**Role:** You are an elite 30+ year veteran ML Engineer, Clinical Data Scientist, and Databricks/Mosaic AI Expert. You have full autonomy to rewrite, optimize, and completely overhaul the existing codebase to create a winning solution. 

**Project Context:** "Readmission Risk Explorer" is a Mosaic AI Agent built on Databricks in partnership with Accenture and Databricks. 
**Dataset:** UCI Diabetes 130-US Hospitals dataset (~101,766 records, 50 features). You have access to the full `diabetic_data.csv` and `IDS_mapping.csv` in your environment.

## 🎯 CORE OBJECTIVES & TARGETS
1. **Optimize the Precision-Recall Tradeoff:** The clinical priority is maximizing the capture of High-Risk patients (high recall for `<30` days readmission), but **not** at the expense of precision. The current code forces a 90% recall by artificially lowering the threshold, which destroys precision and flags too many false positives. Find the legitimate patterns to achieve high recall natively.
2. **Total Code Autonomy:** The provided code (`Data_Preparation.py`, `Model.py`, `Gradio_Output.py`) is flawed. You are authorized to discard bad approaches (e.g., the random clinical imputation hack in data prep) and completely rewrite the pipeline using state-of-the-art ML engineering.
3. **Self-Correction & Backpropagation:** Run your code, evaluate the metrics (PR-AUC, F1, Recall, Confusion Matrix), identify why it is failing or underperforming, and iterate. Do not stop until the results are hackathon-winning quality.
4. **No Hallucinations:** Base every decision, stat, and feature on the actual dataset. There are exactly ~101,766 records and 50 features. Do not hallucinate external clinical data.

---

## 🛠️ STEP-BY-STEP EXECUTION PLAN

You must operate strictly in the following phases. **Do NOT move to the next phase until the current phase is fully executed, tested, and validated by you.**

### PHASE 1: Deep EDA & Data Understanding
*   Load the full `diabetic_data.csv` and `IDS_mapping.csv`.
*   Analyze the true distributions, missingness (e.g., `?`, `None`, or empty strings), and target variable imbalance (`readmitted` = `<30`, `>30`, `NO`).
*   Formulate a robust clinical feature engineering plan. Stop and output your plan before proceeding.

### PHASE 2: Data Preparation & Feature Engineering
*   **Fix the Target:** Binarize the target correctly (`<30` days vs all others).
*   **Drop Hacks:** Remove the random number generator "clinical imputation" currently used for `number_emergency`. Replace it with statistically sound techniques.
*   **Map the IDs:** Use `IDS_mapping.csv` to map `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`. Categorize these intelligently (e.g., group discharge to "Home" vs "SNF/Rehab" vs "Expired"). Note: Drop records where discharge disposition is "Expired" or "Hospice", as they cannot be readmitted.
*   **ICD-9 Grouping:** The columns `diag_1`, `diag_2`, `diag_3` contain hundreds of raw ICD-9 codes. Group these into meaningful clinical categories (e.g., Circulatory, Respiratory, Diabetes, etc.) rather than leaving them as raw strings.
*   **Medication Engineering:** Combine the 24 medication features into useful aggregates (e.g., `num_meds_changed`, `diabetes_med_complexity`).

### PHASE 3: Advanced Modeling & Mosaic AI Integration
*   Train a robust gradient boosting model (XGBoost, LightGBM, or CatBoost).
*   Implement proper handling of class imbalance (e.g., SMOTE, `scale_pos_weight`, or Focal Loss) rather than arbitrarily dropping the decision threshold.
*   Run hyperparameter tuning (e.g., using Optuna or GridSearch) focusing on maximizing **PR-AUC** (Precision-Recall Area Under Curve) and F2-Score (weighing recall higher than precision, but keeping precision accountable).
*   Calculate the LACE score cleanly.
*   Ensure the Databricks/Mosaic AI LangChain implementation uses the best system prompts for the Clinical Case Manager persona. 
*   *Checkpoint: Output the final classification report and confusion matrix. Do not proceed to Phase 4 until you are satisfied that this is a winning model.*

### PHASE 4: Gradio UI Reconstruction
*   *Only begin this phase once Phase 3 yields a highly performant model.*
*   Refactor the Gradio Dashboard to seamlessly integrate with your newly engineered features and improved model logic.
*   Ensure all data visualizations (Plotly charts) accurately reflect the newly created clinical features and actual model SHAP/feature importance.
*   Maintain the high-quality CSS and UI layout, ensuring the system safely handles missing data gracefully.

---

## 🚀 YOUR IMMEDIATE INSTRUCTION:
Acknowledge these instructions. Begin immediately with **PHASE 1**. Write the code to load and analyze `diabetic_data.csv` and `IDS_mapping.csv`, run it, and tell me your clinical feature engineering plan based on what you find. **Do not generate Gradio code yet.**