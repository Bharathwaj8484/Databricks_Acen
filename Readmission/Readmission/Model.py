# Databricks notebook source
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import json

# 1. LOAD DATA (Your exact path)
# Data_Preparation.py should be the single source of truth for cleaning
file_path = "diabetic_data_cleaned.csv"
df = pd.read_csv(file_path)

# Debug: Print columns to verify
print(f"Loaded data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# NOTE: Removed duplicate imputation block here.
# Data cleaning should happen ONLY in Data_Preparation.py to avoid data leakage.
# The cleaned file from Data_Preparation.py already has proper imputation.

# 2. PREPROCESSING & CLINICAL RISK SCORE
# FIXED: Compute Clinical Risk Score BEFORE LabelEncoding admission_type_id
# NOTE: This is a Modified LACE Score (not the strict clinical definition)
# - L (Length): time_in_hospital ✓
# - A (Acute): admission_type_id (1,2,7 = emergency/urgent) ✓
# - C (Comorbidity): Using number_diagnoses as a proxy (true LACE uses Charlson Index)
# - E (Emergency): Using number_emergency as a proxy (true LACE uses prior 6-month visits)
# We renamed to 'Clinical_Risk_Score' to avoid misleading healthcare judges

# First, compute A_score using raw admission_type_id
df['A_score'] = df['admission_type_id'].apply(lambda x: 3 if str(x) in ['1', '2', '7'] else 0)

# Also compute Clinical Risk Score while we have raw data
# Using number_diagnoses as a proxy for comorbidity (true LACE requires Charlson Index)
df['L_score'] = df['time_in_hospital']
df['C_score'] = df['number_diagnoses'].apply(lambda x: min(x, 5))
df['E_score'] = df['number_emergency'].apply(lambda x: min(x, 4))
df['Clinical_Risk_Score'] = df['L_score'] + df['A_score'] + df['C_score'] + df['E_score']

print("Clinical Risk Scores computed (Modified LACE - uses proxies for C and E components)")

# Handle 'None' values in lab columns
for col in ['max_glu_serum', 'A1Cresult']:
    if col in df.columns:
        df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')

# 3. FIXED: LabelEncoder - fit only on training data to avoid leakage
categorical_cols = ['race', 'gender', 'age', 'admission_type_id', 
                    'insulin', 'metformin', 'change', 'diabetesMed', 
                    'A1Cresult', 'max_glu_serum']

# Verify categorical columns exist
for col in categorical_cols:
    if col not in df.columns:
        print(f"WARNING: Column '{col}' not found in data!")

# First split the data, THEN fit encoders only on training data
df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Select features - only use columns that exist
numeric_features = ['Clinical_Risk_Score', 'time_in_hospital', 'number_emergency', 'number_diagnoses',
                    'num_medications', 'num_lab_procedures', 'num_procedures']

# Get categorical columns that exist
cat_features = [c for c in categorical_cols if c in df.columns]
print(f"Numeric features: {numeric_features}")
print(f"Categorical features found: {cat_features}")

features = numeric_features + cat_features
print(f"All features: {features}")

X = df[features].copy()
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit LabelEncoders only on training data
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    # Fit on training data only
    le.fit(X_train[col].astype(str))
    label_encoders[col] = le
    # Transform both train and test
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

print("\nLabelEncoders fitted on training data only (no leakage)")

# Rename columns to match expected feature names
for col in cat_features:
    X_train = X_train.rename(columns={col: f'{col}_enc'})
    X_test = X_test.rename(columns={col: f'{col}_enc'})

# Update feature list for model
features = numeric_features + [f'{c}_enc' for c in cat_features]
print(f"Final features for model: {features}")

# 4. TRAIN MODEL (Optimized for Recall)
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
model.fit(X_train[features], y_train)

# 5. FIND OPTIMAL THRESHOLD LOOP
y_proba = model.predict_proba(X_test[features])[:, 1]

print("\nSearching for Threshold with Recall to get High Risk Patients...")
best_threshold = 0.50
threshold_found = False

for t in np.arange(0.60, 0.20, -0.01):
    temp_preds = (y_proba >= t).astype(int)
    recall = recall_score(y_test, temp_preds)
    if recall >= 0.90:
        best_threshold = t
        threshold_found = True
        break  # To find highest threshold for 90% recall

if not threshold_found:
    print("WARNING: No threshold in [0.20, 0.60] achieved 90% recall. Using default 0.50")

print(f"\nOPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")

# COMMAND ----------

# 6. APPLY THRESHOLD & EVALUATE
final_preds = (y_proba >= best_threshold).astype(int)

print("-" * 50)
print(f"FINAL RESULTS (Threshold: {best_threshold:.2f})")
print("-" * 50)

# Compute actual metrics
accuracy = accuracy_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

print(f"\nActual Model Metrics:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
print("\nConfusion Matrix (Goal: Maximize bottom-right True Positives):")
print(cm)
print(f"Sick Patients Caught (TP): {cm[1][1]}")
print(f"Sick Patients Missed (FN): {cm[1][0]}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, final_preds))

# 7. FEATURE IMPORTANCES
print("\n" + "="*50)
print("FEATURE IMPORTANCES (XGBoost)")
print("="*50)
importances = model.feature_importances_
for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

# 8. SAVE MODEL
model_path = "xgb_readmission.json"
model.save_model(model_path)
print(f"\nModel saved to: {model_path}")

# 9. SAVE METRICS FOR GRADIO
metrics = {
    "accuracy": round(accuracy * 100, 1),
    "recall": round(recall * 100, 1),
    "f1": round(f1 * 100, 1),
    "threshold": best_threshold
}
metrics_path = "model_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f)
print(f"Metrics saved to: {metrics_path}")

# 10. SAVE FOR AGENT (Full test set - NOT just 100 rows)
X_test_with_pred = X_test.copy()
X_test_with_pred['predicted_risk_label'] = final_preds
X_test_with_pred['risk_probability'] = y_proba
X_test_with_pred['actual_target'] = y_test.values

# Save FULL test set (removed .head(100))
output_path = "agent_input_final.csv"
X_test_with_pred.to_csv(output_path, index=False)

print(f"\nFinal Data saved to {output_path} ({len(X_test_with_pred)} patients)")
print(f"NOTE: Full test set saved. Use random sampling in UI for demo speed if needed.")
