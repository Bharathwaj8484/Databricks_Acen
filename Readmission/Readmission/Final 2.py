# =============================================================================
# STAGE 2 — MODEL TRAINING & EVALUATION (Recall-First)
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)

# -----------------------------------------------------------------
# 1. LOAD CLEANED DATA
# -----------------------------------------------------------------
DATA_PATH = "diabetic_data.csv"
df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)

# Re-apply minimal assumptions from Stage 1
df["readmitted_flag"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

# Drop target leakage / IDs
X = df.drop(columns=["readmitted", "readmitted_flag", "patient_nbr"], errors="ignore")
y = df["readmitted_flag"]

# -----------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (STRATIFIED)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# -----------------------------------------------------------------
# 3. FEATURE TYPES
# -----------------------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -----------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# -----------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# -----------------------------------------------------------------
# 5. MODEL — RECALL-FOCUSED LOGISTIC REGRESSION
# -----------------------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 2},  # emphasize high-risk recall
    solver="lbfgs",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", model)
    ]
)

# -----------------------------------------------------------------
# 6. TRAIN
# -----------------------------------------------------------------
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------
# 7. EVALUATION — DEFAULT THRESHOLD
# -----------------------------------------------------------------
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== Classification Report (threshold = 0.5) ===")
print(classification_report(y_test, y_pred, digits=3))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC AUC:", roc_auc_score(y_test, y_prob))

# -----------------------------------------------------------------
# 8. THRESHOLD TUNING — RECALL FIRST
# -----------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Choose threshold where recall >= 0.80 if possible
target_recall = 0.80
idx = np.where(recall >= target_recall)[0]

if len(idx) > 0:
    best_threshold = thresholds[idx[-1]]
else:
    best_threshold = 0.5

y_pred_tuned = (y_prob >= best_threshold).astype(int)

print(f"\n=== Tuned Threshold = {best_threshold:.3f} ===")
print(classification_report(y_test, y_pred_tuned, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
