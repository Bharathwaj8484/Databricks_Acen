# Readmission Risk Explorer — Complete Pipeline Overhaul

Rebuild the entire ML pipeline for the UCI Diabetes 130-US Hospitals dataset to create a hackathon-winning readmission risk prediction system. The current codebase has critical flaws (target leakage, no ICD-9 grouping, no discharge filtering) that must be fixed.

## EDA Key Findings

| Finding | Detail |
|---------|--------|
| **Records** | 101,766 encounters, 71,518 unique patients |
| **Target** | `<30`: 11.2% (11,357), `>30`: 34.9%, `NO`: 53.9% |
| **Strongest predictor** | `number_inpatient` — readmission rate climbs from 8.4% (0 visits) to 31.4% (5 visits) |
| **ICD-9 428 (CHF)** | Highest readmission rate at 14.1% |
| **ICD-9 434 (Cerebrovascular)** | 16.2% readmission rate |
| **Expired/Hospice** | 2,423 records (DID 11,13,14,19,20,21) — must drop |
| **Duplicate patients** | 16,773 patients have multiple encounters |
| **Critical flaw** | Target leakage in `number_emergency` imputation |

> [!CAUTION]
> The existing `Data_Preparation.py` injects **random values into `number_emergency` based on the target variable** (`readmitted == '<30'`). This is **target leakage** that invalidates all metrics. Must be completely removed.

---

## Proposed Changes

### Data Preparation Pipeline

#### [MODIFY] [Data_Preparation.py](file:///c:/Users/bhara/Downloads/AG%20%20Readmission/Data_Preparation.py)

Complete rewrite with the following changes:

1. **Remove target leakage** — Delete the random clinical imputation hack entirely
2. **Drop expired/hospice** — Filter out DID 11, 13, 14, 19, 20, 21 (patients who can't be readmitted)
3. **Drop duplicate encounters** — Keep only the first encounter per `patient_nbr` to avoid data leakage
4. **Drop `Unknown/Invalid` gender** — Remove 3 records
5. **ICD-9 Grouping** — Map `diag_1`, `diag_2`, `diag_3` to 9 clinical categories:
   - Circulatory (390–459, 785), Respiratory (460–519, 786), Digestive (520–579, 787)
   - Diabetes (250.xx), Injury (800–999), Musculoskeletal (710–739)
   - Genitourinary (580–629, 788), Neoplasms (140–239), Mental (290–319)
   - Other (everything else)
6. **Medication engineering** — Create aggregate features from 23 medication columns:
   - `num_meds_changed`: Count of medications with dose change (Up/Down)
   - `num_meds_active`: Count of medications prescribed (not 'No')
   - `diabetes_med_complexity`: Weighted score based on insulin + oral agents
   - `insulin_intensity`: Map No→0, Steady→1, Down→2, Up→3
7. **Clinical interaction features**:
   - `polypharmacy_flag`: `num_medications > 20`
   - `high_utilizer`: `number_inpatient + number_emergency + number_outpatient`
   - `visit_complexity`: `num_procedures * time_in_hospital`
   - `lab_intensity`: `num_lab_procedures / time_in_hospital`
8. **Map IDS** — Use `IDS_mapping.csv` to create grouped categorical features:
   - `admission_type_group`: Emergency/Urgent, Elective, Other
   - `discharge_group`: Home, Transfer/Rehab, Left AMA, Other
   - `admission_source_group`: Emergency, Referral, Transfer, Other
9. **Proper missing value handling** — Use 'Missing' category instead of imputing fake values

---

### Modeling Pipeline

#### [MODIFY] [Model.py](file:///c:/Users/bhara/Downloads/AG%20%20Readmission/Model.py)

Complete rewrite:

1. **Model**: XGBoost with proper hyperparameter tuning via Optuna
2. **Class imbalance**: SMOTE-ENN (combine oversampling + cleaning) on training set only
3. **Tuning objective**: Maximize **PR-AUC** (not accuracy or raw recall)
4. **Threshold selection**: Use PR curve to find the optimal F2-score threshold (weighs recall 2× higher than precision) rather than artificially lowering threshold
5. **LACE Score**: Clean calculation using actual dataset fields
6. **Feature set**: ~30+ engineered features including ICD-9 groups, medication aggregates, clinical interactions
7. **Evaluation**: Full classification report, confusion matrix, PR-AUC, F2-score, ROC-AUC
8. **SHAP**: Feature importance via SHAP TreeExplainer
9. **Mosaic AI Agent**: Improved system prompt with richer patient context from new features
10. **Export**: Save model, SHAP values, and full test set for Gradio

---

### Gradio UI

#### [MODIFY] [Gradio_Output.py](file:///c:/Users/bhara/Downloads/AG%20%20Readmission/Gradio_Output.py)

Refactor to integrate new features:

1. Update patient profile to include ICD-9 diagnosis groups, medication complexity, utilization scores
2. Replace fake "feature importance" chart with actual SHAP-based feature importance
3. Add PR-AUC and F2-score to header stats
4. Update LACE breakdown with proper component display
5. Richer AI agent context with clinical categories
6. Preserve all existing CSS and premium design

---

### Dependencies

#### [MODIFY] [Imports.py](file:///c:/Users/bhara/Downloads/AG%20%20Readmission/Imports.py)

Add: `lightgbm`, `optuna`, `shap`, `gradio`

---

## Verification Plan

### Automated Tests

Run as a single Python script `verify_pipeline.py` that validates:

```bash
python verify_pipeline.py
```

1. **Data Preparation checks:**
   - No expired/hospice records remain (DID 11,13,14,19,20,21)
   - No duplicate `patient_nbr` values
   - No `Unknown/Invalid` gender
   - Target correctly binarized (only 0/1 in target column)
   - All ICD-9 codes mapped to a valid clinical category
   - No NaN in final feature matrix
   - `number_emergency` was NOT imputed based on target (verify by checking correlation)

2. **Model quality checks:**
   - PR-AUC > 0.20 (baseline for this imbalanced dataset)
   - F2-score > 0.25
   - Recall ≥ 60% at a precision ≥ 20%
   - Classification report printed
   - Confusion matrix printed

3. **End-to-end sanity:**
   - Model can predict on a single patient record
   - Output CSV has all required columns for Gradio

### Manual Verification

- The user should run the Gradio UI on Databricks after all phases are complete
- Verify that the dashboard loads, patient lookup works, and AI assessment generates
- Check that the feature importance chart reflects SHAP values, not hardcoded percentages
