# Credit Default Risk Classification (German Credit Dataset)

## Overview

This project builds a complete machine learning pipeline to predict credit default risk using the German Credit dataset. The goal is to identify high-risk customers (defaulters) while balancing model performance and generalization.

The project focuses on:

- Handling class imbalance
- Controlling overfitting
- Comparing multiple models
- Evaluating using business-relevant metrics
- Explaining predictions using SHAP-based feature attribution
- Serving predictions via a FastAPI REST API

---

## Problem Statement

Predict whether a customer is likely to default on credit based on financial and demographic attributes.

- Target Variable: `risk`
  - `1` → Good (Non-defaulter)
  - `2` → Bad (Defaulter)
  - These labels are converted to binary: 0 for Good, 1 for Bad

---

## Dataset

- Source: UCI German Credit Dataset
- Samples: 1000
- Type: Structured tabular data
- Link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

---

## Tech Stack

### Core

- Python 3.9
- NumPy
- pandas

### Visualization

- matplotlib
- seaborn

### Machine Learning

- scikit-learn
- XGBoost
- imbalanced-learn
- shap==0.42.1

### API

- FastAPI
- Uvicorn
- Pydantic

### Utilities

- joblib (model saving)

---

## Project Structure

```
credit-risk/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/                # Data loading & cleaning
│   ├── features/            # Feature engineering
│   ├── preprocessing/       # Encoding & scaling
│   ├── pipelines/           # Model pipelines
│   ├── models/              # Evaluation utilities
│   └── config.py
│
├── models/                  # Saved .joblib model & threshold files
├── reports/                 # Metrics, plots, outputs
├── notebooks/               # EDA notebooks
├── app.py                   # FastAPI application
├── main.py                  # Training entry point
├── environment.yaml
└── README.md
```

---

## Pipeline

1. **Data Loading**

2. **Data Processing**
   - Missing value handling
   - Target mapping

3. **Feature Engineering**
   - Derived features (e.g., credit per duration)

4. **Train-Test Split** (Stratified 80/20)

5. **Model Pipelines**
   - Encoding (OneHot + Ordinal)
   - Scaling (for linear models only)

6. **Cross-Validation**
   - 5-fold Stratified K-Fold CV
   - Metrics: Accuracy, Precision, Recall, F1, AUC with mean ± std

7. **Model Training**
   - Logistic Regression
   - Random Forest
   - XGBoost

8. **Threshold Tuning**
   - Custom recall-floor based threshold selection (min_recall=0.70)
   - Separate threshold saved per model via joblib

9. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC (primary metric)
   - Confusion Matrix
   - Overfitting check via Train AUC vs Test AUC gap

10. **Explainability**
    - SHAP TreeExplainer on Random Forest
    - Per-prediction feature influence scores (normalized to 100%)
    - Returned alongside prediction via API

11. **Reporting**
    - Metrics saved as JSON per model

---

## Models Used

### Logistic Regression

- Baseline model
- `penalty='elasticnet'`, `solver='saga'`
- Good generalization, minimal overfitting gap (~0.02)
- Limited by linear decision boundary — hits performance ceiling at AUC ~0.73

### Random Forest ✅ Best Model

- `class_weight='balanced'` for imbalance handling
- `max_depth=8`, `min_samples_leaf=10` to control overfitting
- Best trade-off: Recall 0.711, AUC 0.780, F1 0.610
- Used for production predictions and SHAP explanations

### XGBoost

- `scale_pos_weight=2.33` for imbalance handling
- Strong regularization: `max_depth=3`, `reg_alpha=0.5`, `reg_lambda=2.0`
- Highest AUC (0.785) but more conservative — requires lower threshold (0.35) to achieve comparable recall
- Significant train/test gap (0.20) despite regularization

---

## Key Techniques

- **Class imbalance handling:** `class_weight='balanced'`, `scale_pos_weight`
- **Overfitting control:** tree depth limitation, regularization (L1+L2)
- **Stratified K-Fold Cross-Validation:** 5-fold, AUC-based gap analysis
- **AUC-based overfitting detection** instead of accuracy
- **Threshold tuning:** recall-floor approach (min_recall=0.70), threshold saved per model
- **Pipeline-based preprocessing:** no data leakage
- **Explainable AI:** SHAP TreeExplainer with normalized influence % scores

---

## Results Summary

### Cross-Validation (5-Fold Stratified)

| Model | CV AUC | Std | Train AUC | Gap |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.720 | ±0.030 | 0.758 | 0.037 |
| Random Forest | 0.774 | ±0.021 | 0.905 | 0.131 |
| XGBoost | 0.772 | ±0.015 | 0.984 | 0.212 |

### After Threshold Tuning (min_recall=0.70)

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.47 | 0.633 | 0.432 | 0.700 | 0.534 | 0.730 |
| Random Forest | 0.48 | 0.727 | 0.533 | 0.711 | 0.610 | 0.780 |
| XGBoost | 0.35 | 0.680 | 0.478 | 0.722 | 0.575 | 0.785 |

### Final Conclusion

Random Forest provides the best balance between recall, precision, F1, and AUC with acceptable overfitting gap, making it the most suitable model for credit risk prediction. It is used as the production model in the API.

> Note: AUC ~0.77–0.78 is consistent with published benchmarks on this dataset (1000 samples, 1994). The ceiling is a data constraint, not a modeling limitation.

---

## API

The trained Random Forest model is served via FastAPI.

### Endpoint

`POST /predict`

### Response

```json
{
  "prediction": 1,
  "probability": 0.72,
  "risk_level": "High Risk",
  "decision": "Likely to Default",
  "shap_influence_values": {
    "Credit Amount": 25.5,
    "Checking Account Status": -18.5,
    "Loan Duration (months)": 11.8,
    "Savings Account": -10.8,
    "Age": 9.3
  }
}
```

`shap_influence_values` shows each feature's relative influence (%) on the prediction. Positive = increases default risk. Negative = reduces default risk. Values sum to 100%.

---

## How to Run

### 1. Create Environment

```
conda env create -f environment.yaml
conda activate credit-risk
```

### 2. Train Models

```
python main.py
```

### 3. Start API

```
uvicorn app:app --reload
```

---

## Key Learnings

- Accuracy is misleading for imbalanced datasets — AUC is the reliable metric
- AUC gap (train vs test) is a better overfitting indicator than accuracy gap
- Recall is the critical metric in credit risk — missing a defaulter costs more than a false alarm
- Threshold tuning is a business decision, not just a technical one — set a recall floor first
- Tree-based models are robust to `drop=None` in OHE; removing it improves SHAP completeness
- SHAP values from dropped OHE categories are missing — normalization corrects for this
- Cross-validation confirms result stability; CV AUC is the trustworthy estimate

---

## Future Improvements

- Hyperparameter tuning (RandomizedSearchCV / Optuna)
- SMOTE integration and comparison
- Feature interaction terms for Logistic Regression improvement
- Cost-sensitive learning with explicit FN/FP cost matrix

---

## Author

Vaibhav Pandey
