from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("models/random_forest_model.joblib")

# Build explainer once at startup — expensive, don't do it per request
# model[-1] accesses the RF classifier inside the sklearn Pipeline
explainer = shap.TreeExplainer(model[-1])

app = FastAPI(title="Credit Default Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://creditdefaultrisk.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CandidateData(BaseModel):
    duration_months: int
    credit_amount: int
    age: int
    checking_account_status: str
    credit_history: str
    savings_account: str
    purpose: str
    property: str
    housing: str
    employment_since: str

def get_feature_names(pipeline):
    # Reconstruct feature names after ColumnTransformer
    ct = pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, transformer, cols in ct.transformers_:
        if transformer == "passthrough" or name == "num":
            feature_names.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)

    return feature_names

# Readable labels for display on frontend
FEATURE_LABELS = {
    # Numerical (passthrough — names come as-is)
    "duration_months": "Loan Duration (months)",
    "credit_amount": "Credit Amount",
    "age": "Age",
    "credit_per_month": "Credit per Month",
    "installment_rate": "Installment Rate",

    # Ordinal (after split("__")[-1], prefix is stripped)
    "checking_account_status": "Checking Account Status",
    "credit_history": "Credit History",
    "savings_account": "Savings Account",

    # Nominal base names (OHE columns matched via startswith)
    "purpose": "Purpose",
    "property": "Property",
    "housing": "Housing",
    "employment_since": "Employment Since",
    "personal_status_sex": "Personal Status",
    "other_debtors": "Other Debtors",
    "other_installment_plans": "Other Installment Plans",
}

def get_shap_values(pipeline, input_df):
    preprocessor = pipeline.named_steps["preprocessor"]

    if "feature_engineering" in pipeline.named_steps:
        fe = pipeline.named_steps["feature_engineering"]
        input_transformed = fe.transform(input_df)
    else:
        input_transformed = input_df

    X_transformed = preprocessor.transform(input_transformed)
    shap_vals = explainer.shap_values(X_transformed)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]

    feature_names = get_feature_names(pipeline)

    # Accumulate OHE columns into base feature
    raw_dict = {}
    for fname, val in zip(feature_names, shap_vals):
        clean = fname.split("__")[-1]
        label = None
        for key, readable in FEATURE_LABELS.items():
            if clean == key or clean.startswith(key + "_"):
                label = readable
                break
        if label is None:
            label = clean
        raw_dict[label] = raw_dict.get(label, 0) + float(val)

    # Rescale to always sum to 100%
    top_total = sum(abs(v) for v in raw_dict.values())
    influence_sorted = {
        k: round((v / top_total) * 100, 1)
        for k, v in sorted(raw_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    }

    return influence_sorted

@app.get("/")
def home():
    return {"message": "Credit Default Risk Prediction API Running"}

@app.post("/predict")
def predict(candidate_data: CandidateData):
    input_df = pd.DataFrame([candidate_data.dict()])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    if probability < 0.3:
        risk_level = "Low Risk"
    elif probability < 0.6:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    shap_values = get_shap_values(model, input_df)

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "decision": "Likely to Default" if prediction == 1 else "Likely Non-Defaulter",
        "shap_values": shap_values
    }