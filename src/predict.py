import pandas as pd
import joblib
from pathlib import Path

TARGET_COLUMN = "Status"
DROP_COLUMNS = ["ID"]

THRESHOLD = 0.5  # chosen based on analysis

BASE_DIR = Path(__file__).resolve().parent      # → /app/src
MODEL_DIR = BASE_DIR / "model"                  # → /app/src/model

MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

FEATURE_COLUMNS = joblib.load(FEATURE_COLUMNS_PATH)

def preprocess_input(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])

    # Drop unused columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical variables
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[FEATURE_COLUMNS]

    return df



def predict_default(input_data: dict):
    # Load model
    model = joblib.load(MODEL_PATH)

    # Preprocess input
    X = preprocess_input(input_data)

    # Predict probability
    prob_default = float(model.predict_proba(X)[:, 1][0])

    # Apply threshold
    prediction = int(prob_default >= THRESHOLD)

    return {
        "default_probability": round(prob_default, 4),
        "risk_label": "High Risk" if prediction == 1 else "Low Risk"
    }


if __name__ == "__main__":
    sample_input = {
        "year": 2022,
        "loan_limit": "C",
        "Gender": "Male",
        "approv_in_adv": "Y",
        "loan_type": "Type1",
        "loan_purpose": "p1",
        "Credit_Worthiness": "Yes",
        "open_credit": 2,
        "business_or_commercial": "No",
        "loan_amount": 250000,
        "rate_of_interest": 7.5,
        "Interest_rate_spread": 0.3,
        "Upfront_charges": 5000,
        "term": 360,
        "Neg_ammortization": "No",
        "interest_only": "No",
        "lump_sum_payment": "No",
        "property_value": 350000,
        "construction_type": "Type1",
        "occupancy_type": "Owner",
        "Secured_by": "Home",
        "total_units": 1,
        "income": 80000,
        "credit_type": "CIB",
        "Credit_Score": 720,
        "co-applicant_credit_type": "None",
        "age": "35-44",
        "submission_of_application": "Online",
        "LTV": 71.4,
        "Region": "North",
        "Security_Type": "Direct",
        "dtir1": 35
    }

    result = predict_default(sample_input)
    print(result)
