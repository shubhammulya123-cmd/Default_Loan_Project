import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Path to processed data
DATA_PATH =BASE_DIR / "data" / "processed" / "loan_data.csv"

# Target column
TARGET_COLUMN = "Status"

# Columns that are critical for modeling
CRITICAL_COLUMNS = [
    "loan_amount",
    "rate_of_interest",
    "income",
    "Credit_Score",
    "property_value",
    "LTV",
    TARGET_COLUMN
]

def validate_data():
    # 1. Check file existence
    if not DATA_PATH.exists():
        raise FileNotFoundError("Processed data file not found.")

    df = pd.read_csv(DATA_PATH)

    # 2. Check required columns
    missing_cols = set(CRITICAL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 3. Check missing values in critical columns
    missing_counts = df[CRITICAL_COLUMNS].isnull().sum()
    missing_columns = missing_counts[missing_counts > 0]

    if not missing_columns.empty:
        print("Warning: Missing values detected in critical columns:")
        for col, count in missing_columns.items():
            print(f"  {col}: {count} missing values")

    # 4. Target sanity check
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    # 5. Basic numeric sanity checks
    if (df["loan_amount"] <= 0).any():
        raise ValueError("Invalid loan_amount: values must be positive.")

    if (df["income"] < 0).any():
        raise ValueError("Invalid income: negative values found.")

    if not df["Credit_Score"].between(300, 900).all():
        print("Warning: Credit_Score values outside expected range (300–900).")

    print("Data validation completed successfully.")

if __name__ == "__main__":
    validate_data()
