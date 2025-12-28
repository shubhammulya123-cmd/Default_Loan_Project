import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

INPUT_PATH = BASE_DIR / "data" / "processed" / "loan_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "loan_data_featured.csv"

TARGET_COLUMN = "Status"
DROP_COLUMNS = ["ID"]

def feature_engineering():
    df = pd.read_csv(INPUT_PATH)

    # 1. Drop unnecessary columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # 2. Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # 3. Handle missing values
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    # 4. Encode categorical variables
    cat_cols = X.select_dtypes(include="object").columns
    encoder = LabelEncoder()

    for col in cat_cols:
        X[col] = encoder.fit_transform(X[col])

    # 5. Recombine features and target
    df_final = pd.concat([X, y], axis=1)

    # 6. Save processed data
    df_final.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed successfully.")
    print(f"Processed data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    feature_engineering()
