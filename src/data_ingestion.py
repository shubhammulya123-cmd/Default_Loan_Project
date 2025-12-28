import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Loan_Default.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "loan_data.csv"

def ingest_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("Raw data file not found")

    df = pd.read_csv(RAW_DATA_PATH)

    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Data ingestion completed.")

if __name__ == "__main__":
    ingest_data()