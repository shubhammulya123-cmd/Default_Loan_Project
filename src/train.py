import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import numpy as np

# -----------------------------
# Path configuration
# -----------------------------

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_PATH = BASE_DIR / "data" / "processed" / "loan_data_featured.csv"

# Model artifact paths
MODEL_DIR = BASE_DIR / "src" / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

TARGET_COLUMN = "Status"

# -----------------------------
# Training function
# -----------------------------

def train_model():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Split features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Save feature schema
    FEATURE_COLUMNS = X.columns.tolist()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate baseline prediction
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    print(f"Feature columns saved at {FEATURE_COLUMNS_PATH}")

    # Probability-based evaluation
    y_prob = model.predict_proba(X_test)[:, 1]

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        print(f"\nThreshold: {threshold}")
        print(classification_report(y_test, y_pred_thresh))

    print("Average predicted default probability:", y_prob.mean())
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    train_model()
