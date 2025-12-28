# Loan Default Prediction – End-to-End ML System

An end-to-end machine learning project that predicts **loan default risk** and serves predictions through a **FastAPI REST API**, fully **Dockerized** for consistent deployment.

This project demonstrates the complete ML lifecycle:
**data ingestion → validation → feature engineering → model training → threshold tuning → API serving → containerization**.

---

## 🚀 Project Overview

The objective of this project is to estimate the **probability of loan default** based on applicant financial and credit information, and classify applicants as **High Risk** or **Low Risk**.

The system is designed with **real-world ML engineering principles**, not just model accuracy.

---

## 🧠 Machine Learning Approach

- **Problem Type:** Binary classification (default vs non-default)
- **Model:** Logistic Regression (baseline, interpretable)
- **Class Imbalance Handling:** `class_weight="balanced"`
- **Primary Metrics:**
  - Recall (default / minority class)
  - ROC-AUC
- **Decision Strategy:** Probability-based prediction with configurable threshold

The model prioritizes **risk detection** over raw accuracy, which is appropriate for credit-risk problems.

---

## 🏗️ Project Structure

project-root/
│
├── api/
│ └── main.py # FastAPI application
│
├── src/
│ ├── data_ingestion.py
│ ├── data_validation.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── predict.py # Prediction pipeline
│ └── model/
│ ├── model.pkl
│ └── feature_columns.pkl
│
├── data/
│ ├── raw/
│ └── processed/
│
├── requirements.txt
├── Dockerfile
└── README.md


🖥️ Run Locally (Without Docker)
1. Install dependencies
pip install -r requirements.txt

2. Train the model
python src/train.py

3. Start the API
uvicorn api.main:app --reload

4. Open Swagger UI
http://127.0.0.1:8000/docs

🐳 Run with Docker (Recommended)
1. Build the image
docker build -t loan-default-api .

2. Run the container
docker run -p 8000:8000 loan-default-api

3. Open the API
http://127.0.0.1:8000/docs

✅ Key Features

End-to-end ML pipeline

Imbalance-aware modeling

Probability-based decisions

Feature consistency between training & inference

Clean API with Pydantic validation

Dockerized for reproducible deployment

Interactive Swagger UI

👤 Author

Shubham Mulya
Engineering background, transitioning into Data Science / Machine Learning Engineering, focused on building production-ready ML systems.
