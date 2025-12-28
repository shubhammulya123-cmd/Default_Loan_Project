from fastapi import FastAPI
from src.predict import predict_default
from pydantic import BaseModel, Field
from typing import Optional


app = FastAPI(title="Loan Default Prediction API")
class PredictionInput(BaseModel):
    loan_amount: float = Field(..., example=250000)
    rate_of_interest: float = Field(..., example=7.5)
    income: float = Field(..., example=80000)
    Credit_Score: int = Field(..., example=720)
    property_value: float = Field(..., example=350000)
    LTV: float = Field(..., example=71.4)
    dtir1: float = Field(..., example=35)

class PredictionOutput(BaseModel):
    default_probability: float
    risk_label: str

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    return predict_default(payload.dict())



