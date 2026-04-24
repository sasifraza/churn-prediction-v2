import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os 

# file paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH= os.path.join(BASE_DIR,"..","saved_models", "model.pkl")


# load models 

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model=None

# create app

app =FastAPI(title= "Churn Prediction API")

# input schema 

class PredictRequest(BaseModel):
    features: dict

# Health endpoint 

@app.get("/health")
def health():
    return { "status": "ok", "model_loaded": model is not None}

# Metrics endpoint

@app.get ("/metric")

def health():
    return {"Accuracy" : 0.7864,
            "Recall_tuned": 0.7587,
            "Precision_tuned": 0.5380,
            "ROC_AUC": 0.8371
            }

# predict endpoint 

# Predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convert dict to dataframe
    df = pd.DataFrame([request.features])
    
    # Get probability and apply threshold
    proba = model.predict_proba(df)[:,1]
    prediction = (proba >= 0.3).astype(int)
    
    return {
        "churn_probability": round(float(proba[0]), 4),
        "churn_prediction": int(prediction[0]),
        "will_churn": bool(prediction[0] == 1)
    }


