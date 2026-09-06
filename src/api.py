import json
import logging
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/heart_disease_model.joblib"
)

METADATA_PATH = os.getenv(
    "METADATA_PATH",
    "models/model_metadata.json"
)


# =========================================================
# Logging configuration
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("heart-disease-api")


# =========================================================
# Load model and metadata
# =========================================================

try:
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

except Exception as exc:
    raise RuntimeError(
        f"Could not load model or metadata: {exc}"
    )


FEATURES = metadata["features"]
GENDER_MAPPING = metadata["gender_mapping"]


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production API for heart disease prediction",
    version="1.0.0"
)


# =========================================================
# Request schema
# =========================================================

class PredictionRequest(BaseModel):
    sno: float = Field(..., description="Sample number")
    age: float = Field(..., description="Patient age")
    gender: str = Field(..., description="male or female")
    cp: float = Field(..., description="Chest pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: float = Field(..., description="Fasting blood sugar")
    restecg: float = Field(..., description="Resting ECG result")
    thalach: float = Field(..., description="Maximum heart rate")
    exang: float = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression")
    slope: float = Field(..., description="Slope of peak exercise ST")
    ca: float = Field(..., description="Number of major vessels")
    thal: float = Field(..., description="Thalassemia")


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# =========================================================
# Prediction endpoint
# =========================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    # -----------------------------------------------------
    # Validate gender
    # -----------------------------------------------------

    gender = request.gender.lower().strip()

    if gender not in GENDER_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid gender '{request.gender}'. "
                f"Expected one of: {list(GENDER_MAPPING.keys())}"
            )
        )


    # -----------------------------------------------------
    # Convert request to model input
    # -----------------------------------------------------

    input_data = {
        "sno": request.sno,
        "age": request.age,
        "gender": GENDER_MAPPING[gender],
        "cp": request.cp,
        "trestbps": request.trestbps,
        "chol": request.chol,
        "fbs": request.fbs,
        "restecg": request.restecg,
        "thalach": request.thalach,
        "exang": request.exang,
        "oldpeak": request.oldpeak,
        "slope": request.slope,
        "ca": request.ca,
        "thal": request.thal
    }


    # Ensure exact feature ordering
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )


    # -----------------------------------------------------
    # Make prediction
    # -----------------------------------------------------

    try:
        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        probability_map = {
            str(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

        prediction_probability = probability_map[
            str(prediction)
        ]

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )


    # -----------------------------------------------------
    # Request timestamp
    # -----------------------------------------------------

    timestamp = datetime.now(timezone.utc).isoformat()


    # -----------------------------------------------------
    # Per-sample logging
    # -----------------------------------------------------

    logger.info(
        "prediction_request "
        "timestamp=%s "
        "inputs=%s "
        "prediction=%s "
        "probability=%.6f",
        timestamp,
        input_data,
        prediction,
        prediction_probability
    )


    # -----------------------------------------------------
    # API response
    # -----------------------------------------------------

    return {
        "prediction": str(prediction),
        "probability": round(
            prediction_probability,
            6
        ),
        "class_probabilities": {
            key: round(value, 6)
            for key, value in probability_map.items()
        },
        "timestamp": timestamp
    }
