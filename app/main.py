import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


app = FastAPI(title="Virtual Diabetes Clinic Triage")
FEATURES = list(PatientFeatures.model_fields)


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(Path(os.getenv("MODEL_PATH", "models/model.joblib")))


@app.get("/health")
def health():
    get_model()
    return {"status": "ok", "model_version": os.getenv("MODEL_VERSION", "v0.1")}


@app.post("/predict")
def predict(patient: PatientFeatures):
    row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    return {"prediction": float(get_model().predict(row)[0])}
