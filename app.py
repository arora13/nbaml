# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from predict import predict_from_text  # reuses your existing function

app = FastAPI(title="NBA-ML", version="0.1.0")


class PredictRequest(BaseModel):
    query: str  # e.g. "GSW vs LAL 2026-10-24"


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "hello": "nba-ml",
        "usage": "POST /predict with {'query': 'GSW vs LAL 2026-10-24'}"
    }


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        return predict_from_text(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
