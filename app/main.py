from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle une seule fois, au démarrage du serveur
    data = joblib.load(MODEL_PATH)
    artifacts["model"] = data["model"]
    artifacts["target_names"] = data["target_names"]
    yield
    artifacts.clear()


app = FastAPI(
    title="API de prédiction Iris",
    description="Mini-projet MLOps : servir un modèle scikit-learn via FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


class IrisInput(BaseModel):
    """Validation d'entrée : toutes les valeurs doivent être des nombres > 0 et <= 20."""

    sepal_length: float = Field(..., gt=0, le=20, description="Longueur du sépale (cm)")
    sepal_width: float = Field(..., gt=0, le=20, description="Largeur du sépale (cm)")
    petal_length: float = Field(..., gt=0, le=20, description="Longueur du pétale (cm)")
    petal_width: float = Field(..., gt=0, le=20, description="Largeur du pétale (cm)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    }


class PredictionOutput(BaseModel):
    prediction: str
    class_id: int
    probabilities: dict[str, float]


@app.get("/")
def root():
    return {"message": "API Iris en ligne. Va sur /docs pour tester."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in artifacts}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    model = artifacts["model"]
    names = artifacts["target_names"]

    class_id = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    return PredictionOutput(
        prediction=names[class_id],
        class_id=class_id,
        probabilities={names[i]: round(float(p), 4) for i, p in enumerate(proba)},
    )