"""API de inferencia que sirve el modelo registrado con el alias champion."""
import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from scripts.config import cargar_configuracion


cfg = cargar_configuracion()
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]
UMBRAL = cfg["metricas"]["umbral"]
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
VERSION = str(MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version)

app = FastAPI(
    title="API de churn",
    description="Predicción de abandono de clientes con el modelo champion de MLflow.",
)


class Cliente(BaseModel):
    """Datos de un cliente. Los nombres son los mismos que las columnas del CSV."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "CreditScore": 619,
                    "Geography": "France",
                    "Gender": "Female",
                    "Age": 42,
                    "Tenure": 2,
                    "Balance": 0.0,
                    "NumOfProducts": 1,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 101348.88,
                    "Complain": 1,
                    "Satisfaction Score": 2,
                    "Card Type": "DIAMOND",
                    "Point Earned": 464,
                }
            ]
        }
    )

    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Complain: int
    satisfaction_score: int = Field(alias="Satisfaction Score")
    card_type: str = Field(alias="Card Type")
    point_earned: int = Field(alias="Point Earned")


class Prediccion(BaseModel):
    probabilidad_churn: float
    prediccion: int
    version_modelo: str


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": VERSION}


@app.post(
    "/predict",
    response_model=Prediccion,
    summary="Predecir churn de un cliente",
    description="Ejemplo de entrada disponible en el esquema de la petición.",
)
def predict(cliente: Cliente):
    datos = pd.DataFrame([cliente.model_dump(by_alias=True)])[list(modelo.feature_names_in_)]
    proba = float(modelo.predict_proba(datos)[0, 1])
    return {
        "probabilidad_churn": round(proba, 4),
        "prediccion": int(proba >= UMBRAL),
        "version_modelo": VERSION,
    }
