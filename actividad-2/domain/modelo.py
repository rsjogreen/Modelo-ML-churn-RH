"""Lógica de negocio del modelo de churn: variables, pipeline y métricas.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables del modelo. Los identificadores (RowNumber, CustomerId, Surname) no se usan.
NUMERICAS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Complain",
    "Satisfaction Score",
    "Point Earned",
]
CATEGORICAS = ["Geography", "Gender", "Card Type"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "Exited"


def resolver_features(seleccion, custom=None):
    """Convierte ALL, NUMERIC o CUSTOM de features.yaml en columnas numéricas y categóricas."""
    if seleccion == "ALL":
        elegidas = VARIABLES
    elif seleccion == "NUMERIC":
        elegidas = NUMERICAS
    elif seleccion == "CUSTOM":
        elegidas = custom or []
    else:
        raise ValueError(f"seleccion de features no soportada: {seleccion!r}")

    desconocidas = [v for v in elegidas if v not in VARIABLES]
    if desconocidas:
        raise ValueError(f"variables desconocidas en 'custom': {desconocidas}")

    if not elegidas:
        raise ValueError("la selección de features no tiene ninguna variable")
    if len(set(elegidas)) != len(elegidas):
        raise ValueError("la selección de features contiene variables duplicadas")
    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return numericas, categoricas


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar y one-hot) más el estimador elegido en la configuración."""
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def calcular_metricas(y_true, proba, umbral):
    """Métricas de clasificación binaria a partir de la probabilidad de churn."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
