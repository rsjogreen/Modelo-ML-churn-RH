"""Pipeline Prefect que puntúa un lote diario usando el alias champion de MLflow."""
import os
from datetime import date

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task

from scripts.config import cargar_configuracion, resolver_ruta
from scripts.load_data import cargar_datos as leer_datos


@task
def cargar_champion(cfg):
    """Carga el modelo por su alias champion y devuelve su versión y run de entrenamiento."""
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre}, versión {version.version}")
    return modelo, str(version.version), version.run_id


@task
def ingestar_lote(cfg):
    """Simula clientes nuevos con una muestra del CSV, sin incluir la etiqueta real."""
    df = leer_datos(resolver_ruta(cfg["datos"]["ruta"])).drop(columns=["Exited"])
    tamano = cfg["inferencia"]["tamano_lote"]
    if tamano < 1:
        raise ValueError("inferencia.tamano_lote debe ser mayor que cero")
    if tamano > len(df):
        raise ValueError(
            f"inferencia.tamano_lote ({tamano}) supera las filas disponibles ({len(df)})"
        )
    return df.sample(n=tamano)


@task
def predecir_y_guardar(modelo, lote, cfg):
    """Calcula predicciones y las guarda en un CSV fechado."""
    variables = list(modelo.feature_names_in_)
    proba = modelo.predict_proba(lote[variables])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_churn"] = proba
    resultado["prediccion"] = (proba >= cfg["metricas"]["umbral"]).astype(int)

    carpeta = resolver_ruta(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta), len(proba), float(proba.mean())


@task
def registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media):
    """Registra el run y referencia del modelo champion usado en MLflow."""
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("probabilidad_media", proba_media)
        mlflow.log_artifact(ruta)


@flow(name="inferencia-churn", log_prints=True)
def inferencia_flow():
    cfg = cargar_configuracion()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingestar_lote(cfg)
    ruta, filas, proba_media = predecir_y_guardar(modelo, lote, cfg)
    registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media)


if __name__ == "__main__":
    inferencia_flow()
