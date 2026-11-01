"""Pipeline Prefect de ingesta, entrenamiento, evaluación y registro en MLflow."""
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split

from domain.modelo import OBJETIVO, calcular_metricas, construir_modelo, resolver_features
from scripts.config import cargar_configuracion, cargar_configuracion_features, resolver_ruta
from scripts.load_data import cargar_datos as leer_datos


def _configurar_mlflow(cfg):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])


def _obtener_champion(client, nombre_modelo):
    try:
        return client.get_model_version_by_alias(nombre_modelo, "champion")
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


@task
def ingestar_datos(ruta):
    """Carga el CSV y valida las columnas requeridas."""
    return leer_datos(ruta)


@task
def entrenar_candidato(df, candidato, cfg, seleccion, variables, numericas, categoricas):
    """Entrena un candidato y guarda el modelo y los parámetros en su run de MLflow."""
    _configurar_mlflow(cfg)
    division = cfg["division"]
    train_df, _ = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )
    X_train, y_train = train_df[variables], train_df[OBJETIVO]
    modelo = construir_modelo(
        candidato["tipo"],
        candidato.get("parametros") or {},
        numericas,
        categoricas,
    )
    modelo.fit(X_train, y_train)
    proba_train = modelo.predict_proba(X_train)[:, 1]

    nombre_run = (
        f"{cfg['mlflow']['nombre_run']}-{candidato['nombre_run']}-{seleccion.lower()}"
    )
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params(
            {
                "modelo": candidato["tipo"],
                **(candidato.get("parametros") or {}),
                "semilla": division["random_state"],
                "umbral": cfg["metricas"]["umbral"],
                "features": seleccion,
                "n_features": len(variables),
            }
        )
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_train, proba_train),
            input_example=X_train.head(3),
        )
        return run.info.run_id, info.model_uri


@task
def evaluar_candidato(run_id, model_uri, df, cfg, seleccion, variables):
    """Evalúa en el holdout y decide si supera el umbral y al champion vigente."""
    _configurar_mlflow(cfg)
    division = cfg["division"]
    _, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )
    X_test, y_test = test_df[variables], test_df[OBJETIVO]
    modelo = mlflow.sklearn.load_model(model_uri)
    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, proba, cfg["metricas"]["umbral"])

    client = MlflowClient()
    for nombre, valor in metricas.items():
        client.log_metric(run_id, nombre, valor)

    nombre_modelo = cfg["registro"]["nombre_modelo"]
    champion = _obtener_champion(client, nombre_modelo)
    auc_champion = None
    if champion is not None:
        auc_champion = client.get_run(champion.run_id).data.metrics["auc"]

    supera_umbral = metricas["auc"] > cfg["registro"]["umbral_auc"]
    supera_champion = auc_champion is None or metricas["auc"] > auc_champion
    elegible = supera_umbral and supera_champion
    print(
        f"Evaluación features={seleccion}: AUC={metricas['auc']:.4f}, "
        f"umbral={cfg['registro']['umbral_auc']:.4f}, "
        f"AUC champion={auc_champion if auc_champion is not None else 'inexistente'}"
    )
    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "metricas": metricas,
        "elegible": elegible,
    }


@task
def registrar_si_mejora(evaluacion, cfg):
    """Registra una versión y mueve champion solo si todavía supera al champion vigente."""
    if not evaluacion["elegible"]:
        print("El candidato no cumple los criterios; el alias champion no cambia.")
        return None

    _configurar_mlflow(cfg)
    client = MlflowClient()
    nombre_modelo = cfg["registro"]["nombre_modelo"]
    auc = evaluacion["metricas"]["auc"]
    champion = _obtener_champion(client, nombre_modelo)

    if auc <= cfg["registro"]["umbral_auc"]:
        print("El candidato ya no cumple el AUC mínimo; no se registra.")
        return None
    if champion is not None:
        auc_champion = client.get_run(champion.run_id).data.metrics["auc"]
        if auc <= auc_champion:
            print(f"AUC {auc:.4f} no supera al champion ({auc_champion:.4f}); el alias no cambia.")
            return None

    version = mlflow.register_model(evaluacion["model_uri"], nombre_modelo)
    client.set_registered_model_alias(nombre_modelo, "champion", version.version)
    client.set_model_version_tag(nombre_modelo, version.version, "estado", "aprobado")
    print(f"Versión {version.version} registrada como champion (AUC {auc:.4f}).")
    return version.version


@flow(name="entrenamiento-churn", log_prints=True)
def entrenamiento_flow():
    cfg = cargar_configuracion()
    cfg_features = cargar_configuracion_features()
    seleccion = cfg_features["seleccion"]
    numericas, categoricas = resolver_features(seleccion, cfg_features.get("custom"))
    variables = numericas + categoricas

    df = ingestar_datos(resolver_ruta(cfg["datos"]["ruta"]))
    evaluaciones = []
    for candidato in cfg["candidatos"]:
        run_id, model_uri = entrenar_candidato(
            df, candidato, cfg, seleccion, variables, numericas, categoricas
        )
        evaluaciones.append(
            evaluar_candidato(run_id, model_uri, df, cfg, seleccion, variables)
        )

    mejor = max(evaluaciones, key=lambda resultado: resultado["metricas"]["auc"])
    registrar_si_mejora(mejor, cfg)
    print(
        f"Mejor run: {mejor['run_id']} | "
        f"AUC={mejor['metricas']['auc']:.4f} | champion elegible={mejor['elegible']}"
    )
    return mejor["run_id"]


if __name__ == "__main__":
    entrenamiento_flow()
