# Actividad 2: pipelines de entrenamiento e inferencia, y API

Proyecto de la actividad 2: pipeline de entrenamiento y pipeline de inferencia en Prefect, conectados por el alias `champion` en MLflow, y una API de FastAPI que sirve el modelo.

MLflow y Prefect se levantan con el `docker-compose.yml` de la raíz del repositorio.

## Estructura

```
actividad-2/
├── configs/
│   ├── config.yaml            # datos, división, candidatos, MLflow, registro e inferencia
│   └── features.yaml          # selección de variables del experimento
├── domain/
│   └── modelo.py              # lógica del modelo: variables, pipeline y métricas (sin Prefect ni MLflow)
├── scripts/
│   ├── config.py              # carga de YAML y resolución de rutas
│   └── load_data.py           # carga del CSV con validación de columnas
├── infraestructure/
│   ├── flows/
│   │   ├── entrenamiento_flow.py   # pipeline de entrenamiento
│   │   ├── inferencia_flow.py      # pipeline de inferencia (se ejecuta cada día)
│   │   └── desplegar.py            # registra los despliegues en Prefect
│   └── api/
│       └── main.py            # API de FastAPI que sirve el modelo champion
├── poetry.lock                # versiones exactas resueltas (no editar a mano)
└── pyproject.toml             # dependencias declaradas con versión fijada
```

El CSV de entrada está en `../data/Customer-Churn-Records.csv` relativo a esta carpeta. Las predicciones diarias se guardan en `../data/`.

## Requisitos

- Python entre 3.12 y 3.14.
- Poetry 2.x.
- MLflow y Prefect levantados con Docker Compose desde la raíz del repositorio.

## Entorno

Desde `actividad-2`:

```powershell
poetry install
```

## Variables de entorno

Define estas variables en cada terminal donde ejecutes un flow o la API:

```powershell
$env:PREFECT_API_URL = "http://localhost:4201/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
$env:PYTHONUTF8 = "1"
```

`MLFLOW_TRACKING_URI` tiene prioridad sobre `mlflow.tracking_uri` de `configs/config.yaml`.
Los puertos locales `4201` y `5001` corresponden al mapeo de puertos predeterminado del `docker-compose.yml` de este repositorio.

`PYTHONUTF8` evita un error de codificación en Windows al cerrar un run de MLflow (ver solución de problemas). Es necesario en la terminal donde ejecutes `desplegar.py`.

**Ejecuta los comandos de ejemplo desde `actividad-2`.** Las rutas YAML también se resuelven desde la ubicación del proyecto, por lo que los flows no dependen del directorio actual.

Levanta primero los servicios desde la raíz del repositorio:

```powershell
docker compose up --build -d mlflow prefect-server
```

## 1. Entrenamiento

Entrena el modelo, lo registra en MLflow y, si supera el umbral mínimo y el AUC del champion actual, le asigna el alias `champion`:

```powershell
poetry run python -m infraestructure.flows.entrenamiento_flow
```

El flow realiza ingesta → entrenamiento → evaluación → registro. Cada candidato crea un run con el tag `fase=entrenamiento`; todos se evalúan con el mismo holdout. Por ejemplo:

- `Versión 1 registrada como champion (AUC 0.9992)`: el modelo pasó a ser champion.
- `El candidato no cumple los criterios...`: el run queda registrado, pero no se crea una nueva versión ni cambia el alias. Los logs de evaluación muestran su AUC, el mínimo y el AUC del champion actual.

El flow carga los datos una vez, entrena cada candidato y elige el de mayor AUC. Solo registra una versión y mueve el alias `champion` si el candidato supera estrictamente `registro.umbral_auc` y el AUC del champion vigente.

**Dónde verlo:**

- En MLflow (http://localhost:5001), experimento `churn-training-mlops`, un run por candidato. Cada run tiene el tag `fase = entrenamiento`, parámetros incluyendo `features` (`ALL`, `NUMERIC` o `CUSTOM`) y `n_features`, las métricas (`auc`, `accuracy`, `precision`, `recall`, `f1`) y el modelo como artefacto.
- En MLflow, en el Model Registry, el modelo `churn-modelo` tiene el alias `champion` y el tag `estado = aprobado` en su versión.
- En Prefect (http://localhost:4201/runs), el flow `entrenamiento-churn`.

### Cambiar las variables del experimento

Edita `configs/features.yaml` para elegir las variables:

- `seleccion: ALL`: todas las variables, numéricas y categóricas.
- `seleccion: NUMERIC`: solo variables numéricas.
- `seleccion: CUSTOM`: solo las columnas listadas en `custom`. Sus nombres deben coincidir exactamente con las columnas del CSV.

Cada run registra el valor de `seleccion` en `features` y la cantidad seleccionada en `n_features`.

### Cambiar los modelos o sus hiperparámetros

Edita la lista `candidatos` en `configs/config.yaml`. Cada candidato tiene `nombre_run`, `tipo` (`logistic_regression` o `gradient_boosting`) y `parametros`. El umbral mínimo para ser champion está en `registro.umbral_auc`, y el umbral de clasificación (probabilidad a partir de la que se predice churn) en `metricas.umbral`.

### Cuántos runs se crean

Cada ejecución crea un run por candidato (`len(candidatos)`).

## 2. Inferencia

Carga el modelo `champion`, puntúa un lote de clientes, guarda las predicciones y registra un run en MLflow:

```powershell
poetry run python -m infraestructure.flows.inferencia_flow
```

Necesitas un champion registrado: ejecuta antes el entrenamiento.

- **Lote:** `inferencia.tamano_lote` clientes tomados al azar del CSV. Como simulación de clientes nuevos, el lote cambia en cada ejecución.
- **Predicciones:** se guardan en `../data/prediccion_<fecha>.csv`. Cada fila contiene las variables del cliente, `probabilidad_churn` y `prediccion` (1 si predice abandono), sin copiar la etiqueta real `Exited`.
- **Run en MLflow:** experimento `churn-training-mlops`, run `inferencia-<fecha>`. Tiene las tags `fase = inferencia`, `modelo_version` (versión del champion usada) y `modelo_run_id` (run de entrenamiento de esa versión); las métricas `filas_predichas` y `probabilidad_media`; y el CSV de predicciones como artefacto.
- **Columnas del modelo:** la inferencia usa las columnas guardadas en el champion; no depende de cambios posteriores en `features.yaml`.

## 3. Despliegues en Prefect

Registra dos despliegues y mantén la terminal abierta:

```powershell
poetry run python -m infraestructure.flows.desplegar
```

- `entrenamiento-churn/entrenamiento-churn`: sin horario. Se lanza a mano.
- `inferencia-churn/inferencia-diaria`: se ejecuta según `inferencia.cron` de `configs/config.yaml` (por defecto, cada día a las 06:00 UTC).

Mientras la terminal esté abierta, Prefect puede lanzar las ejecuciones. Si la cierras, las ejecuciones programadas no se lanzan.

Para lanzar un despliegue a mano, desde otra terminal con las variables definidas:

```powershell
poetry run prefect deployment run "entrenamiento-churn/entrenamiento-churn"
poetry run prefect deployment run "inferencia-churn/inferencia-diaria"
```

También puedes lanzarlos desde la UI de Prefect, en http://localhost:4201/deployments.

## 4. API de inferencia

La API carga el champion al arrancar. Si el entrenamiento asigna un champion nuevo, reinicia la API.

```powershell
poetry run uvicorn infraestructure.api.main:app --port 8000
```

- Documentación interactiva: http://localhost:8000/docs
- Estado y versión del modelo: http://localhost:8000/health
- Predicción para un cliente: `POST http://localhost:8000/predict`

Ejemplo de predicción desde PowerShell, en otra terminal:

```powershell
$cliente = @{
    CreditScore = 619; Geography = "France"; Gender = "Female"; Age = 42
    Tenure = 2; Balance = 0.0; NumOfProducts = 1; HasCrCard = 1
    IsActiveMember = 1; EstimatedSalary = 101348.88; Complain = 1
    "Satisfaction Score" = 2; "Card Type" = "DIAMOND"; "Point Earned" = 464
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $cliente
```

La documentación OpenAPI en `/docs` incluye un ejemplo completo del cliente. La respuesta incluye `probabilidad_churn`, `prediccion` (1 si el cliente abandona, 0 si no) y `version_modelo`.

Respuesta de ejemplo (la probabilidad y versión dependen del champion activo):

```json
{
  "probabilidad_churn": 0.9943,
  "prediccion": 1,
  "version_modelo": "1"
}
```

La API recibe todas las columnas del cliente aunque el modelo use solo algunas; usa únicamente las que necesita el champion. El umbral de clasificación se lee de `metricas.umbral`.

## Orden recomendado

1. Levantar MLflow y Prefect con Docker Compose desde la raíz.
2. Entrenar: `python -m infraestructure.flows.entrenamiento_flow`.
3. Inferir: `python -m infraestructure.flows.inferencia_flow`.
4. Desplegar en Prefect: `python -m infraestructure.flows.desplegar`.
5. Levantar la API: `infraestructure.api.main:app`.

## Solución de problemas

- **`Registered Model with name=churn-modelo not found`:** no hay champion todavía. Ejecuta el entrenamiento y revisa que el AUC supere `registro.umbral_auc`. Cambia la selección en `configs/features.yaml` si hace falta.
- **`UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3c3'`** al terminar un run desde Prefect: la consola usa `cp1252`. Define `$env:PYTHONUTF8 = "1"` en la terminal donde ejecutas `desplegar.py` y vuelve a lanzarlo. Si el run ha quedado en `RUNNING` en MLflow, márcalo como fallido desde la UI.
- **`ModuleNotFoundError: No module named 'scripts'`:** ejecuta `poetry install` de nuevo desde `actividad-2`, para que los paquetes `scripts`, `domain` e `infraestructure` queden instalados.
- **La API no encuentra el modelo al arrancar:** entrena primero y revisa que `MLFLOW_TRACKING_URI` apunte al servidor correcto.
- **`httpx.InvalidURL: Invalid non-printable ASCII character in URL, '\n'`:** la variable `PREFECT_API_URL` tiene un salto de línea al final, probablemente por haberla copiado con un salto. Comprueba su longitud con `$env:PREFECT_API_URL.Length`: debe ser `25`. Si no lo es, vuelve a definirla en una terminal nueva con `$env:PREFECT_API_URL = "http://localhost:4201/api"` y revisa que `$env:MLFLOW_TRACKING_URI.Length` sea `21`.
- **La UI de Prefect no muestra despliegues:** ejecuta `desplegar.py` con `PREFECT_API_URL` definida y deja la terminal abierta.
