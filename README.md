# Modelo de prediccion de churn de empleados

Proyecto educativo para explorar la prediccion de rotacion de empleados con datos completamente sinteticos. El proyecto no utiliza informacion personal ni registros reales.

## Estructura

```text
.
|-- churn_ml/
|   |-- data.py          # Generacion de datos sinteticos
|   |-- model.py         # Entrenamiento y metricas
|   `-- train.py         # Flujo de Prefect y registro en MLflow
|-- actividad-2/notebooks/eda_churn.ipynb  # Analisis exploratorio de la data sintetica
|-- tests/               # Pruebas del baseline
|-- .env.example         # Puertos de los servicios Docker
|-- docker-compose.yml   # MLflow, Prefect y ejecucion del entrenamiento
|-- Dockerfile
`-- requirements.txt    # Dependencias directas con versiones fijadas
```

## Requisitos

- Python 3.12
- Docker Desktop con Docker Compose

## Ejecutar localmente

En PowerShell, desde la raiz del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
$env:PREFECT_HOME = "$PWD\.prefect"
python -m churn_ml.train
```

La ejecucion local almacena los experimentos en `mlruns/` si no se configura `MLFLOW_TRACKING_URI`. `PREFECT_HOME` mantiene la base local de Prefect aislada dentro del proyecto. El entorno `.venv` es local y esta excluido de Git; se reconstruye usando `requirements.txt`.

## Analisis exploratorio de datos sinteticos

Abre [actividad-2/notebooks/eda_churn.ipynb](actividad-2/notebooks/eda_churn.ipynb) en VS Code, selecciona el entorno de `actividad-2` y ejecuta todas las celdas. El notebook genera los registros con `churn_ml.data.generate_employee_data` y los guarda/lee desde `data/churn_empleados_sintetico.csv`; no necesita leer `data/Customer-Churn-Records.csv`. Ajusta `N_SAMPLES` y `RANDOM_STATE` en la celda de configuracion para cambiar el tamaño de la muestra o reproducirla.

El notebook revisa calidad, distribuciones, balance de clases, tasas de churn por grupo y correlaciones. Como los datos y la etiqueta son simulados, los resultados solo sirven para practicar el análisis y no representan relaciones reales sobre empleados.

## Ejecutar con Docker, Prefect y MLflow

```powershell
Copy-Item .env.example .env
docker compose up --build -d mlflow prefect-server
docker compose run --rm trainer
```

- MLflow: <http://localhost:5001>
- Prefect: <http://localhost:4201>

El entrenamiento se ejecuta en el contenedor `trainer`, registra sus metricas y el modelo en MLflow, y Prefect registra la ejecucion del flujo. Los datos de MLflow persisten en el volumen Docker `mlflow_data`.

La implementación completa de la actividad 2 —flows de Prefect, selección de variables YAML, MLflow Model Registry y API FastAPI— está documentada en [actividad-2/README.md](actividad-2/README.md).

Para apagar los servicios sin borrar los experimentos:

```powershell
docker compose down
```

Para eliminar tambien el volumen de MLflow y todos sus experimentos:

```powershell
docker compose down --volumes
```

## Alcance del baseline

El modelo inicial usa regresion logistica y una division entrenamiento/prueba estratificada. Reporta accuracy, precision, recall, F1 y ROC AUC. Los datos sinteticos solo verifican el funcionamiento del pipeline: sus metricas no representan resultados reales ni deben usarse para decisiones laborales.

## Siguientes pasos sugeridos

1. Definir la variable objetivo, ventana temporal y politica de datos con el negocio.
2. Incorporar un conjunto de datos autorizado y anonimizado con un esquema documentado.
3. Revisar calidad, fuga de informacion, desbalance y sesgos antes de comparar modelos.
4. Agregar validacion cruzada, seguimiento de experimentos y criterios de aceptacion.
5. Documentar privacidad, retencion y uso responsable; evaluar riesgos de decisiones automatizadas sobre empleados.
