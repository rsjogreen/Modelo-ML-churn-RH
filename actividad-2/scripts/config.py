"""Carga compartida de la configuración de la actividad 2."""
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "configs"


def cargar_configuracion():
    with (CONFIG_DIR / "config.yaml").open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def cargar_configuracion_features():
    with (CONFIG_DIR / "features.yaml").open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def resolver_ruta(ruta):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else (BASE_DIR / ruta).resolve()
