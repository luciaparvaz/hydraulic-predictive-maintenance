"""Carga del dataset 'Condition Monitoring of Hydraulic Systems' (UCI ID 447).

Nota importante: el prompt original de este proyecto asume que `ucimlrepo.fetch_ucirepo(id=447)`
funciona, siguiendo el patrón estándar de UCI. En la práctica, esta llamada falla con
`DatasetNotFoundError` — el dataset existe en el repositorio pero no está habilitado para
import programático vía `ucimlrepo` (comprobado en este proyecto el 2026-08-20). Este loader
usa exclusivamente la descarga directa del ZIP estático que sí es público (no requiere
autenticación). Nota de auditoría: una versión anterior de este archivo incluía una función
`_try_ucimlrepo()` que pretendía ser un "fallback automático" a ucimlrepo, pero nunca estaba
conectada al pipeline real de carga (código muerto que contradecía su propia documentación) y,
además, no había forma de verificar qué forma de datos devolvería `ucimlrepo` si algún día
UCI habilita este dataset — parsearla a ciegas sin poder probarlo habría sido peor que no
tener fallback. Se eliminó esa función; si UCI habilita el dataset en el futuro, este loader
deberá actualizarse entonces, con la respuesta real delante para poder probarlo.

Limitación conocida: `profile.txt` tiene 5 columnas — las 4 usadas como targets de
clasificación (`cooler_condition`, `valve_condition`, `pump_leakage`,
`accumulator_pressure`) más `stable_flag`. Este loader carga las 5, pero `stable_flag`
no se usa como target en ninguna fase del proyecto — se emplea únicamente como filtro en
`02_feature_engineering.ipynb` (se descartan los ciclos con `stable_flag=1` antes de
construir `X_train`/`X_test`) y la columna se elimina antes de exportar `y_train.parquet`/
`y_test.parquet`. Ningún artefacto en `data/processed/` conserva `stable_flag`.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXTRACT_DIR = RAW_DIR / "hydraulic_raw"
ZIP_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"

# Sensor -> frecuencia de muestreo (Hz). El número de lecturas por ciclo de 60s es freq * 60.
SENSOR_FREQUENCIES_HZ = {
    "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1,
    "VS1": 1, "CE": 1, "CP": 1, "SE": 1,
}

TARGET_COLUMNS = [
    "cooler_condition",       # 3 / 20 / 100
    "valve_condition",        # 73 / 80 / 90 / 100
    "pump_leakage",           # 0 / 1 / 2
    "accumulator_pressure",   # 90 / 100 / 115 / 130
    "stable_flag",            # 0 / 1
]


def _extraction_is_complete(extract_dir: Path) -> bool:
    """True solo si están los 17 archivos de sensor + profile.txt, no solo si la carpeta no está vacía.

    Una extracción interrumpida (proceso matado a mitad del `zf.extractall`) deja el directorio
    existente y no vacío, pero con archivos faltantes — sin esta comprobación, download_hydraulic_dataset()
    la daría por válida y el fallo real (FileNotFoundError en np.loadtxt, dentro de load_sensors())
    aparecería más tarde con un mensaje que no apunta a la causa (extracción incompleta).
    """
    expected = {f"{name}.txt" for name in SENSOR_FREQUENCIES_HZ} | {"profile.txt"}
    present = {p.name for p in extract_dir.glob("*.txt")}
    return expected.issubset(present)


def download_hydraulic_dataset(force: bool = False) -> Path:
    """Descarga y extrae el ZIP de UCI a data/raw/hydraulic_raw/. Devuelve esa ruta."""
    if EXTRACT_DIR.exists() and _extraction_is_complete(EXTRACT_DIR) and not force:
        return EXTRACT_DIR

    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "hydraulic.zip"
    if not zip_path.exists() or force:
        urllib.request.urlretrieve(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXTRACT_DIR)

    if not _extraction_is_complete(EXTRACT_DIR):
        raise RuntimeError(
            f"Extracción incompleta en {EXTRACT_DIR}: faltan uno o más de los 17 archivos de sensor "
            "o profile.txt tras extraer el ZIP. Vuelve a intentar con force=True."
        )

    return EXTRACT_DIR


def load_sensors(force_download: bool = False) -> dict[str, np.ndarray]:
    """Carga cada sensor como un array (2205 ciclos x N lecturas). N depende de la frecuencia
    del sensor (6000 para 100Hz, 600 para 10Hz, 60 para 1Hz)."""
    extract_dir = download_hydraulic_dataset(force=force_download)
    sensors = {}
    for name in SENSOR_FREQUENCIES_HZ:
        path = extract_dir / f"{name}.txt"
        sensors[name] = np.loadtxt(path, delimiter="\t")
    return sensors


def load_targets(force_download: bool = False) -> pd.DataFrame:
    """Carga profile.txt como DataFrame con las 5 columnas objetivo nombradas."""
    extract_dir = download_hydraulic_dataset(force=force_download)
    y = pd.read_csv(extract_dir / "profile.txt", sep="\t", header=None)
    y.columns = TARGET_COLUMNS
    return y


def load_hydraulic_dataset(force_download: bool = False) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Devuelve (sensors, y): dict de arrays por sensor + DataFrame de targets, ambos indexados
    implícitamente por número de ciclo (fila 0..2204)."""
    sensors = load_sensors(force_download=force_download)
    y = load_targets(force_download=force_download)
    return sensors, y


if __name__ == "__main__":
    try:
        from ucimlrepo import fetch_ucirepo

        fetch_ucirepo(id=447)
        print("AVISO: ucimlrepo funcionó (inesperado) — UCI debe haber habilitado el dataset 447.")
        print("Este loader sigue usando descarga directa; considera actualizarlo para usar ucimlrepo.")
    except Exception as e:
        print(f"ucimlrepo falló como se esperaba ({type(e).__name__}). Usando descarga directa.")

    sensors, y = load_hydraulic_dataset()
    print(f"Sensores cargados: {list(sensors.keys())}")
    for name, arr in sensors.items():
        print(f"  {name}: shape={arr.shape}")
    print(f"\nTargets shape: {y.shape}")
    print(y.head())
