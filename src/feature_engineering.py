"""Extracción y selección de features escalares para el dataset UCI Hydraulic System (ID 447).

Refactorización de la lógica ya validada en `notebooks/02_feature_engineering.ipynb`
(ejecutado con 0 errores). No se ha alterado ninguna fórmula ni criterio de selección
respecto al notebook — este módulo solo reorganiza ese mismo código en funciones puras
e importables.

Nota de fidelidad al notebook: los sensores del dataset se muestrean a frecuencias
distintas (100 Hz para PS1-PS6/EPS1, 10 Hz para FS1-FS2, 1 Hz para el resto), por lo
que no existe un único array (timesteps × sensores) por ciclo — cada sensor tiene su
propio número de lecturas por ciclo. El notebook opera, en su lugar, sobre un
`dict[str, np.ndarray]` con un array (n_ciclos, n_lecturas_del_sensor) por sensor —
la misma estructura que devuelve `data_loader.load_hydraulic_dataset()` — y de forma
vectorizada sobre todos los ciclos a la vez, no ciclo a ciclo. Este módulo mantiene
esa misma estructura de entrada por ser la que el notebook ejecutó y validó; forzar
una única matriz (timesteps × sensores) por ciclo habría exigido inventar una
representación que el notebook nunca usa.

Bloques:
    1. Extracción de features candidatas (`compute_level_features`,
       `compute_ps3_peak_seconds`, `compute_differential_features`,
       `extract_candidate_features`).
    2. Cálculo de eta cuadrado (`eta_squared`, `compute_eta2_matrix`).
    3. Pipeline completo de construcción del dataset (`build_feature_matrix`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ── Constantes del módulo ────────────────────────────────────────────────────────

# Los 17 sensores del dataset, en el mismo orden que SENSOR_FREQUENCIES_HZ en
# src/data_loader.py (100 Hz: PS1-PS6, EPS1 · 10 Hz: FS1, FS2 · 1 Hz: TS1-TS4, VS1, CE, CP, SE).
SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
]

# Los 4 targets de componente (sin stable_flag, que se usa como filtro en el notebook,
# no como target de clasificación) — mismo nombre de variable que en el notebook.
COMPONENT_TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]

# Los 5 diferenciales de presión/temperatura definidos explícitamente en la Sección 3
# del notebook (celda 38c8917b) — solo estos pares, no todas las combinaciones posibles
# de sensores. Cada uno resta dos columnas "{sensor}_mean" ya calculadas en el Bloque 1.
DIFFERENTIAL_DEFINITIONS: dict[str, tuple[str, str]] = {
    "PS1m_minus_PS2m": ("PS1_mean", "PS2_mean"),
    "PS2m_minus_PS3m": ("PS2_mean", "PS3_mean"),
    "PS3m_minus_PS5m": ("PS3_mean", "PS5_mean"),
    "PS4m_minus_PS5m": ("PS4_mean", "PS5_mean"),
    "TS3m_minus_TS1m": ("TS3_mean", "TS1_mean"),
}

# Las 73 features seleccionadas, leídas directamente de data/processed/X_train.parquet
# (`pd.read_parquet(...).columns.tolist()`) — no se recalcula el criterio de selección
# (eta^2 < 0.01 en los 4 targets simultáneamente, más la exclusión explícita de
# TS3m_minus_TS1m por redundancia con TS3_mean/TS1_mean) en este módulo, se hardcodea
# el resultado ya validado.
SELECTED_FEATURES: list[str] = [
    "PS1_mean", "PS1_std", "PS1_mean_first_half", "PS1_mean_second_half",
    "PS2_mean", "PS2_std", "PS2_mean_first_half", "PS2_mean_second_half",
    "PS3_mean", "PS3_std", "PS3_mean_first_half", "PS3_mean_second_half",
    "PS4_mean", "PS4_std", "PS4_mean_first_half", "PS4_mean_second_half",
    "PS5_mean", "PS5_std", "PS5_mean_first_half", "PS5_mean_second_half",
    "PS6_mean", "PS6_std", "PS6_mean_first_half", "PS6_mean_second_half",
    "EPS1_mean", "EPS1_std", "EPS1_mean_first_half", "EPS1_mean_second_half",
    "FS1_mean", "FS1_std", "FS1_mean_first_half", "FS1_mean_second_half",
    "FS2_mean", "FS2_std", "FS2_mean_first_half", "FS2_mean_second_half",
    "TS1_mean", "TS1_std", "TS1_mean_first_half", "TS1_mean_second_half",
    "TS2_mean", "TS2_std", "TS2_mean_first_half", "TS2_mean_second_half",
    "TS3_mean", "TS3_std", "TS3_mean_first_half", "TS3_mean_second_half",
    "TS4_mean", "TS4_std", "TS4_mean_first_half", "TS4_mean_second_half",
    "VS1_mean", "VS1_std", "VS1_mean_first_half", "VS1_mean_second_half",
    "CE_mean", "CE_std", "CE_mean_first_half", "CE_mean_second_half",
    "CP_mean", "CP_std", "CP_mean_first_half", "CP_mean_second_half",
    "SE_mean", "SE_std", "SE_mean_first_half", "SE_mean_second_half",
    "PS3_peak_s",
    "PS1m_minus_PS2m", "PS2m_minus_PS3m", "PS3m_minus_PS5m", "PS4m_minus_PS5m",
]


# ── Bloque 1 — Extracción de features candidatas ────────────────────────────────


def compute_level_features(sensors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Media, desviación estándar y medias por mitad temporal, por sensor y por ciclo.

    Para cada sensor calcula 4 features escalares por ciclo: la media global del ciclo
    (nivel medio de la señal), la desviación estándar intra-ciclo (dispersión de las
    lecturas dentro de un mismo ciclo, no entre ciclos), y las medias de la primera y
    segunda mitad temporal del ciclo (para detectar asimetrías o tendencias dentro del
    ciclo que la media global, al promediarlas, ocultaría). Estas cuatro estadísticas
    son la forma más simple de resumir cada serie temporal sin asumir nada sobre su
    forma — un punto de partida neutro antes de features más específicas (fase,
    diferenciales).

    El cálculo está vectorizado sobre el eje de lecturas (axis=1): `arr.mean(axis=1)`
    calcula la media de cada fila (ciclo) para las N filas a la vez, equivalente a un
    bucle en Python ciclo a ciclo pero sin su coste.

    Args:
        sensors: dict sensor -> array de forma (n_ciclos, n_lecturas_del_sensor). El
            número de lecturas depende de la frecuencia de muestreo del sensor (p.ej.
            6000 para un sensor a 100 Hz en un ciclo de 60s, 60 para uno a 1 Hz).

    Returns:
        DataFrame con una fila por ciclo y 4 columnas por sensor:
        `{sensor}_mean`, `{sensor}_std`, `{sensor}_mean_first_half`,
        `{sensor}_mean_second_half`.
    """
    features = {}
    for name, arr in sensors.items():
        half = arr.shape[1] // 2
        features[f"{name}_mean"] = arr.mean(axis=1)
        features[f"{name}_std"] = arr.std(axis=1)
        features[f"{name}_mean_first_half"] = arr[:, :half].mean(axis=1)
        features[f"{name}_mean_second_half"] = arr[:, half:].mean(axis=1)
    return pd.DataFrame(features)


def compute_ps3_peak_seconds(ps3_array: np.ndarray) -> np.ndarray:
    """Posición temporal (en segundos) del pico de presión PS3, tras suavizado.

    PS3 mide la presión en el lado activo del cilindro; el instante en que alcanza su
    máximo dentro del ciclo se desplaza según el estado de la válvula (`valve_condition`),
    incluso cuando el nivel máximo de presión no cambia mucho — es la fase de la señal,
    no su amplitud, la que lleva la señal discriminativa (documentado en la Sección 3
    del EDA). Por eso esta feature no forma parte de `compute_level_features`: no es
    una estadística de nivel, es una posición.

    Se suaviza la señal con una media móvil de 50 muestras (0.5s a 100 Hz) antes de
    tomar el argmax, para no quedarse con el índice de un pico de ruido de una sola
    muestra en vez del pico real de la forma de onda.

    Args:
        ps3_array: array de forma (n_ciclos, n_lecturas), lecturas de PS3 a 100 Hz.

    Returns:
        Array 1D de longitud n_ciclos: posición del pico suavizado, en segundos
        (índice de muestra / 100.0).
    """
    smoothed = uniform_filter1d(ps3_array, size=50, axis=1)
    return smoothed.argmax(axis=1) / 100.0


def compute_differential_features(level_features: pd.DataFrame) -> pd.DataFrame:
    """Diferenciales entre pares de sensores de presión/temperatura definidos a mano.

    Cada diferencial resta las medias de nivel (`{sensor}_mean`, ya calculadas por
    `compute_level_features`) de dos sensores relacionados físicamente por su posición
    en el circuito hidráulico (p.ej. `PS4m_minus_PS5m` aproxima la caída de presión
    entre el lado pasivo del cilindro y la línea de retorno). La hipótesis es que una
    caída de presión anómala entre dos puntos del circuito es más informativa de una
    fuga o degradación que el nivel absoluto en cualquiera de los dos puntos por
    separado. Solo se calculan los 5 pares definidos explícitamente en
    `DIFFERENTIAL_DEFINITIONS` — no todas las combinaciones posibles de sensores.

    Args:
        level_features: DataFrame devuelto por `compute_level_features` (debe
            contener, como mínimo, las columnas `{sensor}_mean` referenciadas en
            `DIFFERENTIAL_DEFINITIONS`).

    Returns:
        DataFrame con una fila por ciclo y una columna por diferencial definido en
        `DIFFERENTIAL_DEFINITIONS` (5 columnas).
    """
    return pd.DataFrame(
        {name: level_features[a] - level_features[b] for name, (a, b) in DIFFERENTIAL_DEFINITIONS.items()}
    )


def extract_candidate_features(sensors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Ensambla todas las features candidatas: nivel + fase + diferenciales.

    Concatena las 68 features de nivel (17 sensores x 4 estadísticas,
    `compute_level_features`), la feature de fase (`PS3_peak_s`,
    `compute_ps3_peak_seconds`) y los 5 diferenciales (`compute_differential_features`)
    en una única tabla de 74 features candidatas — el universo completo de features
    sobre el que actúa la selección por eta^2 del Bloque 3. 74, no 73: incluye
    `TS3m_minus_TS1m`, que el notebook excluye explícitamente por redundancia con
    `TS3_mean`/`TS1_mean` (Sección 3), no porque no se calcule.

    Args:
        sensors: dict sensor -> array (n_ciclos, n_lecturas_del_sensor), mismo
            formato que `compute_level_features`. Debe incluir la clave "PS3".

    Returns:
        DataFrame con una fila por ciclo y 74 columnas (68 de nivel + 1 de fase +
        5 diferenciales).
    """
    level = compute_level_features(sensors)
    phase = pd.DataFrame({"PS3_peak_s": compute_ps3_peak_seconds(sensors["PS3"])})
    diffs = compute_differential_features(level)
    return pd.concat([level, phase, diffs], axis=1)


# ── Bloque 2 — Cálculo de eta cuadrado ───────────────────────────────────────────


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Eta cuadrado (tamaño del efecto de un ANOVA de un factor) entre una feature y un target.

    eta^2 = SS_entre / SS_total, donde SS_total es la suma de cuadrados de todos los
    valores respecto a la media global (varianza total de la feature) y SS_entre es la
    suma, ponderada por tamaño de grupo, de las desviaciones al cuadrado de cada media
    de grupo respecto a la media global (cuánta de esa varianza total se explica por
    saber a qué grupo/nivel del target pertenece cada observación). Es una medida de
    tamaño del efecto univariante y simétrica en escala: eta^2=0 indica que las medias
    de todos los grupos son iguales (el target no mueve la media de la feature en
    absoluto); eta^2=1 indica que el target determina la feature por completo (varianza
    intra-grupo nula). No implica causalidad ni captura relaciones no lineales o de
    varianza (dos grupos con la misma media pero distinta dispersión dan eta^2 bajo
    aunque sean distinguibles).

    Args:
        values: Serie de valores numéricos de una feature, una fila por ciclo.
        groups: Serie categórica (incluye enteros con niveles discretos, como los
            targets de este dataset) del mismo largo que `values` — el grupo/nivel de
            cada ciclo.

    Returns:
        eta^2 en [0, 1]. Devuelve 0.0 si SS_total es 0 (feature constante en toda la
        muestra): sin varianza que explicar, no hay asociación que medir, y 0/0 no
        debe propagarse como NaN.
    """
    frame = pd.DataFrame({"v": values, "g": groups})
    grand_mean = frame["v"].mean()
    ss_total = ((frame["v"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = frame.groupby("g")["v"].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    return ss_between / ss_total


def compute_eta2_matrix(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_columns: list[str] = COMPONENT_TARGETS,
) -> pd.DataFrame:
    """Matriz de eta^2 de cada feature frente a cada target.

    Aplica `eta_squared` a cada combinación (feature, target) — es la tabla que el
    notebook usa (Sección 4) para decidir qué features conservar: una feature se
    excluye si su eta^2 es menor que 0.01 en los 4 targets simultáneamente (ningún
    target la explica ni siquiera débilmente). Esta función reproduce el cálculo de
    esa tabla; no reproduce el criterio de exclusión en sí (eso vive en la constante
    `SELECTED_FEATURES`, ya validada).

    Args:
        features: DataFrame de features candidatas, una fila por ciclo (p.ej. la
            salida de `extract_candidate_features`).
        targets: DataFrame de targets, mismo número de filas y mismo orden que
            `features`. Debe contener las columnas listadas en `target_columns`.
        target_columns: nombres de las columnas de `targets` a evaluar. Por defecto,
            los 4 targets de componente (`COMPONENT_TARGETS`).

    Returns:
        DataFrame con una fila por feature (mismo índice que `features.columns`) y
        una columna por target en `target_columns`.
    """
    return pd.DataFrame(
        {target: [eta_squared(features[feature], targets[target]) for feature in features.columns] for target in target_columns},
        index=features.columns,
    )


# ── Bloque 3 — Pipeline completo de construcción del dataset ───────────────────


def build_feature_matrix(
    sensors: dict[str, np.ndarray],
    y: pd.DataFrame,
    selected_features: list[str] = SELECTED_FEATURES,
) -> pd.DataFrame:
    """Pipeline completo: extracción + eta^2 + selección de las features validadas.

    Aplica `extract_candidate_features` (Bloque 1) a todos los ciclos de `sensors`,
    calcula la matriz de eta^2 de esas features candidatas frente a `y` (Bloque 2,
    `compute_eta2_matrix` — se calcula por completitud del pipeline, pero la selección
    de columnas no depende de recalcular el umbral: usa `selected_features`, la lista
    ya validada en el notebook) y devuelve únicamente esas columnas seleccionadas.

    Esta función no filtra ciclos (outliers, `stable_flag`) ni realiza el split
    train/test — esa es una decisión de preprocesamiento de nivel de ciclo (Sección 0
    del notebook), independiente de la extracción de features, y no forma parte de
    este módulo. Se puede llamar tanto sobre el dataset crudo completo (2205 ciclos)
    como sobre cualquier subconjunto ya filtrado (p.ej. solo el split de train).

    Args:
        sensors: dict sensor -> array (n_ciclos, n_lecturas_del_sensor), formato
            devuelto por `data_loader.load_hydraulic_dataset()`.
        y: DataFrame de targets, mismo número de filas y mismo orden que los arrays
            de `sensors`. Debe contener, como mínimo, las columnas de
            `COMPONENT_TARGETS` para el cálculo de eta^2.
        selected_features: columnas finales a devolver. Por defecto, `SELECTED_FEATURES`
            (las 73 features validadas en el notebook).

    Returns:
        DataFrame con una fila por ciclo y una columna por feature en
        `selected_features` (73 por defecto).
    """
    candidate_features = extract_candidate_features(sensors)
    compute_eta2_matrix(candidate_features, y)  # calculado por completitud del pipeline; no determina la selección
    return candidate_features[selected_features]
