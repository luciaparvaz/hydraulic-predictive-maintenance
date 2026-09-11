"""Entrenamiento y evaluación de los modelos de detección de anomalías (Fase 3) y
clasificación supervisada por componente (Fase 4) para el dataset UCI Hydraulic
System (ID 447).

Refactorización de la lógica ya validada en `notebooks/03_anomaly_detection.ipynb`
y `notebooks/04_failure_prediction.ipynb` (ambos ejecutados con 0 errores). No se
ha alterado ningún hiperparámetro, fórmula ni criterio de selección respecto a esos
notebooks — este módulo solo reorganiza ese mismo código en funciones puras e
importables.

Bloques:
    1. Detección de anomalías, Fase 3 (`train_mahalanobis_detector`,
       `train_isolation_forest_detector`).
    2. Entrenamiento del clasificador supervisado, Fase 4 (`train_classifier`).
    3. Evaluación del clasificador, Fase 4 (`evaluate_classifier`,
       `cross_validate_classifier`).
    4. Selección de modelo por target (`select_best_model`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

COMPONENT_TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]


# ── Bloque 1 — Detección de anomalías (Fase 3) ──────────────────────────────────


def train_mahalanobis_detector(X_train_normal: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    """Distancia de Mahalanobis al cuadrado de cada ciclo de test a la referencia normal.

    Por qué LedoitWolf y no covarianza muestral ordinaria: `X_train_normal` son los
    ciclos donde los 4 componentes están simultáneamente en su estado óptimo — en este
    dataset, una única celda del diseño factorial de 144, con solo 8 ciclos en train
    (Sección 0 de la Fase 3). Con n=8 observaciones y p=73 features, la matriz de
    covarianza muestral tiene rango máximo n-1=7: es singular por construcción
    (66 valores propios exactamente en cero), sin importar la calidad de los datos, y
    no es invertible — requisito indispensable para la distancia de Mahalanobis.
    `LedoitWolf` produce una covarianza regularizada S̃ = (1-α)·S + α·(tr(S)/p)·I,
    una mezcla entre la covarianza muestral S y una identidad escalada por la traza
    media, con el coeficiente de shrinkage α∈[0,1] estimado analíticamente para
    minimizar el error cuadrático esperado frente a la covarianza verdadera. S̃ es
    definida positiva e invertible por construcción, incluso con n≪p.

    El escalado (`StandardScaler`) se ajusta exclusivamente sobre `X_train_normal` —
    nunca sobre `X_test` ni sobre el training set completo — y se aplica después a
    ambos: ajustar sobre cualquier otro conjunto introduciría fuga de la distribución
    de ciclos degradados en la referencia de normalidad.

    Args:
        X_train_normal: features de los ciclos de entrenamiento con los 4 componentes
            en su estado óptimo simultáneamente (la referencia de "normalidad").
        X_test: features del conjunto de test sobre el que se calculan las distancias.

    Returns:
        Array 1D de longitud `len(X_test)`: distancia de Mahalanobis AL CUADRADO de
        cada ciclo de test a la referencia normal (`LedoitWolf.mahalanobis()` de
        scikit-learn ya devuelve la distancia al cuadrado, no la distancia simple).
    """
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    lw = LedoitWolf()
    lw.fit(X_normal_scaled)
    return lw.mahalanobis(X_test_scaled)


def train_isolation_forest_detector(X_train: pd.DataFrame, X_test: pd.DataFrame, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Puntuación de anomalía de Isolation Forest para cada ciclo de test.

    A diferencia de Mahalanobis, Isolation Forest no requiere un conjunto de
    referencia "solo normal": aísla cada punto mediante particiones recursivas
    aleatorias, y los puntos distintos del grueso de la población se aíslan en menos
    cortes — no necesita haber visto antes un conjunto limpio. Por eso se entrena
    sobre `X_train` completo (mezcla de ciclos normales y degradados), evitando así
    la misma crisis de tamaño muestral (n=8) que exige regularización en
    `train_mahalanobis_detector`. Tampoco requiere escalado: los cortes son umbrales
    sobre valores observados de cada feature, invariantes a transformaciones
    monótonas como `StandardScaler`.

    Hiperparámetros (Sección 2 de la Fase 3, sin modificar): `n_estimators=200`,
    `contamination="auto"` — el umbral se determina "as in the original paper"
    (Liu, Ting & Zhou, 2008), NO como un percentil fijo del 10% (verificado
    empíricamente: `contamination="auto"` marca ~22% del training set como anómalo,
    frente a ~10% con `contamination=0.1` explícito — no son equivalentes).

    `decision_function()` de scikit-learn devuelve valores negativos para puntos más
    anómalos; se invierte el signo para que, igual que en `train_mahalanobis_detector`,
    "más alto" signifique siempre "más anómalo".

    Args:
        X_train: features completas de entrenamiento (sin escalar, sin filtrar a
            solo ciclos normales).
        X_test: features del conjunto de test sobre el que se calculan las puntuaciones.
        random_state: semilla del bosque aleatorio, para determinismo. 42 en el notebook.

    Returns:
        Array 1D de longitud `len(X_test)`: puntuación de anomalía de cada ciclo de
        test (`-decision_function`, más alto = más anómalo).
    """
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
    iso.fit(X_train)
    return -iso.decision_function(X_test)


# ── Bloque 2 — Entrenamiento del clasificador supervisado (Fase 4) ─────────────


def train_classifier(target: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE):
    """Entrena el clasificador final (ya seleccionado) para un target de componente.

    El tipo de modelo por target viene de `select_best_model` (Bloque 4) — la
    selección basada en el F1-macro de validación cruzada de la Fase 4, Sección 1.
    `valve_condition` usa un `Pipeline(StandardScaler, LogisticRegression)`; los
    otros tres targets usan `RandomForestClassifier`.

    Por qué `Pipeline` y no escalar `X_train` una sola vez antes de entrenar: el
    `Pipeline` encapsula el `StandardScaler` junto con el clasificador para que,
    cuando este objeto se pase a `cross_validate_classifier` (validación cruzada),
    el escalador se reajuste (`fit`) independientemente en cada fold de
    entrenamiento y solo se aplique (`transform`) al fold de validación — nunca al
    revés. Escalar `X_train` completo una sola vez antes de la validación cruzada
    filtraría información de cada fold de validación hacia el ajuste del escalador
    (fuga de datos), inflando artificialmente el rendimiento estimado. Con
    `Pipeline`, ese riesgo desaparece por construcción: el objeto entero (escalador
    + clasificador) es lo que `cross_val_score` clona y reajusta en cada fold.

    `RandomForestClassifier` no necesita `Pipeline` porque es invariante a la escala
    (sus splits son umbrales directos sobre los valores de cada feature).

    Args:
        target: nombre del target a entrenar — una de `COMPONENT_TARGETS`.
        X_train: features de entrenamiento (73 columnas de la Fase 2).
        y_train: etiquetas de entrenamiento para `target` (una Serie, no un DataFrame).
        random_state: semilla para el modelo (bosque aleatorio o solver de
            Regresión Logística). 42 en el notebook.

    Returns:
        El modelo ya ajustado (`.fit(X_train, y_train)` ya ejecutado): un
        `RandomForestClassifier` o un `Pipeline(StandardScaler, LogisticRegression)`,
        según `select_best_model(target)`.
    """
    model_name = select_best_model(target)

    if model_name == "LogisticRegression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=random_state)),
        ])
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    model.fit(X_train, y_train)
    return model


# ── Bloque 3 — Evaluación del clasificador (Fase 4) ─────────────────────────────


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalúa un clasificador ya ajustado sobre el test set sellado.

    F1-macro (no accuracy ni F1 ponderado) es la métrica reportada porque promedia
    el F1 de cada clase por igual, sin ponderar por su frecuencia — apropiada aquí
    porque, aunque las 4 distribuciones de clase son casi uniformes (Sección 0 de la
    Fase 4), el objetivo operacional es detectar cualquier nivel de degradación con
    la misma atención, no solo el más frecuente. Un accuracy o un F1 ponderado por
    soporte podría enmascarar un mal desempeño en la clase minoritaria si esta
    pesara menos en el promedio.

    Args:
        model: clasificador ya ajustado (`.fit()` ya ejecutado), p.ej. la salida de
            `train_classifier`.
        X_test: features del conjunto de test sellado.
        y_test: etiquetas reales del conjunto de test sellado, para el mismo target
            con el que se entrenó `model`.

    Returns:
        dict con dos claves: `"test_f1_macro"` (float, F1-macro sobre `X_test`) y
        `"predictions"` (`np.ndarray`, las predicciones de `model.predict(X_test)`).
    """
    predictions = model.predict(X_test)
    test_f1_macro = f1_score(y_test, predictions, average="macro")
    return {"test_f1_macro": test_f1_macro, "predictions": predictions}


def cross_validate_classifier(model, X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE) -> float:
    """F1-macro medio de validación cruzada estratificada de 5 folds.

    `StratifiedKFold(n_splits=5, shuffle=True, random_state=...)` — estratificada
    para que cada fold conserve, aproximadamente, la misma proporción de clases que
    el conjunto completo (relevante incluso con clases casi uniformes, para no
    introducir varianza adicional por azar en folds pequeños). Se usa exclusivamente
    sobre `X_train`/`y_train`: usar el test en esta etapa introduciría sesgo de
    selección de modelo, ya que el test debe quedar reservado para la evaluación
    final (`evaluate_classifier`).

    Si `model` es un `Pipeline(StandardScaler, LogisticRegression)`, `cross_val_score`
    clona y reajusta el pipeline completo en cada fold — el escalador solo ve el
    fold de entrenamiento de esa iteración, nunca el de validación (ver el docstring
    de `train_classifier` para el razonamiento completo).

    Args:
        model: un estimador de scikit-learn sin ajustar (no debe llamarse `.fit()`
            antes — `cross_val_score` lo hace internamente en cada fold).
        X_train: features de entrenamiento.
        y_train: etiquetas de entrenamiento para el target de interés.
        random_state: semilla del `StratifiedKFold`. 42 en el notebook.

    Returns:
        float: media del F1-macro sobre los 5 folds.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    return scores.mean()


# ── Bloque 4 — Selección de modelo por target ───────────────────────────────────

# Tabla de consulta explícita: qué tipo de modelo ganó la comparación de F1-macro
# en validación cruzada (Fase 4, Sección 1, 5-fold estratificado sobre X_train/y_train).
# Los cuatro pares (RF, LR) con su delta real:
#   cooler_condition:      RF=1.0000 | LR=1.0000 | delta=+0.0000 -> empate; se elige RF
#                           por consistencia con los otros 3 targets y por su invarianza a la escala.
#   valve_condition:       RF=0.9922 | LR=0.9983 | delta=+0.0061 a favor de LR -> único target
#                           donde el modelo lineal superó al no lineal.
#   pump_leakage:          RF=0.9948 | LR=0.9905 | delta=+0.0043 a favor de RF.
#   accumulator_pressure:  RF=0.9845 | LR=0.8207 | delta=+0.1638 a favor de RF -> la brecha más
#                           grande de los cuatro; señal de que la importancia de features en RF
#                           para este target está dominada por interacciones no lineales
#                           (confirmado en la Fase 4, Sección 2: Spearman ρ=-0.3742 entre el
#                           ranking de importancia RF y el ranking de η² univariante de la Fase 2).
_BEST_MODEL_BY_TARGET: dict[str, str] = {
    "cooler_condition": "RandomForest",
    "valve_condition": "LogisticRegression",
    "pump_leakage": "RandomForest",
    "accumulator_pressure": "RandomForest",
}


def select_best_model(target: str) -> str:
    """Nombre del tipo de modelo seleccionado como mejor para un target, ya decidido.

    Es una tabla de consulta fija, no una re-evaluación: la comparación real
    (validación cruzada de 5 folds sobre `X_train`/`y_train`, F1-macro) ya se hizo en
    la Fase 4, Sección 1, y esta función solo expone su resultado documentado (ver
    los comentarios de `_BEST_MODEL_BY_TARGET` para los números de CV F1-macro y el
    delta que justificó cada elección). No vuelve a entrenar ni evaluar nada.

    Limitación estadística conocida: para `valve_condition` (delta=+0.61pp a favor de
    LR) y `pump_leakage` (delta=+0.43pp a favor de RF), el margen de CV es inferior a
    1 punto porcentual y la elección de modelo **no está respaldada por una prueba de
    significancia estadística**. Un test de McNemar pareado sobre las predicciones de
    test (n=290) para ambos targets da p≥0.5 (no significativo) — el ranking por CV es
    direccionalmente estable entre semillas, pero no es distinguible de ruido de
    muestreo con este tamaño de test. Para `accumulator_pressure` (delta=+16.38pp) la
    diferencia sí es sustancial y McNemar la confirma como significativa incluso tras
    corrección FDR. `cooler_condition` es un empate exacto (delta=0.0000), sin ambigüedad
    posible. Esta limitación no se ha corregido en el código (cambiar la selección de
    modelo alteraría resultados ya validados por ejecución) — se documenta aquí para que
    quien reutilice `train_classifier`/`select_best_model` conozca su alcance real.

    Args:
        target: nombre del target — una de `COMPONENT_TARGETS`.

    Returns:
        `"RandomForest"` o `"LogisticRegression"`.

    Raises:
        KeyError: si `target` no es uno de los 4 targets de componente conocidos.
    """
    return _BEST_MODEL_BY_TARGET[target]
