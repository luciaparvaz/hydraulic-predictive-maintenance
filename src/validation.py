"""Extensiones de validación añadidas tras la revisión de portfolio (Fase 4b).

Cubre cuatro huecos detectados por una auditoría externa sobre `notebooks/04_failure_prediction.ipynb`
y `src/models.py`:

  1. `grouped_generalization_report`: el split train/test (`strat_key` en la Fase 2) estratifica
     por la combinación completa de los 4 targets -- las 144 celdas del diseño factorial -- lo que
     garantiza que cada celda esté representada en ambos lados con réplicas casi idénticas (mismo
     punto de operación físico, solo difiere el ruido de sensor). Esta función mide cuánto de las
     métricas de test reportadas es interpolación dentro de celdas ya vistas, comparando el F1-macro
     del split estratificado original contra un `GroupKFold` que fuerza a que cada celda esté SOLO
     en un lado -- la pregunta real de generalización a un punto de operación nuevo.
  2. `dual_model_predictions` + `mcnemar_bootstrap_report`: el README citaba un IC bootstrap y un
     test de McNemar que no existían como código ejecutable -- solo como texto describiendo un
     resultado no reproducible desde el repo. Aquí se entrenan y evalúan AMBOS modelos (RF y LR)
     para los 4 targets, se guardan sus predicciones, y se calculan bootstrap CI y McNemar de verdad.
  3. `class_weight_report`: el EDA recomienda dos veces `class_weight="balanced"` pero ningún modelo
     de `src/models.py` lo aplica. Se compara CV F1-macro con y sin esa opción para los 4 targets.
  4. `hyperparameter_search_report`: no existía ninguna búsqueda de hiperparámetros en el proyecto
     (los modelos usan valores por defecto salvo `n_estimators=100`/`max_iter=1000`). Se añade una
     búsqueda `RandomizedSearchCV` modesta, ejecutada solo sobre `X_train`/`y_train` (nunca sobre el
     test), con `class_weight` incluido en la rejilla para que la pregunta 3 y la pregunta 4 se
     respondan de forma conjunta y consistente.

Todas las funciones son deterministas (semilla fija, `RANDOM_STATE=42` heredada de `src/models.py`)
y no tocan el test set salvo `dual_model_predictions`/`mcnemar_bootstrap_report`, que lo tocan
exactamente una vez, para el reporte final -- nunca para elegir nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .models import COMPONENT_TARGETS, RANDOM_STATE, select_best_model

__all__ = [
    "build_group_ids",
    "grouped_generalization_report",
    "dual_model_predictions",
    "mcnemar_bootstrap_report",
    "class_weight_report",
    "hyperparameter_search_report",
]


def build_group_ids(y: pd.DataFrame, targets: list[str] = COMPONENT_TARGETS) -> pd.Series:
    """Reconstruye `strat_key` de la Fase 2: una clave de texto por ciclo que identifica la celda
    del diseño factorial a la que pertenece (combinación exacta de los 4 targets). Por construcción,
    todos los ciclos con la misma clave tienen el mismo valor en los 4 targets -- es exactamente lo
    que hace de esta clave un identificador de "punto de operación", no una etiqueta de clase."""
    return y[targets].astype(str).agg("_".join, axis=1)


def _make_model(model_name: str, random_state: int = RANDOM_STATE, class_weight: str | None = None):
    if model_name == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, multi_class="multinomial", solver="lbfgs",
                random_state=random_state, class_weight=class_weight,
            )),
        ])
    return RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=class_weight)


def grouped_generalization_report(
    X_full: pd.DataFrame, y_full: pd.DataFrame, n_splits: int = 5, random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compara, para cada target, el F1-macro del split estratificado original (tal como se reporta
    hoy en el README) contra el F1-macro medio de un `GroupKFold` que impide que ninguna celda del
    diseño factorial aparezca a la vez en entrenamiento y evaluación.

    `X_full`/`y_full` deben ser la concatenación de train+test tras el filtrado de la Fase 2
    (outliers + stable_flag==0) -- es decir, `X_train`+`X_test` y `y_train`+`y_test` reunidos, para
    que el `GroupKFold` pueda repartir las 144 celdas libremente en vez de heredar la partición ya
    contaminada del split original.
    """
    groups = build_group_ids(y_full)
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for target in COMPONENT_TARGETS:
        model_name = select_best_model(target)
        y_t = y_full[target]
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_full, y_t, groups):
            model = _make_model(model_name, random_state=random_state)
            model.fit(X_full.iloc[train_idx], y_t.iloc[train_idx])
            preds = model.predict(X_full.iloc[val_idx])
            fold_scores.append(f1_score(y_t.iloc[val_idx], preds, average="macro"))
        fold_scores = np.array(fold_scores)
        rows.append({
            "target": target,
            "model": model_name,
            "n_groups_total": groups.nunique(),
            "n_folds": n_splits,
            "grouped_cv_f1_macro_mean": fold_scores.mean(),
            "grouped_cv_f1_macro_std": fold_scores.std(),
            "grouped_cv_f1_macro_min": fold_scores.min(),
            "grouped_cv_f1_macro_max": fold_scores.max(),
        })
    return pd.DataFrame(rows)


def dual_model_predictions(
    X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Entrena RandomForest Y LogisticRegression (no solo el elegido por `select_best_model`) para
    los 4 targets sobre train completo, y devuelve un DataFrame con una fila por ciclo de test y
    columnas `{target}_true`, `{target}_pred_rf`, `{target}_pred_lr` -- la pieza que faltaba para
    poder recalcular McNemar/bootstrap desde el repo en vez de solo citarlos en prosa."""
    out = {}
    for target in COMPONENT_TARGETS:
        y_tr = y_train[target]
        y_te = y_test[target]
        rf = _make_model("RandomForest", random_state=random_state)
        lr = _make_model("LogisticRegression", random_state=random_state)
        rf.fit(X_train, y_tr)
        lr.fit(X_train, y_tr)
        out[f"{target}_true"] = y_te.to_numpy()
        out[f"{target}_pred_rf"] = rf.predict(X_test)
        out[f"{target}_pred_lr"] = lr.predict(X_test)
    return pd.DataFrame(out)


def _bootstrap_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                      random_state: int = RANDOM_STATE) -> tuple[float, float, float]:
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    idx = np.arange(n)
    boot_scores = np.empty(n_boot)
    for b in range(n_boot):
        sample_idx = resample(idx, replace=True, n_samples=n, random_state=rng.randint(0, 2**31 - 1))
        boot_scores[b] = f1_score(y_true[sample_idx], y_pred[sample_idx], average="macro")
    point = f1_score(y_true, y_pred, average="macro")
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return point, float(lo), float(hi)


def mcnemar_bootstrap_report(dual_preds: pd.DataFrame, n_boot: int = 1000,
                              random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Para cada target: IC95% bootstrap del F1-macro de RF y de LR por separado (1000 remuestras,
    percentiles 2.5/97.5 -- como citaba el README, pero ahora con el código que lo produce), y un
    test de McNemar pareado sobre `correcto/incorrecto` por ciclo entre RF y LR (la forma estándar
    de aplicar McNemar a una comparación de clasificadores, incluida la multiclase: la tabla 2x2 no
    compara clases, compara "¿acertó el modelo A?" vs. "¿acertó el modelo B?" en el mismo ciclo)."""
    rows = []
    for target in COMPONENT_TARGETS:
        y_true = dual_preds[f"{target}_true"].to_numpy()
        pred_rf = dual_preds[f"{target}_pred_rf"].to_numpy()
        pred_lr = dual_preds[f"{target}_pred_lr"].to_numpy()

        rf_f1, rf_lo, rf_hi = _bootstrap_f1_ci(y_true, pred_rf, n_boot=n_boot, random_state=random_state)
        lr_f1, lr_lo, lr_hi = _bootstrap_f1_ci(y_true, pred_lr, n_boot=n_boot, random_state=random_state + 1)

        rf_correct = (pred_rf == y_true)
        lr_correct = (pred_lr == y_true)
        b = int(np.sum(rf_correct & ~lr_correct))   # RF acierta, LR falla
        c = int(np.sum(~rf_correct & lr_correct))   # LR acierta, RF falla
        table = np.array([[int(np.sum(rf_correct & lr_correct)), b], [c, int(np.sum(~rf_correct & ~lr_correct))]])
        # exact=True (test binomial exacto) recomendado cuando b+c < 25, como aquí
        # (n=290 ciclos de test, discordancia esperada muy por debajo de ese umbral en este dataset).
        result = mcnemar(table, exact=(b + c) < 25)

        rows.append({
            "target": target,
            "rf_test_f1_macro": rf_f1, "rf_ci95_lo": rf_lo, "rf_ci95_hi": rf_hi,
            "lr_test_f1_macro": lr_f1, "lr_ci95_lo": lr_lo, "lr_ci95_hi": lr_hi,
            "n_discordant_rf_only_correct": b, "n_discordant_lr_only_correct": c,
            "mcnemar_statistic": float(result.statistic), "mcnemar_pvalue": float(result.pvalue),
            "mcnemar_exact": bool((b + c) < 25),
        })
    df = pd.DataFrame(rows)
    # Corrección de comparaciones múltiples (Benjamini-Hochberg) sobre los 4 tests de McNemar (uno
    # por target). El docstring de select_best_model() en models.py afirmaba que la significancia de
    # accumulator_pressure sobrevivía "incluso tras corrección FDR" sin que existiera código que la
    # calculara -- exactamente el tipo de cifra citada-pero-no-reproducible que esta revisión corrige.
    reject, p_adj, _, _ = multipletests(df["mcnemar_pvalue"].to_numpy(), alpha=0.05, method="fdr_bh")
    df["mcnemar_pvalue_fdr_bh"] = p_adj
    df["mcnemar_significant_fdr_bh"] = reject
    return df


def class_weight_report(X_train: pd.DataFrame, y_train: pd.DataFrame,
                         random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """CV F1-macro (5-fold estratificado, igual esquema que `cross_validate_classifier`) con y sin
    `class_weight="balanced"`, para el modelo ya elegido de cada target. Responde directamente a si
    la recomendación del EDA (nunca aplicada en `src/models.py`) habría cambiado algo."""
    rows = []
    for target in COMPONENT_TARGETS:
        model_name = select_best_model(target)
        y_t = y_train[target]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for weight_label, weight_value in (("default (None)", None), ("balanced", "balanced")):
            model = _make_model(model_name, random_state=random_state, class_weight=weight_value)
            scores = []
            for tr_idx, val_idx in cv.split(X_train, y_t):
                m = _make_model(model_name, random_state=random_state, class_weight=weight_value)
                m.fit(X_train.iloc[tr_idx], y_t.iloc[tr_idx])
                preds = m.predict(X_train.iloc[val_idx])
                scores.append(f1_score(y_t.iloc[val_idx], preds, average="macro"))
            scores = np.array(scores)
            rows.append({
                "target": target, "model": model_name, "class_weight": weight_label,
                "cv_f1_macro_mean": scores.mean(), "cv_f1_macro_std": scores.std(),
            })
    return pd.DataFrame(rows)


def baseline_dummy_report(X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_test: pd.DataFrame, y_test: pd.DataFrame,
                           random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """F1-macro en test de dos baselines triviales (`most_frequent` y `stratified`) por target, para
    contextualizar qué tan fácil o difícil es cada tarea antes de atribuir el F1~0.99 al modelo."""
    rows = []
    for target in COMPONENT_TARGETS:
        y_tr, y_te = y_train[target], y_test[target]
        for strategy in ("most_frequent", "stratified"):
            dummy = DummyClassifier(strategy=strategy, random_state=random_state)
            dummy.fit(X_train, y_tr)
            preds = dummy.predict(X_test)
            rows.append({
                "target": target, "strategy": strategy,
                "test_f1_macro": f1_score(y_te, preds, average="macro", zero_division=0),
            })
    return pd.DataFrame(rows)


_RF_SEARCH_SPACE = {
    "n_estimators": randint(100, 400),
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"],
}
_LR_SEARCH_SPACE = {
    "clf__C": uniform(0.01, 10),
    "clf__class_weight": [None, "balanced"],
}


def hyperparameter_search_report(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 30,
                                  random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """`RandomizedSearchCV` (5-fold estratificado, F1-macro, solo sobre train) por target, sobre el
    tipo de modelo ya elegido en `select_best_model`. Incluye `class_weight` en la rejilla, así que
    esta función responde a la vez la pregunta de tuning y la de `class_weight` de forma conjunta:
    si "balanced" fuese realmente mejor, la búsqueda lo habría encontrado como óptimo."""
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for target in COMPONENT_TARGETS:
        model_name = select_best_model(target)
        y_t = y_train[target]
        base_model = _make_model(model_name, random_state=random_state)
        space = _RF_SEARCH_SPACE if model_name == "RandomForest" else _LR_SEARCH_SPACE

        search = RandomizedSearchCV(
            base_model, space, n_iter=n_iter, cv=cv, scoring="f1_macro",
            random_state=random_state, n_jobs=-1,
        )
        search.fit(X_train, y_t)

        default_scores = []
        for tr_idx, val_idx in cv.split(X_train, y_t):
            m = _make_model(model_name, random_state=random_state)
            m.fit(X_train.iloc[tr_idx], y_t.iloc[tr_idx])
            default_scores.append(f1_score(y_t.iloc[val_idx], m.predict(X_train.iloc[val_idx]), average="macro"))

        rows.append({
            "target": target, "model": model_name, "n_iter_search": n_iter,
            "cv_f1_macro_default": float(np.mean(default_scores)),
            "cv_f1_macro_tuned": float(search.best_score_),
            "best_params": str(search.best_params_),
        })
    return pd.DataFrame(rows)
