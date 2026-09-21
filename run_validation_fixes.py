#!/usr/bin/env python
"""Fase 4b -- extensiones de validación (ver `src/validation.py` para el detalle de cada una).

Ejecuta las 5 correcciones del plan de acción de la revisión de portfolio:
  1. Generalización agrupada por celda del diseño factorial vs. el split estratificado original.
  2. Predicciones de ambos modelos (RF y LR) en test + bootstrap CI + McNemar real.
  3. Comparación con/sin class_weight="balanced".
  4. (Executive summary -- no requiere cómputo, se escribe directamente en el README.)
  5. Baseline trivial (DummyClassifier) + búsqueda de hiperparámetros.

Lee los artefactos ya existentes en data/processed/ (no vuelve a descargar ni reconstruir las 73
features desde los sensores crudos -- no hace falta: X_train/X_test/y_train/y_test ya están
exportados por la Fase 2 y son la única entrada que estas 5 correcciones necesitan). Escribe cada
resultado como CSV/Parquet en data/processed/, con un nombre `04b_*` para distinguirlos de los
artefactos de las fases originales.

Uso:
    python run_validation_fixes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import validation  # noqa: E402
from src.models import COMPONENT_TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}")


def main() -> None:
    log("Cargando artefactos de la Fase 2 (data/processed/)...")
    X_train = pd.read_parquet(PROCESSED / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED / "y_train.parquet")
    y_test = pd.read_parquet(PROCESSED / "y_test.parquet")
    log(f"  X_train={X_train.shape} X_test={X_test.shape} y_train={y_train.shape} y_test={y_test.shape}")

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    n_groups = validation.build_group_ids(y_full).nunique()
    log(f"  Combinaciones (celdas del diseño factorial) presentes en train+test: {n_groups}")

    # 1. Generalización agrupada por celda vs. split estratificado original -----------------
    log("1/5 -- Generalización agrupada por celda (GroupKFold, k=5) vs. split estratificado...")
    grouped_df = validation.grouped_generalization_report(X_full, y_full, n_splits=5)
    stratified_test_f1 = {}
    for target in COMPONENT_TARGETS:
        model = validation._make_model(validation.select_best_model(target))
        model.fit(X_train, y_train[target])
        from sklearn.metrics import f1_score
        stratified_test_f1[target] = f1_score(y_test[target], model.predict(X_test), average="macro")
    grouped_df["stratified_test_f1_macro"] = grouped_df["target"].map(stratified_test_f1)
    grouped_df["delta_stratified_minus_grouped"] = (
        grouped_df["stratified_test_f1_macro"] - grouped_df["grouped_cv_f1_macro_mean"]
    )
    grouped_df.to_csv(PROCESSED / "04b_grouped_generalization.csv", index=False)
    for _, r in grouped_df.iterrows():
        log(f"    {r['target']:<24} split estratificado={r['stratified_test_f1_macro']:.4f}  "
            f"GroupKFold={r['grouped_cv_f1_macro_mean']:.4f}+/-{r['grouped_cv_f1_macro_std']:.4f}  "
            f"delta={r['delta_stratified_minus_grouped']:+.4f}")

    # 2. Predicciones duales + bootstrap CI + McNemar real -----------------------------------
    log("2/5 -- Entrenando RF y LR para los 4 targets, prediciendo en test...")
    dual_preds = validation.dual_model_predictions(X_train, y_train, X_test, y_test)
    dual_preds.to_parquet(PROCESSED / "04b_predictions_test_dual_model.parquet", index=False)
    log("  Calculando bootstrap CI (1000 remuestras) y McNemar...")
    mcnemar_df = validation.mcnemar_bootstrap_report(dual_preds, n_boot=1000)
    mcnemar_df.to_csv(PROCESSED / "04b_mcnemar_bootstrap.csv", index=False)
    for _, r in mcnemar_df.iterrows():
        log(f"    {r['target']:<24} RF={r['rf_test_f1_macro']:.4f} [{r['rf_ci95_lo']:.4f},{r['rf_ci95_hi']:.4f}]  "
            f"LR={r['lr_test_f1_macro']:.4f} [{r['lr_ci95_lo']:.4f},{r['lr_ci95_hi']:.4f}]  "
            f"McNemar b={r['n_discordant_rf_only_correct']} c={r['n_discordant_lr_only_correct']} "
            f"p={r['mcnemar_pvalue']:.4f}")

    # 3 + 5. class_weight (comparación aislada) + búsqueda de hiperparámetros (incluye class_weight) --
    log("3/5 -- class_weight='balanced' vs. default (CV 5-fold sobre train)...")
    cw_df = validation.class_weight_report(X_train, y_train)
    cw_df.to_csv(PROCESSED / "04b_class_weight_comparison.csv", index=False)
    for target in COMPONENT_TARGETS:
        sub = cw_df[cw_df["target"] == target]
        default_f1 = sub[sub["class_weight"] == "default (None)"]["cv_f1_macro_mean"].iloc[0]
        balanced_f1 = sub[sub["class_weight"] == "balanced"]["cv_f1_macro_mean"].iloc[0]
        log(f"    {target:<24} default={default_f1:.4f}  balanced={balanced_f1:.4f}  "
            f"delta={balanced_f1 - default_f1:+.4f}")

    log("5/5 -- Baseline DummyClassifier + RandomizedSearchCV (n_iter=30, 5-fold, solo train)...")
    baseline_df = validation.baseline_dummy_report(X_train, y_train, X_test, y_test)
    baseline_df.to_csv(PROCESSED / "04b_baseline_dummy.csv", index=False)
    search_df = validation.hyperparameter_search_report(X_train, y_train, n_iter=30)
    search_df.to_csv(PROCESSED / "04b_hyperparameter_search.csv", index=False)
    for _, r in search_df.iterrows():
        log(f"    {r['target']:<24} default={r['cv_f1_macro_default']:.4f}  "
            f"tuned={r['cv_f1_macro_tuned']:.4f}  best_params={r['best_params']}")

    log("Completado. Artefactos escritos en data/processed/04b_*.{csv,parquet}")


if __name__ == "__main__":
    main()
