# Hydraulic Systems Predictive Maintenance (English)

🇪🇸 **Versión canónica en español: [`../README.md`](../README.md)**. This is a translation of that
README. **Important scope note:** unlike the other three projects in this portfolio, this repo does
not have a separate `english/notebooks/` mirror re-executed in English — the notebooks and code
comments themselves are written in Spanish only. This file translates the documentation, not the
underlying analysis code.

![Per-component degradation classification results — confusion matrices for cooler, valve, pump and accumulator](../reports/figures/04_confusion_matrices.png)

---

End-to-end data science project on the **Condition Monitoring of Hydraulic Systems** dataset (UCI ID 447): from raw sensors to a continuous per-cycle health proxy, covering EDA, feature engineering, anomaly detection, per-component supervised classification, and the construction of a composite degradation index.

Each notebook follows the **Decision → Code → Interpretation** pattern: every code cell is preceded by a cell that justifies what is being done and why, and followed by a cell that interprets the actual results obtained on execution — not anticipated results. Every figure in this README is taken directly from running the notebooks (`jupyter nbconvert --execute`, 0 errors across all five).

### Executive summary

This project trains classifiers that flag degradation in 4 components of an industrial hydraulic system (cooler, valve, pump, accumulator) from sensor data alone — the kind of system found in manufacturing, construction, and heavy machinery. The best models correctly classify each component's degradation level between **84% and 100% of the time on data never seen during training** (see "Generalization to new operating points" in Phase 4 for why the honest answer is a range, not a single headline number). As an illustrative, unvalidated scenario: if even a modest share of unplanned downtime at a plant running this kind of equipment could be traced to these 4 failure modes, catching them earlier — instead of only after a breakdown — is exactly the use case this project demonstrates end to end, from raw sensor signal to a validated classifier. This public dataset has no cost data, so no real savings figure is claimed here; that number would have to come from a specific plant's maintenance records.

### The dataset

2,205 60-second cycles from a hydraulic test rig, with 17 sensor channels sampled at 100, 10, or 1 Hz (pressure, flow, temperature, power, efficiency, vibration) and 4 independent target variables describing the state of four physical components:

| Component | Variable | Levels |
|---|---|---|
| Cooler | `cooler_condition` | 3 (close to failure) / 20 (reduced) / 100 (optimal) |
| Valve | `valve_condition` | 73 / 80 / 90 / 100 (optimal) |
| Pump | `pump_leakage` | 0 (no leakage) / 1 / 2 (severe leakage) |
| Accumulator | `accumulator_pressure` | 90 / 100 / 115 / 130 (optimal) |

The four targets combine into a **144-cell factorial design** (3×4×3×4), crossed in a controlled way — this is not a degradation time series from a single machine, but independent laboratory experiments. This structural property shapes much of the project's methodological decisions (see Phase 3 and Limitations).

`ucimlrepo.fetch_ucirepo(id=447)` fails (`DatasetNotFoundError`) — the dataset is not enabled for programmatic import via that library despite being listed in the repository. [`data_loader.py`](../src/data_loader.py) downloads the public static ZIP directly instead; see the module's docstring for the detail of that check.

### Project structure

```
hydraulic_maintenance/
├── data/
│   ├── raw/hydraulic_raw/          # raw sensors + profile.txt, downloaded by data_loader.py
│   └── processed/                  # intermediate and final artifacts per phase (see table below)
├── notebooks/                      # Spanish only — see scope note above
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_failure_prediction.ipynb
│   └── 05_rul_proxy.ipynb
├── src/
│   ├── data_loader.py               # download and load the raw dataset
│   ├── feature_engineering.py       # extraction of the 73 features and selection by η²
│   ├── models.py                    # training/evaluation: detectors (Phase 3) and classifiers (Phase 4)
│   ├── validation.py                # Phase 4b: grouped generalization, dual-model bootstrap/McNemar,
│   │                                 # class_weight and hyperparameter-search checks (see Phase 4)
│   └── visualization.py             # the project's 9 figures, refactored as pure functions
├── run_validation_fixes.py         # Phase 4b entrypoint — writes data/processed/04b_*
├── reports/figures/                # 9 figures — see table below
├── models/                         # reserved; no serialized models (see note below)
├── requirements.txt
├── LICENSE
├── README.md                       # Spanish, canonical
└── english/README.md               # this file
```

**Note on `src/`:** [`data_loader.py`](../src/data_loader.py), [`feature_engineering.py`](../src/feature_engineering.py), [`models.py`](../src/models.py) and [`visualization.py`](../src/visualization.py) are refactored from notebook logic already executed with 0 errors, without altering any formula, hyperparameter, or selection criterion relative to what was validated in the notebooks 01–05. [`validation.py`](../src/validation.py) is different in kind: it is **new** analysis added in Phase 4b (see Phase 4 above), executed via [`run_validation_fixes.py`](../run_validation_fixes.py) rather than a notebook — it was not present in, and does not alter, the original 5 notebooks. `models/` remains empty: no model in the project is serialized to disk. There is no `.pkl`/`.joblib` to load — to get a trained model you have to retrain it by running the notebooks in order (or by calling `train_classifier`/`train_mahalanobis_detector`/`train_isolation_forest_detector` from `src/models.py`, or the functions in `src/validation.py`, on the artifacts in `data/processed/`), not retrieve it from disk.

### Pipeline by phase

#### Phase 1 — EDA ([`01_eda.ipynb`](../notebooks/01_eda.ipynb))
Confirms the 144-combination factorial design, identifies that `cooler_condition` dominates the variance of most sensors, and that the temporal position of the peak in `PS3` (not its level) is the most informative indicator of valve degradation. Detects 2 outlier cycles (617, 371) via group-conditioned MAD, not a global threshold.

#### Phase 2 — Feature engineering ([`02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb))
Reduces each cycle of up to 6,000 readings per sensor to **73 scalar features** (means, standard deviations, temporal halves, cross-sensor differentials, peak position), selected by η² against the 4 targets. Exports `X_train`/`X_test` (1,157/290 cycles, 80/20 split) and `y_train`/`y_test` to Parquet.

#### Phase 3 — Anomaly detection ([`03_anomaly_detection.ipynb`](../notebooks/03_anomaly_detection.ipynb))
Shows, with quantitative evidence, that unsupervised detection **fails on this specific dataset**: the "all 4 components optimal at once" state is only 1 of 144 combinations (8 cycles in train, 0.69%) — the *rarest* combination in the design, not the most common, inverting the premise these methods operate on. Mahalanobis with LedoitWolf regularization reaches AUC-ROC=1.0000 on test, but due to a scaling artifact (a single feature with near-zero variance in the 8 reference cycles) tied to `cooler_condition`, not a reliable multivariate distance. Isolation Forest gets AUC-ROC=**0.2743** — worse than chance — because it detects statistical rarity, and the healthy state is precisely the rarest thing in the design.

#### Phase 4 — Supervised classification ([`04_failure_prediction.ipynb`](../notebooks/04_failure_prediction.ipynb))
Trains an independent classifier per component (Random Forest / Logistic Regression), validated by 5-fold CV and evaluated on a sealed test set:

| Target | Model chosen | Test F1-macro | 95% CI (bootstrap, 1,000 resamples) |
|---|---|---|---|
| `cooler_condition` | Random Forest | 1.0000 | [1.0000, 1.0000] |
| `valve_condition` | Logistic Regression | 0.9965 | [0.9889, 1.0000] |
| `pump_leakage` | Random Forest | 0.9931 | [0.9825, 1.0000] |
| `accumulator_pressure` | Random Forest | 0.9863 | [0.9701, 0.9967] |

*These 95% CIs are now produced by real, reusable code (`src/validation.py::mcnemar_bootstrap_report`, `sklearn.utils.resample`, 1,000 resamples, 2.5/97.5 percentiles), run via `run_validation_fixes.py`. A prior version of this README described this exact computation in prose without shipping the code that produces it, and `predictions_test.parquet` only stored the winning model's predictions per target — so the number could not actually be regenerated from the repository. `data/processed/04b_predictions_test_dual_model.parquet` now stores both models' predictions for all 4 targets, and `04b_mcnemar_bootstrap.csv` the full comparison below.*

A paired McNemar test (on per-cycle correct/incorrect, `statsmodels.stats.contingency_tables.mcnemar`) confirms: for `valve_condition` (1 discordant pair each direction, p=1.0000) and `pump_leakage` (2 vs. 0, p=0.5000) the RF-vs-LR gap is not distinguishable from sampling noise, consistent with what a prior version of this README already stated — now backed by executable, reproducible code instead of only prose. For `accumulator_pressure` the gap is real and large: RF 0.9863 vs. **LR 0.8602** (39 vs. 3 discordant pairs, p=6.6×10⁻⁸) — and, applying Benjamini-Hochberg FDR correction across the 4 targets tested (`multipletests`, α=0.05), still significant (p_FDR=2.7×10⁻⁷). A prior version of this README already claimed the `accumulator_pressure` result "survives FDR correction" without any code computing that correction; it does hold, and now there is code to show it.

**Reproducibility note:** a prior version of this README cited 82.28% for Logistic Regression's test F1-macro on `accumulator_pressure`; recomputing it under the environment pinned in `requirements.txt` (same model definition — `solver="lbfgs"`, `multi_class="multinomial"`, same `random_state=42`) gives 86.02% instead, a ~4pp difference. The most likely cause is scikit-learn version drift in that now-deprecated code path (`multi_class="multinomial"` is removed in scikit-learn 1.7). Either way the substantive conclusion is unchanged — if anything strengthened, since 86.02% is still far below RF's 98.63% and the McNemar p-value leaves no ambiguity.

Central finding: for `accumulator_pressure`, Random Forest's feature importance is **negatively correlated** with its univariate η² from Phase 2 (Spearman ρ=-0.3742) — the features most useful to the model (`FS2_*`) have η²≈0.003, practically null in isolation. This directly explains why Logistic Regression scores far lower (86.02% F1-macro on test, vs. 98.63% with RF): the signal is interaction-based, not a linear mean shift, and no univariate analysis could have anticipated it.

##### Generalization to new operating points (not just held-out cycles)

The 80/20 split (Phase 2, `strat_key`) stratifies by the *full combination* of the 4 targets — i.e., by the 144-cell factorial design itself. With only 144 cells and 7–15 near-identical replicate cycles each (same physical operating point, sensor noise only), **every cell that appears in test also appears in train**: the model has already seen cycles from the exact same operating point it is being tested on, just not that literal cycle. That is a narrower, more optimistic question than "does this generalize to an operating point never seen in training" — the practically relevant one for a system deployed on new equipment or a new combination of component states.

Re-evaluating with `GroupKFold` (`src/validation.py::grouped_generalization_report`, `run_validation_fixes.py`), grouping by that same 144-cell key so that **no cell appears on both sides of any fold**, gives a materially different picture for 2 of the 4 targets:

| Target | Stratified test F1-macro (interpolation) | GroupKFold F1-macro, k=5 (new operating point) | Gap |
|---|---|---|---|
| `cooler_condition` | 1.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `valve_condition` | 0.9965 | 0.9938 ± 0.0077 | 0.0027 |
| `pump_leakage` | 0.9931 | 0.9658 ± 0.0320 | 0.0273 |
| `accumulator_pressure` | 0.9863 | **0.8421 ± 0.0410** | **0.1442** |

`cooler_condition` and `valve_condition` barely move — their signal is close to global/linear across the sensor space, so having seen "a similar cell" isn't doing much work. `pump_leakage` loses ~2.7pp. `accumulator_pressure` loses **14.4pp**: this is precisely the target whose Random Forest signal, per the central finding above, comes from feature *interactions* rather than univariate shifts — interaction-based signal is exactly the kind a tree ensemble can partly memorize per operating point instead of learning as a rule that transfers to a new one, which is why this target is also the most exposed by this correction. Both numbers are reported here on purpose: the stratified test F1 is not wrong, it answers a real (if narrower) question, but it is not evidence of generalization to a genuinely new combination of component states — and a prior version of this README did not make that distinction.

##### Robustness checks: trivial baseline and hyperparameter search

Two gaps a technical reviewer would ask about first, both closed via `run_validation_fixes.py`:

- **Trivial baseline** (`DummyClassifier`, `data/processed/04b_baseline_dummy.csv`): `most_frequent` scores F1-macro 0.10–0.17 across the 4 targets, `stratified` scores 0.26–0.33 — both far below the 0.84–1.00 the real models reach, confirming the achieved scores are not an artifact of an easy/imbalanced target the dummy could already game.
- **Hyperparameter search** (`RandomizedSearchCV`, 5-fold CV on train only, 30 iterations, `class_weight` included in the search space — `data/processed/04b_hyperparameter_search.csv`): no notebook or `src/models.py` had ever tuned hyperparameters (only `n_estimators=100`/`max_iter=1000` were set explicitly, everything else left at scikit-learn defaults). The search finds `cooler_condition` and `valve_condition` already at their ceiling (no improvement possible); `pump_leakage` improves from 0.9948 to 0.9965 CV F1-macro, `accumulator_pressure` from 0.9845 to 0.9871, both via `class_weight="balanced"` combined with `min_samples_leaf=1` and more trees (`n_estimators=256`). The gains are real but small (<0.3pp) — the shipped model (untuned defaults, documented above) is kept as the primary result for simplicity, and this search is reported as a robustness/headroom check, not a replacement.
- **`class_weight="balanced"` in isolation** (`data/processed/04b_class_weight_comparison.csv`): the Phase 1/2 EDA recommends it twice as a precaution against class imbalance, but no model in `src/models.py` ever applied it. Testing it in isolation (holding every other hyperparameter at its current default) gives a mixed, negligible result: no change for `cooler_condition`/`valve_condition`, +0.09pp for `pump_leakage`, **-0.17pp for `accumulator_pressure`**. In isolation it doesn't help; it only helps as part of a broader tuned configuration (see hyperparameter search above, where it is selected jointly with other changes for 2 of 4 targets). The EDA's recommendation was tested rather than applied on faith — the honest conclusion is "not worth it on its own, mildly useful combined with tuning", not "always apply it".

#### Phase 5 — RUL proxy ([`05_rul_proxy.ipynb`](../notebooks/05_rul_proxy.ipynb))
Combines the 4 Phase 4 predictions into a composite health index `H_mean ∈ [0,1]` (mean per-component normalized degradation, complemented to 1). Validated internally (decreases monotonically with the cooler's actual degradation: 0.627→0.502→0.369 by level) and cross-validated against the Phase 3 scores (ρ=-0.567 with Mahalanobis, ρ=-0.190 with Isolation Forest — magnitudes consistent with each detector's AUC in Phase 3). Exports `rul_proxy_test.parquet` as the pipeline's final artifact.

### Artifacts in `data/processed/`

| File | Source | Content |
|---|---|---|
| `X_train.parquet`, `X_test.parquet` | Phase 2 | 73 features per cycle (1,157 / 290 rows) |
| `y_train.parquet`, `y_test.parquet` | Phase 2 | 4 original targets per cycle |
| `eta2_features_vs_targets.csv` | Phase 2 | η² of each candidate feature against each target |
| `anomaly_scores_test.parquet` | Phase 3 | `maha2_test`, `anomaly_score_if_test` (test, 290 rows) |
| `predictions_test.parquet` | Phase 4 | Per-component predictions on test + anomaly label |
| `rul_proxy_test.parquet` | Phase 5 | Health index `H_mean`/`H_max` and per-component contribution |
| `04b_grouped_generalization.csv` | Phase 4b | Stratified vs. `GroupKFold` F1-macro per target (see "Generalization to new operating points") |
| `04b_predictions_test_dual_model.parquet` | Phase 4b | Both RF and LR predictions on test, for all 4 targets (McNemar/bootstrap input) |
| `04b_mcnemar_bootstrap.csv` | Phase 4b | Bootstrap CI + McNemar test, RF vs. LR, per target |
| `04b_class_weight_comparison.csv` | Phase 4b | CV F1-macro with/without `class_weight="balanced"`, in isolation |
| `04b_baseline_dummy.csv` | Phase 4b | `DummyClassifier` (`most_frequent`/`stratified`) test F1-macro, per target |
| `04b_hyperparameter_search.csv` | Phase 4b | `RandomizedSearchCV` results (default vs. tuned CV F1-macro, best params) per target |

**`reports/figures/`** contains the 9 figures generated by the five notebooks:

| File | Phase |
|---|---|
| `01_target_distributions.png` | Phase 1 |
| `01_target_cooccurrence.png` | Phase 1 |
| `01_cycle_shapes_by_condition.png` | Phase 1 |
| `01_pressure_sensor_correlation.png` | Phase 1 |
| `02_eta2_heatmap.png` | Phase 2 |
| `03_roc_and_score_distribution.png` | Phase 3 |
| `04_confusion_matrices.png` | Phase 4 |
| `04_eta2_vs_importance_scatter.png` | Phase 4 |
| `05_health_index_distribution.png` | Phase 5 |

### Methodological limitations

1. **No real time dimension.** The 2,205 cycles are independent experiments, not a progressive degradation series — the Phase 5 proxy ranks relative severity, it does not predict time to failure.
2. **The balanced factorial design is an atypical condition for anomaly detection.** It invalidated Mahalanobis and, especially, Isolation Forest in Phase 3, because the healthy state is the *least* frequent combination, not the most frequent. To be precise: "balanced" refers to the number of replicates per cell (144/144 cells represented in train, 141 match exactly at 8 replicates, actual range 7–15 — verified, not an approximate claim) — it does **not** mean each cell carries a weight proportional to the total. With 144 cells, none can represent more than ~0.7% of the total by construction: the combination of all 4 components simultaneously optimal is 8 of 1,157 training cycles (0.69%), but that is not an underrepresented cell relative to the others — it is the inevitable arithmetic consequence of splitting the population into 144 equal parts.
3. **The health proxy has no time units** and cannot be translated into "days to failure" without real progression-rate data that this dataset does not contain.
4. **The classification results depend on a single random seed (`random_state=42`).** Inter-seed variability has not been evaluated within the documented pipeline (notebooks/README) — there is no repetition with different seeds to confirm that the CV deltas or the per-target model rankings are stable with respect to the specific seed choice.
5. **The sealed test set is less independent than a single F1-macro number suggests.** The 80/20 split stratifies by the full 4-target combination (the 144-cell factorial design), so every cell in test also has near-identical replicate cycles in train. For 2 of 4 targets this barely matters; for `accumulator_pressure` the test F1-macro (0.9863) is ~14pp more optimistic than a `GroupKFold` evaluation that holds out entire operating points (0.8421 ± 0.0410) — see "Generalization to new operating points" in Phase 4 for the full comparison and why this target specifically is the most exposed.

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# ... 02 through 05 in order — each phase consumes artifacts exported by the previous one
python run_validation_fixes.py   # Phase 4b — reads data/processed/{X,y}_{train,test}.parquet from
                                   # Phase 2, writes data/processed/04b_*; ~6 minutes (the hyperparameter
                                   # search accounts for most of that time)
```

*(Commands run from the repo root, not from `english/` — see the scope note at the top: there is no separate English notebook/script tree to run.)*

**Note on `requirements.txt`:** the project's original plan called for `torch`, `xgboost`, `shap`, `optuna`, and `imbalanced-learn` (in particular, a PyTorch autoencoder for Phase 3) — but no notebook or `src/` module imports them in the final implementation: Phase 3 replaced the planned autoencoder with Isolation Forest after discovering there are only 8 normal cycles in train, not enough to train a dense network without severe overfitting (reasoning documented in that section's Decision). Those 5 packages, along with `joblib` (also unused — no model is serialized), were **removed** from `requirements.txt`. The project uses exclusively `pandas`, `numpy`, `scipy` (added — it was missing despite being used in several notebooks and in `src/feature_engineering.py`), `matplotlib`, `seaborn`, `scikit-learn`, `pyarrow` (pandas' parquet engine), and `jupyter` (execution environment); `ucimlrepo` is kept only for the diagnostic check in `src/data_loader.py`.

### Data and license

- **Dataset:** [Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
  (UCI Machine Learning Repository, ID 447; created by ZeMA gGmbH / Universität des Saarlandes) —
  **CC BY 4.0** license (Creative Commons Attribution 4.0 International): allows sharing and
  adapting with attribution. The raw data (531 MB) is not included in the repository — it is
  downloaded on demand with `src/data_loader.py` (see "How to run"). `data/processed/` is
  versioned: these are the pipeline's artifacts/results (~1 MB), not a copy of the raw data.
- **Code in this repository:** MIT.
