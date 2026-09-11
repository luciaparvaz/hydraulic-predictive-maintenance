# Mantenimiento Predictivo de Sistemas Hidráulicos Industriales

Proyecto de ciencia de datos de extremo a extremo sobre el dataset **Condition Monitoring of Hydraulic Systems** (UCI ID 447): desde sensores brutos hasta un proxy de salud continuo por ciclo, pasando por EDA, feature engineering, detección de anomalías, clasificación supervisada por componente y construcción de un índice de degradación compuesto.

Cada notebook sigue el patrón **Decisión → Código → Interpretación**: toda celda de código está precedida por una celda que justifica qué se hace y por qué, y seguida de una celda que interpreta los resultados reales obtenidos al ejecutar — no resultados anticipados. Todas las cifras de este README están tomadas directamente de la ejecución de los notebooks (`jupyter nbconvert --execute`, 0 errores en los cinco).

## El dataset

2205 ciclos de 60 segundos de un banco de pruebas hidráulico, con 17 canales de sensores muestreados a 100, 10 o 1 Hz (presión, caudal, temperatura, potencia, eficiencia, vibración) y 4 variables objetivo independientes que describen el estado de cuatro componentes físicos:

| Componente | Variable | Niveles |
|---|---|---|
| Refrigerador | `cooler_condition` | 3 (cerca de fallo) / 20 (reducido) / 100 (óptimo) |
| Válvula | `valve_condition` | 73 / 80 / 90 / 100 (óptimo) |
| Bomba | `pump_leakage` | 0 (sin fuga) / 1 / 2 (fuga severa) |
| Acumulador | `accumulator_pressure` | 90 / 100 / 115 / 130 (óptimo) |

Los cuatro targets se combinan en un **diseño factorial de 144 celdas** (3×4×3×4), cruzadas de forma controlada — no es una serie temporal de degradación de una sola máquina, sino experimentos de laboratorio independientes. Esta propiedad estructural condiciona buena parte de las decisiones metodológicas del proyecto (ver Fase 3 y Limitaciones).

`ucimlrepo.fetch_ucirepo(id=447)` falla (`DatasetNotFoundError`) — el dataset no está habilitado para import programático vía esa librería pese a estar listado en el repositorio. [`src/data_loader.py`](src/data_loader.py) descarga el ZIP estático público directamente en su lugar; ver el docstring del módulo para el detalle de esa verificación.

## Estructura del proyecto

```
hydraulic_maintenance/
├── data/
│   ├── raw/hydraulic_raw/          # sensores crudos + profile.txt, descargados por data_loader.py
│   └── processed/                  # artefactos intermedios y finales de cada fase (ver tabla abajo)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_failure_prediction.ipynb
│   └── 05_rul_proxy.ipynb
├── src/
│   ├── data_loader.py               # descarga y carga del dataset crudo
│   ├── feature_engineering.py       # extracción de las 73 features y selección por η²
│   ├── models.py                    # entrenamiento/evaluación: detectores (Fase 3) y clasificadores (Fase 4)
│   └── visualization.py             # las 9 figuras del proyecto, refactorizadas como funciones puras
├── reports/figures/                # 9 figuras — ver tabla abajo
├── models/                         # reservado; sin modelos serializados (ver nota abajo)
├── requirements.txt
├── LICENSE
└── README.md
```

**Nota sobre `src/`:** los cuatro módulos están completos — [`data_loader.py`](src/data_loader.py), [`feature_engineering.py`](src/feature_engineering.py), [`models.py`](src/models.py) y [`visualization.py`](src/visualization.py) —, todos refactorizados desde lógica de notebook ya ejecutada con 0 errores, sin alterar ninguna fórmula, hiperparámetro ni criterio de selección respecto a lo validado en los notebooks. `models/` sigue vacío: los modelos de las Fases 3 y 4 se entrenan y evalúan dentro del propio notebook (o mediante `src/models.py`), pero ninguno se serializa a disco. No hay ningún `.pkl`/`.joblib` que cargar — para obtener un modelo entrenado hay que reentrenarlo ejecutando los notebooks en orden (o llamando a `train_classifier`/`train_mahalanobis_detector`/`train_isolation_forest_detector` de `src/models.py` sobre los artefactos de `data/processed/`), no recuperarlo de disco.

## Pipeline por fases

### Fase 1 — EDA ([`01_eda.ipynb`](notebooks/01_eda.ipynb))
Confirma el diseño factorial de 144 combinaciones, identifica que `cooler_condition` domina la varianza de la mayoría de sensores, y que la posición temporal del pico en `PS3` (no su nivel) es el indicador más informativo de degradación de la válvula. Detecta 2 ciclos atípicos (617, 371) mediante MAD condicionado por grupo, no un umbral global.

### Fase 2 — Feature engineering ([`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb))
Reduce cada ciclo de hasta 6000 lecturas por sensor a **73 features escalares** (medias, desviaciones, mitades temporales, diferenciales entre sensores, posición del pico), seleccionadas por η² frente a los 4 targets. Exporta `X_train`/`X_test` (1157/290 ciclos, split 80/20) y `y_train`/`y_test` a Parquet.

### Fase 3 — Detección de anomalías ([`03_anomaly_detection.ipynb`](notebooks/03_anomaly_detection.ipynb))
Muestra, con evidencia cuantitativa, que la detección no supervisada **falla en este dataset concreto**: el estado "los 4 componentes óptimos a la vez" es solo 1 de 144 combinaciones (8 ciclos en train, 0.69%) — la combinación *más rara* del diseño, no la más común, invirtiendo la premisa bajo la que operan estos métodos. Mahalanobis con regularización LedoitWolf alcanza AUC-ROC=1.0000 en test, pero por un artefacto de escalado (una sola feature con varianza casi nula en los 8 ciclos de referencia) ligado a `cooler_condition`, no por una distancia multivariante fiable. Isolation Forest obtiene AUC-ROC=**0.2743** — peor que el azar — porque detecta rareza estadística, y el estado sano es precisamente lo más raro del diseño.

### Fase 4 — Clasificación supervisada ([`04_failure_prediction.ipynb`](notebooks/04_failure_prediction.ipynb))
Entrena un clasificador independiente por componente (Random Forest / Regresión Logística), validado por CV de 5 folds y evaluado en test sellado:

| Target | Modelo elegido | Test F1-macro | IC95% (bootstrap, 1000 remuestras) | Δ vs. CV |
|---|---|---|---|---|
| `cooler_condition` | Random Forest | 1.0000 | [1.0000, 1.0000] | +0.0000 |
| `valve_condition` | Regresión Logística | 0.9965 | [0.9887, 1.0000] | -0.0018 |
| `pump_leakage` | Random Forest | 0.9931 | [0.9813, 1.0000] | -0.0017 |
| `accumulator_pressure` | Random Forest | 0.9863 | [0.9714, 0.9968] | +0.0018 |

*Los IC95% no estaban precalculados en `predictions_test.parquet` (solo contiene las predicciones puntuales) — se calcularon a partir de esas predicciones con bootstrap (`sklearn.utils.resample`, 1000 remuestras, percentiles 2.5/97.5). Nota relevante para la Sección "Fase 4" de arriba: con estos intervalos, el ranking RF-vs-LR de `valve_condition` y `pump_leakage` (Δ de CV <1pp) no es distinguible de ruido de muestreo — un test de McNemar pareado sobre las predicciones de test lo confirma (p≥0.5 en ambos, no significativo). Solo la brecha de `accumulator_pressure` es estadísticamente defendible.*

Hallazgo central: para `accumulator_pressure`, la importancia de features de Random Forest está **negativamente correlacionada** con su η² univariante de la Fase 2 (Spearman ρ=-0.3742) — las features más útiles para el modelo (`FS2_*`) tienen η²≈0.003, prácticamente nulo en aislamiento. Explica directamente por qué Regresión Logística cae a 82.28% F1-macro en ese mismo target (98.44% con RF): la señal es de interacción, no de desplazamiento lineal de medias, y ningún análisis univariante podía haberlo anticipado.

### Fase 5 — Proxy de RUL ([`05_rul_proxy.ipynb`](notebooks/05_rul_proxy.ipynb))
Combina las 4 predicciones de la Fase 4 en un índice de salud compuesto `H_mean ∈ [0,1]` (media de degradación normalizada por componente, complementada a 1). Validado internamente (decrece monótonamente con la degradación real del refrigerador: 0.627→0.502→0.369 según el nivel) y cruzadamente contra los scores de la Fase 3 (ρ=-0.567 con Mahalanobis, ρ=-0.190 con Isolation Forest — magnitudes coherentes con el AUC de cada detector en la Fase 3). Exporta `rul_proxy_test.parquet` como artefacto final del pipeline.

## Artefactos en `data/processed/`

| Archivo | Origen | Contenido |
|---|---|---|
| `X_train.parquet`, `X_test.parquet` | Fase 2 | 73 features por ciclo (1157 / 290 filas) |
| `y_train.parquet`, `y_test.parquet` | Fase 2 | 4 targets originales por ciclo |
| `eta2_features_vs_targets.csv` | Fase 2 | η² de cada feature candidata frente a cada target |
| `anomaly_scores_test.parquet` | Fase 3 | `maha2_test`, `anomaly_score_if_test` (test, 290 filas) |
| `predictions_test.parquet` | Fase 4 | Predicciones por componente en test + etiqueta de anomalía |
| `rul_proxy_test.parquet` | Fase 5 | Índice de salud `H_mean`/`H_max` y contribución por componente |

**`reports/figures/`** contiene las 9 figuras generadas por los cinco notebooks:

| Archivo | Fase |
|---|---|
| `01_target_distributions.png` | Fase 1 |
| `01_target_cooccurrence.png` | Fase 1 |
| `01_cycle_shapes_by_condition.png` | Fase 1 |
| `01_pressure_sensor_correlation.png` | Fase 1 |
| `02_eta2_heatmap.png` | Fase 2 |
| `03_roc_and_score_distribution.png` | Fase 3 |
| `04_confusion_matrices.png` | Fase 4 |
| `04_eta2_vs_importance_scatter.png` | Fase 4 |
| `05_health_index_distribution.png` | Fase 5 |

## Limitaciones metodológicas

1. **Sin dimensión temporal real.** Los 2205 ciclos son experimentos independientes, no una serie de degradación progresiva — el proxy de la Fase 5 ordena severidad relativa, no predice tiempo hasta el fallo.
2. **El diseño factorial equilibrado es una condición atípica para detección de anomalías.** Invalidó Mahalanobis y, sobre todo, Isolation Forest en la Fase 3, porque el estado sano es la combinación *menos* frecuente, no la más frecuente. Precisión: "equilibrado" se refiere al número de réplicas por celda (144/144 celdas representadas en train, 141 coinciden exactamente en 8 réplicas, rango real 7–15 — verificado, no es una afirmación aproximada) — **no** significa que cada celda tenga un peso proporcional al total. Con 144 celdas, ninguna puede representar más de ~0.7% del total por construcción: la combinación de los 4 componentes óptimos simultáneamente son 8 de 1157 ciclos de train (0.69%), pero eso no es una celda infrarrepresentada respecto a las demás — es la consecuencia aritmética inevitable de dividir la población en 144 partes iguales.
3. **El proxy de salud no tiene unidades de tiempo** y no puede traducirse a "días hasta el fallo" sin datos de tasa de progresión real que este dataset no contiene.
4. **Los resultados de clasificación dependen de una única semilla aleatoria (`random_state=42`).** La variabilidad inter-semilla no se ha evaluado dentro del pipeline documentado (notebooks/README) — no hay ninguna repetición con semillas distintas para confirmar que los deltas de CV o los rankings de modelo por target son estables frente a la elección concreta de semilla.

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# ... 02 a 05 en orden — cada fase consume artefactos exportados por la anterior
```

**Nota sobre `requirements.txt`:** el plan original del proyecto preveía `torch`, `xgboost`, `shap`, `optuna` e `imbalanced-learn` (en particular, un autoencoder en PyTorch para la Fase 3) — pero ningún notebook ni módulo de `src/` los importa en la implementación final: la Fase 3 sustituyó el autoencoder planeado por Isolation Forest al descubrir que solo hay 8 ciclos normales en train, insuficientes para entrenar una red densa sin sobreajuste severo (razonamiento documentado en la Decisión de esa sección). Esos 5 paquetes, junto con `joblib` (tampoco usado — ningún modelo se serializa), se **eliminaron** de `requirements.txt`. El proyecto usa exclusivamente `pandas`, `numpy`, `scipy` (añadido — faltaba pese a usarse en varios notebooks y en `src/feature_engineering.py`), `matplotlib`, `seaborn`, `scikit-learn`, `pyarrow` (motor parquet de pandas) y `jupyter` (entorno de ejecución); `ucimlrepo` se mantiene solo por la comprobación diagnóstica en `src/data_loader.py`.

## Datos y licencia

- **Dataset:** [Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
  (UCI Machine Learning Repository, ID 447; creado por ZeMA gGmbH / Universität des Saarlandes) —
  licencia **CC BY 4.0** (Creative Commons Attribution 4.0 International): permite compartir y
  adaptar con atribución. Los datos crudos (531 MB) no se incluyen en el repositorio — se descargan
  bajo demanda con `src/data_loader.py` (ver "Cómo ejecutar"). `data/processed/` sí se versiona:
  son los artefactos/resultados del pipeline (~1 MB), no una copia de los datos crudos.
- **Código de este repositorio:** MIT.
