# Hydraulic Systems Predictive Maintenance · Mantenimiento Predictivo de Sistemas Hidráulicos

🌐 **English version: [`english/README.md`](english/README.md)** — translation of this README; the
notebooks and code comments themselves are written in Spanish only (no separate `english/`
notebook mirror exists for this project — see the note at the top of the English version).

![Per-component degradation classification results — confusion matrices for cooler, valve, pump and accumulator](reports/figures/04_confusion_matrices.png)

---

Proyecto de ciencia de datos de extremo a extremo sobre el dataset **Condition Monitoring of Hydraulic Systems** (UCI ID 447): desde sensores brutos hasta un proxy de salud continuo por ciclo, pasando por EDA, feature engineering, detección de anomalías, clasificación supervisada por componente y construcción de un índice de degradación compuesto.

Cada notebook sigue el patrón **Decisión → Código → Interpretación**: toda celda de código está precedida por una celda que justifica qué se hace y por qué, y seguida de una celda que interpreta los resultados reales obtenidos al ejecutar — no resultados anticipados. Todas las cifras de este README están tomadas directamente de la ejecución de los notebooks (`jupyter nbconvert --execute`, 0 errores en los cinco).

### Resumen ejecutivo

Este proyecto entrena clasificadores que detectan la degradación de 4 componentes de un sistema hidráulico industrial (refrigerador, válvula, bomba, acumulador) a partir únicamente de datos de sensores — el tipo de sistema presente en fabricación, construcción y maquinaria pesada. Los mejores modelos clasifican correctamente el nivel de degradación de cada componente entre el **84% y el 100% de las veces sobre datos que el modelo nunca vio en entrenamiento** (ver "Generalización a puntos de operación nuevos" en la Fase 4 para entender por qué la respuesta honesta es un rango, no una única cifra titular). Como escenario ilustrativo y sin validar: si una fracción incluso modesta de las paradas no planificadas en una planta con este tipo de maquinaria fuera atribuible a estos 4 modos de fallo, detectarlos antes — en vez de descubrirlos tras la avería — es exactamente el caso de uso que este proyecto demuestra de principio a fin, desde la señal cruda del sensor hasta un clasificador validado. Este dataset público no incluye datos de coste, así que no se afirma aquí ninguna cifra real de ahorro; esa cifra tendría que salir de los registros de mantenimiento de una planta concreta.

### El dataset

2205 ciclos de 60 segundos de un banco de pruebas hidráulico, con 17 canales de sensores muestreados a 100, 10 o 1 Hz (presión, caudal, temperatura, potencia, eficiencia, vibración) y 4 variables objetivo independientes que describen el estado de cuatro componentes físicos:

| Componente | Variable | Niveles |
|---|---|---|
| Refrigerador | `cooler_condition` | 3 (cerca de fallo) / 20 (reducido) / 100 (óptimo) |
| Válvula | `valve_condition` | 73 / 80 / 90 / 100 (óptimo) |
| Bomba | `pump_leakage` | 0 (sin fuga) / 1 / 2 (fuga severa) |
| Acumulador | `accumulator_pressure` | 90 / 100 / 115 / 130 (óptimo) |

Los cuatro targets se combinan en un **diseño factorial de 144 celdas** (3×4×3×4), cruzadas de forma controlada — no es una serie temporal de degradación de una sola máquina, sino experimentos de laboratorio independientes. Esta propiedad estructural condiciona buena parte de las decisiones metodológicas del proyecto (ver Fase 3 y Limitaciones).

`ucimlrepo.fetch_ucirepo(id=447)` falla (`DatasetNotFoundError`) — el dataset no está habilitado para import programático vía esa librería pese a estar listado en el repositorio. [`src/data_loader.py`](src/data_loader.py) descarga el ZIP estático público directamente en su lugar; ver el docstring del módulo para el detalle de esa verificación.

### Estructura del proyecto

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
│   ├── validation.py                # Fase 4b: generalización agrupada, bootstrap/McNemar de ambos modelos,
│   │                                 # comprobaciones de class_weight y búsqueda de hiperparámetros (ver Fase 4)
│   └── visualization.py             # las 9 figuras del proyecto, refactorizadas como funciones puras
├── run_validation_fixes.py         # entrypoint de la Fase 4b — escribe data/processed/04b_*
├── reports/figures/                # 9 figuras — ver tabla abajo
├── models/                         # reservado; sin modelos serializados (ver nota abajo)
├── requirements.txt
├── LICENSE
├── README.md
└── english/README.md
```

**Nota sobre `src/`:** [`data_loader.py`](src/data_loader.py), [`feature_engineering.py`](src/feature_engineering.py), [`models.py`](src/models.py) y [`visualization.py`](src/visualization.py) están refactorizados desde lógica de notebook ya ejecutada con 0 errores, sin alterar ninguna fórmula, hiperparámetro ni criterio de selección respecto a lo validado en los notebooks 01-05. [`validation.py`](src/validation.py) es distinto por naturaleza: es análisis **nuevo** añadido en la Fase 4b (ver Fase 4 arriba), ejecutado vía [`run_validation_fixes.py`](run_validation_fixes.py) en vez de un notebook — no estaba presente en los 5 notebooks originales ni los altera. `models/` sigue vacío: ningún modelo del proyecto se serializa a disco. No hay ningún `.pkl`/`.joblib` que cargar — para obtener un modelo entrenado hay que reentrenarlo ejecutando los notebooks en orden (o llamando a `train_classifier`/`train_mahalanobis_detector`/`train_isolation_forest_detector` de `src/models.py`, o a las funciones de `src/validation.py`, sobre los artefactos de `data/processed/`), no recuperarlo de disco.

### Pipeline por fases

#### Fase 1 — EDA ([`01_eda.ipynb`](notebooks/01_eda.ipynb))
Confirma el diseño factorial de 144 combinaciones, identifica que `cooler_condition` domina la varianza de la mayoría de sensores, y que la posición temporal del pico en `PS3` (no su nivel) es el indicador más informativo de degradación de la válvula. Detecta 2 ciclos atípicos (617, 371) mediante MAD condicionado por grupo, no un umbral global.

#### Fase 2 — Feature engineering ([`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb))
Reduce cada ciclo de hasta 6000 lecturas por sensor a **73 features escalares** (medias, desviaciones, mitades temporales, diferenciales entre sensores, posición del pico), seleccionadas por η² frente a los 4 targets. Exporta `X_train`/`X_test` (1157/290 ciclos, split 80/20) y `y_train`/`y_test` a Parquet.

#### Fase 3 — Detección de anomalías ([`03_anomaly_detection.ipynb`](notebooks/03_anomaly_detection.ipynb))
Muestra, con evidencia cuantitativa, que la detección no supervisada **falla en este dataset concreto**: el estado "los 4 componentes óptimos a la vez" es solo 1 de 144 combinaciones (8 ciclos en train, 0.69%) — la combinación *más rara* del diseño, no la más común, invirtiendo la premisa bajo la que operan estos métodos. Mahalanobis con regularización LedoitWolf alcanza AUC-ROC=1.0000 en test, pero por un artefacto de escalado (una sola feature con varianza casi nula en los 8 ciclos de referencia) ligado a `cooler_condition`, no por una distancia multivariante fiable. Isolation Forest obtiene AUC-ROC=**0.2743** — peor que el azar — porque detecta rareza estadística, y el estado sano es precisamente lo más raro del diseño.

#### Fase 4 — Clasificación supervisada ([`04_failure_prediction.ipynb`](notebooks/04_failure_prediction.ipynb))
Entrena un clasificador independiente por componente (Random Forest / Regresión Logística), validado por CV de 5 folds y evaluado en test sellado:

| Target | Modelo elegido | Test F1-macro | IC95% (bootstrap, 1000 remuestras) |
|---|---|---|---|
| `cooler_condition` | Random Forest | 1.0000 | [1.0000, 1.0000] |
| `valve_condition` | Regresión Logística | 0.9965 | [0.9889, 1.0000] |
| `pump_leakage` | Random Forest | 0.9931 | [0.9825, 1.0000] |
| `accumulator_pressure` | Random Forest | 0.9863 | [0.9701, 0.9967] |

*Estos IC95% ahora los produce código real y reutilizable (`src/validation.py::mcnemar_bootstrap_report`, `sklearn.utils.resample`, 1000 remuestras, percentiles 2.5/97.5), ejecutado por `run_validation_fixes.py`. Una versión anterior de este README describía este cálculo exacto en prosa sin publicar el código que lo produce, y `predictions_test.parquet` solo guardaba las predicciones del modelo ganador por target — así que la cifra no se podía regenerar realmente desde el repositorio. `data/processed/04b_predictions_test_dual_model.parquet` guarda ahora las predicciones de ambos modelos para los 4 targets, y `04b_mcnemar_bootstrap.csv` la comparación completa de abajo.*

Un test de McNemar pareado (sobre acierto/error por ciclo, `statsmodels.stats.contingency_tables.mcnemar`) confirma: para `valve_condition` (1 par discordante en cada dirección, p=1.0000) y `pump_leakage` (2 vs. 0, p=0.5000) la brecha RF-vs-LR no es distinguible de ruido de muestreo, tal como ya afirmaba una versión anterior de este README — ahora respaldado por código ejecutable y reproducible, no solo por prosa. Para `accumulator_pressure` la brecha es real y grande: RF 0.9863 vs. **LR 0.8602** (39 vs. 3 pares discordantes, p=6.6×10⁻⁸) — y, aplicando corrección FDR de Benjamini-Hochberg sobre los 4 targets testeados (`multipletests`, α=0.05), sigue siendo significativa (p_FDR=2.7×10⁻⁷). Una versión anterior de este README ya afirmaba que el resultado de `accumulator_pressure` "sobrevive tras corrección FDR" sin que existiera código que calculara esa corrección; sí se sostiene, y ahora hay código que lo demuestra.

**Nota de reproducibilidad:** una versión anterior de este README citaba 82.28% para el F1-macro de test de Regresión Logística en `accumulator_pressure`; recalculándolo bajo el entorno fijado en `requirements.txt` (misma definición de modelo — `solver="lbfgs"`, `multi_class="multinomial"`, mismo `random_state=42`) da 86.02%, una diferencia de ~4pp. La causa más probable es deriva de versión de scikit-learn en ese camino de código ya deprecado (`multi_class="multinomial"` se elimina en scikit-learn 1.7). En cualquier caso la conclusión sustantiva no cambia — si acaso se refuerza, ya que 86.02% sigue muy por debajo del 98.63% de RF y el p-valor de McNemar no deja ambigüedad.

Hallazgo central: para `accumulator_pressure`, la importancia de features de Random Forest está **negativamente correlacionada** con su η² univariante de la Fase 2 (Spearman ρ=-0.3742) — las features más útiles para el modelo (`FS2_*`) tienen η²≈0.003, prácticamente nulo en aislamiento. Explica directamente por qué Regresión Logística puntúa mucho más bajo (86.02% F1-macro en test, frente a 98.63% con RF): la señal es de interacción, no de desplazamiento lineal de medias, y ningún análisis univariante podía haberlo anticipado.

##### Generalización a puntos de operación nuevos (no solo ciclos retenidos)

El split 80/20 (Fase 2, `strat_key`) estratifica por la *combinación completa* de los 4 targets — es decir, por el propio diseño factorial de 144 celdas. Con solo 144 celdas y 7-15 réplicas casi idénticas cada una (mismo punto de operación físico, solo difiere el ruido de sensor), **toda celda presente en test también está presente en train**: el modelo ya ha visto ciclos del mismo punto de operación exacto sobre el que se evalúa, solo que no ese ciclo literal. Es una pregunta más estrecha y optimista que "¿generaliza esto a un punto de operación nunca visto en entrenamiento?" — la pregunta relevante en la práctica para un sistema desplegado sobre equipo nuevo o una combinación nueva de estados de componente.

Reevaluando con `GroupKFold` (`src/validation.py::grouped_generalization_report`, `run_validation_fixes.py`), agrupando por esa misma clave de 144 celdas para que **ninguna celda aparezca a ambos lados de ningún fold**, el panorama cambia de forma sustancial para 2 de los 4 targets:

| Target | Test F1-macro estratificado (interpolación) | F1-macro GroupKFold, k=5 (punto de operación nuevo) | Brecha |
|---|---|---|---|
| `cooler_condition` | 1.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `valve_condition` | 0.9965 | 0.9938 ± 0.0077 | 0.0027 |
| `pump_leakage` | 0.9931 | 0.9658 ± 0.0320 | 0.0273 |
| `accumulator_pressure` | 0.9863 | **0.8421 ± 0.0410** | **0.1442** |

`cooler_condition` y `valve_condition` apenas se mueven — su señal es casi global/lineal en el espacio de sensores, así que haber visto "una celda parecida" no aporta gran cosa. `pump_leakage` pierde ~2.7pp. `accumulator_pressure` pierde **14.4pp**: es precisamente el target cuya señal en Random Forest, según el hallazgo central de arriba, viene de *interacciones* entre features y no de desplazamientos univariantes — una señal basada en interacciones es exactamente el tipo que un ensemble de árboles puede memorizar en parte por punto de operación en lugar de aprender como una regla que transfiere a uno nuevo, y por eso este target es también el más expuesto por esta corrección. Ambas cifras se reportan aquí a propósito: el F1 de test estratificado no está mal calculado, responde a una pregunta real (aunque más estrecha) — pero no es evidencia de generalización a una combinación genuinamente nueva de estados de componente — y una versión anterior de este README no hacía esa distinción.

##### Comprobaciones de robustez: baseline trivial y búsqueda de hiperparámetros

Dos huecos que un revisor técnico preguntaría primero, ambos cerrados vía `run_validation_fixes.py`:

- **Baseline trivial** (`DummyClassifier`, `data/processed/04b_baseline_dummy.csv`): `most_frequent` puntúa F1-macro 0.10-0.17 en los 4 targets, `stratified` puntúa 0.26-0.33 — ambos muy por debajo del 0.84-1.00 que alcanzan los modelos reales, confirmando que las cifras logradas no son un artefacto de un target fácil/desbalanceado que el dummy ya pudiera explotar.
- **Búsqueda de hiperparámetros** (`RandomizedSearchCV`, CV de 5 folds solo sobre train, 30 iteraciones, `class_weight` incluido en el espacio de búsqueda — `data/processed/04b_hyperparameter_search.csv`): ningún notebook ni `src/models.py` había ajustado hiperparámetros nunca (solo `n_estimators=100`/`max_iter=1000` se fijaron explícitamente, el resto quedó en los valores por defecto de scikit-learn). La búsqueda encuentra `cooler_condition` y `valve_condition` ya en su techo (sin mejora posible); `pump_leakage` mejora de 0.9948 a 0.9965 F1-macro de CV, `accumulator_pressure` de 0.9845 a 0.9871, ambos vía `class_weight="balanced"` combinado con `min_samples_leaf=1` y más árboles (`n_estimators=256`). Las mejoras son reales pero pequeñas (<0.3pp) — el modelo ya publicado (valores por defecto sin ajustar, documentado arriba) se mantiene como resultado principal por simplicidad, y esta búsqueda se reporta como comprobación de robustez/margen de mejora, no como reemplazo.
- **`class_weight="balanced"` en aislamiento** (`data/processed/04b_class_weight_comparison.csv`): el EDA de las Fases 1/2 lo recomienda dos veces como precaución ante el desbalance de clases, pero ningún modelo de `src/models.py` llegó a aplicarlo. Probarlo en aislamiento (manteniendo el resto de hiperparámetros en su valor actual por defecto) da un resultado mixto y despreciable: sin cambio en `cooler_condition`/`valve_condition`, +0.09pp en `pump_leakage`, **-0.17pp en `accumulator_pressure`**. En aislamiento no ayuda; solo resulta mínimamente útil como parte de una configuración más amplia ya ajustada (ver la búsqueda de hiperparámetros de arriba, donde se elige junto con otros cambios para 2 de 4 targets). La recomendación del EDA se puso a prueba en vez de aplicarse por fe — la conclusión honesta es "no compensa por sí sola, levemente útil combinada con tuning", no "aplicarla siempre".

#### Fase 5 — Proxy de RUL ([`05_rul_proxy.ipynb`](notebooks/05_rul_proxy.ipynb))
Combina las 4 predicciones de la Fase 4 en un índice de salud compuesto `H_mean ∈ [0,1]` (media de degradación normalizada por componente, complementada a 1). Validado internamente (decrece monótonamente con la degradación real del refrigerador: 0.627→0.502→0.369 según el nivel) y cruzadamente contra los scores de la Fase 3 (ρ=-0.567 con Mahalanobis, ρ=-0.190 con Isolation Forest — magnitudes coherentes con el AUC de cada detector en la Fase 3). Exporta `rul_proxy_test.parquet` como artefacto final del pipeline.

### Artefactos en `data/processed/`

| Archivo | Origen | Contenido |
|---|---|---|
| `X_train.parquet`, `X_test.parquet` | Fase 2 | 73 features por ciclo (1157 / 290 filas) |
| `y_train.parquet`, `y_test.parquet` | Fase 2 | 4 targets originales por ciclo |
| `eta2_features_vs_targets.csv` | Fase 2 | η² de cada feature candidata frente a cada target |
| `anomaly_scores_test.parquet` | Fase 3 | `maha2_test`, `anomaly_score_if_test` (test, 290 filas) |
| `predictions_test.parquet` | Fase 4 | Predicciones por componente en test + etiqueta de anomalía |
| `rul_proxy_test.parquet` | Fase 5 | Índice de salud `H_mean`/`H_max` y contribución por componente |
| `04b_grouped_generalization.csv` | Fase 4b | F1-macro estratificado vs. `GroupKFold` por target (ver "Generalización a puntos de operación nuevos") |
| `04b_predictions_test_dual_model.parquet` | Fase 4b | Predicciones de RF y LR en test, para los 4 targets (input de McNemar/bootstrap) |
| `04b_mcnemar_bootstrap.csv` | Fase 4b | IC bootstrap + test de McNemar, RF vs. LR, por target |
| `04b_class_weight_comparison.csv` | Fase 4b | F1-macro de CV con/sin `class_weight="balanced"`, en aislamiento |
| `04b_baseline_dummy.csv` | Fase 4b | F1-macro de test de `DummyClassifier` (`most_frequent`/`stratified`), por target |
| `04b_hyperparameter_search.csv` | Fase 4b | Resultados de `RandomizedSearchCV` (F1-macro de CV default vs. tuned, mejores params) por target |

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

### Limitaciones metodológicas

1. **Sin dimensión temporal real.** Los 2205 ciclos son experimentos independientes, no una serie de degradación progresiva — el proxy de la Fase 5 ordena severidad relativa, no predice tiempo hasta el fallo.
2. **El diseño factorial equilibrado es una condición atípica para detección de anomalías.** Invalidó Mahalanobis y, sobre todo, Isolation Forest en la Fase 3, porque el estado sano es la combinación *menos* frecuente, no la más frecuente. Precisión: "equilibrado" se refiere al número de réplicas por celda (144/144 celdas representadas en train, 141 coinciden exactamente en 8 réplicas, rango real 7–15 — verificado, no es una afirmación aproximada) — **no** significa que cada celda tenga un peso proporcional al total. Con 144 celdas, ninguna puede representar más de ~0.7% del total por construcción: la combinación de los 4 componentes óptimos simultáneamente son 8 de 1157 ciclos de train (0.69%), pero eso no es una celda infrarrepresentada respecto a las demás — es la consecuencia aritmética inevitable de dividir la población en 144 partes iguales.
3. **El proxy de salud no tiene unidades de tiempo** y no puede traducirse a "días hasta el fallo" sin datos de tasa de progresión real que este dataset no contiene.
4. **Los resultados de clasificación dependen de una única semilla aleatoria (`random_state=42`).** La variabilidad inter-semilla no se ha evaluado dentro del pipeline documentado (notebooks/README) — no hay ninguna repetición con semillas distintas para confirmar que los deltas de CV o los rankings de modelo por target son estables frente a la elección concreta de semilla.
5. **El test sellado es menos independiente de lo que sugiere una única cifra de F1-macro.** El split 80/20 estratifica por la combinación completa de los 4 targets (el diseño factorial de 144 celdas), así que toda celda presente en test tiene también réplicas casi idénticas en train. Para 2 de los 4 targets apenas importa; para `accumulator_pressure` el F1-macro de test (0.9863) es ~14pp más optimista que una evaluación `GroupKFold` que retiene puntos de operación completos (0.8421 ± 0.0410) — ver "Generalización a puntos de operación nuevos" en la Fase 4 para la comparación completa y por qué este target en concreto es el más expuesto.

### Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# ... 02 a 05 en orden — cada fase consume artefactos exportados por la anterior
python run_validation_fixes.py   # Fase 4b — lee data/processed/{X,y}_{train,test}.parquet de la
                                   # Fase 2, escribe data/processed/04b_*; ~6 minutos (la búsqueda de
                                   # hiperparámetros explica la mayor parte de ese tiempo)
```

**Nota sobre `requirements.txt`:** el plan original del proyecto preveía `torch`, `xgboost`, `shap`, `optuna` e `imbalanced-learn` (en particular, un autoencoder en PyTorch para la Fase 3) — pero ningún notebook ni módulo de `src/` los importa en la implementación final: la Fase 3 sustituyó el autoencoder planeado por Isolation Forest al descubrir que solo hay 8 ciclos normales en train, insuficientes para entrenar una red densa sin sobreajuste severo (razonamiento documentado en la Decisión de esa sección). Esos 5 paquetes, junto con `joblib` (tampoco usado — ningún modelo se serializa), se **eliminaron** de `requirements.txt`. El proyecto usa exclusivamente `pandas`, `numpy`, `scipy` (añadido — faltaba pese a usarse en varios notebooks y en `src/feature_engineering.py`), `matplotlib`, `seaborn`, `scikit-learn`, `pyarrow` (motor parquet de pandas) y `jupyter` (entorno de ejecución); `ucimlrepo` se mantiene solo por la comprobación diagnóstica en `src/data_loader.py`.

### Datos y licencia

- **Dataset:** [Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
  (UCI Machine Learning Repository, ID 447; creado por ZeMA gGmbH / Universität des Saarlandes) —
  licencia **CC BY 4.0** (Creative Commons Attribution 4.0 International): permite compartir y
  adaptar con atribución. Los datos crudos (531 MB) no se incluyen en el repositorio — se descargan
  bajo demanda con `src/data_loader.py` (ver "Cómo ejecutar"). `data/processed/` sí se versiona:
  son los artefactos/resultados del pipeline (~1 MB), no una copia de los datos crudos.
- **Código de este repositorio:** MIT.
