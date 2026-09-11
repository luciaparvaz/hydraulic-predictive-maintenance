"""Visualizaciones del proyecto UCI Hydraulic System (ID 447), refactorizadas desde
las celdas de gráficos ya ejecutadas y validadas en los cinco notebooks del pipeline.

Estado real de cada notebook en el momento de este refactor: **solo `01_eda.ipynb`
contenía figuras** (4, todas exportadas a `reports/figures/`) — `02`, `03`, `04` y
`05` importaban `matplotlib`/`seaborn` pero nunca los usaban; toda su Decisión/
Interpretación se comunicaba con tablas de texto. Antes de escribir este módulo se
añadieron las secciones de visualización correspondientes a cada uno de esos cuatro
notebooks (Decisión → Código → Interpretación, ejecutadas con 0 errores, cada una
guardando su figura en `reports/figures/`), y este módulo extrae esa lógica ya
ejecutada — no inventa ninguna visualización que no se haya corrido primero en un
notebook.

Convención común a todas las funciones: `ax=None` (o `axes=None` para las que
producen varios paneles a la vez, ver más abajo) crea su propia figura; si se pasa
un `ax`/`axes` ya existente, la función dibuja sobre él sin crear una figura nueva
— así se pueden componer en un dashboard multi-panel. `save_path`, si no es `None`,
guarda con `fig.savefig(save_path, bbox_inches="tight", dpi=150)`. Ninguna función
llama a `plt.show()` — eso lo decide quien la use.

Nota sobre multi-panel: cuatro de las figuras originales (`plot_target_distributions`,
`plot_cycle_shapes_by_condition`) están compuestas, en el notebook, de varios paneles
independientes (uno por target o por sensor) dentro de una sola figura — no un único
gráfico. Para esas, `axes` acepta una lista/array de Axes (una por panel); si es
`None`, la función crea internamente la grilla que el notebook usó (1×4 o 1×3) en
vez de un único `Axes`.

Bloques:
    1. EDA (`plot_target_distributions`, `plot_target_cooccurrence`,
       `plot_cycle_shapes_by_condition`, `plot_pressure_sensor_correlation`).
    2. Feature engineering (`plot_eta2_heatmap`).
    3. Detección de anomalías (`plot_roc_curves`, `plot_anomaly_score_distribution`).
    4. Clasificación supervisada (`plot_confusion_matrix`, `plot_importance_vs_eta2`).
    5. Proxy de RUL (`plot_health_index_distribution`, `plot_health_by_degradation_level`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

COMPONENT_TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]

__all__ = [
    "plot_target_distributions",
    "plot_target_cooccurrence",
    "plot_cycle_shapes_by_condition",
    "plot_pressure_sensor_correlation",
    "plot_eta2_heatmap",
    "plot_roc_curves",
    "plot_anomaly_score_distribution",
    "plot_confusion_matrix",
    "plot_importance_vs_eta2",
    "plot_health_index_distribution",
    "plot_health_by_degradation_level",
]


def _finish(fig, save_path):
    """Guarda la figura si se pidió `save_path`. No llama a plt.show() — lo decide el llamador."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)


# ── Bloque 1 — EDA (01_eda.ipynb) ────────────────────────────────────────────────


def plot_target_distributions(y: pd.DataFrame, targets: list[str] = COMPONENT_TARGETS, axes=None, save_path=None):
    """Distribución de clases (nº de ciclos) de cada variable objetivo, una barra por nivel.

    Reproduce `reports/figures/01_target_distributions.png`: una grilla 1×N (N=len(targets))
    de gráficos de barras, cada uno con el conteo de ciclos por nivel de un target,
    ordenado por valor de etiqueta (no por frecuencia) y con el número exacto anotado
    sobre cada barra — más preciso que leer la altura a ojo.

    Args:
        y: DataFrame de targets, una fila por ciclo (debe contener las columnas de `targets`).
        targets: nombres de las columnas a graficar, una por panel. Por defecto, los
            4 targets de componente.
        axes: array/lista de `len(targets)` Axes ya existentes para dibujar sobre
            ellos, o `None` para crear una figura nueva de 1×`len(targets)` paneles.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, axes): la figura (`None` si se pasaron `axes` ya existentes) y el
        array de Axes usado.
    """
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4))

    for ax, col in zip(np.atleast_1d(axes), targets):
        counts = y[col].value_counts().sort_index()
        counts.plot(kind="bar", ax=ax, color="#2980b9")
        ax.set_title(col)
        ax.set_xlabel("")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 15, str(v), ha="center", fontsize=8)

    if fig is not None:
        fig.suptitle("Distribución de cada variable objetivo (número de ciclos por clase)", y=1.05)
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, axes


def plot_target_cooccurrence(y: pd.DataFrame, targets: list[str] = COMPONENT_TARGETS, ax=None, save_path=None):
    """Heatmap de correlación de Spearman entre las variables objetivo de componente.

    Reproduce `reports/figures/01_target_cooccurrence.png`. Se usa Spearman, no
    Pearson: los niveles de cada target son ordinales (representan grados de
    degradación) pero no están necesariamente espaciados de forma lineal, y Spearman
    solo asume monotonía, no proporcionalidad entre los valores.

    Args:
        y: DataFrame de targets, una fila por ciclo.
        targets: columnas a incluir en la matriz de correlación.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    corr = y[targets].corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlación de Spearman entre las 4 variables objetivo de componente")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


def plot_cycle_shapes_by_condition(sensors: dict[str, np.ndarray], y: pd.DataFrame, sensor_freqs_hz: dict[str, int], axes=None, save_path=None):
    """Forma media del ciclo (± 1 std) de 3 pares sensor/target elegidos en el EDA.

    Reproduce `reports/figures/01_cycle_shapes_by_condition.png`: 3 paneles fijos —
    VS1 (vibración) por `pump_leakage`, TS3 (temperatura del circuito de refrigeración)
    por `cooler_condition`, y PS3 (presión del lado activo del cilindro) por
    `valve_condition` — los 3 casos de estudio que el EDA identificó como más
    reveladores de la forma en que cada componente se manifiesta en su sensor más
    relacionado. No se generaliza a otros pares sensor/target: son exactamente los
    3 que el notebook decidió graficar, ni más ni menos.

    Para cada nivel del target correspondiente, dibuja la media punto a punto del
    ciclo (agregando sobre todos los ciclos de ese nivel) con una banda de ±1
    desviación estándar alrededor.

    Args:
        sensors: dict sensor -> array (n_ciclos, n_lecturas), formato de
            `data_loader.load_hydraulic_dataset()`. Debe incluir "VS1", "TS3", "PS3".
        y: DataFrame de targets, mismo número de filas y orden que los arrays de `sensors`.
        sensor_freqs_hz: dict sensor -> frecuencia de muestreo en Hz (p.ej.
            `data_loader.SENSOR_FREQUENCIES_HZ`), para expresar el eje temporal en segundos.
        axes: array/lista de 3 Axes ya existentes, o `None` para crear una figura nueva de 1×3.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, axes): la figura (`None` si se pasaron `axes` ya existentes) y el
        array de 3 Axes usado.
    """
    panels = [
        ("VS1", "pump_leakage", "VS1 (vibración) por pump_leakage"),
        ("TS3", "cooler_condition", "TS3 (temp. circuito refrigeración) por cooler_condition"),
        ("PS3", "valve_condition", "PS3 (presión lado activo cilindro) por valve_condition"),
    ]

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (sensor_name, target_col, title) in zip(np.atleast_1d(axes), panels):
        arr = sensors[sensor_name]
        time_axis = np.arange(arr.shape[1]) / sensor_freqs_hz[sensor_name]
        for level in sorted(y[target_col].unique()):
            mask = (y[target_col] == level).values
            mean_curve = arr[mask].mean(axis=0)
            std_curve = arr[mask].std(axis=0)
            ax.plot(time_axis, mean_curve, label=f"{target_col}={level} (n={mask.sum()})")
            ax.fill_between(time_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        ax.set_xlabel("Tiempo dentro del ciclo (s)")
        ax.set_ylabel(sensor_name)
        ax.set_title(title)
        ax.legend(fontsize=7)

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, axes


def plot_pressure_sensor_correlation(sensors: dict[str, np.ndarray], ax=None, save_path=None):
    """Heatmap de correlación de Pearson entre las medias por ciclo de PS1-PS6.

    Reproduce `reports/figures/01_pressure_sensor_correlation.png`. Usa la media
    por ciclo de cada uno de los 6 sensores de presión (no las lecturas crudas) como
    variable de entrada a la correlación — reduce cada ciclo a un escalar por sensor
    antes de correlacionar, igual que el notebook.

    Args:
        sensors: dict sensor -> array (n_ciclos, n_lecturas). Debe incluir PS1-PS6.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ps_cols = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"]
    cycle_means = pd.DataFrame({name: sensors[name].mean(axis=1) for name in ps_cols})
    ps_corr = cycle_means.corr()

    sns.heatmap(ps_corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlación entre presiones PS1-PS6 (media por ciclo)")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


# ── Bloque 2 — Feature engineering (02_feature_engineering.ipynb) ──────────────


def plot_eta2_heatmap(eta2_df: pd.DataFrame, top_n: int = 15, targets: list[str] = COMPONENT_TARGETS, ax=None, save_path=None):
    """Heatmap de η² de las `top_n` features más informativas, por target.

    Reproduce `reports/figures/02_eta2_heatmap.png`. Selecciona las `top_n` features
    con mayor η² máximo entre los `targets` (no las `top_n` de cada target por
    separado) — el mismo criterio que usó el notebook. Nota documentada en esa
    sección: este criterio favorece a targets con señal fuerte Y redundante entre
    muchas columnas (p.ej. `cooler_condition`) frente a targets con una señal
    igualmente fuerte pero concentrada en una sola feature (p.ej. `valve_condition`
    vía `PS3_peak_s`) — esta última puede no aparecer en el top-`top_n` aunque sea
    muy informativa, y esa es una limitación real del método, no un error del gráfico.

    Args:
        eta2_df: DataFrame de η², una fila por feature y una columna por target
            (p.ej. `data/processed/eta2_features_vs_targets.csv`, ya cargado).
        top_n: número de features a mostrar, ordenadas por η² máximo descendente.
        targets: columnas de `eta2_df` a considerar para el ranking y a mostrar.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 8))

    top_features = eta2_df[targets].max(axis=1).sort_values(ascending=False).head(top_n).index.tolist()
    eta2_top = eta2_df.loc[top_features, targets]

    sns.heatmap(eta2_top, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "η²"})
    ax.set_title(f"η² de las {top_n} features más informativas, por target")
    ax.set_xlabel("")
    ax.set_ylabel("")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


# ── Bloque 3 — Detección de anomalías (03_anomaly_detection.ipynb) ─────────────


def plot_roc_curves(y_true, scores_dict: dict[str, np.ndarray], ax=None, save_path=None):
    """Curvas ROC de uno o más detectores, superpuestas, con el AUC en la leyenda.

    Reproduce el panel A de `reports/figures/03_roc_and_score_distribution.png`.
    Recibe únicamente arrays de scores y la etiqueta verdadera — no reentrena ni
    recibe ningún objeto de modelo; `roc_curve`/`roc_auc_score` se calculan aquí
    porque son, en sí mismos, la visualización (no un recómputo del detector).

    Args:
        y_true: etiqueta binaria verdadera (1 = anómalo, 0 = normal), un valor por ciclo.
        scores_dict: dict nombre_del_detector -> array de scores continuos (más alto
            = más anómalo), del mismo largo que `y_true`. P.ej.
            `{"Mahalanobis": maha2_test, "Isolation Forest": anomaly_score_test}`.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))

    colors = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]
    for (name, scores), color in zip(scores_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC")
    ax.legend(fontsize=8, loc="lower right")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


def plot_anomaly_score_distribution(scores, y_true, score_name: str = "anomaly_score", ax=None, save_path=None):
    """Distribución de un score de anomalía, separada por el ground truth real.

    Reproduce el panel B de `reports/figures/03_roc_and_score_distribution.png`.
    Un detector bien calibrado debería mostrar la curva de densidad de los ciclos
    normales desplazada hacia valores más bajos que la de los anómalos; que ambas
    se solapen (o estén invertidas) es, en sí mismo, un resultado a reportar, no un
    fallo del gráfico.

    Args:
        scores: array de scores continuos de anomalía (más alto = más anómalo).
        y_true: etiqueta binaria verdadera (1 = anómalo, 0 = normal), mismo largo que `scores`.
        score_name: nombre del score, usado como etiqueta del eje x.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))

    plot_df = pd.DataFrame({
        score_name: scores,
        "es_anomalo": pd.Series(np.asarray(y_true)).map({0: "Normal (0)", 1: "Anómalo (1)"}),
    })
    sns.histplot(data=plot_df, x=score_name, hue="es_anomalo", element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(f"Distribución de {score_name} por ground truth real")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


# ── Bloque 4 — Clasificación supervisada (04_failure_prediction.ipynb) ─────────


def plot_confusion_matrix(y_true, y_pred, target_name: str, ax=None, save_path=None):
    """Matriz de confusión de un target, con las clases ordenadas ascendentemente.

    Reproduce un panel de `reports/figures/04_confusion_matrices.png`. Las clases
    se etiquetan con sus valores originales (p.ej. 73/80/90/100 para
    `valve_condition`), no con índices 0..k, para que los ejes se lean directamente
    en la escala física del target.

    Args:
        y_true: etiquetas reales, un valor por ciclo.
        y_pred: predicciones del modelo, mismo largo y mismo dominio de clases que `y_true`.
        target_name: nombre del target, usado como título del panel.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    class_names = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(target_name)

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


def plot_importance_vs_eta2(importances: pd.Series, eta2_series: pd.Series, target_name: str, rho: float, ax=None, save_path=None):
    """Dispersión de η² (Fase 2) vs. importancia de Random Forest (Fase 4), por feature.

    Reproduce un panel de `reports/figures/04_eta2_vs_importance_scatter.png`. Cada
    punto es una feature; superpone una recta de mínimos cuadrados (OLS) como
    referencia visual de tendencia lineal — una ayuda gráfica, no lo que mide `rho`
    (correlación de rangos, no de magnitud). El valor de `rho` se anota en el
    título; esta función no lo recalcula, debe venir ya calculado (p.ej. de
    `scipy.stats.spearmanr`).

    Args:
        importances: importancia de Random Forest por feature (Series indexada por
            nombre de feature).
        eta2_series: η² de cada feature frente a `target_name` (Series con el mismo
            índice que `importances`, o alineable a él).
        target_name: nombre del target, usado en el título.
        rho: coeficiente de Spearman ya calculado entre `importances` y `eta2_series`,
            anotado en el título.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    eta2_aligned = eta2_series.reindex(importances.index)

    ax.scatter(eta2_aligned, importances, alpha=0.6, s=25, color="#2980b9")
    slope, intercept = np.polyfit(eta2_aligned, importances, 1)
    x_line = np.array([eta2_aligned.min(), eta2_aligned.max()])
    ax.plot(x_line, slope * x_line + intercept, color="#c0392b", linestyle="--", label="Recta OLS (referencia visual)")
    ax.set_xlabel("η² (Fase 2)")
    ax.set_ylabel("Importancia RF")
    ax.set_title(f"{target_name} — Spearman ρ={rho:.4f}")
    ax.legend(fontsize=8)

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


# ── Bloque 5 — Proxy de RUL (05_rul_proxy.ipynb) ────────────────────────────────


def plot_health_index_distribution(h_mean: pd.Series, bins: int = 23, ax=None, save_path=None):
    """Histograma de la distribución del índice de salud compuesto.

    Reproduce el panel A de `reports/figures/05_health_index_distribution.png`.
    `bins=23` por defecto porque, en el test set del proyecto, `H_mean_pred` solo
    puede tomar 23 valores distintos (Sección 1 de la Fase 5: el índice se
    construye promediando 4 componentes con niveles discretos) — se expone como
    parámetro para que quien reutilice esta función con otro conjunto de datos
    pueda ajustarlo a su propia granularidad.

    Args:
        h_mean: Serie del índice de salud (p.ej. `H_mean_pred`), un valor por ciclo.
        bins: número de bins del histograma.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))

    sns.histplot(h_mean, bins=bins, ax=ax, color="#2980b9")
    ax.set_xlabel(h_mean.name or "H_mean")
    ax.set_title("Distribución del índice de salud")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax


def plot_health_by_degradation_level(h_mean: pd.Series, degradation_level: pd.Series, ax=None, save_path=None):
    """Media ± 1 std del índice de salud, agrupada por nivel de degradación real.

    Reproduce el panel B de `reports/figures/05_health_index_distribution.png` —
    la validación monotónica del proxy: a mayor nivel de degradación real de un
    componente, el índice de salud debería ser, en promedio, menor. Las barras de
    error se calculan aquí (agrupando internamente), no se reciben ya agregadas —
    para eso solo hacen falta las dos Series de entrada.

    Args:
        h_mean: Serie del índice de salud (p.ej. `H_mean_pred`), un valor por ciclo.
        degradation_level: Serie del nivel de degradación real (ordinal, p.ej.
            `deg_true_cooler_condition`), mismo índice/orden que `h_mean`.
        ax: Axes ya existente para dibujar sobre él, o `None` para crear una figura nueva.
        save_path: ruta donde guardar la figura, o `None` para no guardar.

    Returns:
        (fig, ax): la figura (`None` si se pasó un `ax` ya existente) y el Axes usado.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))

    grouped = pd.DataFrame({"h": h_mean.values, "level": degradation_level.values}).groupby("level")["h"].agg(["mean", "std"])

    ax.bar(grouped.index.astype(str), grouped["mean"], yerr=grouped["std"], capsize=6, color="#27ae60")
    ax.set_xlabel(degradation_level.name or "Nivel de degradación real")
    ax.set_ylabel(f"{h_mean.name or 'H_mean'} (media ± 1 std)")
    ax.set_title("Validación monotónica del índice de salud")

    if fig is not None:
        fig.tight_layout()
        _finish(fig, save_path)

    return fig, ax
