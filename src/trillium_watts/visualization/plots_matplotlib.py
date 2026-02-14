"""Matplotlib/seaborn visualization functions for EDA and model analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_missing_values(
    series: pd.Series,
    imputed_series: pd.Series | None = None,
    title: str = "Serie ACTIVA con faltantes",
    min_gap: int = 5,
) -> None:
    """Plot a time series highlighting long stretches of missing values."""
    plt.figure(figsize=(12, 5))
    sns.set_style("whitegrid")

    plt.plot(series.index, series.values, marker="o", linestyle="-", color="#2a9d8f", label="ACTIVA original")

    is_nan = series.isna()
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    nan_ranges = series[is_nan].groupby(nan_groups).apply(lambda x: (x.index[0], x.index[-1]))

    for start, end in nan_ranges:
        gap_length = (end - start).days + 1
        if gap_length >= min_gap:
            plt.axvspan(start, end, color="red", alpha=0.2, label="Intervalos (NaN)")

    if imputed_series is not None:
        imputados = series.isna() & imputed_series.notna()
        plt.plot(imputed_series.index[imputados], imputed_series[imputados], "o", color="orange", label="Valores imputados")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Valor", fontsize=12)
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Matriz de Correlacion",
) -> None:
    """Plot a lower-triangle correlation heatmap."""
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 8},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 60) -> None:
    """Plot autocorrelation and partial autocorrelation functions."""
    plot_acf(series, lags=lags)
    plt.show()
    plot_pacf(series, lags=lags)
    plt.show()


def plot_boxplots(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Plot boxplots for numeric columns to visualize outlier distribution."""
    sns.set_style("whitegrid")
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color="#00b4d8", fliersize=3, linewidth=1.5)
        plt.title(f"Distribucion y outliers en '{col}'", fontweight="bold")
        plt.xlabel(col)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


def plot_outlier_detection(
    original: pd.Series,
    cleaned: pd.Series,
    outlier_mask: pd.Series,
    column_name: str,
) -> None:
    """Plot original vs interpolated values with outlier positions highlighted."""
    sns.set(style="whitegrid")
    df_plot = pd.DataFrame(
        {
            "Tiempo": original.index,
            "Original": original.values,
            "Interpolado": cleaned.values,
            "Es_outlier": outlier_mask.values,
        }
    )

    plt.figure(figsize=(14, 5))
    sns.lineplot(data=df_plot, x="Tiempo", y="Original", label="Original", alpha=0.6)
    sns.lineplot(data=df_plot, x="Tiempo", y="Interpolado", label="Interpolado", linestyle="--")
    plt.scatter(
        df_plot.loc[df_plot["Es_outlier"], "Tiempo"],
        df_plot.loc[df_plot["Es_outlier"], "Interpolado"],
        color="red",
        label="Valores interpolados",
        s=20,
        zorder=5,
    )
    plt.title(f"Interpolacion de Outliers en {column_name}", fontsize=14)
    plt.xlabel("Tiempo")
    plt.ylabel(f"Valor de {column_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_series(df: pd.DataFrame, column: str, title: str = "") -> None:
    """Plot a simple time series line chart."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[column], color="#2a9d8f", linestyle="-", marker="o", markersize=4, label=column)
    plt.title(title or f"{column} a lo largo del tiempo", fontsize=14, fontweight="bold")
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel(f"Valor de {column}", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = 365,
    model: str = "multiplicative",
) -> None:
    """Plot seasonal decomposition into trend, seasonality, and residuals."""
    sns.set_style("whitegrid")
    result = seasonal_decompose(series, model=model, period=period)
    colors = {"observed": "#2a9d8f", "trend": "#264653", "seasonal": "#e76f51", "resid": "#6c757d"}

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    result.observed.plot(ax=axes[0], color=colors["observed"])
    axes[0].set_ylabel("Original")
    axes[0].set_title("Descomposicion estacional - Modelo multiplicativo")

    result.trend.plot(ax=axes[1], color=colors["trend"])
    axes[1].set_ylabel("Tendencia")

    result.seasonal.plot(ax=axes[2], color=colors["seasonal"])
    axes[2].set_ylabel("Estacionalidad")

    result.resid.plot(ax=axes[3], color=colors["resid"])
    axes[3].set_ylabel("Residuo")
    axes[3].set_xlabel("Fecha")

    plt.tight_layout()
    plt.show()


def plot_training_curves(history: dict) -> None:
    """Plot loss and MAE training curves."""
    plt.plot(history["loss"], label="Perdida entrenamiento")
    plt.plot(history["val_loss"], label="Perdida validacion")
    plt.xlabel("Epoca")
    plt.ylabel("MSE")
    plt.title("Curva de perdida (MSE)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(history["mae"], label="MAE entrenamiento")
    plt.plot(history["val_mae"], label="MAE validacion")
    plt.xlabel("Epoca")
    plt.ylabel("MAE")
    plt.title("Curva MAE")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot predicted vs actual target values."""
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="ACTIVA real")
    plt.plot(y_pred, "--", label="ACTIVA predicha")
    plt.title("ACTIVA - Prediccion vs Real")
    plt.xlabel("Indice temporal")
    plt.ylabel("ACTIVA")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
