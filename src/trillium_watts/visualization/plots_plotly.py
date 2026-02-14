"""Plotly visualization functions for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_demand_time_series_figure(
    df: pd.DataFrame,
    color_historical: str = "#1d7a8d",
    color_predicted: str = "#ff6f00",
) -> go.Figure:
    """Create the historical + predicted demand line chart.

    Expects a DataFrame with columns: Fecha, ACTIVA, Tipo.
    """
    fig = px.line(
        df,
        x="Fecha",
        y="ACTIVA",
        color="Tipo",
        color_discrete_map={
            "Historica": color_historical,
            "Predicha": color_predicted,
        },
        labels={
            "Fecha": "Fecha",
            "ACTIVA": "Demanda Energetica (kWh)",
            "Tipo": "Tipo",
        },
        title="Serie Temporal de Demanda Energetica",
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=20, family="Arial", color="#333"),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def create_scenario_comparison_figure(df_results: pd.DataFrame) -> go.Figure:
    """Create a grouped bar chart comparing scenarios on a log scale.

    Expects a DataFrame with columns: Escenario, and one or more metric columns.
    """
    df_melted = df_results.melt(id_vars="Escenario")

    fig = px.bar(
        df_melted,
        x="variable",
        y="value",
        color="Escenario",
        barmode="group",
        labels={"variable": "Indicador", "value": "Valor"},
        title="Comparativa de Beneficios",
    )
    fig.update_layout(xaxis_title="", legend_title="Escenario")
    fig.update_yaxes(type="log", title="Valor (escala logaritmica)")
    return fig
