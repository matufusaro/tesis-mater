from __future__ import annotations

import pandas as pd
import plotly.express as px


def line_margen_mensual(df_month: pd.DataFrame):
    if df_month.empty:
        return None
    df = df_month.copy()
    df["mes"] = df["mes"].astype(int)
    fig = px.line(df, x="mes", y="margen_usd", markers=True, title="Margen mensual (USD)")
    fig.update_layout(xaxis_title="Mes", yaxis_title="USD")
    return fig


def bars_generador_prioridad(df_gp: pd.DataFrame):
    """Barras por mes, separando por prioridad de asignación definida por la comercializadora."""
    if df_gp.empty:
        return None
    df = df_gp.copy()
    df["mes"] = df["mes"].astype(int)
    df["prioridad_asignacion"] = df["prioridad_asignacion"].astype(int).astype(str)
    fig = px.bar(
        df,
        x="mes",
        y="asignado_MWh",
        color="prioridad_asignacion",
        facet_col="nemo_generador",
        facet_col_wrap=2,
        title="Energía asignada por generador y prioridad de asignación (MWh)",
    )
    fig.update_layout(xaxis_title="Mes", yaxis_title="MWh")
    return fig


def donut_top_clientes(df_det: pd.DataFrame, top_n: int = 5):
    if df_det.empty:
        return None
    df = (
        df_det.groupby("nemo_cliente", as_index=False)["margen_usd"]
        .sum()
        .sort_values("margen_usd", ascending=False)
        .head(top_n)
    )
    return px.pie(df, names="nemo_cliente", values="margen_usd", hole=0.5, title=f"Top {top_n} clientes por margen (USD)")
