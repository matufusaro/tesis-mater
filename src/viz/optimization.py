from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core.compliance import cumple_maximo_cliente, cumple_minimo_cliente

TOLERANCIA_VISUAL_VARIACION_MWH = 1e-9


def clientes_incumplidores(
    clientes_anual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve una fila por cliente incumplidor, usando los flags anuales."""
    if clientes_anual.empty:
        return clientes_anual.copy(), clientes_anual.copy()

    required = {"nemo_cliente", "energia_real_anual_mwh", "min_anual_mwh", "max_anual_mwh"}
    missing = required.difference(clientes_anual.columns)
    if missing:
        raise ValueError(f"Faltan columnas de cumplimiento anual: {', '.join(sorted(missing))}")

    clients = clientes_anual.copy()
    if "cumple_min" not in clients:
        clients["cumple_min"] = cumple_minimo_cliente(
            clients["energia_real_anual_mwh"], clients["min_anual_mwh"]
        )
    if "cumple_max" not in clients:
        clients["cumple_max"] = cumple_maximo_cliente(
            clients["energia_real_anual_mwh"],
            pd.to_numeric(clients["max_anual_mwh"], errors="coerce").fillna(float("inf")),
        )
    for column in ("cumple_min", "cumple_max"):
        if clients[column].isna().any() or not clients[column].isin([True, False]).all():
            raise ValueError(f"El campo {column} debe contener valores booleanos válidos.")

    under = clients.loc[~clients["cumple_min"].astype(bool)].drop_duplicates("nemo_cliente").copy()
    over = clients.loc[~clients["cumple_max"].astype(bool)].drop_duplicates("nemo_cliente").copy()
    return under, over


def variaciones_por_cliente(comparison: pd.DataFrame) -> pd.DataFrame:
    """Filtra ruido numérico y ordena diferencias reales sin redondear."""
    changes = comparison.loc[
        comparison["Diferencia (MWh)"].abs() > TOLERANCIA_VISUAL_VARIACION_MWH
    ].copy()
    return changes.sort_values("Diferencia (MWh)", ascending=False, kind="stable")


def precios_venta_ponderados(detalle: pd.DataFrame) -> pd.DataFrame:
    """Calcula ingreso anual dividido por energía asignada para cada cliente."""
    columns = ["nemo_cliente", "precio_venta_ponderado_usd_mwh"]
    if detalle.empty:
        return pd.DataFrame(columns=columns)
    required = {"nemo_cliente", "asignado_MWh", "ingreso_usd"}
    missing = required.difference(detalle.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el precio ponderado: {', '.join(sorted(missing))}")

    data = detalle[["nemo_cliente", "asignado_MWh", "ingreso_usd"]].copy()
    for column in ("asignado_MWh", "ingreso_usd"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data["nemo_cliente"].isna().any() or not np.isfinite(data[["asignado_MWh", "ingreso_usd"]].to_numpy()).all():
        raise ValueError("El detalle contiene clientes o valores no válidos para calcular precios ponderados.")
    annual = data.groupby("nemo_cliente", as_index=False)[["asignado_MWh", "ingreso_usd"]].sum()
    annual["precio_venta_ponderado_usd_mwh"] = (
        annual["ingreso_usd"] / annual["asignado_MWh"].where(annual["asignado_MWh"] > 0)
    )
    return annual[columns]


def grafico_variaciones_clientes(changes: pd.DataFrame) -> go.Figure:
    codes = changes["nemo_cliente"].astype(str)
    names = changes["Cliente"].astype(str)
    labels = [code if code == name else f"{name} ({code})" for code, name in zip(codes, names)]
    values = changes["Diferencia (MWh)"]
    fig = go.Figure(
        go.Bar(
            x=codes,
            y=values,
            name="Diferencia [MWh]",
            marker_color=["#137333" if value > 0 else "#B91C1C" for value in values],
            text=values,
            texttemplate="%{text:,.0f}",
            textposition="outside",
            customdata=labels,
            hovertemplate="%{customdata}<br>%{y:,.0f} MWh<extra></extra>",
            cliponaxis=False,
        )
    )
    prices = pd.to_numeric(changes["precio_venta_ponderado_usd_mwh"], errors="raise")
    valid = prices.notna() & np.isfinite(prices)
    if valid.any():
        fig.add_trace(
            go.Scatter(
                x=codes,
                y=prices,
                yaxis="y2",
                mode="lines+markers",
                name="Precio de venta ponderado [USD/MWh]",
                line=dict(color="#245A81", width=2),
                marker=dict(size=7),
                connectgaps=False,
                customdata=labels,
                hovertemplate="%{customdata}<br>%{y:,.2f} USD/MWh<extra></extra>",
            )
        )
        average = float(prices[valid].mean())
        fig.add_trace(
            go.Scatter(
                x=codes,
                y=[average] * len(codes),
                yaxis="y2",
                mode="lines",
                name="Precio promedio de venta [USD/MWh]",
                line=dict(color="#D9822B", width=2),
                hovertemplate="Promedio: %{y:,.2f} USD/MWh<extra></extra>",
            )
        )
        if valid.sum() >= 2:
            positions = np.arange(len(changes), dtype=float)
            slope, intercept = np.polyfit(positions[valid.to_numpy()], prices[valid].to_numpy(), 1)
            fig.add_trace(
                go.Scatter(
                    x=codes,
                    y=slope * positions + intercept,
                    yaxis="y2",
                    mode="lines",
                    name="Tendencia precio",
                    line=dict(color="#7896A8", width=1.5, dash="dot"),
                    hovertemplate="Tendencia: %{y:,.2f} USD/MWh<extra></extra>",
                )
            )
    fig.update_layout(
        title="Variaciones de asignación por cliente",
        height=600,
        xaxis=dict(
            title="Cliente",
            categoryorder="array",
            categoryarray=codes.tolist(),
            tickmode="array",
            tickvals=codes.tolist(),
            ticktext=codes.tolist(),
            tickangle=-45,
            automargin=True,
            range=[-0.5, 19.5] if len(changes) > 20 else None,
            rangeslider=dict(visible=len(changes) > 20),
        ),
        yaxis=dict(title="Diferencia [MWh]", domain=[0, 0.62], zeroline=True, zerolinewidth=2),
        yaxis2=dict(title="Precio de Venta [USD/MWh]", domain=[0.72, 1], side="right", anchor="x"),
        legend=dict(orientation="h", x=0, y=1.15, yanchor="bottom"),
        margin=dict(l=20, r=20, t=110, b=120),
        bargap=0.3,
    )
    return fig
