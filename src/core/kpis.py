from __future__ import annotations

import pandas as pd


def compute_kpis(df_month: pd.DataFrame) -> dict[str, float]:
    """KPIs básicos a partir del resumen mensual."""
    if df_month is None or df_month.empty:
        return {
            "margen_total_usd": 0.0,
            "energia_total_mwh": 0.0,
            "ingreso_total_usd": 0.0,
            "costo_total_usd": 0.0,
        }

    return {
        "margen_total_usd": float(df_month["margen_usd"].sum()),
        "energia_total_mwh": float(df_month["energia_asignada_mwh"].sum()),
        "ingreso_total_usd": float(df_month["ingreso_usd"].sum()),
        "costo_total_usd": float(df_month["costo_usd"].sum()),
    }
