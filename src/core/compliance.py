from __future__ import annotations

import pandas as pd

TOLERANCIA_CUMPLIMIENTO_MWH = 1.0


def cumple_minimo_cliente(energia: pd.Series, minimo: pd.Series) -> pd.Series:
    return pd.to_numeric(energia, errors="coerce") >= (
        pd.to_numeric(minimo, errors="coerce") - TOLERANCIA_CUMPLIMIENTO_MWH
    )


def cumple_maximo_cliente(energia: pd.Series, maximo: pd.Series) -> pd.Series:
    return pd.to_numeric(energia, errors="coerce") <= (
        pd.to_numeric(maximo, errors="coerce") + TOLERANCIA_CUMPLIMIENTO_MWH
    )


def cumple_top_generador(energia_tomada: pd.Series, minimo_top: pd.Series) -> pd.Series:
    return pd.to_numeric(energia_tomada, errors="coerce") >= (
        pd.to_numeric(minimo_top, errors="coerce") - TOLERANCIA_CUMPLIMIENTO_MWH
    )


def cumple_top_por_faltante(faltante_top: pd.Series) -> pd.Series:
    return pd.to_numeric(faltante_top, errors="coerce") <= TOLERANCIA_CUMPLIMIENTO_MWH


def incumplimiento_sobre_tolerancia(incumplimiento_real: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(incumplimiento_real, errors="coerce")
        .fillna(0.0)
        .sub(TOLERANCIA_CUMPLIMIENTO_MWH)
        .clip(lower=0.0)
    )
