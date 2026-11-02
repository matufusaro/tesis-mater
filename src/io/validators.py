from __future__ import annotations

import pandas as pd


def validate_inputs(
    df_gen: pd.DataFrame,
    df_cons: pd.DataFrame,
    df_lim: pd.DataFrame,
    df_nom: pd.DataFrame,
) -> list[str]:
    """Validación mínima para ejecutar el simulador.

    Se valida:
    - Presencia de columnas obligatorias
    - Unicidad de límites por cliente
    """

    errors: list[str] = []

    gen_cols = {
        "anio",
        "mes",
        "nemo_generador",
        "energia_disponible_mwh",
        "precio_compra_usd_mwh",
    }
    cons_cols = {"anio", "mes", "nemo_cliente", "consumo_mwh"}
    lim_cols = {"nemo_cliente", "min_anual_mwh", "max_anual_mwh"}
    nom_cols = {
        "id_acuerdo",
        "anio",
        "mes",
        "nemo_cliente",
        "nemo_generador",
        "prioridad_cliente",
        "prioridad_asignacion",
        "precio_venta_usd_mwh",
        "nominal_mwh",
        "mes_inicio",
        "mes_fin",
    }

    for name, df, cols in [
        ("Generadores_Energia", df_gen, gen_cols),
        ("Clientes_Consumo", df_cons, cons_cols),
        ("Clientes_Limites", df_lim, lim_cols),
        ("Nominaciones", df_nom, nom_cols),
    ]:
        missing = cols - set(df.columns)
        if missing:
            errors.append(f"[{name}] faltan columnas: {sorted(missing)}")

    if errors:
        return errors

    if df_lim["nemo_cliente"].duplicated().any():
        errors.append("[Clientes_Limites] nemo_cliente debe ser único (1 fila por cliente).")

    bad = df_lim[df_lim["min_anual_mwh"].astype(float) > df_lim["max_anual_mwh"].astype(float)]
    if not bad.empty:
        errors.append("[Clientes_Limites] Hay clientes con min_anual_mwh > max_anual_mwh.")

    return errors
