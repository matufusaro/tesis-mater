from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_excel(path: str | Path) -> dict[str, pd.DataFrame]:
    """Carga la base Excel del modelo.

    Pestañas requeridas:
    - Generadores_Energia: energía disponible mensual por generador (puede incluir columna 'escenario')
    - Clientes_Consumo: consumo mensual por cliente (puede incluir columna 'escenario')
    - Clientes_Limites: mínimos/máximos anuales a asignar por cliente
    - Nominaciones: nominaciones mensuales (acuerdos cliente-generador) y prioridad de asignación

    Nota:
    Si las pestañas de generación/consumo no incluyen 'escenario', se asume 'BASE'.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo Excel: {path}")

    xls = pd.ExcelFile(path)
    required = {"Generadores_Energia", "Clientes_Consumo", "Clientes_Limites", "Nominaciones"}
    missing = required - set(xls.sheet_names)
    if missing:
        raise ValueError(f"Faltan pestañas en el Excel: {sorted(missing)}")

    df_gen = pd.read_excel(path, sheet_name="Generadores_Energia")
    df_cons = pd.read_excel(path, sheet_name="Clientes_Consumo")
    df_lim = pd.read_excel(path, sheet_name="Clientes_Limites")
    df_nom = pd.read_excel(path, sheet_name="Nominaciones")

    # Excel puede conservar filas vacías al final de una tabla; no son registros y
    # pueden causar fallos al convertir campos como anio a entero.
    df_gen = df_gen.dropna(how="all").reset_index(drop=True)
    df_cons = df_cons.dropna(how="all").reset_index(drop=True)
    df_lim = df_lim.dropna(how="all").reset_index(drop=True)
    df_nom = df_nom.dropna(how="all").reset_index(drop=True)

    if "escenario" not in df_gen.columns:
        df_gen["escenario"] = "BASE"
    if "escenario" not in df_cons.columns:
        df_cons["escenario"] = "BASE"

    return {"gen": df_gen, "cons": df_cons, "lim": df_lim, "nom": df_nom}
