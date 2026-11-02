from __future__ import annotations

from pathlib import Path
import pandas as pd


def apply_nominal_overrides(df_nom_base: pd.DataFrame, df_nom_edited: pd.DataFrame) -> pd.DataFrame:
    """Aplica cambios de nominal_mwh hechos en el editor de la app.

    Se espera que ambas tablas contengan las columnas:
    - id_acuerdo, anio, mes, nominal_mwh

    Devuelve df_nom_base con nominal_mwh reemplazado por los valores editados cuando existan.
    """
    key = ["id_acuerdo", "anio", "mes"]

    if df_nom_edited is None or len(df_nom_edited) == 0:
        return df_nom_base.copy()

    # Validaciones mínimas
    for col in key + ["nominal_mwh"]:
        if col not in df_nom_base.columns:
            raise ValueError(f"df_nom_base no contiene la columna requerida: '{col}'")
        if col not in df_nom_edited.columns:
            raise ValueError(f"df_nom_edited no contiene la columna requerida: '{col}'")

    base = df_nom_base.copy()
    edited = df_nom_edited[key + ["nominal_mwh"]].copy()

    edited["nominal_mwh"] = pd.to_numeric(edited["nominal_mwh"], errors="coerce")

    out = base.merge(
        edited.rename(columns={"nominal_mwh": "nominal_mwh_override"}),
        on=key,
        how="left",
    )
    out["nominal_mwh"] = out["nominal_mwh_override"].fillna(out["nominal_mwh"])
    out = out.drop(columns=["nominal_mwh_override"])
    return out


# ---- Opcional: overrides persistidos en CSV (no requerido para la app) ----

def apply_override_nominal(df_nom: pd.DataFrame, override_path: str | Path) -> pd.DataFrame:
    """Aplica overrides desde un CSV con columnas: id_acuerdo, anio, mes, nominal_mwh."""
    override_path = Path(override_path)
    if not override_path.exists():
        return df_nom.copy()

    ov = pd.read_csv(override_path)
    req = {"id_acuerdo", "anio", "mes", "nominal_mwh"}
    if not req.issubset(set(ov.columns)):
        raise ValueError(f"Override inválido. Debe tener columnas {sorted(req)}")

    key = ["id_acuerdo", "anio", "mes"]
    out = df_nom.merge(
        ov[key + ["nominal_mwh"]].rename(columns={"nominal_mwh": "nominal_mwh_override"}),
        on=key,
        how="left",
    )
    out["nominal_mwh"] = out["nominal_mwh_override"].fillna(out["nominal_mwh"])
    return out.drop(columns=["nominal_mwh_override"])
