from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cammesa_simulator import simulate_cammesa
from src.core.optimizer_milp import solve_milp_nominal
from src.io.exporter import results_to_excel_bytes
from src.io.loader import load_excel
from src.io.overrides import apply_nominal_overrides


st.set_page_config(page_title="Asignación de Energía - Demo", layout="wide")
DATA_FILE_DEFAULT = "data/BD_Ejemplo_2025.xlsx"


# -------------------- Helpers básicos --------------------
def _scenario_options(df: pd.DataFrame) -> list[str]:
    if "escenario" not in df.columns:
        return ["(único)"]
    return sorted(df["escenario"].astype(str).unique().tolist())


def _safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)


def _filter_year_scenario(df: pd.DataFrame, anio: int, escenario: str | None) -> pd.DataFrame:
    out = df.copy()
    if "anio" in out.columns:
        out = out[_safe_int_series(out["anio"]) == int(anio)].copy()
    if "escenario" in out.columns and escenario and escenario != "(único)":
        out = out[out["escenario"].astype(str) == str(escenario)].copy()
    return out


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -------------------- Penalidad TOP + Economía anual --------------------
def _top_penalty_total(results: dict[str, pd.DataFrame]) -> float:
    """
    Devuelve penalidad TOP total anual en USD si existe.
    El simulador/tu pipeline puede guardar esto en distintos dataframes/keys,
    por eso buscamos en varios nombres probables.
    """
    for key in ["take_or_pay_anual", "take_or_pay", "top_anual", "penalidades_top"]:
        df = results.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            for col in ["penalidad_usd", "penalidad_top_usd", "penalidad_total_usd", "penalidad_USD"]:
                if col in df.columns:
                    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return 0.0


def _annual_economics(results: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Resume economía anual:
      - ingreso_usd
      - costo_usd
      - margen_bruto_usd
      - penalidad_top_usd (anual)
      - margen_neto_usd
    """
    rm = results.get("resumen_mensual", pd.DataFrame()).copy()

    ingreso = float(pd.to_numeric(rm["ingreso_usd"], errors="coerce").fillna(0.0).sum()) if "ingreso_usd" in rm.columns else 0.0
    costo = float(pd.to_numeric(rm["costo_usd"], errors="coerce").fillna(0.0).sum()) if "costo_usd" in rm.columns else 0.0
    margen_bruto = float(pd.to_numeric(rm["margen_usd"], errors="coerce").fillna(0.0).sum()) if "margen_usd" in rm.columns else (ingreso - costo)

    pen_top = _top_penalty_total(results)
    margen_neto = margen_bruto - pen_top

    return {
        "Ingreso anual (USD)": ingreso,
        "Costo anual (USD)": costo,
        "Margen bruto anual (USD)": margen_bruto,
        "Penalidad TOP anual (USD)": pen_top,
        "Margen neto anual (USD)": margen_neto,
    }


def _kpis(results: dict[str, pd.DataFrame]) -> dict[str, float]:
    rm = results.get("resumen_mensual", pd.DataFrame()).copy()
    na = results.get("no_abastecido", pd.DataFrame()).copy()

    if "energia_asignada_mwh" in rm.columns:
        energia_asignada = float(pd.to_numeric(rm["energia_asignada_mwh"], errors="coerce").fillna(0.0).sum())
    elif "asignado_MWh" in rm.columns:
        energia_asignada = float(pd.to_numeric(rm["asignado_MWh"], errors="coerce").fillna(0.0).sum())
    else:
        energia_asignada = 0.0

    if not na.empty and "no_abastecido_mwh" in na.columns:
        no_abastecido = float(pd.to_numeric(na["no_abastecido_mwh"], errors="coerce").fillna(0.0).sum())
    elif "no_abastecido_MWh" in rm.columns:
        no_abastecido = float(pd.to_numeric(rm["no_abastecido_MWh"], errors="coerce").fillna(0.0).sum())
    else:
        no_abastecido = 0.0

    econ = _annual_economics(results)

    return {
        "Margen neto anual (USD)": float(econ["Margen neto anual (USD)"]),
        "Penalidad TOP anual (USD)": float(econ["Penalidad TOP anual (USD)"]),
        "Energía asignada anual (MWh)": energia_asignada,
        "Energía no abastecida anual (MWh)": no_abastecido,
    }


# -------------------- Tablas derivadas --------------------
def _alloc_by_client_month(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    det = results.get("detalle", pd.DataFrame()).copy()
    if det.empty:
        return pd.DataFrame(columns=["nemo_cliente", "mes", "asignado_mwh"])

    c_cliente = _pick_first_col(det, ["nemo_cliente", "cliente", "Nemo_Cliente", "nemo", "Cliente"])
    c_mes = _pick_first_col(det, ["mes", "Mes"])
    c_e = _pick_first_col(det, ["energia_asignada_mwh", "asignado_mwh", "asignado_MWh", "energia_mwh", "energia_MWh", "MWh", "mwh"])
    if not c_cliente or not c_mes or not c_e:
        return pd.DataFrame(columns=["nemo_cliente", "mes", "asignado_mwh"])

    out = det.groupby([c_cliente, c_mes], as_index=False)[c_e].sum()
    out = out.rename(columns={c_cliente: "nemo_cliente", c_mes: "mes", c_e: "asignado_mwh"})
    out["mes"] = _safe_int_series(out["mes"])
    return out


def _alloc_by_gen_month(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    det = results.get("detalle", pd.DataFrame()).copy()
    if det.empty:
        return pd.DataFrame(columns=["nemo_generador", "mes", "tomado_mwh"])

    c_gen = _pick_first_col(det, ["nemo_generador", "generador", "Nemo_Generador", "respaldo", "Generador"])
    c_mes = _pick_first_col(det, ["mes", "Mes"])
    c_e = _pick_first_col(det, ["energia_asignada_mwh", "asignado_mwh", "asignado_MWh", "energia_mwh", "energia_MWh", "MWh", "mwh"])
    if not c_gen or not c_mes or not c_e:
        return pd.DataFrame(columns=["nemo_generador", "mes", "tomado_mwh"])

    out = det.groupby([c_gen, c_mes], as_index=False)[c_e].sum()
    out = out.rename(columns={c_gen: "nemo_generador", c_mes: "mes", c_e: "tomado_mwh"})
    out["mes"] = _safe_int_series(out["mes"])
    return out


def _consumption_by_client_month(df_cons_y: pd.DataFrame) -> pd.DataFrame:
    if df_cons_y.empty:
        return pd.DataFrame(columns=["nemo_cliente", "mes", "consumo_mwh"])

    c_cliente = _pick_first_col(df_cons_y, ["nemo_cliente", "cliente", "Nemo_Cliente", "nemo", "Cliente"])
    c_mes = _pick_first_col(df_cons_y, ["mes", "Mes"])
    c_cons = _pick_first_col(df_cons_y, ["consumo_mwh", "energia_consumida_mwh", "demanda_mwh", "demanda", "consumo", "energia_mwh", "MWh", "mwh"])
    if not c_cliente or not c_mes or not c_cons:
        return pd.DataFrame(columns=["nemo_cliente", "mes", "consumo_mwh"])

    out = df_cons_y.groupby([c_cliente, c_mes], as_index=False)[c_cons].sum()
    out = out.rename(columns={c_cliente: "nemo_cliente", c_mes: "mes", c_cons: "consumo_mwh"})
    out["mes"] = _safe_int_series(out["mes"])
    return out


def _available_by_gen_month(df_gen_y: pd.DataFrame) -> pd.DataFrame:
    if df_gen_y.empty:
        return pd.DataFrame(columns=["nemo_generador", "mes", "disponible_mwh"])

    c_gen = _pick_first_col(df_gen_y, ["nemo_generador", "generador", "Nemo_Generador", "respaldo", "Generador"])
    c_mes = _pick_first_col(df_gen_y, ["mes", "Mes"])
    c_av = _pick_first_col(df_gen_y, ["energia_disponible_mwh", "generacion_mwh", "energia_generada_mwh", "energia_mwh", "MWh", "mwh"])
    if not c_gen or not c_mes or not c_av:
        return pd.DataFrame(columns=["nemo_generador", "mes", "disponible_mwh"])

    out = df_gen_y.groupby([c_gen, c_mes], as_index=False)[c_av].sum()
    out = out.rename(columns={c_gen: "nemo_generador", c_mes: "mes", c_av: "disponible_mwh"})
    out["mes"] = _safe_int_series(out["mes"])
    return out


def _client_minmax_table(df_lim: pd.DataFrame, results: dict[str, pd.DataFrame], anio: int) -> pd.DataFrame:
    """
    Tabla por cliente con:
      - min_anual_mwh (si no existe en df_lim -> 0)
      - max_anual_mwh (si no existe -> NaN)
      - asignado_anual_mwh (desde results["detalle"])
      - cumple_min (check)
    """
    lim = df_lim.copy()
    if "anio" in lim.columns:
        lim = lim[_safe_int_series(lim["anio"]) == int(anio)].copy()

    c_cliente = _pick_first_col(lim, ["nemo_cliente", "cliente", "Nemo_Cliente", "nemo", "Cliente"])
    c_min = _pick_first_col(lim, ["min_anual_mwh", "min_mwh_anual", "minimo_anual_mwh", "min_mwh"])
    c_max = _pick_first_col(lim, ["max_anual_mwh", "max_mwh_anual", "maximo_anual_mwh", "max_mwh"])

    if not c_cliente:
        return pd.DataFrame(columns=["nemo_cliente", "min_anual_mwh", "max_anual_mwh", "asignado_anual_mwh", "cumple_min"])

    base = lim[[c_cliente]].drop_duplicates().rename(columns={c_cliente: "nemo_cliente"}).copy()

    if c_min:
        mins = lim.groupby(c_cliente, as_index=False)[c_min].min().rename(columns={c_cliente: "nemo_cliente", c_min: "min_anual_mwh"})
        base = base.merge(mins, on="nemo_cliente", how="left")
    if c_max:
        maxs = lim.groupby(c_cliente, as_index=False)[c_max].max().rename(columns={c_cliente: "nemo_cliente", c_max: "max_anual_mwh"})
        base = base.merge(maxs, on="nemo_cliente", how="left")

    if "min_anual_mwh" not in base.columns:
        base["min_anual_mwh"] = 0.0
    base["min_anual_mwh"] = pd.to_numeric(base["min_anual_mwh"], errors="coerce").fillna(0.0)

    if "max_anual_mwh" not in base.columns:
        base["max_anual_mwh"] = np.nan
    base["max_anual_mwh"] = pd.to_numeric(base["max_anual_mwh"], errors="coerce")

    alloc_cm = _alloc_by_client_month(results)
    if alloc_cm.empty:
        base["asignado_anual_mwh"] = 0.0
    else:
        asg = alloc_cm.groupby("nemo_cliente", as_index=False)["asignado_mwh"].sum().rename(columns={"asignado_mwh": "asignado_anual_mwh"})
        base = base.merge(asg, on="nemo_cliente", how="left")
        base["asignado_anual_mwh"] = pd.to_numeric(base["asignado_anual_mwh"], errors="coerce").fillna(0.0)

    base["cumple_min"] = (base["asignado_anual_mwh"] >= base["min_anual_mwh"]).map({True: "✅", False: "❌"})

    return (
        base[["nemo_cliente", "min_anual_mwh", "max_anual_mwh", "asignado_anual_mwh", "cumple_min"]]
        .sort_values("nemo_cliente")
        .reset_index(drop=True)
    )


def _gen_avail_vs_taken_table(df_gen_y: pd.DataFrame, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    av = _available_by_gen_month(df_gen_y)
    tk = _alloc_by_gen_month(results)
    out = av.merge(tk, on=["nemo_generador", "mes"], how="left")
    out["tomado_mwh"] = out["tomado_mwh"].fillna(0.0)
    out["remanente_mwh"] = out["disponible_mwh"] - out["tomado_mwh"]
    return out.sort_values(["nemo_generador", "mes"]).reset_index(drop=True)


def _client_month_overassign(cons_cm: pd.DataFrame, alloc_cm: pd.DataFrame) -> pd.DataFrame:
    if cons_cm.empty:
        return pd.DataFrame(columns=["nemo_cliente", "mes", "consumo_mwh", "asignado_mwh", "exceso_mwh"])
    out = cons_cm.merge(alloc_cm, on=["nemo_cliente", "mes"], how="left")
    out["asignado_mwh"] = out["asignado_mwh"].fillna(0.0)
    out["exceso_mwh"] = out["asignado_mwh"] - out["consumo_mwh"]
    out = out[out["exceso_mwh"] > 1e-6].copy()
    return out.sort_values(["exceso_mwh"], ascending=False).reset_index(drop=True)


def _min_violations_table(df_client_limits: pd.DataFrame) -> pd.DataFrame:
    if df_client_limits.empty:
        return pd.DataFrame(columns=df_client_limits.columns)
    return df_client_limits[df_client_limits["cumple_min"] == "❌"].copy().reset_index(drop=True)


def _dfs_to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=str(sheet)[:31], index=False)
    return buf.getvalue()


def _df_to_excel_bytes(df: pd.DataFrame, sheet: str) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


# -------------------- Resumen automático --------------------
def _summary_text(
    df_clients_cmp: pd.DataFrame,
    df_gens_cmp: pd.DataFrame,
    df_margin_month: pd.DataFrame,
    df_min_viol_opt: pd.DataFrame,
    kpi_base: dict[str, float],
    kpi_opt: dict[str, float],
    econ_base: dict[str, float],
    econ_opt: dict[str, float],
) -> str:
    lines: list[str] = []

    if not df_clients_cmp.empty and "delta_asignado_mwh" in df_clients_cmp.columns:
        top_cli = (
            df_clients_cmp.groupby("nemo_cliente", as_index=False)["delta_asignado_mwh"].sum()
            .sort_values("delta_asignado_mwh", ascending=False)
        )
        top_pos = top_cli.head(5)
        top_neg = top_cli.tail(5).sort_values("delta_asignado_mwh")
        lines.append("**Clientes – cambios de energía (Opt - Base):**")
        if len(top_pos):
            lines.append("- Más energía asignada: " + ", ".join([f"{r.nemo_cliente} ({r.delta_asignado_mwh:,.0f} MWh)" for r in top_pos.itertuples()]))
        if len(top_neg):
            lines.append("- Menos energía asignada: " + ", ".join([f"{r.nemo_cliente} ({r.delta_asignado_mwh:,.0f} MWh)" for r in top_neg.itertuples()]))

        if len(top_pos):
            cli_set = set(top_pos["nemo_cliente"].tolist())
            tmp = df_clients_cmp[df_clients_cmp["nemo_cliente"].isin(cli_set)].copy()
            tmp_m = tmp.groupby(["nemo_cliente", "mes"], as_index=False)["delta_asignado_mwh"].sum()
            tmp_m = tmp_m.sort_values("delta_asignado_mwh", ascending=False).head(8)
            if len(tmp_m):
                lines.append("- Meses destacados (clientes con mayor aumento): " + ", ".join([f"{r.nemo_cliente}-M{int(r.mes)} ({r.delta_asignado_mwh:,.0f})" for r in tmp_m.itertuples()]))

    if not df_gens_cmp.empty and "delta_tomado_mwh" in df_gens_cmp.columns:
        top_gen = (
            df_gens_cmp.groupby("nemo_generador", as_index=False)["delta_tomado_mwh"].sum()
            .sort_values("delta_tomado_mwh", ascending=False)
        )
        top_pos_g = top_gen.head(5)
        top_neg_g = top_gen.tail(5).sort_values("delta_tomado_mwh")
        lines.append("\n**Generadores – cambios de energía tomada (Opt - Base):**")
        if len(top_pos_g):
            lines.append("- Se tomó más energía de: " + ", ".join([f"{r.nemo_generador} ({r.delta_tomado_mwh:,.0f} MWh)" for r in top_pos_g.itertuples()]))
        if len(top_neg_g):
            lines.append("- Se tomó menos energía de: " + ", ".join([f"{r.nemo_generador} ({r.delta_tomado_mwh:,.0f} MWh)" for r in top_neg_g.itertuples()]))

        tmp_m = df_gens_cmp.groupby(["nemo_generador", "mes"], as_index=False)["delta_tomado_mwh"].sum()
        tmp_m_pos = tmp_m.sort_values("delta_tomado_mwh", ascending=False).head(8)
        tmp_m_neg = tmp_m.sort_values("delta_tomado_mwh", ascending=True).head(8)
        if len(tmp_m_pos):
            lines.append("- Meses donde aumentó más la toma: " + ", ".join([f"{r.nemo_generador}-M{int(r.mes)} ({r.delta_tomado_mwh:,.0f})" for r in tmp_m_pos.itertuples()]))
        if len(tmp_m_neg):
            lines.append("- Meses donde cayó más la toma: " + ", ".join([f"{r.nemo_generador}-M{int(r.mes)} ({r.delta_tomado_mwh:,.0f})" for r in tmp_m_neg.itertuples()]))

    if not df_margin_month.empty and {"mes", "delta_margen_usd"}.issubset(df_margin_month.columns):
        inc = df_margin_month.sort_values("delta_margen_usd", ascending=False).head(5)
        dec = df_margin_month.sort_values("delta_margen_usd", ascending=True).head(5)
        lines.append("\n**Margen BRUTO – meses con mayor cambio (Opt - Base):**")
        if len(inc):
            lines.append("- Meses donde subió más: " + ", ".join([f"M{int(r.mes)} (+{r.delta_margen_usd:,.0f} USD)" for r in inc.itertuples()]))
        if len(dec):
            lines.append("- Meses donde bajó más: " + ", ".join([f"M{int(r.mes)} ({r.delta_margen_usd:,.0f} USD)" for r in dec.itertuples()]))

    n_bad = int(len(df_min_viol_opt)) if df_min_viol_opt is not None else 0
    lines.append(f"\n**Incumplimiento de mínimos (Opt):** {n_bad} clientes.")
    if n_bad and "nemo_cliente" in df_min_viol_opt.columns:
        lines.append("- Clientes: " + ", ".join(df_min_viol_opt["nemo_cliente"].astype(str).head(15).tolist()) + ("..." if n_bad > 15 else ""))

    mbn = float(econ_base.get("Margen neto anual (USD)", 0.0))
    mon = float(econ_opt.get("Margen neto anual (USD)", 0.0))
    lines.append(
        f"\n**Economía anual (NETO):** Base {mbn:,.0f} USD | Opt {mon:,.0f} USD | Δ {mon-mbn:,.0f} USD"
    )
    lines.append(
        f"Penalidad TOP anual: Base {float(econ_base.get('Penalidad TOP anual (USD)', 0.0)):,.0f} | "
        f"Opt {float(econ_opt.get('Penalidad TOP anual (USD)', 0.0)):,.0f}"
    )

    return "\n".join(lines)


# -------------------- App --------------------
def main() -> None:
    with st.sidebar:
        st.header("Datos")
        data_file = st.text_input("Archivo Excel de datos", value=DATA_FILE_DEFAULT)
        st.caption("Tip: dejá el archivo dentro de la carpeta 'data/'.")

        st.divider()
        st.subheader("Reglas / Validaciones")
        strict_min = st.checkbox("Mínimos anuales obligatorios (para OPT)", value=True)
        strict_no_overassign = st.checkbox("Prohibir asignación > consumo (reportar y marcar)", value=True)

        try:
            data = load_excel(data_file)
            df_gen = data["gen"]
            df_cons = data["cons"]
            df_lim = data["lim"]
            df_nom_base = data["nom"]
        except Exception as e:
            st.error(f"No se pudo leer el Excel: {e}")
            st.stop()

        anios = sorted(pd.to_numeric(df_nom_base["anio"], errors="coerce").dropna().astype(int).unique().tolist())
        anio = st.selectbox("Año", anios, index=len(anios) - 1)

        gen_scen = st.selectbox("Escenario generación", _scenario_options(df_gen))
        cons_scen = st.selectbox("Escenario consumo", _scenario_options(df_cons))

        st.divider()
        st.caption("Las nominaciones sólo se editan en el Simulador Baseline.")

    tab1, tab2, tab3 = st.tabs(["Simulador Baseline", "Optimizador MILP", "Comparativa"])

    # -------------------- TAB 1: Baseline --------------------
    with tab1:
        st.title("MODELO DE SIMULACIÓN (CAMMESA)")

        st.subheader("Generadores")

        df_gen_y = _filter_year_scenario(df_gen, int(anio), gen_scen)

        generator_count = 0
        available_annual_mwh = 0.0
        min_top_annual_mwh = 0.0

        if not df_gen_y.empty and "nemo_generador" in df_gen_y.columns:
            df_gen_numeric = df_gen_y.copy()
            df_gen_numeric["energia_disponible_mwh"] = pd.to_numeric(
                df_gen_numeric.get("energia_disponible_mwh", pd.Series(dtype=float)), errors="coerce"
            ).fillna(0.0)

            generator_count = int(df_gen_numeric["nemo_generador"].astype(str).nunique())
            available_annual_mwh = float(df_gen_numeric["energia_disponible_mwh"].sum())

            if "take_or_pay_ratio" in df_gen_numeric.columns:
                df_gen_numeric["take_or_pay_ratio"] = pd.to_numeric(
                    df_gen_numeric["take_or_pay_ratio"], errors="coerce"
                ).fillna(0.0)

                gen_top = (
                    df_gen_numeric.groupby(["anio", "nemo_generador"], as_index=False)
                    .agg(
                        energia_disponible_anual_mwh=("energia_disponible_mwh", "sum"),
                        take_or_pay_ratio=("take_or_pay_ratio", "max"),
                    )
                )
                min_top_annual_mwh = float(
                    (gen_top["energia_disponible_anual_mwh"] * gen_top["take_or_pay_ratio"]).sum()
                )

        def _format_es_mwh(value: float) -> str:
            return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

        c1, c2, c3 = st.columns(3)
        c1.metric("Generadores", f"{generator_count:,}")
        c2.metric("Energía disponible", f"{_format_es_mwh(available_annual_mwh)} MWh")
        c3.metric("Energía mínima TOP", f"{_format_es_mwh(min_top_annual_mwh)} MWh")

        st.markdown("**Distribución de energía nominada por prioridad**")

        nom_source = st.session_state.get("baseline_nominal_edited_df", df_nom_base)
        if isinstance(nom_source, pd.DataFrame):
            nom_priority = _filter_year_scenario(nom_source, int(anio), None).copy()
        else:
            nom_priority = _filter_year_scenario(df_nom_base, int(anio), None).copy()

        if not nom_priority.empty and {"prioridad_asignacion", "nominal_mwh"}.issubset(nom_priority.columns):
            nom_priority["mes"] = pd.to_numeric(nom_priority.get("mes", pd.Series(dtype=float)), errors="coerce")
            if {"mes_inicio", "mes_fin"}.issubset(nom_priority.columns):
                nom_priority["mes_inicio"] = pd.to_numeric(nom_priority["mes_inicio"], errors="coerce")
                nom_priority["mes_fin"] = pd.to_numeric(nom_priority["mes_fin"], errors="coerce")
                nom_priority = nom_priority[
                    nom_priority["mes_inicio"].le(nom_priority["mes"])
                    & nom_priority["mes"].le(nom_priority["mes_fin"])
                ].copy()

            nom_priority["prioridad_asignacion"] = pd.to_numeric(
                nom_priority["prioridad_asignacion"], errors="coerce"
            )
            nom_priority["nominal_mwh"] = pd.to_numeric(nom_priority["nominal_mwh"], errors="coerce").fillna(0.0)
            nom_priority = nom_priority.dropna(subset=["prioridad_asignacion"])

            priority_summary = (
                nom_priority.groupby("prioridad_asignacion", as_index=False)["nominal_mwh"]
                .sum()
                .sort_values("prioridad_asignacion")
                .rename(columns={"prioridad_asignacion": "Prioridad", "nominal_mwh": "Energía nominada (MWh)"})
            )
            priority_total = float(priority_summary["Energía nominada (MWh)"].sum())
            priority_summary["Porcentaje"] = (
                priority_summary["Energía nominada (MWh)"].div(priority_total).mul(100.0)
                if priority_total > 0
                else 0.0
            )
            priority_summary["Prioridad"] = priority_summary["Prioridad"].apply(
                lambda v: str(int(v)) if float(v).is_integer() else str(v)
            )

            priority_summary["Etiqueta"] = (
                priority_summary["Energía nominada (MWh)"].map(_format_es_mwh)
                + " MWh · "
                + priority_summary["Porcentaje"].map(lambda v: f"{v:.1f}%")
            )

            priority_labels = priority_summary["Prioridad"].tolist()
            blue_palette = [
                "#0B3C5D", "#1F5A86", "#397CBF", "#69A2D1",
                "#8EBCE0", "#B0CEE8", "#5C718A", "#7F9DBB",
            ]
            color_map = {
                priority: blue_palette[i % len(blue_palette)]
                for i, priority in enumerate(priority_labels)
            }

            stacked = priority_summary.assign(_Fila="Nominado")
            stacked["Etiqueta segmento"] = stacked.apply(
                lambda r: f"P{int(r['Prioridad'])}" if r["Porcentaje"] >= 8.0 else "",
                axis=1,
            ) if stacked["Prioridad"].map(lambda v: str(v).isdigit()).all() else stacked["Porcentaje"].map(lambda v: "" if v < 8.0 else "")

            fig_priority = px.bar(
                stacked,
                x="Energía nominada (MWh)",
                y="_Fila",
                color="Prioridad",
                orientation="h",
                barmode="stack",
                color_discrete_map=color_map,
                category_orders={"Prioridad": priority_labels},
                text="Etiqueta segmento",
            )
            fig_priority.update_layout(
                height=125,
                margin=dict(l=20, r=20, t=10, b=30),
                showlegend=False,
                bargap=0.35,
                xaxis_title="MWh nominados",
                yaxis=dict(showticklabels=False, title=""),
            )
            fig_priority.update_traces(textposition="inside", insidetextanchor="middle", cliponaxis=False)
            st.plotly_chart(fig_priority, use_container_width=True)

            legend_parts = []
            for _, row in priority_summary.iterrows():
                priority_value = str(row["Prioridad"])
                priority_text = f"P{int(float(priority_value))}" if priority_value.replace(".", "", 1).isdigit() else priority_value
                legend_parts.append(
                    f'<span style="color:{color_map[row["Prioridad"]]};">■</span> '
                    f'<b>{priority_text}</b>: {_format_es_mwh(float(row["Energía nominada (MWh)"]))} MWh · {float(row["Porcentaje"]):.1f}%'
                )
            st.markdown(" &nbsp;·&nbsp; ".join(legend_parts), unsafe_allow_html=True)
        else:
            st.info("No hay nominaciones disponibles para mostrar la distribución por prioridad.")

        st.markdown("**Detalle por generador**")

        if not df_gen_y.empty and not nom_priority.empty:
            detail = nom_priority.copy()
            if "nemo_generador" not in detail.columns or "nominal_mwh" not in detail.columns:
                detail = pd.DataFrame()

            if not detail.empty:
                detail["nominal_mwh"] = pd.to_numeric(detail["nominal_mwh"], errors="coerce").fillna(0.0)
                detail["prioridad_asignacion"] = pd.to_numeric(detail["prioridad_asignacion"], errors="coerce")

                gen_detail = (
                    detail.groupby(["nemo_generador", "prioridad_asignacion"], as_index=False)["nominal_mwh"]
                    .sum()
                    .rename(columns={
                        "nemo_generador": "NEMO",
                        "prioridad_asignacion": "Prioridad",
                        "nominal_mwh": "Energía nominada (MWh)",
                    })
                )
                gen_detail["Prioridad"] = gen_detail["Prioridad"].apply(
                    lambda v: str(int(v)) if float(v).is_integer() else str(v)
                )

                names = pd.DataFrame()
                if "nombre_generador" in df_gen_y.columns:
                    names = (
                        df_gen_y.groupby("nemo_generador", as_index=False)["nombre_generador"]
                        .agg(lambda s: " / ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip()))))
                        .rename(columns={"nemo_generador": "NEMO", "nombre_generador": "Generador"})
                    )
                elif "nombre_generador" in nom_priority.columns:
                    names = (
                        nom_priority.groupby("nemo_generador", as_index=False)["nombre_generador"]
                        .agg(lambda s: " / ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip()))))
                        .rename(columns={"nemo_generador": "NEMO", "nombre_generador": "Generador"})
                    )
                gen_detail = gen_detail.merge(names, on="NEMO", how="left")

                if "precio_compra_usd_mwh" in df_gen_y.columns:
                    prices = df_gen_y.copy()
                    prices["precio_compra_usd_mwh"] = pd.to_numeric(prices["precio_compra_usd_mwh"], errors="coerce")
                    prices = prices.dropna(subset=["precio_compra_usd_mwh"])
                    prices = (
                        prices.groupby("nemo_generador", as_index=False)["precio_compra_usd_mwh"]
                        .agg(lambda s: " / ".join(f"{float(v):,.2f}" for v in sorted(set(s))))
                        .rename(columns={
                            "nemo_generador": "NEMO",
                            "precio_compra_usd_mwh": "Precio(s) compra (USD/MWh)",
                        })
                    )
                    gen_detail = gen_detail.merge(prices, on="NEMO", how="left")
                else:
                    gen_detail["Precio(s) compra (USD/MWh)"] = "No disponible"

                if "Generador" not in gen_detail.columns:
                    gen_detail["Generador"] = gen_detail["NEMO"]
                gen_detail["Energía nominada (MWh)"] = gen_detail["Energía nominada (MWh)"].round(2)
                gen_detail = gen_detail[[
                    "Generador",
                    "NEMO",
                    "Prioridad",
                    "Energía nominada (MWh)",
                    "Precio(s) compra (USD/MWh)",
                ]].sort_values(["Generador", "NEMO", "Prioridad"])

                st.dataframe(gen_detail, use_container_width=True, hide_index=True)
            else:
                st.info("No hay nominaciones vigentes para mostrar el detalle por generador.")
        else:
            st.info("No hay datos suficientes para mostrar el detalle por generador.")

        st.caption("Los indicadores consideran el año y escenario seleccionados en la barra lateral.")

        st.subheader("Clientes")
        st.caption("Consumos mensuales y límites anuales cargados desde el archivo Excel.")

        st.subheader("Ejecutar simulación")

        df_nom_y = _filter_year_scenario(df_nom_base, int(anio), None)
        df_nom_y = df_nom_y.copy()
        df_nom_y = df_nom_y.sort_values(["nemo_cliente", "prioridad_asignacion", "nemo_generador", "mes"]).reset_index(drop=True)

        run = st.button("Ejecutar simulación baseline", type="primary")
        if run:
            try:
                edited_nom = st.session_state.get("baseline_nominal_edited_df", df_nom_y)
                if not isinstance(edited_nom, pd.DataFrame):
                    raise TypeError("La edición de nominaciones debe devolver un DataFrame de pandas.")
                df_nom_use = apply_nominal_overrides(df_nom_base, edited_nom)
                df_gen_y = _filter_year_scenario(df_gen, int(anio), gen_scen)
                df_cons_y = _filter_year_scenario(df_cons, int(anio), cons_scen)

                results = simulate_cammesa(df_gen_y, df_cons_y, df_lim, df_nom_use, anio=int(anio))
                st.session_state["baseline_results"] = results
                st.session_state["baseline_meta"] = {"anio": int(anio), "gen": gen_scen, "cons": cons_scen}
                st.session_state["baseline_gen_y"] = df_gen_y
                st.session_state["baseline_cons_y"] = df_cons_y
            except Exception as e:
                st.error(f"Error crítico durante la simulación: {e}")
                st.stop()

        if "baseline_results" in st.session_state:
            results = st.session_state["baseline_results"]
            df_gen_y = st.session_state["baseline_gen_y"]
            df_cons_y = st.session_state["baseline_cons_y"]

            st.subheader("Resultados de simulación")

            rm = results.get("resumen_mensual", pd.DataFrame()).copy()
            energia_total_asignada = 0.0
            margen_bruto_operativo = 0.0
            if not rm.empty:
                if "energia_asignada_mwh" in rm.columns:
                    energia_total_asignada = float(
                        pd.to_numeric(rm["energia_asignada_mwh"], errors="coerce").fillna(0.0).sum()
                    )
                if "margen_usd" in rm.columns:
                    margen_bruto_operativo = float(
                        pd.to_numeric(rm["margen_usd"], errors="coerce").fillna(0.0).sum()
                    )

            no_ab = results.get("no_abastecido", pd.DataFrame()).copy()
            energia_no_abastecida = 0.0
            if not no_ab.empty and "no_abastecido_mwh" in no_ab.columns:
                energia_no_abastecida = float(
                    pd.to_numeric(no_ab["no_abastecido_mwh"], errors="coerce").fillna(0.0).sum()
                )

            top_anual = results.get("take_or_pay_anual", pd.DataFrame()).copy()
            energia_top_incumplida = 0.0
            if not top_anual.empty and "shortfall_anual_mwh" in top_anual.columns:
                top_anual["shortfall_anual_mwh"] = pd.to_numeric(
                    top_anual["shortfall_anual_mwh"], errors="coerce"
                ).fillna(0.0)
                energia_top_incumplida = float(top_anual["shortfall_anual_mwh"].sum())

            def _format_es(value: float) -> str:
                return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

            def _format_es_usd_mwh(value: float) -> str:
                return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            st.markdown("**Balance energético**")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Energía total asignada (MWh)", f"{_format_es(energia_total_asignada)}")
            b2.metric("Energía no abastecida (MWh)", f"{_format_es(energia_no_abastecida)}")
            b3.metric("Energía TOP (Incumplimiento) (MWh)", f"{_format_es(energia_top_incumplida)}")
            with b4:
                st.metric("Margen Bruto Operativo (USD)", f"{_format_es(margen_bruto_operativo)}")
                st.caption("No contempla penalizaciones por incumplimientos TOP")

            clientes_anual = results.get("clientes_anual", pd.DataFrame()).copy()
            cantidad_clientes_bajo_minimo = 0
            mwh_faltantes_minimo = 0.0
            if not clientes_anual.empty and {"nemo_cliente", "energia_real_anual_mwh", "min_anual_mwh"}.issubset(clientes_anual.columns):
                cliente_status = clientes_anual.copy()
                cliente_status["nemo_cliente"] = cliente_status["nemo_cliente"].astype(str)
                cliente_status["energia_real_anual_mwh"] = pd.to_numeric(
                    cliente_status["energia_real_anual_mwh"], errors="coerce"
                ).fillna(0.0)
                cliente_status["min_anual_mwh"] = pd.to_numeric(
                    cliente_status["min_anual_mwh"], errors="coerce"
                ).fillna(0.0)
                cliente_status = cliente_status.groupby("nemo_cliente", as_index=False).agg(
                    energia_real_anual_mwh=("energia_real_anual_mwh", "max"),
                    min_anual_mwh=("min_anual_mwh", "max"),
                )
                cliente_status["faltante_min_mwh"] = (
                    cliente_status["min_anual_mwh"] - cliente_status["energia_real_anual_mwh"]
                ).clip(lower=0.0)
                bajo_minimo = cliente_status[cliente_status["faltante_min_mwh"] > 1e-9]
                cantidad_clientes_bajo_minimo = int(bajo_minimo["nemo_cliente"].nunique())
                mwh_faltantes_minimo = float(bajo_minimo["faltante_min_mwh"].sum())

            cons_cm = _consumption_by_client_month(df_cons_y)
            alloc_cm = _alloc_by_client_month(results)
            df_over_b = _client_month_overassign(cons_cm, alloc_cm)
            cantidad_clientes_sobreasignados = 0
            mwh_excedentes = 0.0
            if not df_over_b.empty and "nemo_cliente" in df_over_b.columns:
                cantidad_clientes_sobreasignados = int(df_over_b["nemo_cliente"].astype(str).nunique())
                if "exceso_mwh" in df_over_b.columns:
                    mwh_excedentes = float(
                        pd.to_numeric(df_over_b["exceso_mwh"], errors="coerce")
                        .fillna(0.0)
                        .sum()
                    )
                else:
                    mwh_excedentes = float(pd.to_numeric(df_over_b["asignado_mwh"], errors="coerce").fillna(0.0).sum() - pd.to_numeric(df_over_b["consumo_mwh"], errors="coerce").fillna(0.0).sum())

            generadores_bajo_top = 0
            nombres_generadores_bajo_top = []
            if not top_anual.empty and "shortfall_anual_mwh" in top_anual.columns and "nemo_generador" in top_anual.columns:
                top_anual["nemo_generador"] = top_anual["nemo_generador"].astype(str)
                incumplidos = top_anual[pd.to_numeric(top_anual["shortfall_anual_mwh"], errors="coerce").fillna(0.0) > 1e-9].copy()
                generadores_bajo_top = int(incumplidos["nemo_generador"].nunique())

                names_by_generator = pd.DataFrame()
                if not df_gen_y.empty and "nombre_generador" in df_gen_y.columns:
                    names_by_generator = (
                        df_gen_y.groupby("nemo_generador", as_index=False)["nombre_generador"]
                        .agg(lambda s: " / ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip()))))
                    )
                    names_by_generator["nemo_generador"] = names_by_generator["nemo_generador"].astype(str)

                for nemo in incumplidos["nemo_generador"].drop_duplicates().sort_values():
                    nombre = nemo
                    if not names_by_generator.empty:
                        match = names_by_generator[names_by_generator["nemo_generador"] == nemo]
                        if not match.empty and str(match.iloc[0]["nombre_generador"]).strip():
                            nombre = str(match.iloc[0]["nombre_generador"])
                    nombres_generadores_bajo_top.append(nombre)

            st.markdown("**Cumplimiento contractual**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Generadores bajo TOP", f"{generadores_bajo_top:,}")
                if nombres_generadores_bajo_top:
                    st.caption(
                        "Incumplieron: " + ", ".join(nombres_generadores_bajo_top[:5])
                        + ("…" if len(nombres_generadores_bajo_top) > 5 else "")
                    )
                    if len(nombres_generadores_bajo_top) > 5:
                        with st.expander("Ver todos los generadores bajo TOP"):
                            st.write("\n".join(f"- {nombre}" for nombre in nombres_generadores_bajo_top))
                else:
                    st.caption("Sin generadores bajo TOP")

            with c2:
                st.metric("Clientes bajo mínimo", f"{cantidad_clientes_bajo_minimo:,}")
                st.caption(f"{_format_es(mwh_faltantes_minimo)} MWh faltantes para alcanzar mínimos")

            with c3:
                st.metric("Clientes sobreasignados", f"{cantidad_clientes_sobreasignados:,}")
                st.caption(f"{_format_es(mwh_excedentes)} MWh excedentes respecto del consumo mensual")

            penalidad_total_top = 0.0
            costos_unitarios_top = pd.DataFrame()
            if not top_anual.empty and "penalidad_anual_usd" in top_anual.columns:
                top_anual["penalidad_anual_usd"] = pd.to_numeric(
                    top_anual["penalidad_anual_usd"], errors="coerce"
                ).fillna(0.0)
                penalidad_total_top = float(top_anual["penalidad_anual_usd"].sum())

            if not top_anual.empty and "penalidad_no_tomado_usd_mwh" in top_anual.columns:
                top_anual["penalidad_no_tomado_usd_mwh"] = pd.to_numeric(
                    top_anual["penalidad_no_tomado_usd_mwh"], errors="coerce"
                ).fillna(0.0)
                costos_unitarios_top = top_anual.groupby("nemo_generador", as_index=False)["penalidad_no_tomado_usd_mwh"]
                costos_unitarios_top = costos_unitarios_top.max()

            st.markdown("**Penalidades TOP**")
            p1, p2, p3 = st.columns(3)
            p1.metric("Energía TOP", f"{_format_es(energia_top_incumplida)} MWh")

            costo_unitario_texto = "No disponible"
            if not costos_unitarios_top.empty:
                valores_costos = sorted(set(float(v) for v in costos_unitarios_top["penalidad_no_tomado_usd_mwh"].tolist()))
                if len(valores_costos) == 1:
                    costo_unitario_texto = f"{_format_es_usd_mwh(valores_costos[0])} USD/MWh"
                else:
                    costo_unitario_texto = "Varía por generador"
            p2.metric("Costo unitario TOP", costo_unitario_texto)
            if not costos_unitarios_top.empty:
                valores_costos = sorted(set(float(v) for v in costos_unitarios_top["penalidad_no_tomado_usd_mwh"].tolist()))
                if len(valores_costos) > 1:
                    with st.expander("Ver costos unitarios TOP por generador"):
                        detalle_costos = costos_unitarios_top.copy()
                        detalle_costos["Costo TOP (USD/MWh)"] = detalle_costos["penalidad_no_tomado_usd_mwh"].map(_format_es_usd_mwh)
                        st.dataframe(
                            detalle_costos[["nemo_generador", "Costo TOP (USD/MWh)"]].rename(columns={"nemo_generador": "Generador"}),
                            use_container_width=True,
                            hide_index=True,
                        )

            p3.metric("Penalidad total", f"{_format_es(penalidad_total_top)} USD")

            st.divider()
            st.subheader("Visualizaciones")

            st.markdown("### Energía mensual asignada (MWh)")
            detalle = results.get("detalle", pd.DataFrame()).copy()
            mes_nombre = {
                1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
            }
            if not detalle.empty and {"mes", "asignado_MWh"}.issubset(detalle.columns):
                detalle["mes"] = pd.to_numeric(detalle["mes"], errors="coerce")
                detalle["asignado_MWh"] = pd.to_numeric(detalle["asignado_MWh"], errors="coerce").fillna(0.0)

                nombres_gen = pd.DataFrame()
                if not df_gen_y.empty and "nombre_generador" in df_gen_y.columns:
                    nombres_gen = (
                        df_gen_y.groupby("nemo_generador", as_index=False)["nombre_generador"]
                        .agg(lambda s: " / ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip()))))
                    )
                    nombres_gen["nemo_generador"] = nombres_gen["nemo_generador"].astype(str)
                    nombres_gen["Etiqueta generador"] = nombres_gen.apply(
                        lambda r: str(r["nombre_generador"]) if str(r["nombre_generador"]).strip() else str(r["nemo_generador"]), axis=1
                    )

                opciones_selector = []
                mapping_selector = {}
                for nemo in sorted(detalle["nemo_generador"].astype(str).dropna().unique().tolist()):
                    etiqueta = nemo
                    if not nombres_gen.empty:
                        match = nombres_gen[nombres_gen["nemo_generador"] == nemo]
                        if not match.empty:
                            etiqueta = str(match.iloc[0]["Etiqueta generador"])
                    etiqueta_visible = etiqueta if etiqueta == nemo else f"{etiqueta} ({nemo})"
                    opciones_selector.append(etiqueta_visible)
                    mapping_selector[etiqueta_visible] = nemo

                generadores_seleccionados = st.multiselect(
                    "Generadores incluidos",
                    options=opciones_selector,
                    default=[],
                    placeholder="Todos los generadores",
                )
                mensual = detalle.copy()
                if generadores_seleccionados:
                    nemos_seleccionados = [mapping_selector[label] for label in generadores_seleccionados]
                    mensual = mensual[mensual["nemo_generador"].astype(str).isin(nemos_seleccionados)].copy()

                mensual_chart = (
                    mensual.groupby("mes", as_index=False)["asignado_MWh"]
                    .sum()
                    .rename(columns={"asignado_MWh": "Energía asignada (MWh)"})
                )
                mensual_chart = mensual_chart[mensual_chart["mes"].between(1, 12)].copy()
                mensual_chart = mensual_chart.set_index("mes").reindex(range(1, 13), fill_value=0).reset_index()
                mensual_chart["Mes"] = mensual_chart["mes"].map(mes_nombre)

                fig_mensual = px.bar(
                    mensual_chart,
                    x="Mes",
                    y="Energía asignada (MWh)",
                    category_orders={"Mes": list(mes_nombre.values())},
                    text=None,
                    color_discrete_sequence=["#397CBF"],
                )
                fig_mensual.update_traces(
                    hovertemplate="%{x}<br>%{y:,.0f} MWh<extra></extra>",
                    textfont=None,
                )
                fig_mensual.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    xaxis_title="Mes",
                    yaxis_title="Energía asignada (MWh)",
                )
                st.plotly_chart(fig_mensual, use_container_width=True)

                suma_mensual = float(mensual_chart["Energía asignada (MWh)"].sum())
                if abs(suma_mensual - energia_total_asignada) <= 1e-6:
                    st.caption(f"Total coincidente: {_format_es(energia_total_asignada)} MWh.")
                else:
                    st.caption("El selector activo no coincide con el total general de la simulación.")
            else:
                st.info("No hay detalle de asignación mensual disponible.")

            st.markdown("### Cumplimiento TOP por generador")
            if not top_anual.empty and {
                "nemo_generador", "energia_disponible_anual_mwh", "min_take_anual_mwh", "energia_tomada_anual_mwh"
            }.issubset(top_anual.columns):
                top_chart = top_anual.copy()
                top_chart["nemo_generador"] = top_chart["nemo_generador"].astype(str)
                top_chart["nombre_hover"] = top_chart["nemo_generador"]
                if not df_gen_y.empty and "nombre_generador" in df_gen_y.columns:
                    nombres_top = (
                        df_gen_y.groupby("nemo_generador", as_index=False)["nombre_generador"]
                        .agg(lambda s: " / ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip()))))
                    )
                    nombres_top["nemo_generador"] = nombres_top["nemo_generador"].astype(str)
                    top_chart = top_chart.merge(
                        nombres_top.rename(columns={"nombre_generador": "nombre_hover"}),
                        on="nemo_generador",
                        how="left",
                        suffixes=("", "_nombre"),
                    )
                    if "nombre_hover_nombre" in top_chart.columns:
                        top_chart["nombre_hover"] = top_chart["nombre_hover_nombre"].fillna(top_chart["nombre_hover"])

                top_chart["Diferencia"] = (
                    pd.to_numeric(top_chart["energia_tomada_anual_mwh"], errors="coerce").fillna(0.0)
                    - pd.to_numeric(top_chart["min_take_anual_mwh"], errors="coerce").fillna(0.0)
                )
                top_chart = top_chart.sort_values(["Diferencia", "nemo_generador"])

                top_melt = top_chart.melt(
                    id_vars=["nombre_hover", "nemo_generador", "Diferencia"],
                    value_vars=["min_take_anual_mwh", "energia_tomada_anual_mwh"],
                    var_name="Serie",
                    value_name="MWh",
                )
                top_melt["Serie"] = top_melt["Serie"].map({
                    "min_take_anual_mwh": "Mínimo TOP",
                    "energia_tomada_anual_mwh": "Asignado",
                })

                fig_top = px.bar(
                    top_melt,
                    x="nombre_hover",
                    y="MWh",
                    color="Serie",
                    barmode="group",
                    custom_data=["nemo_generador", "Serie"],
                    color_discrete_map={"Mínimo TOP": "#6A9FC3", "Asignado": "#0B3C5D"},
                    category_orders={"x": top_chart["nombre_hover"].tolist()},
                )
                fig_top.update_traces(
                    hovertemplate="Generador: %{x}<br>NEMO: %{customdata[0]}<br>%{customdata[1]}: %{y:,.0f} MWh<extra></extra>"
                )
                fig_top.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=120),
                    legend_title_text="Serie",
                    xaxis_title="Generador",
                    yaxis_title="Energía anual (MWh)",
                )
                st.plotly_chart(fig_top, use_container_width=True)

                top_table = top_chart.copy()
                top_table["Generador"] = top_table["nombre_hover"]
                top_table["Energía total disponible"] = pd.to_numeric(top_table["energia_disponible_anual_mwh"], errors="coerce").fillna(0.0).round(0).astype(int)
                top_table["Energía mínima TOP"] = pd.to_numeric(top_table["min_take_anual_mwh"], errors="coerce").fillna(0.0).round(0).astype(int)
                top_table["Energía asignada"] = pd.to_numeric(top_table["energia_tomada_anual_mwh"], errors="coerce").fillna(0.0).round(0).astype(int)
                top_table["Diferencia"] = top_table["Diferencia"].round(0).astype(int)
                top_table = top_table[["Generador", "Energía total disponible", "Energía mínima TOP", "Energía asignada", "Diferencia"]]
                st.dataframe(
                    top_table.style.map(
                        lambda v: "color: #B91C1C; font-weight: 700" if isinstance(v, (int, float, np.integer, np.floating)) and v < 0 else "",
                        subset=["Diferencia"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No hay datos suficientes en `take_or_pay_anual` para comparar TOP por generador.")

            st.markdown("### Clientes bajo mínimo contractual anual")
            if not clientes_anual.empty and {"nemo_cliente", "energia_real_anual_mwh", "min_anual_mwh"}.issubset(clientes_anual.columns):
                min_clientes = clientes_anual.copy()
                min_clientes["nemo_cliente"] = min_clientes["nemo_cliente"].astype(str)
                min_clientes["min_anual_mwh"] = pd.to_numeric(min_clientes["min_anual_mwh"], errors="coerce").fillna(0.0)
                min_clientes["energia_real_anual_mwh"] = pd.to_numeric(min_clientes["energia_real_anual_mwh"], errors="coerce").fillna(0.0)
                min_clientes["Diferencia"] = min_clientes["energia_real_anual_mwh"] - min_clientes["min_anual_mwh"]
                min_clientes = min_clientes[min_clientes["Diferencia"] < -1e-9].copy()

                nombres_clientes = pd.DataFrame()
                if not df_cons_y.empty and "nombre_cliente" in df_cons_y.columns:
                    nombres_clientes = (
                        df_cons_y.groupby("nemo_cliente", as_index=False)["nombre_cliente"]
                        .agg(lambda s: " / ".join(sorted(set(str(v) for v in s.dropna() if str(v).strip()))))
                    )
                    nombres_clientes["nemo_cliente"] = nombres_clientes["nemo_cliente"].astype(str)

                if not min_clientes.empty:
                    min_clientes["Cliente"] = min_clientes["nemo_cliente"]
                    if not nombres_clientes.empty:
                        min_clientes = min_clientes.merge(
                            nombres_clientes.rename(columns={"nombre_cliente": "Cliente"}),
                            on="nemo_cliente",
                            how="left",
                            suffixes=("", "_nombre"),
                        )
                        if "Cliente_nombre" in min_clientes.columns:
                            min_clientes["Cliente"] = min_clientes["Cliente_nombre"].fillna(min_clientes["Cliente"])

                    min_clientes = min_clientes.sort_values("Diferencia")
                    min_melt = min_clientes.melt(
                        id_vars=["Cliente", "nemo_cliente", "Diferencia"],
                        value_vars=["min_anual_mwh", "energia_real_anual_mwh"],
                        var_name="Serie",
                        value_name="MWh",
                    )
                    min_melt["Serie"] = min_melt["Serie"].map({
                        "min_anual_mwh": "Mínimo comprometido",
                        "energia_real_anual_mwh": "Asignado",
                    })
                    fig_min = px.bar(
                        min_melt,
                        x="Cliente",
                        y="MWh",
                        color="Serie",
                        barmode="group",
                        custom_data=["nemo_cliente"],
                        color_discrete_map={"Mínimo comprometido": "#6A9FC3", "Asignado": "#B91C1C"},
                        category_orders={"Cliente": min_clientes["Cliente"].tolist()},
                    )
                    fig_min.update_traces(hovertemplate="Cliente: %{x}<br>NEMO: %{customdata[0]}<br>%{y:,.0f} MWh<extra></extra>")
                    fig_min.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=100), legend_title_text="Serie", xaxis_title="Cliente", yaxis_title="Energía anual (MWh)")
                    st.plotly_chart(fig_min, use_container_width=True)

                    min_table = min_clientes.copy()
                    min_table["Energía mínima comprometida"] = min_table["min_anual_mwh"].round(0).astype(int)
                    min_table["Energía asignada"] = min_table["energia_real_anual_mwh"].round(0).astype(int)
                    min_table["Diferencia"] = min_table["Diferencia"].round(0).astype(int)
                    min_table = min_table[["Cliente", "Energía mínima comprometida", "Energía asignada", "Diferencia"]]
                    st.dataframe(
                        min_table.style.map(
                            lambda v: "color: #B91C1C; font-weight: 700" if isinstance(v, (int, float, np.integer, np.floating)) and v < 0 else "",
                            subset=["Diferencia"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("Cumplimiento total: ningún cliente se encuentra bajo su mínimo contractual anual.")
            else:
                st.info("No hay datos suficientes en `clientes_anual` para verificar mínimos contractuales.")

            st.markdown("### Clientes sobreasignados frente al máximo contractual anual")
            if not clientes_anual.empty and {"nemo_cliente", "energia_real_anual_mwh", "max_anual_mwh"}.issubset(clientes_anual.columns):
                max_clientes = clientes_anual.copy()
                max_clientes["nemo_cliente"] = max_clientes["nemo_cliente"].astype(str)
                max_clientes["max_anual_mwh"] = pd.to_numeric(max_clientes["max_anual_mwh"], errors="coerce")
                max_clientes["energia_real_anual_mwh"] = pd.to_numeric(max_clientes["energia_real_anual_mwh"], errors="coerce").fillna(0.0)
                max_clientes = max_clientes[max_clientes["max_anual_mwh"].notna()].copy()
                max_clientes["Diferencia"] = max_clientes["energia_real_anual_mwh"] - max_clientes["max_anual_mwh"]
                max_clientes = max_clientes[max_clientes["Diferencia"] > 1e-9].copy()

                cantidad_max_sobre = int(max_clientes["nemo_cliente"].nunique()) if not max_clientes.empty else 0
                st.metric("Clientes sobreasignados frente al máximo anual", f"{cantidad_max_sobre:,}")

                if not max_clientes.empty:
                    max_clientes["Cliente"] = max_clientes["nemo_cliente"]
                    if not nombres_clientes.empty:
                        max_clientes = max_clientes.merge(
                            nombres_clientes.rename(columns={"nombre_cliente": "Cliente nombre"}),
                            on="nemo_cliente",
                            how="left",
                        )
                        if "Cliente nombre" in max_clientes.columns:
                            max_clientes["Cliente"] = max_clientes["Cliente nombre"].fillna(max_clientes["Cliente"])
                    max_clientes = max_clientes.sort_values("Diferencia", ascending=False)
                    max_table = max_clientes[["Cliente", "max_anual_mwh", "energia_real_anual_mwh", "Diferencia"]].copy()
                    max_table = max_table.rename(columns={
                        "max_anual_mwh": "Energía máxima acordada",
                        "energia_real_anual_mwh": "Energía asignada",
                    })
                    max_table["Energía máxima acordada"] = max_table["Energía máxima acordada"].round(0).astype(int)
                    max_table["Energía asignada"] = max_table["Energía asignada"].round(0).astype(int)
                    max_table["Diferencia"] = max_table["Diferencia"].round(0).astype(int)
                    st.dataframe(
                        max_table.style.map(
                            lambda v: "color: #B91C1C; font-weight: 700" if isinstance(v, (int, float, np.integer, np.floating)) and v > 0 else "",
                            subset=["Diferencia"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("No existen sobreasignaciones frente al máximo contractual anual.")
            else:
                st.info("No hay datos suficientes en `clientes_anual` para verificar sobreasignaciones frente a `max_anual_mwh`.")

            st.caption("Los resultados mensuales frente al consumo y los máximos contractuales anuales se presentan por separado.")

            cons_cm = _consumption_by_client_month(df_cons_y)
            alloc_cm = _alloc_by_client_month(results)
            df_over_b = _client_month_overassign(cons_cm, alloc_cm)
            if df_over_b.empty:
                st.success("OK: no se detectaron casos de asignación mayor al consumo.")
            else:
                st.error(f"Se detectaron {len(df_over_b)} filas cliente-mes con asignación > consumo.")

            st.divider()
            st.subheader("Exportación Excel")
            econ_b = _annual_economics(results)
            df_econ_b = pd.DataFrame([econ_b]).T.reset_index()
            df_econ_b.columns = ["Concepto", "USD"]
            df_limits_b = _client_minmax_table(df_lim, results, int(anio))
            df_gen_take_b = _gen_avail_vs_taken_table(df_gen_y, results)

            xlsx_all_b = _dfs_to_excel_bytes(
                {
                    "economia_anual": df_econ_b,
                    "resumen_mensual": results["resumen_mensual"],
                    "por_generador_pr": results["por_generador_prioridad"],
                    "detalle": results["detalle"],
                    "minmax_cliente": df_limits_b,
                    "gen_disp_vs_tom": df_gen_take_b,
                    "overassign": df_over_b,
                }
            )
            st.download_button(
                "Descargar TODO Baseline (Excel)",
                data=xlsx_all_b,
                file_name=f"baseline_todo_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                "Descargar resultados Baseline (Excel - formato estándar)",
                data=results_to_excel_bytes(results),
                file_name=f"resultados_baseline_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.divider()
            st.subheader("Detalle técnico y auditoría")

            with st.expander("Economía anual", expanded=False):
                st.dataframe(df_econ_b, use_container_width=True, hide_index=True)

            with st.expander("Resumen mensual y compras por generador", expanded=False):
                st.markdown("**Resumen mensual**")
                st.dataframe(results["resumen_mensual"], use_container_width=True, hide_index=True)
                st.markdown("**Compras por generador (PG1/PG2 internas)**")
                st.dataframe(results["por_generador_prioridad"], use_container_width=True, hide_index=True)

            with st.expander("Detalle de transacciones", expanded=False):
                st.dataframe(results["detalle"], use_container_width=True, hide_index=True)

            with st.expander("Mínimo / máximo anual por cliente", expanded=False):
                st.dataframe(df_limits_b, use_container_width=True, hide_index=True)

            with st.expander("Generadores: disponible vs tomado", expanded=False):
                st.dataframe(df_gen_take_b, use_container_width=True, hide_index=True)

            with st.expander("Validación: asignación mayor al consumo", expanded=False):
                if df_over_b.empty:
                    st.success("OK: no se detectaron casos de asignación mayor al consumo.")
                else:
                    st.dataframe(df_over_b, use_container_width=True, hide_index=True)

            st.caption("Las tablas técnicas conservan las fórmulas, periodizaciones y resultados originales del simulador.", help=None)

        st.divider()
        st.subheader("Edición de nominaciones mensuales")
        st.markdown("**`nominal_mwh` — nominaciones mensuales**")
        edited = st.data_editor(
            df_nom_y,
            use_container_width=True,
            disabled=[c for c in df_nom_y.columns if c not in ["nominal_mwh"]],
            hide_index=True,
        )
        st.session_state["baseline_nominal_edited_df"] = edited

    # -------------------- TAB 2: Optimizador MILP --------------------
    with tab2:
        st.subheader("Optimizador MILP")

        st.write(
            "La MILP propone **nominaciones** (nominal_mwh por acuerdo-mes) para maximizar el margen esperado "
            "bajo las restricciones cargadas. Luego se re-simula para obtener asignación real."
        )

        st.info(
            "Si querés que los **mínimos sean obligatorios** y el sistema no alcanza a cubrirlos, el resultado "
            "debe considerarse **inaceptable**. Si está tildado 'Mínimos anuales obligatorios', se bloquea OPT si hay incumplimientos."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_block = st.number_input("Mínimo por activación (P2+) [MWh]", min_value=0.0, value=50.0, step=10.0)
        with col2:
            pen = st.number_input("Penalización fija por activación P2+ [USD]", min_value=0.0, value=0.0, step=50.0)
        with col3:
            tlim = st.number_input("Límite de tiempo solver [s]", min_value=5, value=30, step=5)
        with col4:
            run_opt = st.button("Ejecutar optimización MILP", type="primary")

        if run_opt:
            df_gen_y = _filter_year_scenario(df_gen, int(anio), gen_scen)
            df_cons_y = _filter_year_scenario(df_cons, int(anio), cons_scen)

            try:
                sol = solve_milp_nominal(
                    df_gen=df_gen_y,
                    df_cons=df_cons_y,
                    df_lim=df_lim,
                    df_nom_template=df_nom_base,
                    anio=int(anio),
                    gen_scenario=None,
                    cons_scenario=None,
                    min_block_mwh_p2=float(min_block),
                    fixed_penalty_p2_usd=float(pen),
                    solver_time_limit_s=int(tlim),
                )
            except ModuleNotFoundError:
                st.error("No está instalada la biblioteca 'pulp'. Instalá dependencias con: `pip install -r requirements.txt`")
                st.stop()

            results_opt = simulate_cammesa(df_gen_y, df_cons_y, df_lim, sol.df_nom_opt, anio=int(anio))
            df_limits_o = _client_minmax_table(df_lim, results_opt, int(anio))
            df_min_viol = _min_violations_table(df_limits_o)
            cons_cm = _consumption_by_client_month(df_cons_y)
            alloc_cm = _alloc_by_client_month(results_opt)
            df_over_o = _client_month_overassign(cons_cm, alloc_cm)

            st.session_state["opt_solution"] = sol
            st.session_state["opt_results"] = results_opt
            st.session_state["opt_meta"] = {"anio": int(anio), "gen": gen_scen, "cons": cons_scen}
            st.session_state["opt_gen_y"] = df_gen_y
            st.session_state["opt_cons_y"] = df_cons_y
            st.session_state["opt_limits"] = df_limits_o
            st.session_state["opt_overassign"] = df_over_o
            st.session_state["opt_min_viol"] = df_min_viol
            st.session_state["opt_blocked"] = bool(strict_min and not df_min_viol.empty)

        if "opt_solution" in st.session_state:
            sol = st.session_state["opt_solution"]
            st.markdown(f"**Estado solver:** {sol.status}")
            st.markdown(f"**Objetivo (USD):** {sol.objective_usd:,.0f}")
            st.markdown("### Nominaciones propuestas (preview)")
            st.dataframe(sol.df_nom_opt.head(200), use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar nominaciones optimizadas (Excel)",
                data=_df_to_excel_bytes(sol.df_nom_opt, "nominaciones_opt"),
                file_name=f"nominaciones_opt_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if "opt_results" in st.session_state:
            results = st.session_state["opt_results"]
            df_gen_y = st.session_state["opt_gen_y"]
            df_cons_y = st.session_state["opt_cons_y"]
            k = _kpis(results)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Margen neto anual (USD)", f"{k['Margen neto anual (USD)']:,.0f}")
            c2.metric("Penalidad TOP anual (USD)", f"{k['Penalidad TOP anual (USD)']:,.0f}")
            c3.metric("Energía asignada anual (MWh)", f"{k['Energía asignada anual (MWh)']:,.0f}")
            c4.metric("Energía no abastecida anual (MWh)", f"{k['Energía no abastecida anual (MWh)']:,.0f}")

            st.divider()
            st.markdown("### Economía anual (OPT)")
            econ_o = _annual_economics(results)
            df_econ_o = pd.DataFrame([econ_o]).T.reset_index()
            df_econ_o.columns = ["Concepto", "USD"]
            st.dataframe(df_econ_o, use_container_width=True, hide_index=True)

            df_limits_o = st.session_state.get("opt_limits", _client_minmax_table(df_lim, results, int(anio)))
            df_min_viol = st.session_state.get("opt_min_viol", _min_violations_table(df_limits_o))
            df_over_o = st.session_state.get("opt_overassign", _client_month_overassign(_consumption_by_client_month(df_cons_y), _alloc_by_client_month(results)))

            if strict_min and not df_min_viol.empty:
                st.error(f"OPT BLOQUEADO: se incumple mínimo anual en {len(df_min_viol)} clientes.")
            if strict_no_overassign and not df_over_o.empty:
                st.error(f"OPT ALERTA: se detectó asignación > consumo en {len(df_over_o)} filas cliente-mes.")

            st.divider()
            st.markdown("### Resultados (Optimizado)")
            st.markdown("**Resumen mensual**")
            st.dataframe(results["resumen_mensual"], use_container_width=True, hide_index=True)
            st.markdown("**Compras por generador (PG1/PG2 internas)**")
            st.dataframe(results["por_generador_prioridad"], use_container_width=True, hide_index=True)
            st.markdown("**Detalle de transacciones**")
            st.dataframe(results["detalle"], use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Mínimo / Máximo anual por cliente (OPT)")
            st.dataframe(df_limits_o, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### Generadores: disponible vs tomado (OPT)")
            df_gen_take_o = _gen_avail_vs_taken_table(df_gen_y, results)
            st.dataframe(df_gen_take_o, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### Validación: asignación > consumo (OPT)")
            if df_over_o.empty:
                st.success("OK: no se detectaron casos de asignación mayor al consumo.")
            else:
                st.dataframe(df_over_o, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Exportación (OPT)")
            xlsx_all_o = _dfs_to_excel_bytes(
                {
                    "economia_anual": df_econ_o,
                    "resumen_mensual": results["resumen_mensual"],
                    "por_generador_pr": results["por_generador_prioridad"],
                    "detalle": results["detalle"],
                    "minmax_cliente": df_limits_o,
                    "gen_disp_vs_tom": df_gen_take_o,
                    "overassign": df_over_o,
                    "min_viol": df_min_viol,
                }
            )
            st.download_button(
                "Descargar TODO OPT (Excel)",
                data=xlsx_all_o,
                file_name=f"opt_todo_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button(
                "Descargar resultados OPT (Excel - formato estándar)",
                data=results_to_excel_bytes(results),
                file_name=f"resultados_opt_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # -------------------- TAB 3: Comparativa --------------------
    with tab3:
        st.subheader("Comparativa + Resumen")
        if "baseline_results" not in st.session_state or "opt_results" not in st.session_state:
            st.warning("Primero ejecutá una simulación Baseline y una Optimización MILP.")
            return

        b = st.session_state["baseline_results"]
        o = st.session_state["opt_results"]
        df_gen_b = st.session_state.get("baseline_gen_y", _filter_year_scenario(df_gen, int(anio), gen_scen))
        df_cons_b = st.session_state.get("baseline_cons_y", _filter_year_scenario(df_cons, int(anio), cons_scen))
        df_gen_o = st.session_state.get("opt_gen_y", df_gen_b)
        kb = _kpis(b)
        ko = _kpis(o)
        econ_b = _annual_economics(b)
        econ_o = _annual_economics(o)

        df_kpi = pd.DataFrame({"Indicador": list(kb.keys()), "Baseline": list(kb.values()), "Optimizado": [ko[k] for k in kb.keys()]})
        df_kpi["Delta"] = df_kpi["Optimizado"] - df_kpi["Baseline"]
        st.markdown("### KPIs")
        st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Economía anual: Base vs Opt (Bruto / Penalidad TOP / Neto)")
        df_econ_cmp = pd.DataFrame({"Concepto": list(econ_b.keys()), "Baseline": list(econ_b.values()), "Optimizado": [econ_o[k] for k in econ_b.keys()]})
        df_econ_cmp["Delta"] = df_econ_cmp["Optimizado"] - df_econ_cmp["Baseline"]
        st.dataframe(df_econ_cmp, use_container_width=True, hide_index=True)

        cons_cm = _consumption_by_client_month(df_cons_b)
        alloc_b = _alloc_by_client_month(b).rename(columns={"asignado_mwh": "asignado_baseline_mwh"})
        alloc_o = _alloc_by_client_month(o).rename(columns={"asignado_mwh": "asignado_opt_mwh"})
        df_clients = cons_cm.merge(alloc_b, on=["nemo_cliente", "mes"], how="left").merge(alloc_o, on=["nemo_cliente", "mes"], how="left")
        df_clients["asignado_baseline_mwh"] = df_clients["asignado_baseline_mwh"].fillna(0.0)
        df_clients["asignado_opt_mwh"] = df_clients["asignado_opt_mwh"].fillna(0.0)
        df_clients["delta_asignado_mwh"] = df_clients["asignado_opt_mwh"] - df_clients["asignado_baseline_mwh"]
        df_clients = df_clients.sort_values(["nemo_cliente", "mes"]).reset_index(drop=True)

        av_gm = _available_by_gen_month(df_gen_b)
        taken_b = _alloc_by_gen_month(b).rename(columns={"tomado_mwh": "tomado_baseline_mwh"})
        taken_o = _alloc_by_gen_month(o).rename(columns={"tomado_mwh": "tomado_opt_mwh"})
        df_gens = av_gm.merge(taken_b, on=["nemo_generador", "mes"], how="left").merge(taken_o, on=["nemo_generador", "mes"], how="left")
        df_gens["tomado_baseline_mwh"] = df_gens["tomado_baseline_mwh"].fillna(0.0)
        df_gens["tomado_opt_mwh"] = df_gens["tomado_opt_mwh"].fillna(0.0)
        df_gens["delta_tomado_mwh"] = df_gens["tomado_opt_mwh"] - df_gens["tomado_baseline_mwh"]
        df_gens = df_gens.sort_values(["nemo_generador", "mes"]).reset_index(drop=True)

        st.divider()
        st.markdown("### Tabla: Cliente-Mes (Consumo + Base + Opt + Delta)")
        st.dataframe(df_clients[["nemo_cliente", "mes", "consumo_mwh", "asignado_baseline_mwh", "asignado_opt_mwh", "delta_asignado_mwh"]], use_container_width=True, hide_index=True)
        st.markdown("### Tabla: Generador-Mes (Disponible + Base + Opt + Delta)")
        st.dataframe(df_gens[["nemo_generador", "mes", "disponible_mwh", "tomado_baseline_mwh", "tomado_opt_mwh", "delta_tomado_mwh"]], use_container_width=True, hide_index=True)

        df_min_viol_opt = st.session_state.get("opt_min_viol", pd.DataFrame())
        if not df_min_viol_opt.empty:
            st.divider()
            st.error("### Incumplimientos de mínimo (OPT)")
            st.dataframe(df_min_viol_opt, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("## Gráficos")
        st.markdown("### 1) Energía anual por generador – Base vs Opt")
        g1 = df_gens.groupby("nemo_generador", as_index=False)[["tomado_baseline_mwh", "tomado_opt_mwh"]].sum().rename(columns={"nemo_generador": "Generador", "tomado_baseline_mwh": "Base", "tomado_opt_mwh": "Opt"})
        g1m = g1.melt(id_vars=["Generador"], value_vars=["Base", "Opt"], var_name="Escenario", value_name="Energia_MWh")
        st.plotly_chart(px.bar(g1m, x="Generador", y="Energia_MWh", color="Escenario", barmode="group"), use_container_width=True)

        st.markdown("### 2) Energía asignada total por mes – Base vs Opt")
        g2 = df_clients.groupby("mes", as_index=False)[["asignado_baseline_mwh", "asignado_opt_mwh"]].sum().rename(columns={"mes": "Mes", "asignado_baseline_mwh": "Base", "asignado_opt_mwh": "Opt"}).sort_values("Mes")
        g2m = g2.melt(id_vars=["Mes"], value_vars=["Base", "Opt"], var_name="Escenario", value_name="Energia_asignada_MWh")
        st.plotly_chart(px.bar(g2m, x="Mes", y="Energia_asignada_MWh", color="Escenario", barmode="group"), use_container_width=True)

        st.markdown("### 3) Diferencia de margen BRUTO por mes – (Opt - Base)")
        rm_b = b.get("resumen_mensual", pd.DataFrame()).copy()
        rm_o = o.get("resumen_mensual", pd.DataFrame()).copy()
        df_margin = pd.DataFrame(columns=["mes", "margen_base_usd", "margen_opt_usd", "delta_margen_usd"])
        if not rm_b.empty and not rm_o.empty and {"mes", "margen_usd"}.issubset(rm_b.columns) and {"mes", "margen_usd"}.issubset(rm_o.columns):
            df_margin = rm_b[["mes", "margen_usd"]].merge(rm_o[["mes", "margen_usd"]], on="mes", suffixes=("_base", "_opt")).rename(columns={"margen_usd_base": "margen_base_usd", "margen_usd_opt": "margen_opt_usd"})
            df_margin["delta_margen_usd"] = df_margin["margen_opt_usd"] - df_margin["margen_base_usd"]
            df_margin = df_margin.sort_values("mes").reset_index(drop=True)
        st.plotly_chart(px.bar(df_margin, x="mes", y="delta_margen_usd"), use_container_width=True)

        st.divider()
        st.markdown("## Resumen automático")
        st.markdown(_summary_text(df_clients_cmp=df_clients, df_gens_cmp=df_gens, df_margin_month=df_margin, df_min_viol_opt=df_min_viol_opt, kpi_base=kb, kpi_opt=ko, econ_base=econ_b, econ_opt=econ_o))

        st.divider()
        st.markdown("## Exportación (todo en Excel)")
        xlsx_cmp = _dfs_to_excel_bytes({
            "kpis": df_kpi,
            "economia_anual_cmp": df_econ_cmp,
            "clientes_mes": df_clients[["nemo_cliente", "mes", "consumo_mwh", "asignado_baseline_mwh", "asignado_opt_mwh", "delta_asignado_mwh"]],
            "generadores_mes": df_gens[["nemo_generador", "mes", "disponible_mwh", "tomado_baseline_mwh", "tomado_opt_mwh", "delta_tomado_mwh"]],
            "margen_bruto_mes": df_margin,
            "min_viol_opt": df_min_viol_opt,
        })
        st.download_button("Descargar Comparativa + Resumen (Excel)", data=xlsx_cmp, file_name=f"comparativa_completa_{anio}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()