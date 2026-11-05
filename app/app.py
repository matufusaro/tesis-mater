from __future__ import annotations

"""
App Streamlit - Asignación de energía (demo)

Pestañas:
  1) Simulador Baseline
  2) Optimizador MILP
  3) Comparativa + Resumen + (3 gráficos)

Cambios principales (respecto a tu versión):
- Se agrega economía anual explícita (Ingreso, Costo, Margen Bruto, Penalidad TOP anual, Margen Neto).
- KPIs pasan a mostrar Margen NETO anual + Penalidad TOP anual.
- Comparativa incorpora tabla de economía anual Baseline vs Optimizado.
- Exportación Excel incluye la hoja de economía anual.
"""

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
    st.title("Asignación de energía renovable – Simulador y Optimizador (Demo)")

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
        st.subheader("Simulador Baseline")

        df_nom_y = df_nom_base[_safe_int_series(df_nom_base["anio"]) == int(anio)].copy()
        df_nom_y = df_nom_y.sort_values(["nemo_cliente", "prioridad_asignacion", "nemo_generador", "mes"]).reset_index(drop=True)

        st.markdown("**Edición de nominaciones mensuales (nominal_mwh)**")
        edited = st.data_editor(
            df_nom_y,
            use_container_width=True,
            disabled=[c for c in df_nom_y.columns if c not in ["nominal_mwh"]],
            hide_index=True,
        )

        run = st.button("Ejecutar simulación baseline", type="primary")
        if run:
            df_nom_use = apply_nominal_overrides(df_nom_base, edited)
            df_gen_y = _filter_year_scenario(df_gen, int(anio), gen_scen)
            df_cons_y = _filter_year_scenario(df_cons, int(anio), cons_scen)

            results = simulate_cammesa(df_gen_y, df_cons_y, df_lim, df_nom_use, anio=int(anio))
            st.session_state["baseline_results"] = results
            st.session_state["baseline_meta"] = {"anio": int(anio), "gen": gen_scen, "cons": cons_scen}
            st.session_state["baseline_gen_y"] = df_gen_y
            st.session_state["baseline_cons_y"] = df_cons_y

        if "baseline_results" in st.session_state:
            results = st.session_state["baseline_results"]
            df_gen_y = st.session_state["baseline_gen_y"]
            df_cons_y = st.session_state["baseline_cons_y"]

            k = _kpis(results)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Margen neto anual (USD)", f"{k['Margen neto anual (USD)']:,.0f}")
            c2.metric("Penalidad TOP anual (USD)", f"{k['Penalidad TOP anual (USD)']:,.0f}")
            c3.metric("Energía asignada anual (MWh)", f"{k['Energía asignada anual (MWh)']:,.0f}")
            c4.metric("Energía no abastecida anual (MWh)", f"{k['Energía no abastecida anual (MWh)']:,.0f}")

            st.divider()
            st.markdown("### Economía anual (Baseline)")
            econ_b = _annual_economics(results)
            df_econ_b = pd.DataFrame([econ_b]).T.reset_index()
            df_econ_b.columns = ["Concepto", "USD"]
            st.dataframe(df_econ_b, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Resultados (Baseline)")
            st.markdown("**Resumen mensual**")
            st.dataframe(results["resumen_mensual"], use_container_width=True, hide_index=True)

            st.markdown("**Compras por generador (PG1/PG2 internas)**")
            st.dataframe(results["por_generador_prioridad"], use_container_width=True, hide_index=True)

            st.markdown("**Detalle de transacciones**")
            st.dataframe(results["detalle"], use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Mínimo / Máximo anual por cliente (Baseline)")
            df_limits_b = _client_minmax_table(df_lim, results, int(anio))
            st.dataframe(df_limits_b, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Generadores: disponible vs tomado (Baseline)")
            df_gen_take_b = _gen_avail_vs_taken_table(df_gen_y, results)
            st.dataframe(df_gen_take_b, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Validación: asignación > consumo (Baseline)")
            cons_cm = _consumption_by_client_month(df_cons_y)
            alloc_cm = _alloc_by_client_month(results)
            df_over_b = _client_month_overassign(cons_cm, alloc_cm)
            if df_over_b.empty:
                st.success("OK: no se detectaron casos de asignación mayor al consumo.")
            else:
                st.error(f"Se detectaron {len(df_over_b)} filas cliente-mes con asignación > consumo.")
                st.dataframe(df_over_b, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Exportación (Baseline)")
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

    # -------------------- TAB 2: Optimización MILP --------------------
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

        df_kpi = pd.DataFrame(
            {"Indicador": list(kb.keys()), "Baseline": list(kb.values()), "Optimizado": [ko[k] for k in kb.keys()]}
        )
        df_kpi["Delta"] = df_kpi["Optimizado"] - df_kpi["Baseline"]
        st.markdown("### KPIs")
        st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Economía anual: Base vs Opt (Bruto / Penalidad TOP / Neto)")
        df_econ_cmp = pd.DataFrame(
            {
                "Concepto": list(econ_b.keys()),
                "Baseline": list(econ_b.values()),
                "Optimizado": [econ_o[k] for k in econ_b.keys()],
            }
        )
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
        st.dataframe(
            df_clients[["nemo_cliente", "mes", "consumo_mwh", "asignado_baseline_mwh", "asignado_opt_mwh", "delta_asignado_mwh"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Tabla: Generador-Mes (Disponible + Base + Opt + Delta)")
        st.dataframe(
            df_gens[["nemo_generador", "mes", "disponible_mwh", "tomado_baseline_mwh", "tomado_opt_mwh", "delta_tomado_mwh"]],
            use_container_width=True,
            hide_index=True,
        )

        df_min_viol_opt = st.session_state.get("opt_min_viol", pd.DataFrame())
        if not df_min_viol_opt.empty:
            st.divider()
            st.error("### Incumplimientos de mínimo (OPT)")
            st.dataframe(df_min_viol_opt, use_container_width=True, hide_index=True)

        # -------------------- SOLO 3 gráficos --------------------
        st.divider()
        st.markdown("## Gráficos")

        st.markdown("### 1) Energía anual por generador – Base vs Opt")
        g1 = (
            df_gens.groupby("nemo_generador", as_index=False)[["tomado_baseline_mwh", "tomado_opt_mwh"]].sum()
            .rename(columns={"nemo_generador": "Generador", "tomado_baseline_mwh": "Base", "tomado_opt_mwh": "Opt"})
        )
        g1m = g1.melt(id_vars=["Generador"], value_vars=["Base", "Opt"], var_name="Escenario", value_name="Energia_MWh")
        fig1 = px.bar(g1m, x="Generador", y="Energia_MWh", color="Escenario", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 2) Energía asignada total por mes – Base vs Opt")
        g2 = (
            df_clients.groupby("mes", as_index=False)[["asignado_baseline_mwh", "asignado_opt_mwh"]].sum()
            .rename(columns={"mes": "Mes", "asignado_baseline_mwh": "Base", "asignado_opt_mwh": "Opt"})
            .sort_values("Mes")
        )
        g2m = g2.melt(id_vars=["Mes"], value_vars=["Base", "Opt"], var_name="Escenario", value_name="Energia_asignada_MWh")
        fig2 = px.bar(g2m, x="Mes", y="Energia_asignada_MWh", color="Escenario", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 3) Diferencia de margen BRUTO por mes – (Opt - Base)")
        rm_b = b.get("resumen_mensual", pd.DataFrame()).copy()
        rm_o = o.get("resumen_mensual", pd.DataFrame()).copy()
        df_margin = pd.DataFrame(columns=["mes", "margen_base_usd", "margen_opt_usd", "delta_margen_usd"])
        if not rm_b.empty and not rm_o.empty and {"mes", "margen_usd"}.issubset(rm_b.columns) and {"mes", "margen_usd"}.issubset(rm_o.columns):
            df_margin = (
                rm_b[["mes", "margen_usd"]].merge(rm_o[["mes", "margen_usd"]], on="mes", suffixes=("_base", "_opt"))
                .rename(columns={"margen_usd_base": "margen_base_usd", "margen_usd_opt": "margen_opt_usd"})
            )
            df_margin["delta_margen_usd"] = df_margin["margen_opt_usd"] - df_margin["margen_base_usd"]
            df_margin = df_margin.sort_values("mes").reset_index(drop=True)

        fig3 = px.bar(df_margin, x="mes", y="delta_margen_usd")
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            f"Nota: el gráfico muestra Δ de margen BRUTO mensual. Penalidad TOP es ANUAL. "
            f"Δ Margen NETO anual = {econ_o['Margen neto anual (USD)'] - econ_b['Margen neto anual (USD)']:,.0f} USD."
        )

        st.divider()
        st.markdown("## Resumen automático")
        summary = _summary_text(
            df_clients_cmp=df_clients,
            df_gens_cmp=df_gens,
            df_margin_month=df_margin,
            df_min_viol_opt=df_min_viol_opt,
            kpi_base=kb,
            kpi_opt=ko,
            econ_base=econ_b,
            econ_opt=econ_o,
        )
        st.markdown(summary)

        st.divider()
        st.markdown("## Exportación (todo en Excel)")
        xlsx_cmp = _dfs_to_excel_bytes(
            {
                "kpis": df_kpi,
                "economia_anual_cmp": df_econ_cmp,
                "clientes_mes": df_clients[["nemo_cliente", "mes", "consumo_mwh", "asignado_baseline_mwh", "asignado_opt_mwh", "delta_asignado_mwh"]],
                "generadores_mes": df_gens[["nemo_generador", "mes", "disponible_mwh", "tomado_baseline_mwh", "tomado_opt_mwh", "delta_tomado_mwh"]],
                "margen_bruto_mes": df_margin,
                "min_viol_opt": df_min_viol_opt,
            }
        )
        st.download_button(
            "Descargar Comparativa + Resumen (Excel)",
            data=xlsx_cmp,
            file_name=f"comparativa_completa_{anio}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
