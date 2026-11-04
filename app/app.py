from __future__ import annotations

"""App Streamlit - Asignación de energía (demo).

Pestañas:
  1) Simulador Baseline (replica lógica de asignación tipo CAMMESA).
  2) Optimizador MILP (propone nominaciones y vuelve a simular).
  3) Comparativa (baseline vs optimizado).
"""

import sys
from pathlib import Path

# Permite resolver el paquete `src` aunque Streamlit inicie con `app/` en sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.core.cammesa_simulator import simulate_cammesa
from src.core.optimizer_milp import solve_milp_nominal
from src.io.exporter import results_to_excel_bytes
from src.io.loader import load_excel
from src.io.overrides import apply_nominal_overrides


st.set_page_config(page_title="Asignación de Energía - Demo", layout="wide")


DATA_FILE_DEFAULT = "data/BD_Ejemplo_2026_escenarios.xlsx"


def _scenario_options(df: pd.DataFrame) -> list[str]:
    if "escenario" not in df.columns:
        return ["(único)"]
    return sorted(df["escenario"].astype(str).unique().tolist())


def _filter_year_scenario(df: pd.DataFrame, anio: int, escenario: str | None) -> pd.DataFrame:
    out = df[df["anio"].astype(int) == int(anio)].copy()
    if "escenario" in out.columns and escenario and escenario != "(único)":
        out = out[out["escenario"].astype(str) == str(escenario)].copy()
    return out


def _kpis(results: dict[str, pd.DataFrame]) -> dict[str, float]:
    rm = results["resumen_mensual"].copy()
    k = {
        "Margen anual (USD)": float(rm["margen_usd"].sum()),
        "Energía asignada anual (MWh)": float(rm["asignado_MWh"].sum()),
        "Energía no abastecida anual (MWh)": float(rm["no_abastecido_MWh"].sum()),
    }
    return k


def main() -> None:
    st.title("Asignación de energía renovable – Simulador y Optimizador (Demo)")

    with st.sidebar:
        st.header("Datos")
        data_file = st.text_input("Archivo Excel de datos", value=DATA_FILE_DEFAULT)
        st.caption("Tip: dejá el archivo dentro de la carpeta 'data/'.")

        # Cargar
        try:
            data = load_excel(data_file)
            df_gen = data["gen"]
            df_cons = data["cons"]
            df_lim = data["lim"]
            df_nom_base = data["nom"]
        except Exception as e:
            st.error(f"No se pudo leer el Excel: {e}")
            st.stop()

        anios = sorted(set(df_nom_base["anio"].astype(int).unique().tolist()))
        anio = st.selectbox("Año", anios, index=len(anios) - 1)

        gen_scen = st.selectbox("Escenario generación", _scenario_options(df_gen))
        cons_scen = st.selectbox("Escenario consumo", _scenario_options(df_cons))

        st.divider()
        st.caption("Las nominaciones sólo se editan en el Simulador Baseline.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Simulador Baseline", "Optimizador MILP", "Comparativa"])

    # -------------------- TAB 1: Baseline --------------------
    with tab1:
        st.subheader("Simulador Baseline")
        st.write(
            "En esta pestaña se replica la lógica de asignación a partir de las nominaciones "
            "cargadas. El resultado se puede exportar a Excel en formato de tablas."
        )

        # Nominaciones del año seleccionado
        df_nom_y = df_nom_base[df_nom_base["anio"].astype(int) == int(anio)].copy()
        df_nom_y = df_nom_y.sort_values(["nemo_cliente", "prioridad_asignacion", "nemo_generador", "mes"])\
                       .reset_index(drop=True)

        st.markdown("**Edición de nominaciones mensuales (nominal_mwh)**")
        edited = st.data_editor(
            df_nom_y,
            use_container_width=True,
            disabled=[
                c
                for c in df_nom_y.columns
                if c not in ["nominal_mwh"]
            ],
            hide_index=True,
        )

        run = st.button("Ejecutar simulación baseline", type="primary")
        if run:
            # Aplicar overrides (edición) sólo al nominal
            df_nom_use = apply_nominal_overrides(df_nom_base, edited)

            df_gen_y = _filter_year_scenario(df_gen, anio, gen_scen)
            df_cons_y = _filter_year_scenario(df_cons, anio, cons_scen)

            results = simulate_cammesa(df_gen_y, df_cons_y, df_lim, df_nom_use, anio=anio)
            st.session_state["baseline_results"] = results
            st.session_state["baseline_meta"] = {"anio": anio, "gen": gen_scen, "cons": cons_scen}

        if "baseline_results" in st.session_state:
            results = st.session_state["baseline_results"]
            k = _kpis(results)

            c1, c2, c3 = st.columns(3)
            c1.metric("Margen anual (USD)", f"{k['Margen anual (USD)']:,.0f}")
            c2.metric("Energía asignada anual (MWh)", f"{k['Energía asignada anual (MWh)']:,.0f}")
            c3.metric("Energía no abastecida anual (MWh)", f"{k['Energía no abastecida anual (MWh)']:,.0f}")

            st.divider()
            st.markdown("### Resultados")

            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("**Resumen mensual**")
                st.dataframe(results["resumen_mensual"], use_container_width=True, hide_index=True)
            with colB:
                st.markdown("**Compras por generador (PG1/PG2 internas)**")
                st.dataframe(results["por_generador_prioridad"], use_container_width=True, hide_index=True)

            st.markdown("**Detalle de transacciones**")
            st.dataframe(results["detalle"], use_container_width=True, hide_index=True)

            xlsx_bytes = results_to_excel_bytes(results)
            st.download_button(
                "Descargar resultados (Excel)",
                data=xlsx_bytes,
                file_name=f"resultados_baseline_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # -------------------- TAB 2: Optimización MILP --------------------
    with tab2:
        st.subheader("Optimizador MILP")
        st.write(
            "Esta pestaña propone **nominaciones** mediante un modelo de Programación Lineal Mixta "
            "(MILP) y luego vuelve a simular la asignación. El componente entero (binario) se usa "
            "para representar la 'activación' de acuerdos de menor prioridad (p.ej. prioridad_asignacion=2)."
        )

        st.info(
            "Nota: si el solver devuelve **Infeasible**, significa que con las restricciones actuales "
            "(oferta/demanda y mínimos/máximos anuales) no existe una solución factible para ese escenario."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_block = st.number_input("Mínimo por activación (P2+) [MWh]", min_value=0.0, value=50.0, step=10.0)
        with col2:
            pen = st.number_input("Penalización fija por activación P2+ [USD]", min_value=0.0, value=100.0, step=50.0)
        with col3:
            tlim = st.number_input("Límite de tiempo solver [s]", min_value=5, value=30, step=5)
        with col4:
            run_opt = st.button("Ejecutar optimización MILP", type="primary")

        if run_opt:
            df_gen_y = _filter_year_scenario(df_gen, anio, gen_scen)
            df_cons_y = _filter_year_scenario(df_cons, anio, cons_scen)

            try:
                sol = solve_milp_nominal(
                    df_gen=df_gen_y,
                    df_cons=df_cons_y,
                    df_lim=df_lim,
                    df_nom_template=df_nom_base,
                    anio=anio,
                    gen_scenario=None,  # ya filtramos
                    cons_scenario=None,
                    min_block_mwh_p2=float(min_block),
                    fixed_penalty_p2_usd=float(pen),
                    solver_time_limit_s=int(tlim),
                )
            except ModuleNotFoundError:
                st.error(
                    "No está instalada la biblioteca 'pulp'. Instalá dependencias con: `pip install -r requirements.txt`"
                )
                st.stop()

            st.session_state["opt_solution"] = sol

            # Re-simular con nominaciones optimizadas
            results_opt = simulate_cammesa(df_gen_y, df_cons_y, df_lim, sol.df_nom_opt, anio=anio)
            st.session_state["opt_results"] = results_opt
            st.session_state["opt_meta"] = {"anio": anio, "gen": gen_scen, "cons": cons_scen}

        if "opt_solution" in st.session_state:
            sol = st.session_state["opt_solution"]
            st.markdown(f"**Estado solver:** {sol.status}  ")
            st.markdown(f"**Objetivo (USD):** {sol.objective_usd:,.0f}")

            st.markdown("### Nominaciones propuestas")
            st.dataframe(sol.df_nom_opt.head(200), use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar nominaciones optimizadas (Excel)",
                data=_df_to_excel_bytes(sol.df_nom_opt, "nominaciones"),
                file_name=f"nominaciones_opt_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if "opt_results" in st.session_state:
            results = st.session_state["opt_results"]
            k = _kpis(results)
            c1, c2, c3 = st.columns(3)
            c1.metric("Margen anual (USD)", f"{k['Margen anual (USD)']:,.0f}")
            c2.metric("Energía asignada anual (MWh)", f"{k['Energía asignada anual (MWh)']:,.0f}")
            c3.metric("Energía no abastecida anual (MWh)", f"{k['Energía no abastecida anual (MWh)']:,.0f}")

            xlsx_bytes = results_to_excel_bytes(results)
            st.download_button(
                "Descargar resultados optimización (Excel)",
                data=xlsx_bytes,
                file_name=f"resultados_opt_{anio}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # -------------------- TAB 3: Comparativa --------------------
    with tab3:
        st.subheader("Comparativa")
        st.write("Comparación entre el último Baseline ejecutado y la última Optimización ejecutada.")

        if "baseline_results" not in st.session_state or "opt_results" not in st.session_state:
            st.warning("Primero ejecutá una simulación Baseline y una Optimización MILP.")
        else:
            b = st.session_state["baseline_results"]
            o = st.session_state["opt_results"]
            kb = _kpis(b)
            ko = _kpis(o)

            df_cmp = pd.DataFrame(
                {
                    "Indicador": list(kb.keys()),
                    "Baseline": list(kb.values()),
                    "Optimizado": [ko[k] for k in kb.keys()],
                }
            )
            df_cmp["Delta"] = df_cmp["Optimizado"] - df_cmp["Baseline"]
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

            st.markdown("### Margen mensual")
            rm_b = b["resumen_mensual"].copy()
            rm_o = o["resumen_mensual"].copy()
            rm = rm_b[["mes", "margen_usd"]].merge(
                rm_o[["mes", "margen_usd"]], on="mes", suffixes=("_baseline", "_opt")
            )
            st.line_chart(rm.set_index("mes"))


def _df_to_excel_bytes(df: pd.DataFrame, sheet: str) -> bytes:
    import io

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


if __name__ == "__main__":
    main()
