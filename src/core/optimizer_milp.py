from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class MilpSolution:
    status: str
    objective_usd: float
    df_nom_opt: pd.DataFrame
    df_vars: pd.DataFrame


def solve_milp_nominal(
    df_gen: pd.DataFrame,
    df_cons: pd.DataFrame,
    df_lim: pd.DataFrame,
    df_nom_template: pd.DataFrame,  # NO se usa (compatibilidad con app)
    anio: int,
    gen_scenario: str | None = None,
    cons_scenario: str | None = None,
    min_block_mwh_p2: float = 0.0,     # no se usa
    fixed_penalty_p2_usd: float = 0.0, # no se usa
    solver_time_limit_s: int = 30,
) -> MilpSolution:
    """
    Optimizador "desde cero":
    - Crea asignaciones x[c,g,m] usando SOLO generación + consumo + límites + precio de venta del cliente.
    - Construye df_nom_opt (nominaciones) desde cero.

    Requisitos de datos:
    - df_cons debe incluir columna: precio_venta_usd_mwh (por cliente-mes o constante repetida).
    - df_gen debe incluir: energia_disponible_mwh, precio_compra_usd_mwh, take_or_pay_ratio, penalidad_no_tomado_usd_mwh
      (TOP anual por generador, ratio y penalidad constantes).
    - df_lim: min_anual_mwh y max_anual_mwh por cliente (si falta, se asume 0 / +inf).
    """
    import pulp  # type: ignore

    # ---------- filtrar año/escenario ----------
    g = df_gen.copy()
    c = df_cons.copy()
    lim = df_lim.copy()

    if "anio" in g.columns:
        g = g[pd.to_numeric(g["anio"], errors="coerce").fillna(-1).astype(int) == int(anio)].copy()
    if "anio" in c.columns:
        c = c[pd.to_numeric(c["anio"], errors="coerce").fillna(-1).astype(int) == int(anio)].copy()
    if "anio" in lim.columns:
        lim = lim[pd.to_numeric(lim["anio"], errors="coerce").fillna(int(anio)).astype(int) == int(anio)].copy()

    if gen_scenario is not None and "escenario" in g.columns:
        g = g[g["escenario"].astype(str) == str(gen_scenario)].copy()
    if cons_scenario is not None and "escenario" in c.columns:
        c = c[c["escenario"].astype(str) == str(cons_scenario)].copy()

    # ---------- validar columnas ----------
    required_gen = ["mes", "nemo_generador", "energia_disponible_mwh", "precio_compra_usd_mwh"]
    for col in required_gen:
        if col not in g.columns:
            raise KeyError(f"df_gen: falta columna '{col}'")

    required_cons = ["mes", "nemo_cliente", "consumo_mwh", "precio_venta_usd_mwh"]
    for col in required_cons:
        if col not in c.columns:
            raise KeyError(f"df_cons: falta columna '{col}' (agregala al Excel)")

    # TOP (si no están, default 0)
    if "take_or_pay_ratio" not in g.columns:
        g["take_or_pay_ratio"] = 0.0
    if "penalidad_no_tomado_usd_mwh" not in g.columns:
        g["penalidad_no_tomado_usd_mwh"] = 0.0

    # ---------- normalizar numéricos ----------
    g["mes"] = pd.to_numeric(g["mes"], errors="coerce").astype(int)
    g["energia_disponible_mwh"] = pd.to_numeric(g["energia_disponible_mwh"], errors="coerce").fillna(0.0)
    g["precio_compra_usd_mwh"] = pd.to_numeric(g["precio_compra_usd_mwh"], errors="coerce").fillna(0.0)
    g["take_or_pay_ratio"] = pd.to_numeric(g["take_or_pay_ratio"], errors="coerce").fillna(0.0)
    g["penalidad_no_tomado_usd_mwh"] = pd.to_numeric(g["penalidad_no_tomado_usd_mwh"], errors="coerce").fillna(0.0)

    c["mes"] = pd.to_numeric(c["mes"], errors="coerce").astype(int)
    c["consumo_mwh"] = pd.to_numeric(c["consumo_mwh"], errors="coerce").fillna(0.0)
    c["precio_venta_usd_mwh"] = pd.to_numeric(c["precio_venta_usd_mwh"], errors="coerce")

    # si hay duplicados cliente-mes, los sumamos (consumo) y promediamos PV
    c = (
        c.groupby(["nemo_cliente", "mes"], as_index=False)
        .agg(consumo_mwh=("consumo_mwh", "sum"), precio_venta_usd_mwh=("precio_venta_usd_mwh", "mean"))
    )

    # generación: agrupar generador-mes
    g = (
        g.groupby(["nemo_generador", "mes"], as_index=False)
        .agg(
            energia_disponible_mwh=("energia_disponible_mwh", "sum"),
            precio_compra_usd_mwh=("precio_compra_usd_mwh", "mean"),
            take_or_pay_ratio=("take_or_pay_ratio", "max"),
            penalidad_no_tomado_usd_mwh=("penalidad_no_tomado_usd_mwh", "max"),
        )
    )

    # límites cliente
    if "min_anual_mwh" not in lim.columns:
        lim["min_anual_mwh"] = 0.0
    if "max_anual_mwh" not in lim.columns:
        lim["max_anual_mwh"] = np.inf
    lim["min_anual_mwh"] = pd.to_numeric(lim["min_anual_mwh"], errors="coerce").fillna(0.0)
    lim["max_anual_mwh"] = pd.to_numeric(lim["max_anual_mwh"], errors="coerce")
    lim["max_anual_mwh"] = lim["max_anual_mwh"].where(lim["max_anual_mwh"].notna(), np.inf)

    # ---------- sets ----------
    months: List[int] = sorted(c["mes"].unique().tolist()) or list(range(1, 13))
    clients: List[str] = sorted(c["nemo_cliente"].astype(str).unique().tolist())
    gens: List[str] = sorted(g["nemo_generador"].astype(str).unique().tolist())

    if not clients or not gens:
        return MilpSolution(status="NoData", objective_usd=0.0, df_nom_opt=pd.DataFrame(), df_vars=pd.DataFrame())

    # mapas
    cons = {(str(r.nemo_cliente), int(r.mes)): float(r.consumo_mwh) for r in c.itertuples(index=False)}
    sell = {(str(r.nemo_cliente), int(r.mes)): float(r.precio_venta_usd_mwh) for r in c.itertuples(index=False)}

    avail = {(str(r.nemo_generador), int(r.mes)): float(r.energia_disponible_mwh) for r in g.itertuples(index=False)}
    buy = {(str(r.nemo_generador), int(r.mes)): float(r.precio_compra_usd_mwh) for r in g.itertuples(index=False)}

    # TOP anual por generador (ratio y penalidad constantes)
    top_year = (
        g.groupby("nemo_generador", as_index=False)
        .agg(
            disponible_anual=("energia_disponible_mwh", "sum"),
            ratio=("take_or_pay_ratio", "max"),
            pen=("penalidad_no_tomado_usd_mwh", "max"),
        )
    )
    top_req = {str(r.nemo_generador): float(r.ratio) * float(r.disponible_anual) for r in top_year.itertuples(index=False)}
    top_pen = {str(r.nemo_generador): float(r.pen) for r in top_year.itertuples(index=False)}

    min_year = {str(r.nemo_cliente): float(r.min_anual_mwh) for r in lim.itertuples(index=False)} if "nemo_cliente" in lim.columns else {}
    max_year = {str(r.nemo_cliente): float(r.max_anual_mwh) for r in lim.itertuples(index=False)} if "nemo_cliente" in lim.columns else {}
    for cl in clients:
        min_year.setdefault(cl, 0.0)
        max_year.setdefault(cl, float("inf"))

    # ---------- modelo ----------
    model = pulp.LpProblem("OPT_DESDE_CERO", pulp.LpMaximize)

    x: Dict[Tuple[str, str, int], pulp.LpVariable] = {}
    short: Dict[str, pulp.LpVariable] = {}

    # variables x[c,g,m]
    for cl in clients:
        for ge in gens:
            for m in months:
                ub = min(cons.get((cl, m), 0.0), avail.get((ge, m), 0.0))
                x[(cl, ge, m)] = pulp.LpVariable(f"x_{cl}_{ge}_{m}", lowBound=0, upBound=float(ub), cat=pulp.LpContinuous)

    # shortfall anual por generador
    for ge in gens:
        short[ge] = pulp.LpVariable(f"short_{ge}", lowBound=0, cat=pulp.LpContinuous)

    # objetivo: margen - penalidad TOP anual
    obj = []
    for cl in clients:
        for ge in gens:
            for m in months:
                pv = sell.get((cl, m), 0.0)
                pc = buy.get((ge, m), 0.0)
                obj.append((pv - pc) * x[(cl, ge, m)])

    for ge in gens:
        obj.append(-top_pen.get(ge, 0.0) * short[ge])

    model += pulp.lpSum(obj)

    # restricciones demanda cliente-mes
    for cl in clients:
        for m in months:
            model += pulp.lpSum([x[(cl, ge, m)] for ge in gens]) <= float(cons.get((cl, m), 0.0)), f"demand_{cl}_{m}"

    # restricciones oferta generador-mes
    for ge in gens:
        for m in months:
            model += pulp.lpSum([x[(cl, ge, m)] for cl in clients]) <= float(avail.get((ge, m), 0.0)), f"supply_{ge}_{m}"

    # mínimos/máximos anuales cliente
    for cl in clients:
        tot = pulp.lpSum([x[(cl, ge, m)] for ge in gens for m in months])
        model += tot >= float(min_year.get(cl, 0.0)), f"min_year_{cl}"
        mx = float(max_year.get(cl, float("inf")))
        if np.isfinite(mx):
            model += tot <= mx, f"max_year_{cl}"

    # TOP anual generador (short >= requerido - tomado)
    for ge in gens:
        taken = pulp.lpSum([x[(cl, ge, m)] for cl in clients for m in months])
        model += short[ge] >= float(top_req.get(ge, 0.0)) - taken, f"top_{ge}"

    # resolver
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(solver_time_limit_s))
    model.solve(solver)

    status = pulp.LpStatus.get(model.status, str(model.status))
    objective = float(pulp.value(model.objective) or 0.0)

    # salida df_vars
    rows = []
    for (cl, ge, m), var in x.items():
        val = float(var.value() or 0.0)
        if val > 1e-9:
            rows.append({"anio": int(anio), "mes": int(m), "nemo_cliente": cl, "nemo_generador": ge, "x_opt_mwh": val})
    df_vars = pd.DataFrame(rows)

    # construir df_nom_opt desde cero (formato compatible)
    if df_vars.empty:
        df_nom_opt = pd.DataFrame(
            columns=["id_acuerdo", "anio", "mes", "nemo_cliente", "nemo_generador", "prioridad_asignacion",
                     "precio_venta_usd_mwh", "nominal_mwh", "mes_inicio", "mes_fin"]
        )
    else:
        df_nom_opt = df_vars.rename(columns={"x_opt_mwh": "nominal_mwh"}).copy()
        df_nom_opt["id_acuerdo"] = "AUTO_" + df_nom_opt["nemo_cliente"].astype(str) + "_" + df_nom_opt["nemo_generador"].astype(str)
        df_nom_opt["prioridad_asignacion"] = 1
        df_nom_opt["precio_venta_usd_mwh"] = df_nom_opt.apply(lambda r: sell.get((str(r["nemo_cliente"]), int(r["mes"])), 0.0), axis=1)
        df_nom_opt["mes_inicio"] = 1
        df_nom_opt["mes_fin"] = 12
        df_nom_opt = df_nom_opt[
            ["id_acuerdo", "anio", "mes", "nemo_cliente", "nemo_generador", "prioridad_asignacion",
             "precio_venta_usd_mwh", "nominal_mwh", "mes_inicio", "mes_fin"]
        ].copy()

    return MilpSolution(status=status, objective_usd=objective, df_nom_opt=df_nom_opt, df_vars=df_vars)