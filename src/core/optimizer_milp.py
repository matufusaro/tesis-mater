from __future__ import annotations

"""Optimizador MILP (demo).

Esta implementación está pensada para la **app de demostración** y para la tesis:
no intenta reproducir todas las sutilezas contractuales, sino proveer un
esqueleto coherente para:

1) Elegir nominaciones (energía nominal) por acuerdo y por mes.
2) Respetar límites de generación (oferta) y consumo (demanda).
3) Cumplir mínimos/máximos anuales por cliente (tabla Clientes_Limites).
4) Introducir una decisión discreta (variable binaria) para "activar" acuerdos
   de menor preferencia (p.ej. prioridad_asignacion=2), mostrando por qué una
   MILP aporta expresividad respecto de una PL continua.

La función pública principal es :func:`solve_milp_nominal`.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

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
    df_nom_template: pd.DataFrame,
    anio: int,
    gen_scenario: str | None = None,
    cons_scenario: str | None = None,
    min_block_mwh_p2: float = 50.0,
    fixed_penalty_p2_usd: float = 100.0,
    solver_time_limit_s: int = 30,
) -> MilpSolution:
    """Resuelve una MILP para proponer nominaciones mensuales.

    Parameters
    ----------
    df_gen, df_cons, df_lim, df_nom_template:
        DataFrames cargados desde Excel.
    anio:
        Año a optimizar.
    gen_scenario, cons_scenario:
        Si existen columnas 'escenario' en generación/consumo, filtra por el
        escenario elegido.
    min_block_mwh_p2:
        Mínimo de energía si se "activa" un acuerdo con prioridad_asignacion >= 2.
        Se implementa con una binaria (x >= min_block * y).
    fixed_penalty_p2_usd:
        Penalización fija por activar acuerdos de prioridad 2 (para desalentar su
        uso salvo necesidad).
    solver_time_limit_s:
        Límite de tiempo del solver.
    """

    # Import local para que la app pueda correr aunque el usuario todavía no
    # haya instalado PuLP (se informa el error en UI).
    import pulp  # type: ignore

    # --------------- Filtrado base ---------------
    df_gen_y = df_gen[df_gen["anio"].astype(int) == int(anio)].copy()
    df_cons_y = df_cons[df_cons["anio"].astype(int) == int(anio)].copy()
    df_nom_y = df_nom_template[df_nom_template["anio"].astype(int) == int(anio)].copy()

    # Vigencia
    df_nom_y = df_nom_y[
        (df_nom_y["mes"].astype(int) >= df_nom_y["mes_inicio"].astype(int))
        & (df_nom_y["mes"].astype(int) <= df_nom_y["mes_fin"].astype(int))
    ].copy()

    if gen_scenario is not None and "escenario" in df_gen_y.columns:
        df_gen_y = df_gen_y[df_gen_y["escenario"].astype(str) == str(gen_scenario)].copy()
    if cons_scenario is not None and "escenario" in df_cons_y.columns:
        df_cons_y = df_cons_y[df_cons_y["escenario"].astype(str) == str(cons_scenario)].copy()

    # Mapas rápidos
    gen_av = {
        (int(r.anio), int(r.mes), str(r.nemo_generador)): float(r.energia_disponible_mwh)
        for r in df_gen_y.itertuples(index=False)
    }
    gen_price = {
        (int(r.anio), int(r.mes), str(r.nemo_generador)): float(r.precio_compra_usd_mwh)
        for r in df_gen_y.itertuples(index=False)
    }
    cons = {
        (int(r.anio), int(r.mes), str(r.nemo_cliente)): float(r.consumo_mwh)
        for r in df_cons_y.itertuples(index=False)
    }

    # --------------- Modelo ---------------
    model = pulp.LpProblem("Nominal_MILP", pulp.LpMaximize)

    x: Dict[int, pulp.LpVariable] = {}
    y: Dict[int, pulp.LpVariable] = {}

    # Precalcular filas válidas (tienen oferta y demanda definidas)
    df_nom_y = df_nom_y.reset_index(drop=True)
    valid_rows = []
    for i, r in df_nom_y.iterrows():
        key_g = (int(r["anio"]), int(r["mes"]), str(r["nemo_generador"]))
        key_c = (int(r["anio"]), int(r["mes"]), str(r["nemo_cliente"]))
        if key_g not in gen_av or key_c not in cons:
            continue
        valid_rows.append(i)

    if not valid_rows:
        return MilpSolution(status="NoData", objective_usd=0.0, df_nom_opt=df_nom_y, df_vars=pd.DataFrame())

    # Variables y objetivo
    obj_terms = []
    for i in valid_rows:
        r = df_nom_y.loc[i]
        key_g = (int(r["anio"]), int(r["mes"]), str(r["nemo_generador"]))
        buy = float(gen_price[key_g])
        sell = float(r["precio_venta_usd_mwh"])
        margin = sell - buy

        # Cota superior por fila (acotamos por demanda mensual del cliente)
        key_c = (int(r["anio"]), int(r["mes"]), str(r["nemo_cliente"]))
        ub = float(cons[key_c])

        x[i] = pulp.LpVariable(f"x_{i}", lowBound=0, upBound=ub, cat=pulp.LpContinuous)
        y[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Activación: si se usa el acuerdo, x puede ser > 0; si no, x debe ser 0
        model += x[i] <= ub * y[i], f"link_up_{i}"

        # Para acuerdos de prioridad 2+, imponemos un mínimo por activación
        pr = int(r["prioridad_asignacion"])
        if pr >= 2:
            model += x[i] >= float(min_block_mwh_p2) * y[i], f"min_block_p2_{i}"
            obj_terms.append(-float(fixed_penalty_p2_usd) * y[i])

        obj_terms.append(float(margin) * x[i])

    model += pulp.lpSum(obj_terms)

    # --------------- Restricciones ---------------
    # Oferta por generador-mes
    for (yy, mm, g), av in gen_av.items():
        # restringir solo si ese (g,m) aparece en filas válidas
        idx = [i for i in valid_rows if (int(df_nom_y.loc[i, "anio"]), int(df_nom_y.loc[i, "mes"]), str(df_nom_y.loc[i, "nemo_generador"])) == (yy, mm, g)]
        if not idx:
            continue
        model += pulp.lpSum([x[i] for i in idx]) <= float(av), f"supply_{g}_{yy}_{mm}"

    # Demanda por cliente-mes (no asignar más que el consumo real)
    # (podría ser redundante por la cota superior y por sumatoria, pero la dejamos explícita)
    for (yy, mm, c), d in cons.items():
        idx = [i for i in valid_rows if (int(df_nom_y.loc[i, "anio"]), int(df_nom_y.loc[i, "mes"]), str(df_nom_y.loc[i, "nemo_cliente"])) == (yy, mm, c)]
        if not idx:
            continue
        model += pulp.lpSum([x[i] for i in idx]) <= float(d), f"demand_{c}_{yy}_{mm}"

    # Mínimos/máximos anuales por cliente
    for r in df_lim.itertuples(index=False):
        c = str(r.nemo_cliente)
        min_a = float(r.min_anual_mwh)
        max_a = float(r.max_anual_mwh)
        idx = [i for i in valid_rows if str(df_nom_y.loc[i, "nemo_cliente"]) == c]
        if not idx:
            continue
        model += pulp.lpSum([x[i] for i in idx]) >= min_a, f"min_year_{c}"
        model += pulp.lpSum([x[i] for i in idx]) <= max_a, f"max_year_{c}"

    # --------------- Resolver ---------------
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(solver_time_limit_s))
    model.solve(solver)

    status = pulp.LpStatus.get(model.status, str(model.status))
    obj = float(pulp.value(model.objective) or 0.0)

    # --------------- Salidas ---------------
    df_vars = []
    for i in valid_rows:
        df_vars.append(
            {
                "row": i,
                "id_acuerdo": df_nom_y.loc[i, "id_acuerdo"],
                "anio": int(df_nom_y.loc[i, "anio"]),
                "mes": int(df_nom_y.loc[i, "mes"]),
                "nemo_cliente": str(df_nom_y.loc[i, "nemo_cliente"]),
                "nemo_generador": str(df_nom_y.loc[i, "nemo_generador"]),
                "prioridad_asignacion": int(df_nom_y.loc[i, "prioridad_asignacion"]),
                "x_opt_mwh": float(x[i].value() or 0.0),
                "y_activo": int(round(float(y[i].value() or 0.0))),
            }
        )
    df_vars = pd.DataFrame(df_vars)

    df_nom_opt = df_nom_y.copy()
    if not df_vars.empty:
        df_nom_opt.loc[df_vars["row"].values, "nominal_mwh"] = df_vars["x_opt_mwh"].values

    return MilpSolution(status=status, objective_usd=obj, df_nom_opt=df_nom_opt, df_vars=df_vars)
