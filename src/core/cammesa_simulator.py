from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _simulate_priority(
    df_ag: pd.DataFrame,
    demand_rem: dict[str, float],
    av: float,
) -> tuple[pd.DataFrame, float]:
    """Asigna energía dentro de una prioridad de asignación para un (generador, mes).

    Regla usada (coherente con lo conversado):
    - Si la energía disponible no alcanza a cubrir la suma nominal, la asignación es proporcional al nominal.
    - Si un cliente se satura por falta de consumo real, su excedente se reasigna (misma prioridad) proporcionalmente
      al nominal remanente de los acuerdos que aún pueden recibir energía.

    Parameters
    ----------
    df_ag: filas de acuerdos para un mismo generador, mes y prioridad_asignacion
    demand_rem: demanda real remanente por cliente (nemo_cliente -> MWh)
    av: energía disponible remanente (MWh)

    Returns
    -------
    assigned_df: df_ag con columna asignado_MWh
    av_rem: energía que queda sin asignar dentro de esta prioridad
    """
    if df_ag.empty or av <= EPS:
        return df_ag.assign(asignado_MWh=0.0), float(av)

    df = df_ag.copy()
    # pesos: nominal declarado en el mes
    nominal = df["nominal_mwh"].astype(float).to_numpy()
    # capacidad máxima por fila (por consumo real remanente del cliente y por nominal)
    cap = np.array([
        min(float(nominal[i]), float(demand_rem.get(str(df.iloc[i]["nemo_cliente"]), 0.0)))
        for i in range(len(df))
    ], dtype=float)

    assigned = np.zeros(len(df), dtype=float)

    # Iterar para reasignar excedentes por saturación de demanda
    av_left = float(av)
    active = cap > EPS

    while av_left > EPS and active.any():
        weights = nominal.copy()
        weights[~active] = 0.0
        wsum = float(weights.sum())
        if wsum <= EPS:
            break
        alloc = av_left * (weights / wsum)
        # no asignar por encima del cap remanente
        rem_cap = cap - assigned
        alloc = np.minimum(alloc, np.maximum(0.0, rem_cap))
        delta = float(alloc.sum())
        if delta <= EPS:
            break
        assigned += alloc
        av_left -= delta

        # actualizar demanda remanente por cliente (sumando lo asignado a todas las filas del cliente)
        # Nota: un cliente puede aparecer en más de una fila dentro del mismo generador/mes/priority.
        #       Se descuenta de manera acumulada.
        for i in range(len(df)):
            if alloc[i] > EPS:
                c = str(df.iloc[i]["nemo_cliente"])
                demand_rem[c] = float(demand_rem.get(c, 0.0) - alloc[i])

        # recalcular cap y active en base a la nueva demanda remanente
        cap = np.array([
            min(float(nominal[i]), float(demand_rem.get(str(df.iloc[i]["nemo_cliente"]), 0.0)))
            for i in range(len(df))
        ], dtype=float)
        active = (cap - assigned) > EPS

    df["asignado_MWh"] = assigned
    return df, av_left


def simulate_cammesa(
    df_gen: pd.DataFrame,
    df_cons: pd.DataFrame,
    df_lim: pd.DataFrame,
    df_nom: pd.DataFrame,
    anio: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Simulador de asignación (baseline) inspirado en la lógica operativa discutida.

    Supuestos clave en esta versión (sin penalidades):
    - La energía disponible mensual por generador es un único volumen (no se compra PG1/PG2 explícita).
    - La comercializadora define el orden de asignación mediante 'prioridad_asignacion' (1,2,...).
    - No se puede asignar a un cliente más de lo que consume (demanda real del mes).
    - La reasignación de excedentes por saturación es proporcional al nominal, dentro de la misma prioridad.
    - Los mínimos/máximos se chequean de manera anual, a nivel cliente (tabla Clientes_Limites).
    """

    # Filtrar año
    df_gen_y = df_gen[df_gen["anio"].astype(int) == int(anio)].copy()
    df_cons_y = df_cons[df_cons["anio"].astype(int) == int(anio)].copy()
    df_nom_y = df_nom[df_nom["anio"].astype(int) == int(anio)].copy()

    # Vigencia contractual (mes_inicio/mes_fin)
    df_nom_y = df_nom_y[(df_nom_y["mes"].astype(int) >= df_nom_y["mes_inicio"].astype(int)) &
                        (df_nom_y["mes"].astype(int) <= df_nom_y["mes_fin"].astype(int))].copy()

    assigned_rows: list[pd.DataFrame] = []
    left_rows: list[dict] = []

    # Para cada mes, armar demanda remanente por cliente
    for mes in range(1, 13):
        cons_m = df_cons_y[df_cons_y["mes"].astype(int) == mes].copy()
        demand_rem = {str(r["nemo_cliente"]): float(r["consumo_mwh"]) for _, r in cons_m.iterrows()}

        gen_m = df_gen_y[df_gen_y["mes"].astype(int) == mes].copy()
        nom_m = df_nom_y[df_nom_y["mes"].astype(int) == mes].copy()

        # Procesar generador por generador
        for _, g in gen_m.iterrows():
            nemo_g = str(g["nemo_generador"])
            av = float(g["energia_disponible_mwh"])

            ag = nom_m[nom_m["nemo_generador"].astype(str) == nemo_g].copy()
            if ag.empty:
                left_rows.append({"anio": anio, "mes": mes, "nemo_generador": nemo_g, "energia_no_asignada_mwh": av})
                continue

            # Orden de asignación definido por la comercializadora
            for p in sorted(ag["prioridad_asignacion"].astype(int).unique().tolist()):
                if av <= EPS:
                    break
                ag_p = ag[ag["prioridad_asignacion"].astype(int) == int(p)].copy()
                assigned_df, av = _simulate_priority(ag_p, demand_rem, av)
                assigned_rows.append(assigned_df)

            left_rows.append({"anio": anio, "mes": mes, "nemo_generador": nemo_g, "energia_no_asignada_mwh": av})

    detalle = pd.concat(assigned_rows, ignore_index=True) if assigned_rows else pd.DataFrame()
    energia_no_asignada = pd.DataFrame(left_rows)

    # Agregar precios de compra desde Generadores_Energia
    if not detalle.empty:
        detalle = detalle.merge(
            df_gen_y[["anio", "mes", "nemo_generador", "precio_compra_usd_mwh"]],
            on=["anio", "mes", "nemo_generador"],
            how="left",
        )
        detalle["ingreso_usd"] = detalle["asignado_MWh"].astype(float) * detalle["precio_venta_usd_mwh"].astype(float)
        detalle["costo_usd"] = detalle["asignado_MWh"].astype(float) * detalle["precio_compra_usd_mwh"].astype(float)
        detalle["margen_usd"] = detalle["ingreso_usd"] - detalle["costo_usd"]

    # No abastecido por cliente-mes
    if not df_cons_y.empty:
        asg_cm = (detalle.groupby(["anio", "mes", "nemo_cliente"], as_index=False)["asignado_MWh"].sum()
                  if not detalle.empty else pd.DataFrame(columns=["anio", "mes", "nemo_cliente", "asignado_MWh"]))
        no_ab = df_cons_y.merge(asg_cm, on=["anio", "mes", "nemo_cliente"], how="left")
        no_ab["asignado_MWh"] = no_ab["asignado_MWh"].fillna(0.0)
        no_ab["no_abastecido_mwh"] = np.maximum(0.0, no_ab["consumo_mwh"].astype(float) - no_ab["asignado_MWh"].astype(float))
        no_abastecido = no_ab[["anio", "mes", "nemo_cliente", "consumo_mwh", "asignado_MWh", "no_abastecido_mwh"]]
    else:
        no_abastecido = pd.DataFrame()

    # Resumen mensual
    if not detalle.empty:
        resumen_mensual = detalle.groupby(["anio", "mes"], as_index=False).agg(
            energia_asignada_mwh=("asignado_MWh", "sum"),
            ingreso_usd=("ingreso_usd", "sum"),
            costo_usd=("costo_usd", "sum"),
            margen_usd=("margen_usd", "sum"),
        )
    else:
        resumen_mensual = pd.DataFrame(columns=["anio", "mes", "energia_asignada_mwh", "ingreso_usd", "costo_usd", "margen_usd"])

    # Resumen por generador y prioridad
    if not detalle.empty:
        por_gp = detalle.groupby(["anio", "mes", "nemo_generador", "prioridad_asignacion"], as_index=False)["asignado_MWh"].sum()
    else:
        por_gp = pd.DataFrame(columns=["anio", "mes", "nemo_generador", "prioridad_asignacion", "asignado_MWh"])

    # Chequeo anual por cliente
    if not detalle.empty:
        asg_anual = detalle.groupby(["anio", "nemo_cliente"], as_index=False)["asignado_MWh"].sum().rename(columns={"asignado_MWh": "energia_real_anual_mwh"})
    else:
        asg_anual = pd.DataFrame(columns=["anio", "nemo_cliente", "energia_real_anual_mwh"])

    clientes_anual = df_lim.merge(asg_anual, on="nemo_cliente", how="left")
    clientes_anual["anio"] = int(anio)
    clientes_anual["energia_real_anual_mwh"] = clientes_anual["energia_real_anual_mwh"].fillna(0.0)
    clientes_anual["cumple_min"] = clientes_anual["energia_real_anual_mwh"].astype(float) >= clientes_anual["min_anual_mwh"].astype(float) - 1e-6
    clientes_anual["cumple_max"] = clientes_anual["energia_real_anual_mwh"].astype(float) <= clientes_anual["max_anual_mwh"].astype(float) + 1e-6

    return {
        "detalle": detalle,
        "resumen_mensual": resumen_mensual,
        "por_generador_prioridad": por_gp,
        "no_abastecido": no_abastecido,
        "energia_no_asignada_generador": energia_no_asignada,
        "clientes_anual": clientes_anual,
    }
