from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _simulate_priority(
    df_ag: pd.DataFrame,
    demand_rem: dict[str, float],
    av: float,
) -> tuple[pd.DataFrame, float]:
    """Asigna energía dentro de una prioridad de asignación para un (generador, mes)."""
    if df_ag.empty or av <= EPS:
        return df_ag.assign(asignado_MWh=0.0), float(av)

    df = df_ag.copy()
    nominal = pd.to_numeric(df["nominal_mwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    cap = np.array(
        [
            min(float(nominal[i]), float(demand_rem.get(str(df.iloc[i]["nemo_cliente"]), 0.0)))
            for i in range(len(df))
        ],
        dtype=float,
    )

    assigned = np.zeros(len(df), dtype=float)
    av_left = float(av)
    active = cap > EPS

    while av_left > EPS and active.any():
        weights = nominal.copy()
        weights[~active] = 0.0
        wsum = float(weights.sum())
        if wsum <= EPS:
            break

        alloc = av_left * (weights / wsum)
        rem_cap = cap - assigned
        alloc = np.minimum(alloc, np.maximum(0.0, rem_cap))
        delta = float(alloc.sum())
        if delta <= EPS:
            break

        assigned += alloc
        av_left -= delta

        for i in range(len(df)):
            if alloc[i] > EPS:
                c = str(df.iloc[i]["nemo_cliente"])
                demand_rem[c] = float(demand_rem.get(c, 0.0) - alloc[i])

        cap = np.array(
            [
                min(float(nominal[i]), float(demand_rem.get(str(df.iloc[i]["nemo_cliente"]), 0.0)))
                for i in range(len(df))
            ],
            dtype=float,
        )
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
    """Simulador de asignación con robustez + Take-or-Pay ANUAL (margen neto anual)."""

    df_gen_y = df_gen.copy()
    df_cons_y = df_cons.copy()
    df_nom_y = df_nom.copy()

    if "anio" in df_gen_y.columns:
        df_gen_y = df_gen_y[pd.to_numeric(df_gen_y["anio"], errors="coerce").fillna(-1).astype(int) == int(anio)].copy()
    if "anio" in df_cons_y.columns:
        df_cons_y = df_cons_y[pd.to_numeric(df_cons_y["anio"], errors="coerce").fillna(-1).astype(int) == int(anio)].copy()
    if "anio" in df_nom_y.columns:
        df_nom_y = df_nom_y[pd.to_numeric(df_nom_y["anio"], errors="coerce").fillna(-1).astype(int) == int(anio)].copy()

    # Vigencia contractual
    if {"mes_inicio", "mes_fin"}.issubset(df_nom_y.columns):
        df_nom_y["mes"] = pd.to_numeric(df_nom_y["mes"], errors="coerce")
        df_nom_y["mes_inicio"] = pd.to_numeric(df_nom_y["mes_inicio"], errors="coerce")
        df_nom_y["mes_fin"] = pd.to_numeric(df_nom_y["mes_fin"], errors="coerce")
        df_nom_y = df_nom_y[(df_nom_y["mes"] >= df_nom_y["mes_inicio"]) & (df_nom_y["mes"] <= df_nom_y["mes_fin"])].copy()

    # Consumo: agrupar
    df_cons_y["mes"] = pd.to_numeric(df_cons_y["mes"], errors="coerce")
    df_cons_y["consumo_mwh"] = pd.to_numeric(df_cons_y["consumo_mwh"], errors="coerce").fillna(0.0)
    df_cons_y = (
        df_cons_y.dropna(subset=["mes", "nemo_cliente"])
        .groupby(["anio", "mes", "nemo_cliente"], as_index=False)["consumo_mwh"].sum()
    )
    df_cons_y["mes"] = df_cons_y["mes"].astype(int)

    # Generación: agrupar + TOP (constante anual por generador)
    df_gen_y["mes"] = pd.to_numeric(df_gen_y["mes"], errors="coerce")
    df_gen_y["energia_disponible_mwh"] = pd.to_numeric(df_gen_y["energia_disponible_mwh"], errors="coerce").fillna(0.0)
    df_gen_y["precio_compra_usd_mwh"] = pd.to_numeric(df_gen_y["precio_compra_usd_mwh"], errors="coerce").fillna(0.0)

    if "take_or_pay_ratio" not in df_gen_y.columns:
        df_gen_y["take_or_pay_ratio"] = 0.0
    if "penalidad_no_tomado_usd_mwh" not in df_gen_y.columns:
        df_gen_y["penalidad_no_tomado_usd_mwh"] = 0.0

    df_gen_y["take_or_pay_ratio"] = pd.to_numeric(df_gen_y["take_or_pay_ratio"], errors="coerce").fillna(0.0)
    df_gen_y["penalidad_no_tomado_usd_mwh"] = pd.to_numeric(df_gen_y["penalidad_no_tomado_usd_mwh"], errors="coerce").fillna(0.0)

    df_gen_y = (
        df_gen_y.dropna(subset=["anio", "mes", "nemo_generador"])
        .groupby(["anio", "mes", "nemo_generador"], as_index=False)
        .agg(
            energia_disponible_mwh=("energia_disponible_mwh", "sum"),
            precio_compra_usd_mwh=("precio_compra_usd_mwh", "mean"),
            take_or_pay_ratio=("take_or_pay_ratio", "max"),
            penalidad_no_tomado_usd_mwh=("penalidad_no_tomado_usd_mwh", "max"),
        )
    )
    df_gen_y["mes"] = df_gen_y["mes"].astype(int)

    assigned_rows: list[pd.DataFrame] = []
    left_rows: list[dict] = []

    for mes in range(1, 13):
        cons_m = df_cons_y[df_cons_y["mes"].astype(int) == mes].copy()
        demand_rem = {str(r["nemo_cliente"]): float(r["consumo_mwh"]) for _, r in cons_m.iterrows()}

        gen_m = df_gen_y[df_gen_y["mes"].astype(int) == mes].copy()
        nom_m = df_nom_y[df_nom_y["mes"].astype(int) == mes].copy()

        for _, g in gen_m.iterrows():
            nemo_g = str(g["nemo_generador"])
            av = float(g["energia_disponible_mwh"])

            ag = nom_m[nom_m["nemo_generador"].astype(str) == nemo_g].copy()
            if ag.empty:
                left_rows.append({"anio": int(anio), "mes": int(mes), "nemo_generador": nemo_g, "energia_no_asignada_mwh": av})
                continue

            if "prioridad_asignacion" in ag.columns:
                priorities = sorted(ag["prioridad_asignacion"].astype(int).unique().tolist())
            else:
                priorities = [1]
                ag["prioridad_asignacion"] = 1

            for p in priorities:
                if av <= EPS:
                    break
                ag_p = ag[ag["prioridad_asignacion"].astype(int) == int(p)].copy()
                assigned_df, av = _simulate_priority(ag_p, demand_rem, av)
                assigned_rows.append(assigned_df)

            left_rows.append({"anio": int(anio), "mes": int(mes), "nemo_generador": nemo_g, "energia_no_asignada_mwh": av})

    detalle = pd.concat(assigned_rows, ignore_index=True) if assigned_rows else pd.DataFrame()
    energia_no_asignada = pd.DataFrame(left_rows)

    # merge precio compra
    if not detalle.empty:
        detalle = detalle.merge(
            df_gen_y[["anio", "mes", "nemo_generador", "precio_compra_usd_mwh"]],
            on=["anio", "mes", "nemo_generador"],
            how="left",
        )
        detalle["precio_compra_usd_mwh"] = pd.to_numeric(detalle["precio_compra_usd_mwh"], errors="coerce").fillna(0.0)
        detalle["precio_venta_usd_mwh"] = pd.to_numeric(detalle["precio_venta_usd_mwh"], errors="coerce").fillna(0.0)
        detalle["asignado_MWh"] = pd.to_numeric(detalle["asignado_MWh"], errors="coerce").fillna(0.0)

        detalle["ingreso_usd"] = detalle["asignado_MWh"] * detalle["precio_venta_usd_mwh"]
        detalle["costo_usd"] = detalle["asignado_MWh"] * detalle["precio_compra_usd_mwh"]
        detalle["margen_usd"] = detalle["ingreso_usd"] - detalle["costo_usd"]

    # no abastecido
    if not df_cons_y.empty:
        asg_cm = (
            detalle.groupby(["anio", "mes", "nemo_cliente"], as_index=False)["asignado_MWh"].sum()
            if not detalle.empty
            else pd.DataFrame(columns=["anio", "mes", "nemo_cliente", "asignado_MWh"])
        )
        no_ab = df_cons_y.merge(asg_cm, on=["anio", "mes", "nemo_cliente"], how="left")
        no_ab["asignado_MWh"] = pd.to_numeric(no_ab["asignado_MWh"], errors="coerce").fillna(0.0)
        no_ab["no_abastecido_mwh"] = np.maximum(0.0, no_ab["consumo_mwh"].astype(float) - no_ab["asignado_MWh"].astype(float))
        no_abastecido = no_ab[["anio", "mes", "nemo_cliente", "consumo_mwh", "asignado_MWh", "no_abastecido_mwh"]]
    else:
        no_abastecido = pd.DataFrame()

    # resumen mensual (sin TOP, porque TOP es anual)
    if not detalle.empty:
        resumen_mensual = detalle.groupby(["anio", "mes"], as_index=False).agg(
            energia_asignada_mwh=("asignado_MWh", "sum"),
            ingreso_usd=("ingreso_usd", "sum"),
            costo_usd=("costo_usd", "sum"),
            margen_usd=("margen_usd", "sum"),
        )
    else:
        resumen_mensual = pd.DataFrame(columns=["anio", "mes", "energia_asignada_mwh", "ingreso_usd", "costo_usd", "margen_usd"])

    # por generador-prioridad
    if not detalle.empty:
        por_gp = detalle.groupby(["anio", "mes", "nemo_generador", "prioridad_asignacion"], as_index=False)["asignado_MWh"].sum()
    else:
        por_gp = pd.DataFrame(columns=["anio", "mes", "nemo_generador", "prioridad_asignacion", "asignado_MWh"])

    # chequeo anual por cliente
    if not detalle.empty:
        asg_anual = detalle.groupby(["anio", "nemo_cliente"], as_index=False)["asignado_MWh"].sum().rename(columns={"asignado_MWh": "energia_real_anual_mwh"})
    else:
        asg_anual = pd.DataFrame(columns=["anio", "nemo_cliente", "energia_real_anual_mwh"])

    clientes_anual = df_lim.merge(asg_anual, on="nemo_cliente", how="left")
    clientes_anual["anio"] = int(anio)
    clientes_anual["energia_real_anual_mwh"] = pd.to_numeric(clientes_anual["energia_real_anual_mwh"], errors="coerce").fillna(0.0)

    if "min_anual_mwh" not in clientes_anual.columns:
        clientes_anual["min_anual_mwh"] = 0.0
    if "max_anual_mwh" not in clientes_anual.columns:
        clientes_anual["max_anual_mwh"] = np.inf

    clientes_anual["min_anual_mwh"] = pd.to_numeric(clientes_anual["min_anual_mwh"], errors="coerce").fillna(0.0)
    clientes_anual["max_anual_mwh"] = pd.to_numeric(clientes_anual["max_anual_mwh"], errors="coerce").where(pd.notna(clientes_anual["max_anual_mwh"]), np.inf)

    clientes_anual["cumple_min"] = clientes_anual["energia_real_anual_mwh"].astype(float) >= clientes_anual["min_anual_mwh"].astype(float) - 1e-6
    clientes_anual["cumple_max"] = clientes_anual["energia_real_anual_mwh"].astype(float) <= clientes_anual["max_anual_mwh"].astype(float) + 1e-6

    # -------------------- TOP ANUAL --------------------
    if not df_gen_y.empty:
        # disponibilidad anual por generador
        gen_year = (
            df_gen_y.groupby(["anio", "nemo_generador"], as_index=False)
            .agg(
                energia_disponible_anual_mwh=("energia_disponible_mwh", "sum"),
                take_or_pay_ratio=("take_or_pay_ratio", "max"),
                penalidad_no_tomado_usd_mwh=("penalidad_no_tomado_usd_mwh", "max"),
            )
        )

        if not detalle.empty:
            taken_year = (
                detalle.groupby(["anio", "nemo_generador"], as_index=False)["asignado_MWh"].sum()
                .rename(columns={"asignado_MWh": "energia_tomada_anual_mwh"})
            )
        else:
            taken_year = pd.DataFrame(columns=["anio", "nemo_generador", "energia_tomada_anual_mwh"])

        top_anual = gen_year.merge(taken_year, on=["anio", "nemo_generador"], how="left")
        top_anual["energia_tomada_anual_mwh"] = pd.to_numeric(top_anual["energia_tomada_anual_mwh"], errors="coerce").fillna(0.0)
        top_anual["min_take_anual_mwh"] = top_anual["take_or_pay_ratio"].astype(float) * top_anual["energia_disponible_anual_mwh"].astype(float)
        top_anual["shortfall_anual_mwh"] = np.maximum(0.0, top_anual["min_take_anual_mwh"].astype(float) - top_anual["energia_tomada_anual_mwh"].astype(float))
        top_anual["penalidad_anual_usd"] = top_anual["shortfall_anual_mwh"].astype(float) * top_anual["penalidad_no_tomado_usd_mwh"].astype(float)

        penalidad_total = float(top_anual["penalidad_anual_usd"].sum())
    else:
        top_anual = pd.DataFrame()
        penalidad_total = 0.0

    # resumen anual (con TOP)
    margen_bruto = float(resumen_mensual["margen_usd"].sum()) if "margen_usd" in resumen_mensual.columns else 0.0
    resumen_anual = pd.DataFrame(
        [
            {
                "anio": int(anio),
                "margen_bruto_usd": margen_bruto,
                "penalidad_take_or_pay_anual_usd": penalidad_total,
                "margen_neto_usd": margen_bruto - penalidad_total,
            }
        ]
    )

    return {
        "detalle": detalle,
        "resumen_mensual": resumen_mensual,
        "resumen_anual": resumen_anual,
        "por_generador_prioridad": por_gp,
        "no_abastecido": no_abastecido,
        "energia_no_asignada_generador": energia_no_asignada,
        "clientes_anual": clientes_anual,
        "take_or_pay_anual": top_anual,
    }
