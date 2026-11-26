from __future__ import annotations

import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from src.io.exporter import results_to_excel_bytes
from src.viz.optimization import (
    clientes_incumplidores,
    grafico_variaciones_clientes,
    precios_venta_ponderados,
    variaciones_por_cliente,
)


class OptimizationViewTests(unittest.TestCase):
    def test_selects_exported_flags_once_per_client(self) -> None:
        rows = pd.DataFrame(
            {
                "nemo_cliente": ["A", "B", "B", "C", "D"],
                "energia_real_anual_mwh": [10.0, 10.0, 10.0, 10.0, 10.0],
                "min_anual_mwh": [10.5, 11.01, 11.01, 0.0, 11.01],
                "max_anual_mwh": [9.5, 20.0, 20.0, 8.99, 20.0],
                "cumple_min": [True, False, False, True, True],
                "cumple_max": [True, True, True, False, True],
            }
        )
        under, over = clientes_incumplidores(rows)
        self.assertEqual(under["nemo_cliente"].tolist(), ["B"])
        self.assertEqual(over["nemo_cliente"].tolist(), ["C"])
        # Flags are authoritative even if a separate comparison would disagree.
        self.assertNotIn("D", under["nemo_cliente"].tolist())

        workbook = load_workbook(
            io.BytesIO(results_to_excel_bytes({"clientes_anual": rows})), data_only=True
        )
        values = list(workbook["clientes_anual"].values)
        exported = pd.DataFrame(values[1:], columns=values[0])
        exported_under, exported_over = clientes_incumplidores(exported)
        self.assertEqual(under["nemo_cliente"].tolist(), exported_under["nemo_cliente"].tolist())
        self.assertEqual(over["nemo_cliente"].tolist(), exported_over["nemo_cliente"].tolist())

    def test_fallback_uses_contract_tolerance(self) -> None:
        rows = pd.DataFrame(
            {
                "nemo_cliente": ["ok", "min", "max"],
                "energia_real_anual_mwh": [10.0, 10.0, 10.0],
                "min_anual_mwh": [11.0, 11.01, 0.0],
                "max_anual_mwh": [9.0, 20.0, 8.99],
            }
        )
        under, over = clientes_incumplidores(rows)
        self.assertEqual(under["nemo_cliente"].tolist(), ["min"])
        self.assertEqual(over["nemo_cliente"].tolist(), ["max"])

    def test_variation_chart_filters_sorts_and_preserves_precision(self) -> None:
        data = pd.DataFrame(
            {
                "nemo_cliente": ["negative", "zero", "small", "positive", "tiny", "negative2"],
                "Cliente": ["Negative", "Zero", "Small", "Positive", "Tiny", "Negative 2"],
                "Diferencia (MWh)": [-2.7, 0.0, 0.25, 3.2, 1e-10, -0.5],
                "precio_venta_ponderado_usd_mwh": [40.0, 50.0, 20.0, 10.0, 60.0, 30.0],
            }
        )
        changes = variaciones_por_cliente(data)
        self.assertEqual(changes["nemo_cliente"].tolist(), ["positive", "small", "negative2", "negative"])
        self.assertEqual(changes["Diferencia (MWh)"].tolist(), [3.2, 0.25, -0.5, -2.7])
        figure = grafico_variaciones_clientes(changes)
        self.assertEqual(list(figure.data[0].y), [3.2, 0.25, -0.5, -2.7])
        self.assertEqual(list(figure.data[0].marker.color), ["#137333", "#137333", "#B91C1C", "#B91C1C"])
        self.assertEqual(list(figure.layout.xaxis.categoryarray), changes["nemo_cliente"].tolist())
        self.assertEqual(list(figure.data[1].y), [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(list(figure.data[2].y), [25.0] * 4)
        np.testing.assert_allclose(figure.data[3].y, [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(figure.data[0].yaxis, None)
        self.assertEqual(figure.data[1].yaxis, "y2")
        self.assertEqual(figure.data[3].line.dash, "dot")
        self.assertEqual(figure.layout.yaxis.title.text, "Diferencia [MWh]")
        self.assertEqual(figure.layout.yaxis2.title.text, "Precio de Venta [USD/MWh]")
        self.assertLess(figure.layout.yaxis.domain[1], figure.layout.yaxis2.domain[0])

    def test_weighted_price_and_missing_observations(self) -> None:
        detail = pd.DataFrame(
            {
                "nemo_cliente": ["A", "A", "B", "C", "D"],
                "asignado_MWh": [1.0, 3.0, 2.0, 0.0, 4.0],
                "ingreso_usd": [10.0, 90.0, 40.0, 0.0, 160.0],
            }
        )
        prices = precios_venta_ponderados(detail).set_index("nemo_cliente")
        self.assertEqual(prices.loc["A", "precio_venta_ponderado_usd_mwh"], 25.0)
        self.assertTrue(pd.isna(prices.loc["C", "precio_venta_ponderado_usd_mwh"]))

        changes = pd.DataFrame(
            {
                "nemo_cliente": ["A", "C", "B", "D"],
                "Cliente": ["A", "C", "B", "D"],
                "Diferencia (MWh)": [4.0, 3.0, -1.0, -2.0],
            }
        ).merge(prices, on="nemo_cliente", how="left")
        figure = grafico_variaciones_clientes(changes)
        self.assertEqual(len(figure.data[0].y), 4)
        self.assertTrue(pd.isna(figure.data[1].y[1]))
        np.testing.assert_allclose(figure.data[2].y, [85.0 / 3.0] * 4)
        slope, intercept = np.polyfit([0, 2, 3], [25.0, 20.0, 40.0], 1)
        np.testing.assert_allclose(figure.data[3].y, slope * np.arange(4) + intercept)

        single = grafico_variaciones_clientes(changes.iloc[[0, 1]])
        self.assertEqual(len(single.data), 3)
        unavailable = grafico_variaciones_clientes(changes.iloc[[1]])
        self.assertEqual(len(unavailable.data), 1)

    def test_streamlit_counts_and_empty_states_follow_export_flags(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=120).run()
        self.assertEqual([tab.label for tab in app.tabs], ["Simulador Baseline", "Optimizador MILP", "Comparativa"])
        app = app.button[0].click().run()
        app = app.button[1].click().run()
        self.assertFalse(app.exception)

        results = app.session_state["opt_results"]
        workbook = load_workbook(io.BytesIO(results_to_excel_bytes(results)), data_only=True)
        rows = list(workbook["clientes_anual"].values)
        under, over = clientes_incumplidores(pd.DataFrame(rows[1:], columns=rows[0]))
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(int(metrics["Clientes bajo mínimo"]), len(under))
        self.assertEqual(int(metrics["Clientes sobre máximo anual"]), len(over))
        clients = results["clientes_anual"].copy()
        clients["cumple_min"] = True
        clients["cumple_max"] = True
        results["clientes_anual"] = clients
        app.session_state["opt_results"] = results
        app = app.run()
        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Clientes bajo mínimo"], "0")
        self.assertEqual(metrics["Clientes sobre máximo anual"], "0")
        messages = "\n".join(element.value for element in app.markdown)
        self.assertIn("Todos los clientes cumplen el mínimo contractual", messages)
        self.assertIn("Ningún cliente excedió su máximo contractual", messages)
        no_violation_charts = len(app.get("plotly_chart"))

        results = app.session_state["opt_results"]
        clients = results["clientes_anual"].copy()
        clients.loc[clients.index[0], "cumple_min"] = False
        clients.loc[clients.index[0], "energia_real_anual_mwh"] = clients.loc[clients.index[0], "min_anual_mwh"] - 2.0
        clients.loc[clients.index[0], "faltante_min_mwh"] = 2.0
        clients.loc[clients.index[0], "faltante_min_incumplido_mwh"] = 1.0
        clients.loc[clients.index[1], "cumple_max"] = False
        clients.loc[clients.index[1], "energia_real_anual_mwh"] = clients.loc[clients.index[1], "max_anual_mwh"] + 2.0
        clients.loc[clients.index[1], "exceso_max_mwh"] = 2.0
        clients.loc[clients.index[1], "exceso_max_incumplido_mwh"] = 1.0
        clients = pd.concat([clients, clients.iloc[[0, 1]]], ignore_index=True)
        results["clientes_anual"] = clients
        app.session_state["opt_results"] = results
        app = app.run()
        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        workbook = load_workbook(io.BytesIO(results_to_excel_bytes(results)), data_only=True)
        rows = list(workbook["clientes_anual"].values)
        under, over = clientes_incumplidores(pd.DataFrame(rows[1:], columns=rows[0]))
        self.assertEqual(int(metrics["Clientes bajo mínimo"]), len(under))
        self.assertEqual(int(metrics["Clientes sobre máximo anual"]), len(over))
        self.assertEqual((len(under), len(over)), (1, 1))
        self.assertEqual(len(app.get("plotly_chart")), no_violation_charts + 2)


if __name__ == "__main__":
    unittest.main()
