from __future__ import annotations

import io
import unittest

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.core import compliance
from src.core.cammesa_simulator import simulate_cammesa
from src.core.compliance import (
    TOLERANCIA_CUMPLIMIENTO_MWH,
    cumple_maximo_cliente,
    cumple_minimo_cliente,
    cumple_top_generador,
    incumplimiento_sobre_tolerancia,
)
from src.core.optimizer_milp import solve_milp_nominal
from src.io.exporter import results_to_excel_bytes


class ContractComplianceTests(unittest.TestCase):
    differences = [0.0, 0.0001, 0.50, 1.0, 1.01]
    expected = [True, True, True, True, False]

    def test_contract_classifiers_use_unrounded_absolute_tolerance(self) -> None:
        differences = pd.Series(self.differences, dtype=float)
        energy = pd.Series([10.0] * len(differences))

        self.assertEqual(TOLERANCIA_CUMPLIMIENTO_MWH, 1.0)
        self.assertEqual(
            cumple_minimo_cliente(energy, energy + differences).tolist(),
            self.expected,
        )
        self.assertEqual(
            cumple_maximo_cliente(energy, energy - differences).tolist(),
            self.expected,
        )
        self.assertEqual(
            cumple_top_generador(energy - differences, energy).tolist(),
            self.expected,
        )
        np.testing.assert_allclose(
            incumplimiento_sobre_tolerancia(differences).to_numpy(),
            [0.0, 0.0, 0.0, 0.0, 0.01],
        )

    def _simulate(
        self,
        *,
        allocation: float,
        minimum: float,
        maximum: float,
        top_shortfall: float,
    ) -> dict[str, pd.DataFrame]:
        available = allocation + top_shortfall
        df_gen = pd.DataFrame(
            [
                {
                    "anio": 2026,
                    "mes": 1,
                    "nemo_generador": "G1",
                    "energia_disponible_mwh": available,
                    "precio_compra_usd_mwh": 2.0,
                    "take_or_pay_ratio": 1.0,
                    "penalidad_no_tomado_usd_mwh": 7.0,
                }
            ]
        )
        df_cons = pd.DataFrame(
            [{"anio": 2026, "mes": 1, "nemo_cliente": "C1", "consumo_mwh": allocation}]
        )
        df_lim = pd.DataFrame(
            [
                {
                    "nemo_cliente": "C1",
                    "min_anual_mwh": minimum,
                    "max_anual_mwh": maximum,
                }
            ]
        )
        df_nom = pd.DataFrame(
            [
                {
                    "anio": 2026,
                    "mes": 1,
                    "nemo_cliente": "C1",
                    "nemo_generador": "G1",
                    "prioridad_asignacion": 1,
                    "nominal_mwh": allocation,
                    "precio_venta_usd_mwh": 5.0,
                }
            ]
        )
        return simulate_cammesa(df_gen, df_cons, df_lim, df_nom, anio=2026)

    def test_simulation_preserves_real_values_and_emits_consistent_flags(self) -> None:
        allocation = 10.0
        for difference, expected in zip(self.differences, self.expected):
            with self.subTest(difference=difference, contract="minimum"):
                result = self._simulate(
                    allocation=allocation,
                    minimum=allocation + difference,
                    maximum=np.inf,
                    top_shortfall=0.0,
                )
                client = result["clientes_anual"].iloc[0]
                self.assertEqual(bool(client["cumple_min"]), expected)
                self.assertAlmostEqual(client["energia_real_anual_mwh"], allocation)
                self.assertAlmostEqual(client["faltante_min_mwh"], difference)
                self.assertAlmostEqual(
                    client["faltante_min_incumplido_mwh"], max(0.0, difference - 1.0)
                )
                self.assertAlmostEqual(
                    result["resumen_mensual"]["energia_asignada_mwh"].sum(), allocation
                )
                self.assertAlmostEqual(result["resumen_mensual"]["margen_usd"].sum(), 30.0)

            with self.subTest(difference=difference, contract="maximum"):
                result = self._simulate(
                    allocation=allocation,
                    minimum=0.0,
                    maximum=allocation - difference,
                    top_shortfall=0.0,
                )
                client = result["clientes_anual"].iloc[0]
                self.assertEqual(bool(client["cumple_max"]), expected)
                self.assertAlmostEqual(client["energia_real_anual_mwh"], allocation)
                self.assertAlmostEqual(client["exceso_max_mwh"], difference)
                self.assertAlmostEqual(
                    client["exceso_max_incumplido_mwh"], max(0.0, difference - 1.0)
                )
                self.assertAlmostEqual(
                    result["resumen_mensual"]["energia_asignada_mwh"].sum(), allocation
                )
                self.assertAlmostEqual(result["resumen_mensual"]["margen_usd"].sum(), 30.0)

            with self.subTest(difference=difference, contract="TOP"):
                result = self._simulate(
                    allocation=allocation,
                    minimum=0.0,
                    maximum=np.inf,
                    top_shortfall=difference,
                )
                top = result["take_or_pay_anual"].iloc[0]
                self.assertEqual(bool(top["cumple_top"]), expected)
                self.assertAlmostEqual(top["energia_tomada_anual_mwh"], allocation)
                self.assertAlmostEqual(top["shortfall_anual_mwh"], difference)
                self.assertAlmostEqual(
                    top["shortfall_incumplido_anual_mwh"], max(0.0, difference - 1.0)
                )
                self.assertAlmostEqual(top["penalidad_anual_usd"], difference * 7.0)
                self.assertAlmostEqual(
                    result["resumen_mensual"]["energia_asignada_mwh"].sum(), allocation
                )
                self.assertAlmostEqual(result["resumen_mensual"]["margen_usd"].sum(), 30.0)

    def test_standard_excel_exports_the_same_annual_classifications(self) -> None:
        for difference, expected in zip(self.differences, self.expected):
            with self.subTest(difference=difference):
                results = self._simulate(
                    allocation=10.0,
                    minimum=10.0 + difference,
                    maximum=10.0 - difference,
                    top_shortfall=difference,
                )
                workbook = load_workbook(
                    io.BytesIO(results_to_excel_bytes(results)), data_only=True
                )

                clients = workbook["clientes_anual"]
                client_headers = [cell.value for cell in clients[1]]
                client_values = dict(zip(client_headers, [cell.value for cell in clients[2]]))
                self.assertEqual(client_values["cumple_min"], expected)
                self.assertEqual(client_values["cumple_max"], expected)
                self.assertAlmostEqual(client_values["faltante_min_mwh"], difference)
                self.assertAlmostEqual(
                    client_values["faltante_min_incumplido_mwh"],
                    max(0.0, difference - 1.0),
                )
                self.assertAlmostEqual(client_values["exceso_max_mwh"], difference)
                self.assertAlmostEqual(
                    client_values["exceso_max_incumplido_mwh"],
                    max(0.0, difference - 1.0),
                )

                top = workbook["take_or_pay_anual"]
                top_headers = [cell.value for cell in top[1]]
                top_values = dict(zip(top_headers, [cell.value for cell in top[2]]))
                self.assertEqual(top_values["cumple_top"], expected)
                self.assertAlmostEqual(top_values["shortfall_anual_mwh"], difference)
                self.assertAlmostEqual(
                    top_values["shortfall_incumplido_anual_mwh"],
                    max(0.0, difference - 1.0),
                )
                self.assertAlmostEqual(top_values["penalidad_anual_usd"], difference * 7.0)

    def test_solver_solution_is_independent_of_classification_tolerance(self) -> None:
        df_gen = pd.DataFrame(
            [
                {
                    "anio": 2026,
                    "mes": 1,
                    "nemo_generador": "G1",
                    "energia_disponible_mwh": 10.0,
                    "precio_compra_usd_mwh": 2.0,
                    "take_or_pay_ratio": 0.0,
                    "penalidad_no_tomado_usd_mwh": 7.0,
                }
            ]
        )
        df_cons = pd.DataFrame(
            [
                {
                    "anio": 2026,
                    "mes": 1,
                    "nemo_cliente": "C1",
                    "consumo_mwh": 10.0,
                    "precio_venta_usd_mwh": 5.0,
                }
            ]
        )
        df_lim = pd.DataFrame(
            [{"nemo_cliente": "C1", "min_anual_mwh": 4.0, "max_anual_mwh": 6.0}]
        )
        arguments = {
            "df_gen": df_gen,
            "df_cons": df_cons,
            "df_lim": df_lim,
            "df_nom_template": pd.DataFrame(),
            "anio": 2026,
            "solver_time_limit_s": 5,
        }

        baseline = solve_milp_nominal(**arguments)
        original_tolerance = compliance.TOLERANCIA_CUMPLIMIENTO_MWH
        try:
            compliance.TOLERANCIA_CUMPLIMIENTO_MWH = 99.0
            changed_tolerance = solve_milp_nominal(**arguments)
        finally:
            compliance.TOLERANCIA_CUMPLIMIENTO_MWH = original_tolerance

        pd.testing.assert_frame_equal(baseline.df_vars, changed_tolerance.df_vars)
        self.assertEqual(baseline.objective_usd, changed_tolerance.objective_usd)
        self.assertEqual(baseline.df_vars["x_opt_mwh"].sum(), 6.0)


if __name__ == "__main__":
    unittest.main()
