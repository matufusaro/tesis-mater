from __future__ import annotations

import io
from typing import Dict

import pandas as pd


def results_to_excel_bytes(results: Dict[str, pd.DataFrame]) -> bytes:
    """Convierte el diccionario de resultados del simulador/optimizador a un .xlsx.

    Se exporta cada tabla en una hoja distinta para que quede "tipo tabla" en Excel
    (no como un CSV con separadores).
    """

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Orden sugerido
        sheet_order = [
            "resumen_mensual",
            "por_generador_prioridad",
            "clientes_anual",
            "no_abastecido",
            "energia_no_asignada_generador",
            "detalle",
        ]
        used = set()
        for name in sheet_order:
            if name in results and isinstance(results[name], pd.DataFrame):
                results[name].to_excel(writer, sheet_name=_safe_sheet(name), index=False)
                used.add(name)
        # Cualquier otra tabla adicional
        for name, df in results.items():
            if name in used or not isinstance(df, pd.DataFrame):
                continue
            df.to_excel(writer, sheet_name=_safe_sheet(name), index=False)

    return output.getvalue()


def _safe_sheet(name: str) -> str:
    # Excel limita a 31 chars
    s = name.replace("/", "_")
    return s[:31]
