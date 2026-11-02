from __future__ import annotations

import pandas as pd
import io


def dfs_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Convierte un dict {sheet_name: dataframe} en un .xlsx en memoria.

    Escribe tablas con columnas (no CSV por comas), para que se vea igual que en pantalla.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe = str(name)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
            # Formato 'tabla' para mejor legibilidad
            ws = writer.sheets[safe]
            nrows, ncols = df.shape
            if nrows == 0 or ncols == 0:
                continue
            columns = [{"header": str(c)} for c in df.columns]
            ws.add_table(0, 0, nrows, ncols - 1, {"columns": columns, "style": "Table Style Medium 9"})
            ws.freeze_panes(1, 0)
    return output.getvalue()
