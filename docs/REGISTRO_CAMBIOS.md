# Registro de cambios y decisiones

## 2026-11-02 — Importación original

Versión de referencia: 92124e0a44bbc66a28816a9d358f29e543873482. Fuente: asignacion-energia.zip. Se conservan sin cambios los 20 archivos de código, datos, documentación y resultados seleccionados. Se excluyen cachés __pycache__ y data/test_write.txt. README.md inicial conservado.

## Decisiones confirmadas

- TOP: permitir incumplimiento y descontar la penalidad del margen. Falta verificar parámetros contractuales y fórmula antes de implementarlo.
- Registrar cada modificación, motivo, verificación e impacto en resultados.
- Cambios comerciales en datos requieren fundamento; no imputar precios o compromisos desconocidos.

## Diagnóstico preliminar (no corregido todavía)

- Ambos Excel contienen 48 filas completamente vacías en Nominaciones que bloquean conversiones de año.
- Redistribución: caso con oferta 100 MWh, nominales [100,100] y demandas [20,100] devuelve [20,50], dejando 30 MWh asignables.
- Indicadores de app usan columnas ausentes del resumen del simulador.
- App procesa resultados de solver sin exigir factibilidad.
- Comparación no exige escenario y año coincidentes.
- prioridad_cliente no se utiliza para ordenar abastecimiento entre generadores.
- Optimización actual no implementa TOP por generador de la tesis; penalidad fija P2 no equivale a TOP.

## Ejecuciones exploratorias

Solo se eliminaron filas completamente vacías en memoria. Motor original, año 2026:

| Archivo / escenario | Asignación MWh | Margen USD |
|---|---:|---:|
| BD_Ejemplo_2026.xlsx / BASE | 83026.11902027289 | 621577.3810326237 |
| BD_Ejemplo_2026_escenarios.xlsx / Baseline | 85837.94754782163 | 647754.0286550095 |

Estos resultados no son óptimos validados. No se ejecutó PuLP: no estaba instalado en el entorno de revisión.

## Orden de trabajo

1. Validación de entradas y pruebas de conservación, demanda y redistribución.
2. Correcciones reproducibles de motor e interfaz y medición del impacto.
3. Precisar prioridades, TOP y contratos a partir de tesis y datos.
4. Optimización y contraste contra simulación; distinguir óptimo probado de solución factible.
5. Importación de datos e interfaz; documentación de defensa.


## 2026-11-04 — Corrección de importación al arrancar

Se actualizó `app/app.py` para agregar la raíz del proyecto (`tesis-mater/`) a `sys.path` a partir de la ubicación del propio archivo. Esto corrige `ModuleNotFoundError: No module named 'src'` al lanzar Streamlit. Verificación: inspección del recorrido de rutas; aún pendiente confirmación de ejecución en el Windows del usuario. No cambia cálculos ni resultados.


## 2026-11-04 — Limpieza de filas vacías en la carga Excel

El cargador ahora elimina registros completamente vacíos de las cuatro hojas leídas y reinicia sus índices. Esto corrige `IntCastingNaNError` al convertir `Nominaciones.anio` y evita que filas vacías lleguen al simulador. Las filas con algún dato se conservan. Validación: cambio revisado; ejecución en el Windows del usuario pendiente. No altera los datos comerciales válidos.


## 2026-11-04 — Alinear indicadores con resultados del simulador

La app ahora calcula energía asignada desde `resumen_mensual.energia_asignada_mwh` y energía no abastecida desde `no_abastecido.no_abastecido_mwh`, nombres realmente devueltos por el simulador. El KeyError ocurría después de cargar datos y correr la simulación; no era un fallo de lectura de las bases. Verificación: contraste con las columnas de salida de `simulate_cammesa`; confirmación de ejecución en Windows pendiente.
