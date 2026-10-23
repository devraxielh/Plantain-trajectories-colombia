"""
====================================================================================================
ORQUESTADOR MAESTRO DEL PROTOCOLO EXPERIMENTAL: correr_experimento.py
Lee todos los hiperparámetros desde el archivo plano 'parametros_experimento.txt' y ejecuta
secuencialmente las fases enumeradas:
  - Fase 0: fase_0_ingesta_y_clima.py
  - Fase 1: fase_1_clustering_dtw_y_topologia.py
  - Fase 2: fase_2_cadenas_markov_ergodicidad.py
  - Fase 3: fase_3_validacion_rolling_y_xai_shap.py
  - Fase 4: fase_4_integracion_resiliencia_independiente.py
====================================================================================================
"""

import os
import sys
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from organizacion_resultados import registrar_ejecucion, guardar_reproducibilidad, finalizar_paquete
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fase_0_ingesta_y_clima import cargar_parametros_planos, ejecutar_fase_0, solicitar_archivo_inicial, leer_bytes_sin_bloqueo
from fase_1_clustering_dtw_y_topologia import ejecutar_fase_1
from fase_2_cadenas_markov_ergodicidad import ejecutar_fase_2
from fase_3_validacion_rolling_y_xai_shap import ejecutar_fase_3
from fase_4_integracion_resiliencia_independiente import ejecutar_fase_4
from validaciones_aporte_metodologico import ejecutar_validaciones_adicionales



def configurar_estilo_grafico(dpi: int):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def ejecutar_experimento():
    inicio = time.perf_counter()
    fecha_inicio = datetime.now(timezone.utc).isoformat()
    ruta_parametros = sys.argv[2] if len(sys.argv) > 2 else "parametros_experimento.txt"
    params = cargar_parametros_planos(ruta_parametros)
    np.random.seed(params["SEMILLA_ALEATORIA"])
    configurar_estilo_grafico(params["DPI_FIGURAS"])

    archivo_cli = sys.argv[1] if len(sys.argv) > 1 else None
    ruta_inicial = solicitar_archivo_inicial(params, archivo_cli)
    documentos = [p for p in [Path("README.md"), Path("readme.txt")] if p.is_file()]
    entradas = sorted(Path(".").glob("*.py")) + [Path(ruta_parametros), Path(ruta_inicial), Path("requirements.txt")] + documentos
    entradas = list(dict.fromkeys(entradas))
    hashes_antes = {str(p): hashlib.sha256(leer_bytes_sin_bloqueo(str(p))).hexdigest() for p in entradas}

    # Ejecución secuencial de las 5 fases enumeradas (0 a 4)
    contexto = ejecutar_fase_0(params, ruta_inicial)
    contexto = ejecutar_fase_1(contexto, params)
    contexto = ejecutar_fase_2(contexto, params)
    contexto = ejecutar_fase_3(contexto, params)
    contexto = ejecutar_validaciones_adicionales(contexto, params)
    contexto = ejecutar_fase_4(contexto, params)
    cobertura = contexto["df_cobertura"]
    perfiles = contexto["df_perfiles"]
    resumen_perfiles = perfiles.groupby("Cluster_DTW").agg(
        N_Departamentos=("Departamento", "size"),
        Media_Rendimientos_Departamentales=("Rend_Media (ton/ha)", "mean"),
        Media_Areas_Departamentales=("Area_Media (ha)", "mean"),
        Media_Pendiente_Sen=("Sen_Slope_Rend (ton/ha/anio)", "mean"),
        Media_CV_Residual=("CV_Residual_Destendenciado (%)", "mean"))
    resumen_principal = (
        "RESULTADOS PRINCIPALES: PERFILES DE TRANSICIÓN PRODUCTIVA Y SIMILITUD TERRITORIAL\n"
        + f"Periodo completo disponible: {contexto['anio_min']}-{contexto['anio_max']}.\n"
        + f"Registros antes de exclusiones: {contexto['n_filas_antes_exclusiones']}; excluidos explícitamente: {int(contexto['df_exclusiones']['N_Filas_Excluidas'].sum())}. Ver T00; el original se conserva intacto.\n"
        + f"Filas del panel: {len(contexto['df'])}; departamentos: {len(cobertura)}; incluidos en clustering/grafo: {int(cobertura['Incluido_Clustering_y_Grafo'].sum())}.\n"
        + f"Rendimientos válidos observados: {int(cobertura['N_Rendimientos_Validos'].sum())}; sin imputación de rendimientos ni de años ausentes.\n"
        + f"Transiciones observadas entre años consecutivos: {len(contexto['df_trans'])}.\n"
        + "\nPerfiles DTW (medias departamentales, unidades: rendimiento t/ha, pendiente t/ha/año, CV %):\n"
        + resumen_perfiles.round(3).to_string() + "\n\n"
        + f"Partición: modo={contexto['seleccion_adaptativa']['modo']}, algoritmo={contexto['seleccion_adaptativa']['algoritmo']}, k={contexto['seleccion_adaptativa']['k']}; estado={contexto['seleccion_adaptativa']['estado']}.\n"
        + contexto["seleccion_adaptativa"]["motivo"] + "\n"
        + f"Geometría principal definida por objetivo: {params['GEOMETRIA_PRINCIPAL_ADAPTATIVA']}.\n"
        + "Comparación separada de escala y forma (no se elige una geometría por comparar sus silhouettes):\n"
        + contexto["comparacion_geometrias"]["comparacion"].round(4).to_string(index=False) + "\n\n"
        + "T19 resume ambas preguntas, T20 compara asignaciones y crecimiento, T21/T22 conservan sus matrices de distancia.\n"
        + "T16 compara los candidatos, T17 registra cada perturbación y T18 la estabilidad de cada grupo. Ver seleccion_adaptativa.json para la regla y sus umbrales.\n"
        + "T11 registra toda la cobertura, T12 los perfiles y variabilidad residual, T13 la movilidad observada y T14 las matrices por perfil.\n"
        + "Los estados Markov se definen con cuantiles de todo el periodo. Los clusters describen trayectorias, no identifican tecnificación ni sus causas.\n"
        + "Ausencia de tendencia detectada no demuestra estancamiento. Contrastes de tendencia exploratorios con ajuste Holm entre departamentos.\n"
        + "Las distancias DTW se calculan sobre años observados compartidos y se normalizan por raíz de su número; T15 registra ese soporte.\n"
        + "Comparar soportes distintos puede afectar la partición; no se fabrican los años faltantes ni se excluyen departamentos por empezar después.\n"
        + "El artículo principal se centra en perfiles, crecimiento diferenciado y persistencia observada. T05 (largo plazo), T07-T10, Fig5-Fig8 son complementarios; no validan los perfiles ni prueban causalidad.\n"
    )
    resumen_principal += "\n" + contexto["resumen_validaciones_adicionales"]
    Path(contexto["dir_result"], "RESULTADOS_PRINCIPALES.txt").write_text(resumen_principal, encoding="utf-8")
    # Regenerar textos desde los valores reales; sustituye explicaciones antiguas.
    import pandas as pd
    for csv in Path(contexto["dir_tablas"]).glob("T*.csv"):
        tabla = pd.read_csv(csv)
        csv.with_suffix(".txt").write_text(
            "TABLA GENERADA DESDE " + csv.name + "\nVer NOTAS_METODOLOGICAS.txt para alcance y limitaciones.\n\n"
            + tabla.to_string(index=False) + "\n", encoding="utf-8")
    notas_figuras = {
        "Fig1": f"Análisis principal tras exclusiones T00: geometría {params['GEOMETRIA_PRINCIPAL_ADAPTATIVA']}; estado {contexto['seleccion_adaptativa']['estado']}. Selección según T16-T18; T19/T20 contrastan escala y forma; bandas descriptivas, no validación de perfiles.",
        "Fig2": "Grafo de similitud DTW, MST y vecinos. No representa conexiones físicas ni causalidad.",
        "Fig3": "Análisis principal observado: probabilidades y conteos de transición, movilidad por perfil y distribución empírica anual de estados. El nombre histórico del archivo se conserva por compatibilidad. Las proyecciones condicionadas de largo plazo están sólo en T05 complementaria.",
        "Fig4": "Pendientes Sen y Mann-Kendall ordinario, sin ajuste de autocorrelación; inferencia exploratoria.",
        "Fig5": "Análisis complementario supervisado, no validación del clustering/Markov: estimación retrospectiva OOT con clima observado. Panel a: R2 medio anual. Panel b: R2 agrupado XGBoost dinámico.",
        "Fig6": "Contribuciones XGBoost TreeSHAP únicamente en años OOT, no efectos causales.",
        "Fig7": "LOWESS de TreeSHAP OOT. IC bootstrap cruzado departamento/año condicional a modelos fijos, sin reentrenamiento.",
        "Fig8": "Ratios observados y contribuciones TreeSHAP OOT; índice compuesto descriptivo, sin atribución causal.",
        "Fig9": "Contraste principal de escala y evolución: conteos de departamentos por cruce de asignaciones. Los silhouettes de geometrías diferentes no se usan para escoger la pregunta científica.",
        "Fig10": "Ablación climática pareada con cinco algoritmos y dos bases productivas. Ganancia RMSE positiva favorece añadir clima observado. IC bootstrap por bloques anuales OOT, condicionales a modelos fijos; no causalidad. T23-T25.",
        "Fig11": "PAM k=2/3/4 con DTW y RMS euclídeo sin alineamiento; iguales series, normalización y ventanas retiradas. P10 describe sensibilidad determinista, no intervalo de confianza. No establece verdad externa de clusters. T26-T29.",
    }
    for figura in Path(contexto["dir_figuras"]).glob("Fig*.png"):
        figura.with_suffix(".txt").write_text(notas_figuras[figura.name.split("_")[0]] + "\nVer NOTAS_METODOLOGICAS.txt.\n", encoding="utf-8")
    notas = documentos[0].read_text(encoding="utf-8") if documentos else (
        "Consulte RESULTADOS_PRINCIPALES.txt, VALIDACIONES_ADICIONALES.txt y manifest.json.\n"
        "Clustering retrospectivo; clima observado; sin inferencia causal.\n")
    Path(contexto["dir_result"], "NOTAS_METODOLOGICAS.txt").write_text(notas, encoding="utf-8")
    Path(contexto["dir_result"], "INDICE_RESULTADOS.txt").write_text(
        "ANÁLISIS PRINCIPAL: perfiles temporales, crecimiento y transiciones observadas.\n"
        "Fig1, Fig2, Fig3, Fig4, Fig9 y Fig11; T00-T04, T11-T22 y T26-T29; RESULTADOS_PRINCIPALES.txt.\n"
        "ANÁLISIS COMPLEMENTARIO: ML, SHAP, indicadores de resiliencia y proyecciones condicionadas.\n"
        "Fig5-Fig8 y Fig10; T05-T10 y T23-T25. VALIDACIONES_ADICIONALES.txt resume los tres contrastes nuevos. Las proyecciones de largo plazo no son hechos observados.\n", encoding="utf-8")
    hashes_despues = {str(p): hashlib.sha256(leer_bytes_sin_bloqueo(str(p))).hexdigest() for p in entradas}
    if hashes_antes != hashes_despues:
        raise RuntimeError("Un archivo de entrada/código cambió durante la ejecución; resultados no certificados.")
    guardar_reproducibilidad(contexto["dir_result"], entradas, leer_bytes_sin_bloqueo)
    archivos_salida = sorted(p for p in Path(contexto["dir_result"]).rglob("*") if p.is_file() and p.name != "manifest.json")
    salidas = [{"path": p.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "kind": "result"} for p in archivos_salida]
    manifiesto = {
        "schema_version": 2, "claim_id": "PLATANO_PROTOCOLO_CORREGIDO",
        "repository": {"commit": "unavailable: carpeta sin repositorio Git", "dirty": True},
        "command": [sys.executable, *sys.argv],
        "environment": {"software": [{"name": "Python", "version": platform.python_version()}]
            + [{"name": n, "version": version(n)} for n in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "networkx", "seaborn", "xgboost", "lightgbm", "shap", "statsmodels", "openpyxl"]],
            "hardware": platform.platform() + " / " + platform.machine() + " / " + str(os.cpu_count()) + " logical CPUs"},
        "mathematics": {"assertion_tested": "Comparación escala/forma tras exclusiones T00; PAM k=2/3/4 con DTW y RMS sin alineamiento sobre iguales soportes y perturbaciones; diez ablaciones sin/con clima con iguales modelos, cohortes y folds; perfiles, crecimiento y transiciones observadas; ML/SHAP y largo plazo condicionados.",
            "coefficient_domain": "IEEE 754 float64; modelos externos pueden usar float32",
            "conventions": "DTW vectorial, clustering retrospectivo; evaluación temporal con clima observado; IC condicionales por panel",
            "inputs": [str(p) for p in entradas], "bounds": {"anio_min": contexto["anio_min"], "anio_max": contexto["anio_max"], "filas": len(contexto["df"]), "k_seleccionado": contexto["seleccion_adaptativa"]["k"], "algoritmo_seleccionado": contexto["seleccion_adaptativa"]["algoritmo"]},
            "non_claims": ["No prueba causalidad, pronóstico ex ante, estacionariedad ni validez universal de índices.", "Los IC SHAP no incluyen incertidumbre de reentrenamiento."]},
        "randomness": {"used": True, "generator": "NumPy default_rng PCG64 y random_state de bibliotecas", "seed": params["SEMILLA_ALEATORIA"]},
        "run": {"started_at": fecha_inicio, "runtime_seconds": time.perf_counter() - inicio, "exit_status": 0, "status": "completed",
            "resource_limits": {"memory_bytes": None, "cpu_seconds": None, "max_cores": None, "max_threads": None}},
        "outputs": salidas,
        "checks": ["TreeSHAP reconstruye predicciones OOT: np.allclose atol=1e-4", "Hashes de código/configuración/entrada sin cambios durante ejecución", "Guardado de figuras PNG/PDF con errores propagados", "Exclusión explícita y registrada; dos geometrías evaluadas independientemente", "DTW no supera la ruta diagonal RMS: tolerancia 1e-10", "Ablaciones pareadas: mismas filas y años; entrenamiento estrictamente anterior; predicciones finitas"],
        "result": "Ejecución finalizada; ver RESULTADOS_PRINCIPALES.txt y tablas T11-T15 para cobertura, perfiles y transiciones.",
        "residual_risks": ["Pocos años y dependencia temporal limitan inferencia.", "Un punto climático no representa todo un departamento.", "Descargas externas pueden revisarse en ejecuciones futuras."],
        "input_artifacts": [{"path": p.as_posix(), "sha256": hashes_antes[str(p)], "sha256_after": hashes_despues[str(p)]} for p in entradas],
        "declared_results": [p["path"] for p in salidas],
    }
    Path(contexto["dir_result"], "manifest.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 95)
    print(f"EXPERIMENTO COMPLETO FINALIZADO CON ÉXITO PARA: '{contexto['nombre_cultivo']}'")
    print(f"  * Análisis principal: {len(cobertura)} departamentos, {contexto['anio_min']}-{contexto['anio_max']}, {len(contexto['df'])} filas")
    print(f"  * Perfiles y transiciones:     {contexto['dir_result']}\\RESULTADOS_PRINCIPALES.txt")
    print(f"  * Archivo plano de parámetros: {ruta_parametros}")
    print(f"  * Archivo inicial intacto:     {contexto['ruta_inicial']}")
    print(f"  * Datasets procesados en:      {contexto['dir_datasets']}\\")
    print(f"  * Tablas y validaciones en:    {contexto['dir_tablas']}\\RESULTADOS_METRICAS_Y_VALIDACIONES.xlsx")
    print(f"  * Figuras ({params['DPI_FIGURAS']} DPI) en:        {contexto['dir_figuras']}\\")
    print("=" * 95)


def main():
    ruta_parametros = sys.argv[2] if len(sys.argv) > 2 else "parametros_experimento.txt"
    params = cargar_parametros_planos(ruta_parametros)
    directorio = params["CARPETA_RESULTADOS"]
    with registrar_ejecucion(directorio, "experimento_completo"):
        ejecutar_experimento()
    finalizar_paquete(directorio)


if __name__ == "__main__":
    main()
