"""
====================================================================================================
FASE 0: PARSER DE ARCHIVO PLANO DE PARÁMETROS, INGESTA EN MODO SOLO LECTURA, DESCARGA CLIMÁTICA
        (NASA POWER API & NOAA ONI) Y PREPROCESAMIENTO ANTI-LEAKAGE
====================================================================================================
"""

import calendar
import hashlib
import ctypes
import io
import json
import msvcrt
import os
from pathlib import Path
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd


def cargar_parametros_planos(ruta_txt: str = "parametros_experimento.txt") -> dict:
    """
    Lee todos los hiperparámetros, umbrales, rutas y coordenadas desde el archivo plano 'parametros_experimento.txt'.
    """
    if not os.path.exists(ruta_txt):
        raise FileNotFoundError(f"No se encontró el archivo plano de parámetros: '{ruta_txt}'")

    raw_dict = {}
    coordenadas = {}

    with open(ruta_txt, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = [p.strip() for p in linea.split("=", 1)]
            if clave.startswith("COORD_DEPTO_"):
                depto_key = clave.replace("COORD_DEPTO_", "").replace("_", " ").strip().upper()
                partes = [x.strip() for x in valor.split("|")]
                if len(partes) == 3:
                    coordenadas[depto_key] = {
                        "municipio": partes[0],
                        "lat": float(partes[1]),
                        "lon": float(partes[2]),
                    }
            else:
                raw_dict[clave] = valor

    def _bool(k, default=True):
        val = raw_dict.get(k, str(default)).strip().lower()
        if val not in ("true", "1", "yes", "si", "sí", "false", "0", "no"):
            raise ValueError(f"Booleano no reconocido para {k}: {val}")
        return val in ("true", "1", "yes", "si", "sí")

    def _int(k, default=0):
        valor = float(raw_dict.get(k, default))
        if not valor.is_integer():
            raise ValueError(f"{k} requiere un entero, no {valor}.")
        return int(valor)

    def _float(k, default=0.0):
        return float(raw_dict.get(k, default))

    def _list_int(k, default="2,3,4,5,6"):
        return [int(x.strip()) for x in raw_dict.get(k, default).split(",") if x.strip()]

    def _list_float(k, default="25,50,75"):
        return [float(x.strip()) for x in raw_dict.get(k, default).split(",") if x.strip()]

    def _list_str(k, default=""):
        return [x.strip() for x in raw_dict.get(k, default).split(",") if x.strip()]

    params = {
        "PREGUNTAR_ARCHIVO_INICIAL": _bool("PREGUNTAR_ARCHIVO_INICIAL", True),
        "ARCHIVO_INICIAL_POR_DEFECTO": raw_dict.get("ARCHIVO_INICIAL_POR_DEFECTO", "PLATANO.xlsx"),
        "CARPETA_RESULTADOS": raw_dict.get("CARPETA_RESULTADOS", "result"),
        "SEMILLA_ALEATORIA": _int("SEMILLA_ALEATORIA", 42),
        "PROPORCION_TRAIN_HISTORICO": _float("PROPORCION_TRAIN_HISTORICO", 0.80),
        "ANIO_INICIO_ANALISIS": _int("ANIO_INICIO_ANALISIS", 0),
        "ANIO_FIN_ANALISIS": _int("ANIO_FIN_ANALISIS", 0),
        "EXCLUIR_DEPARTAMENTOS": _list_str("EXCLUIR_DEPARTAMENTOS", ""),
        "MOTIVO_EXCLUSION_DEPARTAMENTOS": raw_dict.get("MOTIVO_EXCLUSION_DEPARTAMENTOS", "Exclusión explícita configurada por el investigador"),
        "USAR_CACHE_CLIMA": _bool("USAR_CACHE_CLIMA", True),
        "DPI_FIGURAS": _int("DPI_FIGURAS", 300),
        # NASA POWER y NOAA ONI
        "NASA_POWER_COMUNIDAD": raw_dict.get("NASA_POWER_COMUNIDAD", "AG"),
        "NASA_PARAMETROS": raw_dict.get("NASA_PARAMETROS", "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN,RH2M"),
        "NASA_MAX_WORKERS": _int("NASA_MAX_WORKERS", 8),
        "NASA_TIMEOUT_SEGUNDOS": _int("NASA_TIMEOUT_SEGUNDOS", 35),
        "NOAA_ONI_URL": raw_dict.get("NOAA_ONI_URL", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"),
        "UMBRAL_DEFICIT_HIDRICO_MENSUAL_MM": _float("UMBRAL_DEFICIT_HIDRICO_MENSUAL_MM", 100.0),
        "UMBRAL_ONI_EL_NINO": _float("UMBRAL_ONI_EL_NINO", 0.5),
        "UMBRAL_ONI_LA_NINA": _float("UMBRAL_ONI_LA_NINA", -0.5),
        "MIN_TRIMESTRES_FASE_ENSO": _int("MIN_TRIMESTRES_FASE_ENSO", 5),
        # Fase 1: DTW y Topología
        "MINIMO_ANIOS_COMPARTIDOS_DTW": _int("MINIMO_ANIOS_COMPARTIDOS_DTW", 5),
        "VENTANA_SAKOE_CHIBA": _int("VENTANA_SAKOE_CHIBA", 2),
        "GRILLA_CLUSTERS_K": _list_int("GRILLA_CLUSTERS_K", "2,3,4,5,6"),
        "K_CLUSTERS_FIJO": _int("K_CLUSTERS_FIJO", 0),
        "ALGORITMO_CLUSTER_FIJO": raw_dict.get("ALGORITMO_CLUSTER_FIJO", "pam").strip().lower(),
        "GEOMETRIA_PRINCIPAL_ADAPTATIVA": raw_dict.get("GEOMETRIA_PRINCIPAL_ADAPTATIVA", "forma").strip().lower(),
        "ALGORITMOS_ADAPTATIVOS": _list_str("ALGORITMOS_ADAPTATIVOS", "pam,average,complete"),
        "BLOQUES_ESTABILIDAD_ANIOS": _list_int("BLOQUES_ESTABILIDAD_ANIOS", "1,2"),
        "SILHOUETTE_MINIMA_ADAPTATIVA": _float("SILHOUETTE_MINIMA_ADAPTATIVA", 0.25),
        "ARI_P10_MINIMO": _float("ARI_P10_MINIMO", 0.75),
        "JACCARD_P10_MINIMO": _float("JACCARD_P10_MINIMO", 0.75),
        "ARI_COBERTURA_MINIMO": _float("ARI_COBERTURA_MINIMO", 0.75),
        "FRACCION_REPLICAS_VALIDAS_MINIMA": _float("FRACCION_REPLICAS_VALIDAS_MINIMA", 0.90),
        "TOLERANCIA_SILHOUETTE_ADAPTATIVA": _float("TOLERANCIA_SILHOUETTE_ADAPTATIVA", 0.02),
        "REINICIOS_KMEDOIDS_PAM": _int("REINICIOS_KMEDOIDS_PAM", 15),
        "MAX_ITER_KMEDOIDS": _int("MAX_ITER_KMEDOIDS", 100),
        "VECINOS_KNN_GRAFO": _int("VECINOS_KNN_GRAFO", 2),
        "ALPHA_SIGNIFICANCIA_ESTADISTICA": _float("ALPHA_SIGNIFICANCIA_ESTADISTICA", 0.05),
        # Fase 2: Cadenas de Markov
        "CUANTILES_ESTADOS_MARKOV": _list_float("CUANTILES_ESTADOS_MARKOV", "25,50,75"),
        "ALPHA_DIRICHLET_BAYES": _float("ALPHA_DIRICHLET_BAYES", 0.5),
        "ALPHA_DIRICHLET_BOOTSTRAP": _float("ALPHA_DIRICHLET_BOOTSTRAP", 0.05),
        "UMBRAL_CELDA_ESCASA": _int("UMBRAL_CELDA_ESCASA", 5),
        "REPLICAS_BOOTSTRAP_MARKOV": _int("REPLICAS_BOOTSTRAP_MARKOV", 1000),
        # Fase 3: Rolling-Origin, Modelos ML y TreeSHAP
        "ANIOS_EVALUACION_ROLLING_ORIGIN": _int("ANIOS_EVALUACION_ROLLING_ORIGIN", 6),
        "REPLICAS_BOOTSTRAP_ABLACION": _int("REPLICAS_BOOTSTRAP_ABLACION", 2000),
        "VARIABLES_PREDICTORAS_XAI": _list_str("VARIABLES_PREDICTORAS_XAI"),
        "RF_N_ESTIMATORS": _int("RF_N_ESTIMATORS", 500),
        "RF_MAX_DEPTH": _int("RF_MAX_DEPTH", 10),
        "RF_MIN_SAMPLES_LEAF": _int("RF_MIN_SAMPLES_LEAF", 2),
        "XGB_N_ESTIMATORS": _int("XGB_N_ESTIMATORS", 350),
        "XGB_MAX_DEPTH": _int("XGB_MAX_DEPTH", 4),
        "XGB_LEARNING_RATE": _float("XGB_LEARNING_RATE", 0.04),
        "XGB_SUBSAMPLE": _float("XGB_SUBSAMPLE", 0.85),
        "XGB_COLSAMPLE_BYTREE": _float("XGB_COLSAMPLE_BYTREE", 0.85),
        "LGBM_N_ESTIMATORS": _int("LGBM_N_ESTIMATORS", 350),
        "LGBM_MAX_DEPTH": _int("LGBM_MAX_DEPTH", 5),
        "LGBM_NUM_LEAVES": _int("LGBM_NUM_LEAVES", 20),
        "LGBM_LEARNING_RATE": _float("LGBM_LEARNING_RATE", 0.04),
        "LGBM_SUBSAMPLE": _float("LGBM_SUBSAMPLE", 0.85),
        "GBM_N_ESTIMATORS": _int("GBM_N_ESTIMATORS", 300),
        "GBM_MAX_DEPTH": _int("GBM_MAX_DEPTH", 4),
        "GBM_LEARNING_RATE": _float("GBM_LEARNING_RATE", 0.04),
        "GBM_SUBSAMPLE": _float("GBM_SUBSAMPLE", 0.85),
        "ELASTICNET_ALPHA": _float("ELASTICNET_ALPHA", 0.10),
        "ELASTICNET_L1_RATIO": _float("ELASTICNET_L1_RATIO", 0.50),
        "FRACCION_SUAVIZADO_LOWESS": _float("FRACCION_SUAVIZADO_LOWESS", 0.38),
        "REPLICAS_BOOTSTRAP_SHAP": _int("REPLICAS_BOOTSTRAP_SHAP", 200),
        "NODOS_GRILLA_LOWESS": _int("NODOS_GRILLA_LOWESS", 120),
        "FRACCION_MINIMA_ESTABILIDAD_CRUCE": _float("FRACCION_MINIMA_ESTABILIDAD_CRUCE", 0.25),
        # Fase 4: Resiliencia Independiente
        "UMBRAL_CHOQUE_CLIMATICO_Z": _float("UMBRAL_CHOQUE_CLIMATICO_Z", 1.00),
        "UMBRAL_CHOQUE_ONI_ABS": _float("UMBRAL_CHOQUE_ONI_ABS", 0.75),
        "PESO_INDICE_RESISTENCIA_RS": _float("PESO_INDICE_RESISTENCIA_RS", 0.40),
        "PESO_INDICE_RECUPERACION_RC": _float("PESO_INDICE_RECUPERACION_RC", 0.30),
        "PESO_INDICE_ESTABILIDAD_ST": _float("PESO_INDICE_ESTABILIDAD_ST", 0.30),
        # Visualización
        "PALETA_CLUSTERS": {
            "C1_DTW": raw_dict.get("COLOR_CLUSTER_C1", "#6C757D"),
            "C2_DTW": raw_dict.get("COLOR_CLUSTER_C2", "#2A9D8F"),
            "C3_DTW": raw_dict.get("COLOR_CLUSTER_C3", "#E9C46A"),
            "C4_DTW": raw_dict.get("COLOR_CLUSTER_C4", "#E63946"),
        },
        "NOMBRES_CLUSTERS": {
            "C1_DTW": raw_dict.get("ETIQUETA_CLUSTER_C1", "C1: Periferia Tradicional (Bajo Rend.)"),
            "C2_DTW": raw_dict.get("ETIQUETA_CLUSTER_C2", "C2: Transición en Expansión Moderada"),
            "C3_DTW": raw_dict.get("ETIQUETA_CLUSTER_C3", "C3: Cinturón Comercial Consolidado"),
            "C4_DTW": raw_dict.get("ETIQUETA_CLUSTER_C4", "C4: Élite Agroindustrial (Alto Rend.)"),
        },
        "PALETA_ENSO": {
            "El Niño": raw_dict.get("COLOR_ENSO_EL_NINO", "#D62828"),
            "La Niña": raw_dict.get("COLOR_ENSO_LA_NINA", "#1D3557"),
            "Neutro / Transición": raw_dict.get("COLOR_ENSO_NEUTRO", "#8D99AE"),
        },
        "COORDENADAS_DEPARTAMENTOS": coordenadas,
    }
    if len(params["CUANTILES_ESTADOS_MARKOV"]) != 3:
        raise ValueError("Markov requiere tres cuantiles para cuatro estados fijos en el tiempo.")
    if params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"] not in {"escala", "forma"}:
        raise ValueError("La geometría principal debe ser escala o forma.")
    if not params["GRILLA_CLUSTERS_K"] or min(params["GRILLA_CLUSTERS_K"]) < 2:
        raise ValueError("La grilla adaptativa debe contener valores k >= 2.")
    if not params["ALGORITMOS_ADAPTATIVOS"] or not set(params["ALGORITMOS_ADAPTATIVOS"]).issubset({"pam", "average", "complete"}):
        raise ValueError("Algoritmos adaptativos admitidos: pam, average, complete.")
    if params["K_CLUSTERS_FIJO"] < 0 or (params["K_CLUSTERS_FIJO"] and (
            params["K_CLUSTERS_FIJO"] not in params["GRILLA_CLUSTERS_K"] or
            params["ALGORITMO_CLUSTER_FIJO"] not in params["ALGORITMOS_ADAPTATIVOS"])):
        raise ValueError("La partición fija debe pertenecer a la grilla y a los algoritmos evaluados; 0 activa selección adaptativa.")
    if not params["BLOQUES_ESTABILIDAD_ANIOS"] or min(params["BLOQUES_ESTABILIDAD_ANIOS"]) < 1:
        raise ValueError("Los bloques temporales deben tener longitudes positivas.")
    for clave in ["SILHOUETTE_MINIMA_ADAPTATIVA", "ARI_P10_MINIMO", "JACCARD_P10_MINIMO", "ARI_COBERTURA_MINIMO", "FRACCION_REPLICAS_VALIDAS_MINIMA", "TOLERANCIA_SILHOUETTE_ADAPTATIVA"]:
        if not 0 <= params[clave] <= 1:
            raise ValueError(f"{clave} debe estar entre 0 y 1.")
    if not 0 < params["PROPORCION_TRAIN_HISTORICO"] < 1:
        raise ValueError("PROPORCION_TRAIN_HISTORICO debe estar entre 0 y 1.")
    if any(params[k] < 1 for k in ["NASA_MAX_WORKERS", "REPLICAS_BOOTSTRAP_MARKOV", "REPLICAS_BOOTSTRAP_SHAP", "REPLICAS_BOOTSTRAP_ABLACION", "NODOS_GRILLA_LOWESS"]):
        raise ValueError("Workers, réplicas y nodos deben ser positivos.")
    if params["ALPHA_DIRICHLET_BAYES"] <= 0 or params["ALPHA_DIRICHLET_BOOTSTRAP"] <= 0:
        raise ValueError("Los parámetros Dirichlet deben ser positivos.")
    if not 0 < params["FRACCION_SUAVIZADO_LOWESS"] <= 1 or not 0 < params["FRACCION_MINIMA_ESTABILIDAD_CRUCE"] <= 1:
        raise ValueError("Las fracciones LOWESS/estabilidad deben estar en (0,1].")
    if params["NODOS_GRILLA_LOWESS"] < 3 or params["VENTANA_SAKOE_CHIBA"] < 0:
        raise ValueError("LOWESS requiere al menos tres nodos y DTW una ventana no negativa.")
    if params["MINIMO_ANIOS_COMPARTIDOS_DTW"] < 2:
        raise ValueError("DTW requiere al menos dos años compartidos.")
    cuantiles = params["CUANTILES_ESTADOS_MARKOV"]
    if not 0 < cuantiles[0] < cuantiles[1] < cuantiles[2] < 100:
        raise ValueError("Cuantiles Markov deben ser crecientes y estar entre 0 y 100.")
    return params


def normalizar_texto_depto(nombre: str, catalogo_coords: dict) -> str:
    n_clean = str(nombre).strip().upper()
    if n_clean in catalogo_coords:
        return n_clean
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", n_clean)
        if unicodedata.category(c) != "Mn" or c in ("Ñ", "ñ")
    )
    if sin_tildes in catalogo_coords:
        return sin_tildes
    sin_enie = "".join(c for c in unicodedata.normalize("NFD", n_clean) if unicodedata.category(c) != "Mn")
    if sin_enie in catalogo_coords:
        return sin_enie
    return n_clean


def filtrar_periodo_analisis(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    inicio = params.get("ANIO_INICIO_ANALISIS", 0) or int(df["Anio"].min())
    fin = params.get("ANIO_FIN_ANALISIS", 0) or int(df["Anio"].max())
    if inicio > fin:
        raise ValueError("El año inicial no puede ser posterior al final.")
    resultado = df.loc[df["Anio"].between(inicio, fin)].copy()
    if resultado.empty:
        raise ValueError("El periodo seleccionado no contiene observaciones.")
    return resultado


def aplicar_exclusiones_departamentos(df: pd.DataFrame, params: dict) -> tuple:
    """Exclusiones explícitas de alcance; no alteran el archivo ni dependen del clustering."""
    def clave(value):
        text = "".join(c for c in unicodedata.normalize("NFD", str(value).upper()) if unicodedata.category(c) != "Mn")
        text = " ".join(text.split())
        return "SAN ANDRES Y PROVIDENCIA" if text == "ARCHIPIELAGO DE SAN ANDRES" else text

    columnas = ["Departamento_Excluido", "N_Filas_Excluidas", "Primer_Anio", "Ultimo_Anio", "Motivo"]
    requested = {clave(d) for d in params.get("EXCLUIR_DEPARTAMENTOS", [])}
    keys = df["Departamento"].map(clave)
    records = []
    for dep in sorted(requested):
        rows = df.loc[keys == dep]
        records.append({"Departamento_Excluido": dep, "N_Filas_Excluidas": len(rows),
            "Primer_Anio": rows["Anio"].min() if len(rows) else np.nan,
            "Ultimo_Anio": rows["Anio"].max() if len(rows) else np.nan,
            "Motivo": params.get("MOTIVO_EXCLUSION_DEPARTAMENTOS", "Exclusión explícita configurada")})
    result = df.loc[~keys.isin(requested)].copy()
    if result.empty:
        raise ValueError("La exclusión dejó el análisis sin observaciones.")
    return result, pd.DataFrame(records, columns=columnas)


def agregar_rezagos_anuales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Departamento", "Anio"]).copy()
    columnas = {"Area_Lag1 (ha)": "Area (ha)", "Rendimiento_Lag1 (ton/ha)": "Rendimiento (ton/ha)",
                "Precip_Lag1 (mm)": "Precipitacion_Acumulada_Anual (mm)", "ONI_Lag1 (C)": "ONI_Media_Anual (C)"}
    for lag, original in columnas.items():
        df[lag] = df.groupby("Departamento")[original].shift(1)
    sin_anio_previo = df.groupby("Departamento")["Anio"].diff().ne(1)
    df.loc[sin_anio_previo, list(columnas)] = np.nan
    return df


def solicitar_archivo_inicial(params: dict, archivo_cli: str = None) -> str:
    if archivo_cli and os.path.exists(archivo_cli):
        print(f"[*] Archivo inicial recibido por parámetro CLI: '{archivo_cli}'")
        return archivo_cli

    default_file = params["ARCHIVO_INICIAL_POR_DEFECTO"]
    if not params["PREGUNTAR_ARCHIVO_INICIAL"] and os.path.exists(default_file):
        return default_file

    candidatos = [
        f for f in os.listdir(".")
        if os.path.isfile(f) and f.lower().endswith((".xlsx", ".xls", ".csv")) and not f.startswith("~$")
    ]
    if default_file not in candidatos and candidatos:
        default_file = candidatos[0]

    print("\n" + "=" * 90)
    print("SELECCIÓN DINÁMICA DE ARCHIVO INICIAL (SOLO LECTURA - EL ORIGINAL NO SE MODIFICA)")
    print("=" * 90)
    if candidatos:
        print("Archivos de datos detectados en la carpeta actual:")
        for idx, f in enumerate(candidatos, 1):
            print(f"  [{idx}] {f}")

    try:
        if sys.stdin is not None and sys.stdin.isatty():
            entrada = input(f"\nIngrese el nombre/número del archivo inicial [Enter para '{default_file}']: ").strip()
        else:
            entrada = ""
    except EOFError:
        entrada = ""

    if not entrada:
        ruta_elegida = default_file
    elif entrada.isdigit() and 1 <= int(entrada) <= len(candidatos):
        ruta_elegida = candidatos[int(entrada) - 1]
    else:
        ruta_elegida = entrada.strip('"').strip("'")

    if not os.path.exists(ruta_elegida):
        raise FileNotFoundError(f"No se encontró el archivo '{ruta_elegida}'.")

    print(f"[*] Usando archivo inicial (solo lectura): '{ruta_elegida}'")
    return ruta_elegida


def leer_bytes_sin_bloqueo(ruta_archivo: str) -> bytes:
    try:
        return Path(ruta_archivo).read_bytes()
    except PermissionError:
        GENERIC_READ = 0x80000000
        FILE_SHARE_ALL = 0x00000001 | 0x00000002 | 0x00000004
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x80
        from ctypes import wintypes
        create_file = ctypes.windll.kernel32.CreateFileW
        create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            os.path.abspath(ruta_archivo), GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None
        )
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError()
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        with os.fdopen(fd, "rb") as f:
            raw_bytes = f.read()
        return raw_bytes


def leer_excel_sin_bloqueo(ruta_archivo: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(leer_bytes_sin_bloqueo(ruta_archivo)), sheet_name=0)


def preparar_datos_agricolas(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Lee las seis variables por nombre; admite columnas adicionales y otro orden.

    El encabezado histórico ha/ton se corrige sólo si los valores coinciden con
    producción/área. No se calcula ni imputa un rendimiento con área cero.
    """
    def clave(nombre):
        texto = unicodedata.normalize("NFKD", str(nombre).strip().lower())
        return "".join(c for c in texto if c.isalnum() and not unicodedata.combining(c))

    aliases = {
        "Anio": {"ano", "anio", "ao", "year"},
        "Departamento": {"departamento", "department"},
        "Producto": {"producto", "product", "cultivo"},
        "Area (ha)": {"areaha", "area", "areacosechadaha"},
        "Produccion (ton)": {"produccionton", "producciont", "produccion"},
        "Rendimiento (ton/ha)": {"rendimientotonha", "rendimientotha", "rendimientohaton", "rendimiento"},
    }
    if df_raw.empty:
        raise ValueError("El archivo agrícola no contiene observaciones.")
    renombrar = {}
    for destino, opciones in aliases.items():
        candidatos = [c for c in df_raw.columns if clave(c) in opciones]
        if len(candidatos) != 1:
            raise ValueError(f"Se requiere una columna inequívoca para {destino}; encontradas: {candidatos}.")
        renombrar[candidatos[0]] = destino
    df = df_raw.rename(columns=renombrar).copy()
    if df[list(aliases)].isna().any().any():
        raise ValueError("Las seis variables agrícolas requeridas no pueden contener celdas vacías.")
    for col in ["Departamento", "Producto"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
        if df[col].eq("").any():
            raise ValueError(f"Identificadores vacíos en {col}.")
    for col in ["Anio", "Area (ha)", "Produccion (ton)", "Rendimiento (ton/ha)"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"Valores negativos o no finitos en {col}.")
    if not df["Anio"].mod(1).eq(0).all():
        raise ValueError("El año debe ser un entero; no se truncan años fraccionarios.")
    df["Anio"] = df["Anio"].astype(int)
    if df["Producto"].nunique() != 1 or df.duplicated(["Anio", "Departamento"]).any():
        raise ValueError("Se requiere un único producto y claves año/departamento sin duplicados.")
    positivos = df["Area (ha)"] > 0
    if ((~positivos) & df["Produccion (ton)"].gt(0)).any():
        raise ValueError("Hay producción positiva con área cero; revise el archivo original.")
    cociente = df.loc[positivos, "Produccion (ton)"] / df.loc[positivos, "Area (ha)"]
    if not np.allclose(df.loc[positivos, "Rendimiento (ton/ha)"], cociente, atol=0.0051, rtol=0):
        raise ValueError("El rendimiento no coincide con producción/área a la precisión de 0.01 t/ha; no se asumen unidades.")
    df.loc[~positivos, "Rendimiento (ton/ha)"] = np.nan
    df.attrs["auditoria_ingesta"] = {
        "columnas_originales": list(map(str, df_raw.columns)),
        "mapeo_columnas": {str(k): v for k, v in renombrar.items()},
        "unidad_rendimiento_verificada": "t/ha, verificada mediante producción/área; sin transformar valores",
        "rendimientos_indefinidos_area_cero": int((~positivos).sum()),
        "tolerancia_absoluta_t_ha": 0.0051,
    }
    return df


def descargar_oni_noaa(start_year: int, end_year: int, params: dict) -> pd.DataFrame:
    url = params["NOAA_ONI_URL"]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=params["NASA_TIMEOUT_SEGUNDOS"]) as resp:
        contenido_oni = resp.read().decode("utf-8")
        lines = contenido_oni.splitlines()
    if params.get("DIRECTORIO_RESPUESTAS_CLIMA"):
        with open(os.path.join(params["DIRECTORIO_RESPUESTAS_CLIMA"], "noaa_oni_original.txt"), "w", encoding="utf-8") as archivo:
            archivo.write(contenido_oni)

    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 4:
            seas, yr, anom = parts[0], int(parts[1]), float(parts[3])
            if start_year - 1 <= yr <= end_year + 1:
                records.append({"Anio": yr, "Trimestre": seas, "ONI_Anom": anom})

    df_oni_raw = pd.DataFrame(records)
    if df_oni_raw.empty:
        raise ValueError("NOAA no devolvió temporadas ONI para el periodo solicitado.")
    orden = {s: i for i, s in enumerate(
        ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"])}
    df_oni_raw["Mes"] = df_oni_raw["Trimestre"].map(orden)
    df_oni_raw = df_oni_raw.sort_values(["Anio", "Mes"]).reset_index(drop=True)
    u_nino = params["UMBRAL_ONI_EL_NINO"]
    u_nina = params["UMBRAL_ONI_LA_NINA"]
    min_trim = params["MIN_TRIMESTRES_FASE_ENSO"]
    # Identificar episodios consecutivos antes de resumir por año, incluidos cruces de diciembre.
    valores = df_oni_raw["ONI_Anom"].to_numpy()
    fases = np.where(valores >= u_nino, 1, np.where(valores <= u_nina, -1, 0))
    tiempos = (df_oni_raw["Anio"] * 12 + df_oni_raw["Mes"]).to_numpy()
    episodios = np.zeros(len(fases), dtype=int)
    inicio = 0
    for fin in range(1, len(fases) + 1):
        if fin == len(fases) or fases[fin] != fases[inicio] or tiempos[fin] != tiempos[fin - 1] + 1:
            if fases[inicio] != 0 and fin - inicio >= min_trim:
                episodios[inicio:fin] = fases[inicio]
            inicio = fin
    df_oni_raw["Episodio"] = episodios

    resumen_oni = []
    for yr, grp in df_oni_raw.groupby("Anio"):
        if not start_year <= yr <= end_year:
            continue
        if len(grp) != 12 or grp["Mes"].nunique() != 12:
            raise ValueError(f"ONI incompleto para {yr}; se requieren 12 temporadas.")
        anoms = grp["ONI_Anom"].values
        media_oni = float(np.mean(anoms))
        max_oni = float(np.max(anoms))
        min_oni = float(np.min(anoms))
        abs_max_oni = float(anoms[np.argmax(np.abs(anoms))])
        meses_nino = int(np.sum(anoms >= u_nino))
        meses_nina = int(np.sum(anoms <= u_nina))
        n_ep, n_en = (grp["Episodio"] == 1).sum(), (grp["Episodio"] == -1).sum()
        # Resumen anual propio: fase con más temporadas pertenecientes a episodios reconocidos.
        fase = "El Niño" if n_ep > n_en else ("La Niña" if n_en > n_ep else "Neutro / Transición")
        resumen_oni.append({
            "Anio": yr,
            "ONI_Media_Anual (C)": round(media_oni, 3),
            "ONI_Max_Anual (C)": round(max_oni, 2),
            "ONI_Min_Anual (C)": round(min_oni, 2),
            "ONI_Extremo_Anual (C)": round(abs_max_oni, 2),
            "Trimestres_El_Nino (ONI>=0.5)": meses_nino,
            "Trimestres_La_Nina (ONI<=-0.5)": meses_nina,
            "Fase_ENSO": fase,
            "Metodo_Fase_ENSO": "Mayoría anual de temporadas en episodios consecutivos; empate/transición=neutro",
        })
    return pd.DataFrame(resumen_oni)


def consultar_nasa_power_depto(depto_orig: str, start_year: int, end_year: int, params: dict) -> list:
    coords_dict = params["COORDENADAS_DEPARTAMENTOS"]
    clave_norm = normalizar_texto_depto(depto_orig, coords_dict)
    if clave_norm not in coords_dict:
        raise ValueError(f"No hay coordenadas documentadas para {depto_orig}.")
    info = coords_dict[clave_norm]
    lat, lon = info["lat"], info["lon"]
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters={params['NASA_PARAMETROS']}&community={params['NASA_POWER_COMUNIDAD']}"
        f"&longitude={lon}&latitude={lat}&start={start_year}&end={end_year}&format=JSON"
    )
    for intento in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=params["NASA_TIMEOUT_SEGUNDOS"]) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            if intento == 2:
                raise RuntimeError(f"Error descargando {depto_orig}: {e}")
            time.sleep(1.5 * (intento + 1))

    param_data = data["properties"]["parameter"]
    unidades_esperadas = {"PRECTOTCORR": "mm/day", "ALLSKY_SFC_SW_DWN": "MJ/m^2/day"}
    for variable, unidad in unidades_esperadas.items():
        if data.get("parameters", {}).get(variable, {}).get("units") != unidad:
            raise ValueError(f"Unidad NASA inesperada para {variable}: no se asume una conversión.")
    if params.get("DIRECTORIO_RESPUESTAS_CLIMA"):
        nombre = "".join(c if c.isalnum() else "_" for c in depto_orig)
        with open(os.path.join(params["DIRECTORIO_RESPUESTAS_CLIMA"], f"nasa_{nombre}.json"), "w", encoding="utf-8") as archivo:
            json.dump({"url": url, "respuesta": data}, archivo, ensure_ascii=False)
    u_def = params["UMBRAL_DEFICIT_HIDRICO_MENSUAL_MM"]
    rows = []
    for yr in range(start_year, end_year + 1):
        precip_mm, t2m, t2m_max, t2m_min, rad_sw, rh2m = [], [], [], [], [], []
        for m in range(1, 13):
            key = f"{yr}{m:02d}"
            days_in_m = calendar.monthrange(yr, m)[1]
            p_val = param_data["PRECTOTCORR"].get(key, -999)
            t_val = param_data["T2M"].get(key, -999)
            tmax_val = param_data["T2M_MAX"].get(key, -999)
            tmin_val = param_data["T2M_MIN"].get(key, -999)
            rad_val = param_data["ALLSKY_SFC_SW_DWN"].get(key, -999)
            rh_val = param_data["RH2M"].get(key, -999)

            if p_val != -999:
                precip_mm.append(p_val * days_in_m)
            if t_val != -999:
                t2m.append(t_val)
            if tmax_val != -999:
                t2m_max.append(tmax_val)
            if tmin_val != -999:
                t2m_min.append(tmin_val)
            if rad_val != -999:
                rad_sw.append(rad_val)
            if rh_val != -999:
                rh2m.append(rh_val)

        if any(len(v) != 12 or not np.isfinite(v).all()
               for v in (precip_mm, t2m, t2m_max, t2m_min, rad_sw, rh2m)):
            raise ValueError(f"NASA POWER incompleto: {depto_orig}, {yr}; se requieren 12 meses por variable.")
        p_anual = float(np.sum(precip_mm))
        p_cv = float(np.std(precip_mm, ddof=1) / np.mean(precip_mm)) if np.mean(precip_mm) > 0 else 0.0
        rows.append({
            "Anio": yr,
            "Departamento": depto_orig,
            "Municipio_Ref_Climatica": info["municipio"],
            "Latitud_Ref": lat,
            "Longitud_Ref": lon,
            "Precipitacion_Acumulada_Anual (mm)": round(p_anual, 1),
            "Precipitacion_CV_Intraanual": round(p_cv, 3),
            "Meses_Deficit_Hidrico (<100mm)": int(np.sum(np.array(precip_mm) < u_def)),
            "Temp_Media_Anual (C)": round(float(np.mean(t2m)), 2),
            "Temp_Maxima_Media_Anual (C)": round(float(np.mean(t2m_max)), 2),
            "Mayor_Temp_Maxima_Mensual (C)": round(float(np.max(t2m_max)), 2),
            "Temp_Minima_Media_Anual (C)": round(float(np.mean(t2m_min)), 2),
            "Radiacion_Solar_Media (MJ/m2/dia)": round(float(np.mean(rad_sw)), 2),
            "Humedad_Relativa_Media (%)": round(float(np.mean(rh2m)), 2),
        })
    return rows


def ejecutar_fase_0(params: dict, ruta_inicial: str) -> dict:
    """
    Ejecuta la Fase 0 completa:
      - Lee las seis variables requeridas por nombre, con columnas extra opcionales.
      - Descarga o recupera de caché las series NASA POWER y NOAA ONI.
      - Conserva objetivos indefinidos como faltantes, sin imputarlos.
      - Construye referencias históricas descriptivas y rezagos de años consecutivos.
    """
    dir_result = params["CARPETA_RESULTADOS"]
    dir_datasets = os.path.join(dir_result, "datasets")
    dir_tablas = os.path.join(dir_result, "tablas")
    dir_figuras = os.path.join(dir_result, "figuras")
    for d in (dir_result, dir_datasets, dir_tablas, dir_figuras):
        os.makedirs(d, exist_ok=True)
    params["DIRECTORIO_RESPUESTAS_CLIMA"] = os.path.join(dir_datasets, "fuentes_climaticas")
    os.makedirs(params["DIRECTORIO_RESPUESTAS_CLIMA"], exist_ok=True)

    if ruta_inicial.lower().endswith(".csv"):
        df_raw = pd.read_csv(ruta_inicial)
    else:
        df_raw = leer_excel_sin_bloqueo(ruta_inicial)

    df_base = preparar_datos_agricolas(df_raw)
    Path(dir_datasets, "auditoria_ingesta.json").write_text(
        json.dumps(df_base.attrs["auditoria_ingesta"], indent=2, ensure_ascii=False), encoding="utf-8")
    df_base = filtrar_periodo_analisis(df_base, params)
    n_filas_antes_exclusiones = len(df_base)
    df_base, df_exclusiones = aplicar_exclusiones_departamentos(df_base, params)
    df_exclusiones.to_csv(os.path.join(dir_tablas, "T00_Exclusiones_Documentadas.csv"), index=False, encoding="utf-8-sig")

    productos_unicos = df_base["Producto"].dropna().unique()
    if len(productos_unicos) != 1:
        raise ValueError("El archivo debe contener exactamente un producto.")
    if df_base.duplicated(["Anio", "Departamento"]).any():
        raise ValueError("Existen duplicados año/departamento; no se agregan automáticamente.")
    for columna in ["Area (ha)", "Produccion (ton)", "Rendimiento (ton/ha)"]:
        df_base[columna] = pd.to_numeric(df_base[columna], errors="raise")
        if (df_base[columna] < 0).any():
            raise ValueError(f"Valores negativos en {columna}.")
    # Reconocer porcentajes sólo cuando coinciden con participaciones calculadas del archivo.
    for magnitud, original, corregido in [
        ("Produccion (ton)", "Produccion Nacional (ton)", "Participacion_Produccion_Nacional (%)"),
        ("Area (ha)", "Area Nacional (ha)", "Participacion_Area_Nacional (%)"),
    ]:
        if original not in df_base.columns:
            continue
        total = df_base.groupby("Anio")[magnitud].transform("sum").replace(0, np.nan)
        participacion = 100 * df_base[magnitud] / total
        if np.allclose(df_base[original], participacion, atol=0.0051, rtol=0):
            df_base = df_base.rename(columns={original: corregido})
    nombre_cultivo = productos_unicos[0] if len(productos_unicos) > 0 else os.path.splitext(os.path.basename(ruta_inicial))[0].upper()
    slug_cultivo = "".join(c if c.isalnum() else "_" for c in nombre_cultivo).strip("_")

    anio_min, anio_max = int(df_base["Anio"].min()), int(df_base["Anio"].max())
    anios_unicos = sorted(df_base["Anio"].unique())
    n_anios_total = len(anios_unicos)
    if n_anios_total < 4 or not 0 < params["PROPORCION_TRAIN_HISTORICO"] < 1:
        raise ValueError("Se requieren al menos cuatro años y una proporción train entre 0 y 1.")
    idx_corte = max(1, int(np.floor(n_anios_total * params["PROPORCION_TRAIN_HISTORICO"]))) - 1
    anio_corte_train = int(anios_unicos[idx_corte])
    anio_medio_markov = int(anios_unicos[n_anios_total // 2 - 1])

    print("\n" + "=" * 95)
    print(f"[FASE 0] INGESTA DINÁMICA Y CLIMA PARA: {nombre_cultivo} ({anio_min}–{anio_max})")
    print(f"  * Archivo inicial (intacto / solo lectura): {ruta_inicial} ({len(df_base)} filas)")
    if not df_exclusiones.empty:
        print(f"  * Exclusiones explícitas: {int(df_exclusiones['N_Filas_Excluidas'].sum())} filas; ver T00.")
    print("=" * 95)

    departamentos_lista = sorted(df_base["Departamento"].unique().tolist())
    ruta_cache_clima = os.path.join(dir_datasets, "cache_nasa_power_departamentos.csv")
    ruta_cache_oni = os.path.join(dir_datasets, "cache_noaa_oni_anual.csv")
    ruta_meta_cache = os.path.join(dir_datasets, "cache_nasa_metadata.json")
    firma_solicitud = hashlib.sha256(json.dumps({k: params[k] for k in [
        "NASA_POWER_COMUNIDAD", "NASA_PARAMETROS", "COORDENADAS_DEPARTAMENTOS",
        "UMBRAL_DEFICIT_HIDRICO_MENSUAL_MM"]}, sort_keys=True).encode()).hexdigest()

    usar_cache = False
    if params["USAR_CACHE_CLIMA"] and all(os.path.exists(p) for p in [ruta_cache_clima, ruta_cache_oni, ruta_meta_cache]):
        with open(ruta_meta_cache, encoding="utf-8") as archivo:
            metadata_cache = json.load(archivo)
        df_c_cache = pd.read_csv(ruta_cache_clima)
        df_o_cache = pd.read_csv(ruta_cache_oni)
        claves_esperadas = pd.MultiIndex.from_product([departamentos_lista, range(anio_min, anio_max + 1)])
        claves_cache = pd.MultiIndex.from_frame(df_c_cache[["Departamento", "Anio"]])
        coords_ok = all(
            normalizar_texto_depto(d, params["COORDENADAS_DEPARTAMENTOS"]) in params["COORDENADAS_DEPARTAMENTOS"]
            and np.allclose(df_c_cache.loc[df_c_cache["Departamento"] == d, ["Latitud_Ref", "Longitud_Ref"]],
                            [params["COORDENADAS_DEPARTAMENTOS"][normalizar_texto_depto(d, params["COORDENADAS_DEPARTAMENTOS"])]["lat"],
                             params["COORDENADAS_DEPARTAMENTOS"][normalizar_texto_depto(d, params["COORDENADAS_DEPARTAMENTOS"])]["lon"]])
            for d in departamentos_lista)
        if (claves_esperadas.isin(claves_cache).all() and not claves_cache.duplicated().any()
                and coords_ok and not df_c_cache.isna().any().any()
                and params["UMBRAL_DEFICIT_HIDRICO_MENSUAL_MM"] == 100.0
                and params["NASA_POWER_COMUNIDAD"] == "AG"
                and params["NASA_PARAMETROS"] == "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN,RH2M"
                and metadata_cache.get("firma_solicitud") == firma_solicitud
                and metadata_cache.get("sha256_csv") == hashlib.sha256(Path(ruta_cache_clima).read_bytes()).hexdigest()):
            usar_cache = True
            df_clima = df_c_cache[df_c_cache["Departamento"].isin(departamentos_lista) & df_c_cache["Anio"].between(anio_min, anio_max)].copy()
            # La caché histórica ONI usaba conteos sin continuidad; recalcular desde la fuente.
            df_oni = descargar_oni_noaa(anio_min, anio_max, params)
            df_oni.to_csv(ruta_cache_oni, index=False, encoding="utf-8-sig")
            print(f"      -> NASA POWER desde caché local validada en cobertura/coordenadas; ONI recalculado desde NOAA.")

    if not usar_cache:
        print(f"      -> Descargando serie oficial ONI (NOAA CPC) para {anio_min}-{anio_max}...")
        df_oni = descargar_oni_noaa(anio_min, anio_max, params)
        print(f"      -> Descargando series agroclimáticas NASA POWER API para {len(departamentos_lista)} departamentos...")
        clima_rows = []
        with ThreadPoolExecutor(max_workers=params["NASA_MAX_WORKERS"]) as executor:
            futures = {executor.submit(consultar_nasa_power_depto, d, anio_min, anio_max, params): d for d in departamentos_lista}
            for idx, future in enumerate(as_completed(futures), 1):
                d_name = futures[future]
                clima_rows.extend(future.result())
                print(f"         [{idx:02d}/{len(departamentos_lista):02d}] {d_name} completado")
        df_clima = pd.DataFrame(clima_rows)
        df_clima.to_csv(ruta_cache_clima, index=False, encoding="utf-8-sig")
        df_oni.to_csv(ruta_cache_oni, index=False, encoding="utf-8-sig")
        with open(ruta_cache_clima, "rb") as archivo:
            hash_cache = hashlib.sha256(archivo.read()).hexdigest()
        with open(ruta_meta_cache, "w", encoding="utf-8") as archivo:
            json.dump({"firma_solicitud": firma_solicitud, "sha256_csv": hash_cache,
                       "origen": "NASA POWER API; respuestas originales en fuentes_climaticas",
                       "anios": [anio_min, anio_max]}, archivo, indent=2)

    df_clima = df_clima.rename(columns={"Temp_Maxima_Extrema_Anual (C)": "Mayor_Temp_Maxima_Mensual (C)"})
    df = df_base.merge(df_clima, on=["Anio", "Departamento"], how="left", validate="one_to_one").merge(df_oni, on="Anio", how="left", validate="many_to_one")
    if df[["Precipitacion_Acumulada_Anual (mm)", "ONI_Media_Anual (C)"]].isna().any().any():
        raise ValueError("El cruce climático dejó observaciones sin datos.")
    df = df.sort_values(["Departamento", "Anio"]).reset_index(drop=True)

    ruta_enriquecido_xlsx = os.path.join(dir_datasets, f"{slug_cultivo}_CLIMA_ENSO.xlsx")
    ruta_enriquecido_csv = os.path.join(dir_datasets, f"{slug_cultivo}_CLIMA_ENSO.csv")
    df.to_excel(ruta_enriquecido_xlsx, index=False)
    df.to_csv(ruta_enriquecido_csv, index=False, encoding="utf-8-sig")

    # No imputar el objetivo. Un cociente con área cero es indefinido, no un rendimiento observado.
    df["Rendimiento_Valido"] = (df["Area (ha)"] > 0) & df["Rendimiento (ton/ha)"].notna()
    df.loc[~df["Rendimiento_Valido"], "Rendimiento (ton/ha)"] = np.nan

    # Referencia histórica descriptiva; la evaluación supervisada la reajusta en cada fold.
    df_train_base = df  # Referencia descriptiva de toda la historia; ML la reajusta por fold.
    stats_base = df_train_base.groupby("Departamento").agg(
        p_mean=("Precipitacion_Acumulada_Anual (mm)", "mean"),
        p_std=("Precipitacion_Acumulada_Anual (mm)", "std"),
        t_mean=("Temp_Maxima_Media_Anual (C)", "mean"),
        t_std=("Temp_Maxima_Media_Anual (C)", "std"),
        r_mean=("Rendimiento (ton/ha)", "mean"),
    )
    df = df.merge(stats_base, on="Departamento", how="left")
    df["Anomalia_Precip_Depto (Z)"] = ((df["Precipitacion_Acumulada_Anual (mm)"] - df["p_mean"]) / df["p_std"].replace(0, 1)).round(3)
    df["Anomalia_Temp_Max_Depto (Z)"] = ((df["Temp_Maxima_Media_Anual (C)"] - df["t_mean"]) / df["t_std"].replace(0, 1)).round(3)
    df["Brecha_Rendimiento_Depto (ton/ha)"] = (df["Rendimiento (ton/ha)"] - df["r_mean"]).round(3)
    df = df.drop(columns=["p_mean", "p_std", "t_mean", "t_std", "r_mean"])

    df = agregar_rezagos_anuales(df)

    df["Split_Temporal"] = np.where(df["Anio"] <= anio_corte_train, f"TRAIN_{anio_min}_{anio_corte_train}", f"TEST_{anio_corte_train+1}_{anio_max}")

    return {
        "df": df,
        "df_exclusiones": df_exclusiones,
        "n_filas_antes_exclusiones": n_filas_antes_exclusiones,
        "nombre_cultivo": nombre_cultivo,
        "slug_cultivo": slug_cultivo,
        "ruta_inicial": ruta_inicial,
        "anio_min": anio_min,
        "anio_max": anio_max,
        "anio_corte_train": anio_corte_train,
        "anio_medio_markov": anio_medio_markov,
        "dir_result": dir_result,
        "dir_datasets": dir_datasets,
        "dir_tablas": dir_tablas,
        "dir_figuras": dir_figuras,
    }


# ====================================================================================================
# UTILIDADES DE ESTILO DE PUBLICACIÓN CIENTÍFICA (Q1 JOURNAL STANDARD - ACADEMIC ENGLISH, NO TITLES)
# ====================================================================================================
TRADUCCION_VARS_EN = {
    "Area_Lag1 (ha)": "Lagged Harvested Area (ha)",
    "Radiacion_Solar_Media (MJ/m2/dia)": "Mean Solar Radiation (MJ m$^{-2}$ d$^{-1}$)",
    "Temp_Maxima_Media_Anual (C)": "Mean Maximum Temperature (°C)",
    "Humedad_Relativa_Media (%)": "Mean Relative Humidity (%)",
    "Precipitacion_Acumulada_Anual (mm)": "Annual Cumulative Precipitation (mm)",
    "Precipitacion_CV_Intraanual": "Intra-Annual Precipitation CV",
    "Precip_Lag1 (mm)": "Lagged Annual Precipitation (mm)",
    "Anomalia_Temp_Max_Depto (Z)": "Max. Temperature Anomaly ($Z$-score)",
    "Anomalia_Precip_Depto (Z)": "Precipitation Anomaly ($Z$-score)",
    "ONI_Lag1 (C)": "Lagged Oceanic Niño Index ONI$_{t-1}$ (°C)",
    "ONI_Media_Anual (C)": "Annual Mean Oceanic Niño Index ONI$_t$ (°C)",
    "Rendimiento (ton/ha)": "Crop Yield (t ha$^{-1}$)",
    "Rendimiento_Lag1 (ton/ha)": "Lagged Crop Yield (t ha$^{-1}$)",
}

TRADUCCION_ENSO_EN = {
    "El Niño": "El Niño",
    "La Niña": "La Niña",
    "Neutro / Transición": "Neutral",
}


def configurar_estilo_publicacion(plt_module) -> None:
    plt_module.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def guardar_figura_segura(fig_obj, dir_figuras: str, name: str, dpi: int, plt_module) -> None:
    """Guarda PNG y PDF en memoria primero para evitar bloqueos transitorios de sincronización de OneDrive."""
    for ext, fmt_kw in [("png", {"dpi": dpi}), ("pdf", {})]:
        buf = io.BytesIO()
        fig_obj.savefig(buf, format=ext, bbox_inches="tight", pad_inches=0.08, **fmt_kw)
        data_bytes = buf.getvalue()
        ruta_dest = os.path.join(dir_figuras, f"{name}.{ext}")
        for intento in range(4):
            try:
                with open(ruta_dest, "wb") as f_out:
                    f_out.write(data_bytes)
                break
            except OSError:
                if intento == 3:
                    plt_module.close(fig_obj)
                    raise
                time.sleep(0.35 * (intento + 1))
    plt_module.close(fig_obj)


def indices_bootstrap_panel(df: pd.DataFrame, rng) -> np.ndarray:
    """Bootstrap cruzado por departamento y año (pigeonhole); no trata filas como IID."""
    if not {"Departamento", "Anio"}.issubset(df.columns):
        raise ValueError("El bootstrap requiere Departamento y Anio.")
    deps, dep_codes = np.unique(df["Departamento"].to_numpy(), return_inverse=True)
    years, year_codes = np.unique(df["Anio"].to_numpy(), return_inverse=True)
    for _ in range(100):
        dep_weights = np.bincount(rng.integers(len(deps), size=len(deps)), minlength=len(deps))
        year_weights = np.bincount(rng.integers(len(years), size=len(years)), minlength=len(years))
        indices = np.repeat(np.arange(len(df)), dep_weights[dep_codes] * year_weights[year_codes])
        if len(indices):
            return indices
    raise ValueError("No se pudo generar una réplica no vacía del panel.")


def formatear_depto_titulo(nombre: str) -> str:
    especiales = {
        "VALLE DEL CAUCA": "Valle del Cauca",
        "NORTE DE SANTANDER": "Norte de Santander",
        "SAN ANDRES Y PROVIDENCIA": "San Andrés & Prov.",
        "ARCHIPIELAGO DE SAN ANDRES": "San Andrés & Prov.",
        "LA GUAJIRA": "La Guajira",
        "NARINO": "Nariño",
        "NARIÑO": "Nariño",
        "BOGOTA": "Bogotá D.C.",
        "ATLANTICO": "Atlántico",
        "BOLIVAR": "Bolívar",
        "BOYACA": "Boyacá",
        "CAQUETA": "Caquetá",
        "CHOCO": "Chocó",
        "CORDOBA": "Córdoba",
        "GUAINIA": "Guainía",
        "QUINDIO": "Quindío",
        "VAUPES": "Vaupés",
    }
    u = str(nombre).strip().upper()
    return especiales.get(u, u.title())


def repel_labels_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    texts: list,
    xlim: tuple,
    ylim: tuple,
    char_w_frac: float = 0.011,
    box_h_frac: float = 0.042,
    node_r_frac: float = 0.032,
    n_iter: int = 320,
    seed: int = 42,
) -> tuple:
    """
    Algoritmo determinista de repulsión 2D para colocar etiquetas de texto en gráficos de dispersión
    y redes topológicas sin solapamiento entre etiquetas ni sobre los nodos.
    """
    rng = np.random.default_rng(seed)
    n = len(xs)
    span_x = float(xlim[1] - xlim[0])
    span_y = float(ylim[1] - ylim[0])

    # Coordenadas normalizadas en [0, 1]
    nx_pts = (np.asarray(xs, dtype=float) - xlim[0]) / max(span_x, 1e-9)
    ny_pts = (np.asarray(ys, dtype=float) - ylim[0]) / max(span_y, 1e-9)

    half_w = np.array([max(len(str(t)), 4) * char_w_frac * 0.52 for t in texts])
    half_h = np.full(n, box_h_frac * 0.52)

    # Inicializar etiquetas desplazadas hacia los costados libres respecto a la mediana en X
    med_x = np.median(nx_pts)
    order_y = np.argsort(ny_pts)
    lx = np.zeros(n)
    ly = np.zeros(n)
    for rank_k, i in enumerate(order_y):
        side = -1.0 if (nx_pts[i] < med_x or (abs(nx_pts[i] - med_x) < 0.03 and rank_k % 2 == 0)) else 1.0
        lx[i] = nx_pts[i] + side * (node_r_frac + half_w[i] + 0.025)
        ly[i] = ny_pts[i] + ((rank_k % 3) - 1) * 0.018

    for it in range(n_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # 1. Repulsión etiqueta-etiqueta (cajas rectangulares con margen de seguridad estricto)
        for i in range(n):
            for j in range(i + 1, n):
                req_x = half_w[i] + half_w[j] + 0.014
                req_y = half_h[i] + half_h[j] + 0.014
                dx = lx[i] - lx[j]
                dy = ly[i] - ly[j]
                ov_x = req_x - abs(dx)
                ov_y = req_y - abs(dy)
                if ov_x > 0 and ov_y > 0:
                    sx = 1.0 if (dx > 0 or (dx == 0 and i > j)) else -1.0
                    sy = 1.0 if (dy > 0 or (dy == 0 and i > j)) else -1.0
                    # Separar prioritariamente en el eje de menor solapamiento relativo
                    if (ov_x / req_x) < (ov_y / req_y):
                        fx[i] += sx * ov_x * 0.85
                        fx[j] -= sx * ov_x * 0.85
                        fy[i] += sy * ov_y * 0.25
                        fy[j] -= sy * ov_y * 0.25
                    else:
                        fy[i] += sy * ov_y * 0.85
                        fy[j] -= sy * ov_y * 0.85
                        fx[i] += sx * ov_x * 0.25
                        fx[j] -= sx * ov_x * 0.25

        # 2. Repulsión etiqueta-nodo (para todos los nodos)
        for i in range(n):
            for j in range(n):
                req_x = half_w[i] + node_r_frac + 0.008
                req_y = half_h[i] + node_r_frac + 0.008
                dx = lx[i] - nx_pts[j]
                dy = ly[i] - ny_pts[j]
                ov_x = req_x - abs(dx)
                ov_y = req_y - abs(dy)
                if ov_x > 0 and ov_y > 0:
                    sx = 1.0 if dx >= 0 else -1.0
                    sy = 1.0 if dy >= 0 else -1.0
                    fx[i] += sx * ov_x * 0.70
                    fy[i] += sy * ov_y * 0.70

        # 3. Atracción elástica suave hacia su propio nodo ancla
        for i in range(n):
            dx_a = nx_pts[i] - lx[i]
            dy_a = ny_pts[i] - ly[i]
            dist_a = np.hypot(dx_a, dy_a)
            if dist_a > (node_r_frac + 0.065):
                fx[i] += dx_a * 0.12
                fy[i] += dy_a * 0.12

        damp = 1.0 - 0.55 * (it / n_iter)
        lx += np.clip(fx, -0.04, 0.04) * damp
        ly += np.clip(fy, -0.04, 0.04) * damp

        # Mantener estrictamente dentro de los márgenes de la figura
        lx = np.clip(lx, half_w + 0.015, 1.0 - half_w - 0.015)
        ly = np.clip(ly, half_h + 0.020, 1.0 - half_h - 0.020)

    out_x = xlim[0] + lx * span_x
    out_y = ylim[0] + ly * span_y
    return out_x, out_y
