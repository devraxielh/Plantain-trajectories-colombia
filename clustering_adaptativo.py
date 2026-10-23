"""Selección auditable de k y algoritmo sobre una única geometría DTW bivariada.

Ejemplo independiente, sin descargas climáticas:
    python clustering_adaptativo.py PLATANO.xlsx --salida result_adaptativo

La ejecución completa también llama a este módulo desde ejecutar_fase_1.
Los umbrales son criterios operativos configurables; no pruebas de optimalidad.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from fase_0_ingesta_y_clima import (
    cargar_parametros_planos, filtrar_periodo_analisis,
    leer_bytes_sin_bloqueo, leer_excel_sin_bloqueo, aplicar_exclusiones_departamentos,
    preparar_datos_agricolas,
)
from fase_1_clustering_dtw_y_topologia import dtw_sobre_observaciones, kmedoids_pam


ARCHIVOS_TABLAS = {
    "seleccion": "T16_Seleccion_Adaptativa_Clusters.csv",
    "perturbaciones": "T17_Estabilidad_Perturbaciones.csv",
    "grupos": "T18_Estabilidad_por_Grupo.csv",
}


def matriz_bivariada_observada(rendimientos, areas, params):
    """Reajusta las escalas con el periodo recibido; conserva todos los departamentos."""
    if not rendimientos.index.equals(areas.index) or not rendimientos.columns.equals(areas.columns):
        raise ValueError("Rendimiento y área deben tener las mismas claves y años.")
    if len(rendimientos.columns) < params["MINIMO_ANIOS_COMPARTIDOS_DTW"]:
        raise ValueError("No hay suficientes años compartidos para esta sensibilidad.")
    r = rendimientos.to_numpy(dtype=float)
    a = np.log1p(areas.to_numpy(dtype=float))
    geometria = params.get("GEOMETRIA_ADAPTATIVA", "escala")
    if geometria == "escala":
        sd_r, sd_a = np.nanstd(r), np.nanstd(a)
        if not np.isfinite([sd_r, sd_a]).all() or min(sd_r, sd_a) <= 0:
            raise ValueError("No hay variación suficiente para definir DTW bivariado.")
        r, a = (r - np.nanmean(r)) / sd_r, (a - np.nanmean(a)) / sd_a
    elif geometria == "forma":
        # Centrar y escalar cada variable de cada departamento elimina niveles y amplitudes.
        # Una trayectoria constante queda en cero; no se fabrican tendencias.
        variables = []
        for values in (r, a):
            means = np.nanmean(values, axis=1, keepdims=True)
            stds = np.nanstd(values, axis=1, keepdims=True)
            variables.append((values - means) / np.where(stds > 0, stds, 1.0))
        r, a = variables
    else:
        raise ValueError("Geometría desconocida; se admite escala o forma.")
    n = len(r)
    D, N = np.zeros((n, n)), np.zeros((n, n), dtype=int)
    years = rendimientos.columns.to_numpy(dtype=int)
    for i in range(n):
        N[i, i] = int((np.isfinite(r[i]) & np.isfinite(a[i])).sum())
        for j in range(i + 1, n):
            distance, overlap = dtw_sobre_observaciones(
                np.column_stack([r[i], a[i]]), np.column_stack([r[j], a[j]]),
                years, params["VENTANA_SAKOE_CHIBA"], params["MINIMO_ANIOS_COMPARTIDOS_DTW"])
            D[i, j] = D[j, i] = distance
            N[i, j] = N[j, i] = overlap
    return D, N


def ajustar_particion(D, algoritmo, k, params):
    if not 2 <= k < len(D):
        raise ValueError("La partición requiere 2 <= k < número de departamentos.")
    if algoritmo == "pam":
        labels, _ = kmedoids_pam(D, k, max_iter=params["MAX_ITER_KMEDOIDS"],
            n_restarts=params["REINICIOS_KMEDOIDS_PAM"], seed=params["SEMILLA_ALEATORIA"])
    elif algoritmo in {"average", "complete"}:
        labels = cut_tree(linkage(squareform(D), method=algoritmo), n_clusters=[k]).ravel() + 1
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")
    if len(np.unique(labels)) != k:
        raise ValueError("El algoritmo no produjo el número solicitado de grupos.")
    return np.asarray(labels)


def jaccard_por_grupo(referencia, alternativa):
    """Mejor coincidencia por grupo, invariante a los nombres de las etiquetas."""
    reference, other = np.asarray(referencia), np.asarray(alternativa)
    resultado = {}
    for label in np.unique(reference):
        conjunto = reference == label
        resultado[int(label)] = max(
            float(np.sum(conjunto & (other == nuevo)) / np.sum(conjunto | (other == nuevo)))
            for nuevo in np.unique(other))
    return resultado


def decidir_seleccion(tabla, params):
    """Regla determinista: filtros de respaldo, banda de silhouette y menor k."""
    result = tabla.copy()
    valid = result["Evaluable"].astype(bool) & np.isfinite(result["Silhouette"])
    result["Cumple_Criterios"] = (
        valid & (result["Silhouette"] >= params["SILHOUETTE_MINIMA_ADAPTATIVA"])
        & (result["ARI_Temporal_P10"] >= params["ARI_P10_MINIMO"])
        & (result["Jaccard_Min_Grupo_P10"] >= params["JACCARD_P10_MINIMO"])
        & (result["ARI_Cobertura_Comun"] >= params["ARI_COBERTURA_MINIMO"])
        & (result["Fraccion_Replicas_Validas"] >= params["FRACCION_REPLICAS_VALIDAS_MINIMA"])
        & (result["N_Replicas_Validas"] >= 3))
    respaldada = bool(result["Cumple_Criterios"].any())
    pool = result.loc[result["Cumple_Criterios"] if respaldada else valid]
    if pool.empty:
        raise ValueError("Ninguna partición es evaluable; no se inventa una solución.")
    top = float(pool["Silhouette"].max())
    pool = pool[pool["Silhouette"] >= top - params["TOLERANCIA_SILHOUETTE_ADAPTATIVA"]]
    chosen = pool.sort_values(["k", "Silhouette", "ARI_Temporal_P10", "Algoritmo"],
        ascending=[True, False, False, True], na_position="last").index[0]
    fijo = params.get("K_CLUSTERS_FIJO", 0)
    if fijo:
        solicitado = valid & result["k"].eq(fijo) & result["Algoritmo"].eq(params["ALGORITMO_CLUSTER_FIJO"])
        if solicitado.sum() != 1:
            raise ValueError("La partición fija solicitada no es evaluable o no está en la grilla.")
        chosen = result.index[solicitado][0]
        respaldada = bool(result.loc[chosen, "Cumple_Criterios"])
    result["Seleccionada"] = False
    result.loc[chosen, "Seleccionada"] = True
    row = result.loc[chosen]
    decision = {
        "modo": "fijo_solicitado" if fijo else "adaptativo",
        "k": int(row["k"]), "algoritmo": str(row["Algoritmo"]),
        "cumple_criterios": respaldada,
        "estado": "Respaldo_segun_criterios_configurados" if respaldada else "Exploratoria_sin_respaldo_suficiente",
        "regla": "Filtrar separación/estabilidad/cobertura; dentro de la tolerancia de mejor silhouette elegir menor k, luego mayor silhouette, ARI y nombre de algoritmo.",
        "motivo": "Se satisfacen los criterios operativos configurados." if respaldada else
            "Ningún candidato satisface todos los criterios. Se conserva una partición exploratoria y se informa la falta de respaldo.",
        "no_afirma": "No demuestra optimalidad universal, causalidad ni existencia de un número verdadero de grupos.",
    }
    if fijo:
        decision["estado"] = "Tipologia_fija_" + ("cumple_criterios" if respaldada else "exploratoria_sin_respaldo_suficiente")
        decision["regla"] = "k y algoritmo fijados explícitamente en configuración; grilla conservada como diagnóstico."
        decision["motivo"] = (f"Tipología solicitada de {fijo} grupos mediante {params['ALGORITMO_CLUSTER_FIJO']}; "
            + ("cumple los criterios configurados, sin demostrar optimalidad." if respaldada else
               "no cumple todos los criterios configurados y se presenta como exploratoria, aunque otros candidatos sí puedan cumplirlos."))
    return result, decision


def seleccionar_clustering_adaptativo(rendimientos, areas, params, matriz_base=None, informar=True):
    years = rendimientos.columns.to_numpy(dtype=int)
    if len(years) < params["MINIMO_ANIOS_COMPARTIDOS_DTW"]:
        raise ValueError("Periodo demasiado corto para DTW.")
    D = matriz_base if matriz_base is not None else matriz_bivariada_observada(rendimientos, areas, params)[0]
    candidates = [(alg, k) for alg in dict.fromkeys(params["ALGORITMOS_ADAPTATIVOS"])
                  for k in sorted(set(params["GRILLA_CLUSTERS_K"]))]
    reference, summaries = {}, {}
    for algorithm, k in candidates:
        key = (algorithm, k)
        try:
            reference[key] = ajustar_particion(D, algorithm, k, params)
            _, counts = np.unique(reference[key], return_counts=True)
            summaries[key] = {"Algoritmo": algorithm, "k": k, "Evaluable": True,
                "Silhouette": float(silhouette_score(D, reference[key], metric="precomputed")),
                "Min_Departamentos_Grupo": int(counts.min()), "N_Grupos_Unitarios": int((counts == 1).sum()),
                "Error_Base": ""}
        except ValueError as exc:
            summaries[key] = {"Algoritmo": algorithm, "k": k, "Evaluable": False,
                "Silhouette": np.nan, "Min_Departamentos_Grupo": np.nan, "N_Grupos_Unitarios": np.nan,
                "Error_Base": str(exc)}

    # Retirar ventanas contiguas del calendario; nunca permutar ni inventar años.
    windows = []
    for length in sorted(set(params["BLOQUES_ESTABILIDAD_ANIOS"])):
        for start in range(len(years) - length + 1):
            removed = years[start:start + length]
            if length < len(years):
                windows.append(removed)
    if not windows:
        raise ValueError("No existen perturbaciones temporales posibles con estos bloques.")
    replicates, group_samples = [], {key: {int(c): [] for c in np.unique(labels)} for key, labels in reference.items()}
    for number, removed in enumerate(windows, 1):
        keep = ~np.isin(years, removed)
        try:
            D_test, _ = matriz_bivariada_observada(rendimientos.loc[:, keep], areas.loc[:, keep], params)
            error_distance = ""
        except ValueError as exc:
            D_test, error_distance = None, str(exc)
        for key, labels in reference.items():
            algorithm, k = key
            record = {"Replica": number, "Anios_Retirados": ",".join(map(str, removed)),
                "Algoritmo": algorithm, "k": k, "Valida": False, "ARI": np.nan, "Jaccard_Min_Grupo": np.nan}
            try:
                if D_test is None:
                    raise ValueError(error_distance)
                alternative = ajustar_particion(D_test, algorithm, k, params)
                jaccard = jaccard_por_grupo(labels, alternative)
                record.update(Valida=True, ARI=float(adjusted_rand_score(labels, alternative)),
                              Jaccard_Min_Grupo=min(jaccard.values()))
                for label, score in jaccard.items():
                    group_samples[key][label].append(score)
            except ValueError as exc:
                record["Motivo_No_Evaluable"] = str(exc)
            replicates.append(record)
        if informar and (number == 1 or number % 5 == 0 or number == len(windows)):
            print(f"      -> Estabilidad temporal: {number}/{len(windows)} ventanas evaluadas", flush=True)
    perturbations = pd.DataFrame(replicates)

    common = (rendimientos.notna() & areas.notna()).all(axis=0)
    common_years = years[common.to_numpy()]
    try:
        D_common, _ = matriz_bivariada_observada(rendimientos.loc[:, common], areas.loc[:, common], params)
        error_common = ""
    except ValueError as exc:
        D_common, error_common = None, str(exc)

    groups = []
    for key, row in summaries.items():
        algorithm, k = key
        valid_rows = perturbations.loc[(perturbations["Algoritmo"] == algorithm)
                                      & (perturbations["k"] == k) & perturbations["Valida"]] if not perturbations.empty else pd.DataFrame()
        n_valid = len(valid_rows)
        row.update(N_Replicas_Previstas=len(windows), N_Replicas_Validas=n_valid,
            Fraccion_Replicas_Validas=n_valid / len(windows),
            ARI_Temporal_P10=float(valid_rows["ARI"].quantile(.10)) if n_valid else np.nan,
            ARI_Temporal_Media=float(valid_rows["ARI"].mean()) if n_valid else np.nan,
            Jaccard_Min_Grupo_P10=np.nan, ARI_Cobertura_Comun=np.nan,
            Anios_Cobertura_Comun=",".join(map(str, common_years)), N_Anios_Cobertura_Comun=len(common_years),
            Error_Cobertura=error_common)
        if key not in reference:
            continue
        labels = reference[key]
        quantiles = []
        for label, values in group_samples[key].items():
            q10 = float(np.quantile(values, .10)) if values else np.nan
            quantiles.append(q10)
            groups.append({"Algoritmo": algorithm, "k": k, "Grupo_Referencia": label,
                "N_Departamentos": int((labels == label).sum()),
                "Departamentos": " | ".join(rendimientos.index[labels == label].astype(str)),
                "Jaccard_P10": q10, "Jaccard_Media": float(np.mean(values)) if values else np.nan,
                "N_Replicas_Validas": len(values), "Caso_Unitario": int((labels == label).sum()) == 1})
        row["Jaccard_Min_Grupo_P10"] = min(quantiles) if quantiles else np.nan
        if D_common is not None:
            try:
                labels_common = ajustar_particion(D_common, algorithm, k, params)
                row["ARI_Cobertura_Comun"] = float(adjusted_rand_score(labels, labels_common))
            except ValueError as exc:
                row["Error_Cobertura"] = str(exc)
    table, decision = decidir_seleccion(pd.DataFrame(summaries.values()), params)
    key = (decision["algoritmo"], decision["k"])
    decision["reglas_configuradas"] = {name: params[name] for name in [
        "SILHOUETTE_MINIMA_ADAPTATIVA", "ARI_P10_MINIMO", "JACCARD_P10_MINIMO", "ARI_COBERTURA_MINIMO",
        "FRACCION_REPLICAS_VALIDAS_MINIMA", "TOLERANCIA_SILHOUETTE_ADAPTATIVA", "BLOQUES_ESTABILIDAD_ANIOS"]}
    decision["geometria"] = params.get("GEOMETRIA_ADAPTATIVA", "escala")
    decision["normalizacion"] = ("Z global por variable" if decision["geometria"] == "escala" else
        "Z por variable y departamento; elimina niveles y amplitudes, conserva patrón temporal")
    decision["anios_comunes_sensibilidad"] = common_years.tolist()
    decision["percentiles_sensibilidad"] = "P10 resume perturbaciones deterministas; no es un intervalo de confianza."
    return {"labels": reference[key], "seleccion": table, "decision": decision,
            "perturbaciones": perturbations, "grupos": pd.DataFrame(groups)}


def comparar_escala_y_forma(rendimientos, areas, params, matrices_base=None, informar=True):
    """Dos preguntas científicas distintas; no maximiza silhouette entre geometrías."""
    results, matrices, comparisons, assignments = {}, {}, [], pd.DataFrame({"Departamento": rendimientos.index})
    for geom, prefix in [("escala", "S"), ("forma", "E")]:
        geom_params = dict(params, GEOMETRIA_ADAPTATIVA=geom)
        matrices[geom] = matrices_base[geom] if matrices_base is not None else matriz_bivariada_observada(rendimientos, areas, geom_params)[0]
        if informar:
            print(f"      -> Análisis separado de {geom}", flush=True)
        result = seleccionar_clustering_adaptativo(rendimientos, areas, geom_params, matriz_base=matrices[geom], informar=informar)
        results[geom] = result
        chosen = result["seleccion"].loc[result["seleccion"]["Seleccionada"]].iloc[0]
        comparisons.append({"Geometria": geom, "Principal": geom == params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"],
            "Algoritmo": result["decision"]["algoritmo"], "k": result["decision"]["k"],
            "Estado": result["decision"]["estado"], "Silhouette_Dentro_Geometria": chosen["Silhouette"],
            "ARI_Temporal_P10": chosen["ARI_Temporal_P10"], "Jaccard_Min_Grupo_P10": chosen["Jaccard_Min_Grupo_P10"],
            "ARI_Cobertura_Comun": chosen["ARI_Cobertura_Comun"], "N_Grupos_Unitarios": chosen["N_Grupos_Unitarios"],
            "Alcance": "Escala y forma responden preguntas distintas; sus silhouettes no se usan para elegir una geometría"})
        order = pd.DataFrame({"label": result["labels"], "mean": rendimientos.mean(axis=1).to_numpy()}).groupby("label")["mean"].mean().sort_values().index
        mapping = {old: f"{prefix}{new}" for new, old in enumerate(order, 1)}
        assignments[f"Perfil_{geom.title()}"] = [mapping[label] for label in result["labels"]]
    ari = float(adjusted_rand_score(results["escala"]["labels"], results["forma"]["labels"]))
    table = pd.DataFrame(comparisons)
    table["ARI_Entre_Escala_y_Forma"] = ari
    assignments["Rendimiento_Medio_Observado (ton/ha)"] = rendimientos.mean(axis=1).to_numpy()
    assignments["Area_Media_Observada (ha)"] = areas.mean(axis=1).to_numpy()
    return {"resultados": results, "matrices": matrices, "comparacion": table,
            "asignaciones": assignments, "principal": results[params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"]]}


def guardar_comparacion(comparacion, directorio):
    folder = Path(directorio)
    folder.mkdir(parents=True, exist_ok=True)
    comparacion["comparacion"].to_csv(folder / "T19_Comparacion_Escala_y_Forma.csv", index=False, encoding="utf-8-sig")
    comparacion["asignaciones"].to_csv(folder / "T20_Asignaciones_Escala_y_Forma.csv", index=False, encoding="utf-8-sig")
    for geom, result in comparacion["resultados"].items():
        guardar_seleccion(result, folder / ("comparacion_" + geom))


def guardar_seleccion(resultado, directorio):
    folder = Path(directorio)
    folder.mkdir(parents=True, exist_ok=True)
    for key, filename in ARCHIVOS_TABLAS.items():
        resultado[key].to_csv(folder / filename, index=False, encoding="utf-8-sig")
    (folder / "seleccion_adaptativa.json").write_text(
        json.dumps(resultado["decision"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivo", help="Excel o CSV: año, departamento, producto, área, producción y rendimiento; columnas adicionales opcionales.")
    parser.add_argument("--parametros", default="parametros_experimento.txt")
    parser.add_argument("--salida", default=None,
                        help="Por defecto: CARPETA_RESULTADOS/seleccion_independiente.")
    args = parser.parse_args()
    params = cargar_parametros_planos(args.parametros)
    if args.salida is None:
        args.salida = str(Path(params["CARPETA_RESULTADOS"]) / "seleccion_independiente")
    from organizacion_resultados import registrar_ejecucion
    with registrar_ejecucion(args.salida, "seleccion_adaptativa"):
        ejecutar_seleccion(args, params)


def ejecutar_seleccion(args, params):
    start = time.perf_counter()
    sources = [Path(args.archivo), Path(args.parametros), Path(__file__),
               Path(__file__).with_name("fase_0_ingesta_y_clima.py"), Path(__file__).with_name("fase_1_clustering_dtw_y_topologia.py")]
    before = {str(p): hashlib.sha256(leer_bytes_sin_bloqueo(str(p))).hexdigest() for p in sources}
    raw = pd.read_csv(args.archivo) if args.archivo.lower().endswith(".csv") else leer_excel_sin_bloqueo(args.archivo)
    df = preparar_datos_agricolas(raw)
    if df[["Anio", "Departamento", "Producto"]].isna().any().any():
        raise ValueError("Año, departamento y producto no pueden ser nulos.")
    df["Anio"] = pd.to_numeric(df["Anio"], errors="raise").astype(int)
    df["Departamento"] = df["Departamento"].astype(str).str.strip().str.upper()
    df["Producto"] = df["Producto"].astype(str).str.strip().str.upper()
    df = filtrar_periodo_analisis(df, params)
    df, exclusiones = aplicar_exclusiones_departamentos(df, params)
    if df["Producto"].nunique() != 1 or df.duplicated(["Departamento", "Anio"]).any():
        raise ValueError("Se requiere un producto y observaciones únicas año/departamento.")
    for col in ["Area (ha)", "Rendimiento (ton/ha)"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if (df[col] < 0).any():
            raise ValueError(f"Valores negativos en {col}.")
    df.loc[df["Area (ha)"] <= 0, "Rendimiento (ton/ha)"] = np.nan
    rend = df.pivot(index="Departamento", columns="Anio", values="Rendimiento (ton/ha)")
    area = df.pivot(index="Departamento", columns="Anio", values="Area (ha)")
    print(f"Selección adaptativa: {len(df)} filas, {len(rend)} departamentos, {rend.columns.min()}-{rend.columns.max()}", flush=True)
    comparison = comparar_escala_y_forma(rend, area, params)
    result = comparison["principal"]
    after = {str(p): hashlib.sha256(leer_bytes_sin_bloqueo(str(p))).hexdigest() for p in sources}
    if before != after:
        raise RuntimeError("Las entradas cambiaron durante la ejecución.")
    guardar_seleccion(result, args.salida)
    guardar_comparacion(comparison, args.salida)
    exclusiones.to_csv(Path(args.salida) / "T00_Exclusiones_Documentadas.csv", index=False, encoding="utf-8-sig")
    order = pd.DataFrame({"label": result["labels"], "mean": rend.mean(axis=1).to_numpy()}).groupby("label")["mean"].mean().sort_values().index
    mapping = {old: f"C{new}_DTW" for new, old in enumerate(order, 1)}
    pd.DataFrame({"Departamento": rend.index, "Cluster_DTW": [mapping[c] for c in result["labels"]],
        "N_Anios_Observados": rend.notna().sum(axis=1).to_numpy(), "Estado_Seleccion": result["decision"]["estado"]}).to_csv(
        Path(args.salida) / "Asignaciones_Adaptativas.csv", index=False, encoding="utf-8-sig")
    record = {"command": [sys.executable, *sys.argv], "python": platform.python_version(),
        "duracion_segundos": time.perf_counter() - start, "semilla": params["SEMILLA_ALEATORIA"],
        "entradas_sha256_antes": before, "entradas_sha256_despues": after,
        "resultado": result["decision"], "estado_ejecucion": "completed"}
    record["geometria_principal"] = params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"]
    record["resultados_por_geometria"] = {geom: item["decision"] for geom, item in comparison["resultados"].items()}
    (Path(args.salida) / "ejecucion_adaptativa.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Seleccionado: {result['decision']['algoritmo']}, k={result['decision']['k']}. {result['decision']['estado']}")
    print(f"Resultados: {Path(args.salida).resolve()}")


if __name__ == "__main__":
    main()
