"""
====================================================================================================
FASE 1: CLUSTERING DE SERIES TEMPORALES CON DYNAMIC TIME WARPING (DTW) COMPATIBLE
        (K-MEDOIDS PAM + JERÁRQUICO AVERAGE/COMPLETE, SIN WARD), MANN-KENDALL / SEN Y GRAFO MST
====================================================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from fase_0_ingesta_y_clima import configurar_estilo_publicacion, formatear_depto_titulo, guardar_figura_segura, repel_labels_2d


def dtw_distance(s1: np.ndarray, s2: np.ndarray, window: int = 2, anios=None) -> float:
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            if anios is not None and abs(anios[i - 1] - anios[j - 1]) > window:
                continue
            cost = float(np.sum((s1[i - 1] - s2[j - 1]) ** 2))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(np.sqrt(dtw[n, m]))


def dtw_sobre_observaciones(s1, s2, anios, window, minimo_anios):
    """Alineamiento en años compartidos reales; RMS por número de años, sin imputación."""
    a, b = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    valid_a = np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)
    valid_b = np.isfinite(b) if b.ndim == 1 else np.isfinite(b).all(axis=1)
    valid = valid_a & valid_b
    n = int(valid.sum())
    if n < minimo_anios:
        raise ValueError(f"Sólo hay {n} años compartidos; se requieren {minimo_anios}. No se inventan observaciones.")
    return dtw_distance(a[valid], b[valid], window=window, anios=np.asarray(anios)[valid]) / np.sqrt(n), n


def dunn_index(D: np.ndarray, labels: np.ndarray) -> float:
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return 0.0
    max_intra = 0.0
    min_inter = np.inf
    for c1 in unique_clusters:
        idx1 = np.where(labels == c1)[0]
        if len(idx1) > 1:
            max_intra = max(max_intra, float(np.max(D[np.ix_(idx1, idx1)])))
        for c2 in unique_clusters:
            if c1 < c2:
                idx2 = np.where(labels == c2)[0]
                min_inter = min(min_inter, float(np.min(D[np.ix_(idx1, idx2)])))
    return float(min_inter / max_intra) if max_intra > 0 else 0.0


def kmedoids_pam(D: np.ndarray, k: int, max_iter: int = 100, n_restarts: int = 15, seed: int = 42) -> tuple:
    n = D.shape[0]
    if not 1 <= k < n or n_restarts < 1 or max_iter < 1:
        raise ValueError("PAM requiere 1 <= k < n y al menos una iteración/reinicio.")
    if not np.isfinite(D).all() or not np.allclose(D, D.T) or np.any(D < 0):
        raise ValueError("La matriz de distancias debe ser finita, simétrica y no negativa.")
    best_cost = np.inf
    best_labels = None
    best_medoids = None

    for s_offset in range(n_restarts):
        rng = np.random.default_rng(seed + s_offset)
        medoids = [int(rng.integers(0, n))]
        for _ in range(1, k):
            dist_sq = np.min(D[:, medoids], axis=1) ** 2
            dist_sq[medoids] = 0.0
            if np.sum(dist_sq) == 0:
                dist_sq[:] = 1.0
                dist_sq[medoids] = 0.0
            probs = dist_sq / np.sum(dist_sq)
            medoids.append(int(rng.choice(n, p=probs)))
        medoids = np.array(medoids)

        for _ in range(max_iter):
            # Paso SWAP de PAM: evaluar intercambios entre cada medoide y cada no-medoide.
            current_cost = float(D[:, medoids].min(axis=1).sum())
            swap_cost, best_swap = current_cost, None
            candidatos = np.setdiff1d(np.arange(n), medoids)
            for c in range(k):
                otros = np.delete(medoids, c)
                sin_c = D[:, otros].min(axis=1) if len(otros) else np.full(n, np.inf)
                costes = np.minimum(sin_c[:, None], D[:, candidatos]).sum(axis=0)
                mejor = int(np.argmin(costes))
                if costes[mejor] < swap_cost - 1e-12:
                    best_swap = medoids.copy()
                    best_swap[c] = candidatos[mejor]
                    swap_cost = float(costes[mejor])
            if best_swap is None:
                break
            medoids = best_swap

        labels = np.argmin(D[:, medoids], axis=1)
        labels[medoids] = np.arange(k)  # Resolver empates asignando cada medoide a sí mismo.
        total_cost = float(np.sum([D[i, medoids[labels[i]]] for i in range(n)]))
        if total_cost < best_cost:
            best_cost = total_cost
            best_labels = labels + 1
            best_medoids = medoids

    return best_labels, best_medoids


def mann_kendall_sen_test(y: np.ndarray, t: np.ndarray = None, alpha: float = 0.05) -> dict:
    if t is None:
        t = np.arange(len(y))
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    valid = np.isfinite(y) & np.isfinite(t)
    y, t = y[valid], t[valid]
    n = len(y)
    if n < 2:
        return {"MK_Tau": np.nan, "MK_Z": np.nan, "MK_p_value": np.nan,
                "Sen_Slope (unid/anio)": np.nan, "Clasificacion_Tendencia": "Datos insuficientes"}
    s = 0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            s += np.sign(diff)
            if t[j] != t[i]:
                slopes.append(diff / (t[j] - t[i]))

    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    z = 0.0 if var_s <= 0 else ((s - 1) / np.sqrt(var_s) if s > 0 else ((s + 1) / np.sqrt(var_s) if s < 0 else 0.0))
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1)) if n > 1 else 0.0
    sen_slope = float(np.median(slopes)) if slopes else 0.0
    tendencia = (
        f"Creciente Significativa (p<{alpha})"
        if (p_val < alpha and sen_slope > 0)
        else (f"Decreciente Significativa (p<{alpha})" if (p_val < alpha and sen_slope < 0) else f"Sin tendencia detectada (p>={alpha}); no demuestra estancamiento")
    )
    return {
        "MK_Tau": round(float(tau), 3),
        "MK_Z": round(float(z), 3),
        "MK_p_value": float(p_val),
        "Sen_Slope (unid/anio)": round(sen_slope, 3),
        "Clasificacion_Tendencia": tendencia,
    }


def ejecutar_fase_1(ctx: dict, params: dict) -> dict:
    print("[FASE 1] Clustering DTW compatible (K-Medoids PAM + Jerárquico Average/Complete + Mann-Kendall + Grafo MST)...")

    df = ctx["df"]
    nombre_cultivo = ctx["nombre_cultivo"]
    anio_min, anio_max = ctx["anio_min"], ctx["anio_max"]
    dir_tablas, dir_figuras = ctx["dir_tablas"], ctx["dir_figuras"]
    seed = params["SEMILLA_ALEATORIA"]
    w_dtw = params["VENTANA_SAKOE_CHIBA"]
    n_restarts = params["REINICIOS_KMEDOIDS_PAM"]
    max_iter = params["MAX_ITER_KMEDOIDS"]
    alpha_stat = params["ALPHA_SIGNIFICANCIA_ESTADISTICA"]
    dpi = params["DPI_FIGURAS"]
    paleta_c = params["PALETA_CLUSTERS"]
    nombres_c = params["NOMBRES_CLUSTERS"]

    piv_rend_raw = df.pivot(index="Departamento", columns="Anio", values="Rendimiento (ton/ha)")
    piv_area_raw = df.pivot(index="Departamento", columns="Anio", values="Area (ha)")
    # Todo el panel: se conservan NA y cada distancia usa su soporte observado compartido.
    piv_rend = piv_rend_raw.copy()
    piv_area = piv_area_raw.copy()
    piv_rend_z = piv_rend.sub(piv_rend.mean(axis=1), axis=0).div(piv_rend.std(axis=1, ddof=0).replace(0, 1), axis=0)
    primeras_areas = piv_area.apply(lambda s: s.dropna().iloc[0], axis=1).replace(0, np.nan)
    piv_area_idx = piv_area.div(primeras_areas, axis=0)

    deptos_bal = piv_rend.index.tolist()
    n_d = len(deptos_bal)
    if n_d < 3:
        raise ValueError("Se requieren al menos tres departamentos para comparar particiones.")
    R_mag = piv_rend.values
    R_z = piv_rend_z.values
    if np.nanstd(R_mag) == 0 or np.nanstd(np.log1p(piv_area.values)) == 0:
        raise ValueError("DTW bivariado requiere variación en rendimiento y log-área.")
    R_glob = (R_mag - np.nanmean(R_mag)) / np.nanstd(R_mag)
    A_glob = (np.log1p(piv_area.values) - np.nanmean(np.log1p(piv_area.values))) / np.nanstd(np.log1p(piv_area.values))

    D_mag = np.zeros((n_d, n_d))
    D_shape = np.zeros((n_d, n_d))
    D_multi = np.zeros((n_d, n_d))
    N_compartidos = np.zeros((n_d, n_d), dtype=int)
    anios_dtw = piv_rend.columns.to_numpy(dtype=int)
    min_compartidos = params["MINIMO_ANIOS_COMPARTIDOS_DTW"]
    for i in range(n_d):
        for j in range(i + 1, n_d):
            D_mag[i, j] = D_mag[j, i] = dtw_sobre_observaciones(R_mag[i], R_mag[j], anios_dtw, w_dtw, min_compartidos)[0]
            D_shape[i, j] = D_shape[j, i] = dtw_sobre_observaciones(R_z[i], R_z[j], anios_dtw, w_dtw, min_compartidos)[0]
            distancia, n_comp = dtw_sobre_observaciones(
                np.column_stack([R_glob[i], A_glob[i]]),
                np.column_stack([R_glob[j], A_glob[j]]), anios_dtw, w_dtw, min_compartidos)
            D_multi[i, j] = D_multi[j, i] = distancia
            N_compartidos[i, j] = N_compartidos[j, i] = n_comp
    np.fill_diagonal(N_compartidos, piv_rend.notna().sum(axis=1).to_numpy())
    from clustering_adaptativo import matriz_bivariada_observada
    D_escala = D_multi.copy()
    D_forma, _ = matriz_bivariada_observada(piv_rend, piv_area, dict(params, GEOMETRIA_ADAPTATIVA="forma"))
    geometria_principal = params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"]
    D_multi = D_forma if geometria_principal == "forma" else D_escala
    for nombre, matrix in [("T21_DTW_Distancias_Escala", D_escala), ("T22_DTW_Distancias_Forma", D_forma)]:
        pd.DataFrame(matrix, index=deptos_bal, columns=deptos_bal).to_csv(
            os.path.join(dir_tablas, nombre + ".csv"), index_label="Departamento", encoding="utf-8-sig")
    df_cobertura = pd.DataFrame({
        "Departamento": deptos_bal,
        "N_Filas_Originales": df.groupby("Departamento").size().reindex(deptos_bal).to_numpy(),
        "N_Rendimientos_Validos": piv_rend.notna().sum(axis=1).to_numpy(),
        "Primer_Anio_Observado": piv_rend.apply(lambda s: s.first_valid_index(), axis=1).to_numpy(),
        "Ultimo_Anio_Observado": piv_rend.apply(lambda s: s.last_valid_index(), axis=1).to_numpy(),
        "N_AnioDepto_Ausentes": piv_rend.shape[1] - df.groupby("Departamento").size().reindex(deptos_bal).to_numpy(),
        "Incluido_Clustering_y_Grafo": True,
        "Geometria_Principal": geometria_principal,
        "Metodo_DTW": "Años observados compartidos, ventana en años, distancia dividida por raíz de N compartidos; sin imputación",
    })
    df_cobertura.to_csv(os.path.join(dir_tablas, "T11_Cobertura_Total_Datos.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(N_compartidos, index=deptos_bal, columns=deptos_bal).to_csv(
        os.path.join(dir_tablas, "T15_DTW_Numero_Anios_Compartidos.csv"), encoding="utf-8-sig")

    grid_rows = []
    for enfoque, D_mat in [
        ("DTW_Escala_Bivariada (Rend + logArea)", D_escala),
        ("DTW_Forma_Bivariada_ZDepto", D_forma),
        ("DTW_Magnitud_Rendimiento (ton/ha)", D_mag),
        ("DTW_Forma_ZScore (Tendencia)", D_shape),
    ]:
        dist_cond = squareform(D_mat)
        Z_avg = linkage(dist_cond, method="average")
        Z_comp = linkage(dist_cond, method="complete")
        coph_avg, _ = cophenet(Z_avg, dist_cond)
        coph_comp, _ = cophenet(Z_comp, dist_cond)

        for k in params["GRILLA_CLUSTERS_K"]:
            if not 2 <= k < n_d:
                continue
            for alg_name, lbls, coph_val in [
                ("K-Medoids (PAM)", kmedoids_pam(D_mat, k=k, max_iter=max_iter, n_restarts=n_restarts, seed=seed)[0], np.nan),
                ("Jerárquico Average (UPGMA)", fcluster(Z_avg, t=k, criterion="maxclust"), coph_avg),
                ("Jerárquico Complete (CLINK)", fcluster(Z_comp, t=k, criterion="maxclust"), coph_comp),
            ]:
                if not 2 <= len(np.unique(lbls)) < n_d:
                    continue
                grid_rows.append({
                    "Enfoque_DTW": enfoque,
                    "Algoritmo_Compatible_DTW": alg_name,
                    "Numero_Clusters (k)": k,
                    "Silhouette_Score (DTW)": round(silhouette_score(D_mat, lbls, metric="precomputed"), 4),
                    "Dunn_Index (DTW)": round(dunn_index(D_mat, lbls), 4),
                    "Correlacion_Cofenetica (r_coph)": round(float(coph_val), 4) if pd.notna(coph_val) else None,
                    "N_Departamentos": n_d,
                    "Min_Anios_Compartidos": int(N_compartidos[np.triu_indices(n_d, 1)].min()),
                    "Alcance": "Validación interna descriptiva; DTW sobre soportes compartidos de distinta longitud",
                })

    df_grid_dtw = pd.DataFrame(grid_rows)
    df_grid_dtw.to_csv(os.path.join(dir_tablas, "T01_Metricas_Validacion_Clustering_DTW.csv"), index=False, encoding="utf-8-sig")

    Z_avg_multi = linkage(squareform(D_multi), method="average")
    Z_comp_multi = linkage(squareform(D_multi), method="complete")
    coph_avg_multi, _ = cophenet(Z_avg_multi, squareform(D_multi))

    from clustering_adaptativo import comparar_escala_y_forma, guardar_comparacion, guardar_seleccion, ajustar_particion
    comparacion = comparar_escala_y_forma(piv_rend, piv_area, params, matrices_base={"escala": D_escala, "forma": D_forma})
    adaptativo = comparacion["principal"]
    k_main = adaptativo["decision"]["k"]
    algoritmo_principal = adaptativo["decision"]["algoritmo"]
    nombre_algoritmo = {"pam": "PAM", "average": "UPGMA", "complete": "Complete"}[algoritmo_principal]
    labels_kmed = adaptativo["labels"]  # Alias interno histórico; puede proceder de cualquiera de los tres métodos.
    labels_avg = ajustar_particion(D_multi, "average", k_main, params)
    labels_comp = ajustar_particion(D_multi, "complete", k_main, params)
    print(f"      -> Partición {adaptativo['decision']['modo']}: {nombre_algoritmo}, k={k_main}; {adaptativo['decision']['estado']}", flush=True)
    medoids_idx = {}
    for label in np.unique(labels_kmed):
        indices = np.flatnonzero(labels_kmed == label)
        medoids_idx[label] = indices[np.argmin(D_multi[np.ix_(indices, indices)].sum(axis=1))]

    orden_kmed = (
        pd.DataFrame({"c": labels_kmed, "r": piv_rend.mean(axis=1).values})
        .groupby("c")["r"].mean().sort_values().index.tolist()
    )
    relabel_kmed = {old: f"C{new}_DTW" for new, old in enumerate(orden_kmed, 1)}
    medoides_nombres = {f"C{new}_DTW": deptos_bal[medoids_idx[old]] for new, old in enumerate(orden_kmed, 1)}
    medias_grupo = pd.DataFrame({"c": labels_kmed, "r": piv_rend.mean(axis=1).to_numpy()}).groupby("c")["r"].mean()
    areas_grupo = pd.DataFrame({"c": labels_kmed, "a": piv_area.mean(axis=1).to_numpy()}).groupby("c")["a"].mean()
    colores_previos = params["PALETA_CLUSTERS"]
    orden_clusters = [relabel_kmed[c] for c in orden_kmed]
    paleta_c = {key: colores_previos.get(key, matplotlib.colors.to_hex(plt.get_cmap("tab10")(i % 10)))
                for i, key in enumerate(orden_clusters)}
    nombres_c = {relabel_kmed[old]: (f"C{i}: Trajectory Profile" if geometria_principal == "forma" else
                 f"C{i}: Yield {medias_grupo[old]:.2f} t/ha; Area {areas_grupo[old]:,.0f} ha")
                 for i, old in enumerate(orden_kmed, 1)}
    params["PALETA_CLUSTERS"], params["NOMBRES_CLUSTERS"] = paleta_c, nombres_c
    adaptativo["grupos"]["Cluster_Principal_Si_Seleccionado"] = [
        relabel_kmed.get(label, "") if alg == algoritmo_principal and k == k_main else ""
        for alg, k, label in adaptativo["grupos"][["Algoritmo", "k", "Grupo_Referencia"]].itertuples(index=False, name=None)]
    guardar_seleccion(adaptativo, dir_tablas)
    comparacion["asignaciones"]["Cluster_Principal"] = [relabel_kmed[c] for c in labels_kmed]
    guardar_comparacion(comparacion, dir_tablas)

    clusters_depto = pd.DataFrame({
        "Departamento": deptos_bal,
        "Algoritmo_Principal": algoritmo_principal,
        "Geometria_Principal": geometria_principal,
        "Estado_Seleccion": adaptativo["decision"]["estado"],
        "Cluster_DTW_Principal": [relabel_kmed[c] for c in labels_kmed],
        "Medoide_Representativo_Cluster": [medoides_nombres[relabel_kmed[c]] for c in labels_kmed],
        "Cluster_Jerarquico_Average": [f"Avg_G{c}" for c in labels_avg],
        "Cluster_Jerarquico_Complete": [f"Comp_G{c}" for c in labels_comp],
        f"Rendimiento_{anio_min}": piv_rend.iloc[:, 0].values,
        f"Rendimiento_{anio_max}": piv_rend.iloc[:, -1].values,
        f"Cambio_Rend_{anio_min}_{anio_max} (%)": ((piv_rend.iloc[:, -1] - piv_rend.iloc[:, 0]) / piv_rend.iloc[:, 0].replace(0, np.nan) * 100).round(1).values,
        f"Cambio_Area_{anio_min}_{anio_max} (%)": ((piv_area.iloc[:, -1] - piv_area.iloc[:, 0]) / piv_area.iloc[:, 0].replace(0, np.nan) * 100).round(1).values,
    })

    df = df.merge(
        clusters_depto[["Departamento", "Cluster_DTW_Principal", "Cluster_Jerarquico_Average", "Cluster_Jerarquico_Complete"]],
        on="Departamento", how="left"
    )
    if df["Cluster_DTW_Principal"].isna().any():
        raise ValueError("Hay departamentos sin asignación: no se excluyen silenciosamente del clustering.")

    mk_rows = []
    for depto, grp in df.groupby("Departamento"):
        grp = grp.sort_values("Anio")
        res_r = mann_kendall_sen_test(grp["Rendimiento (ton/ha)"].values, grp["Anio"].values, alpha=alpha_stat)
        res_a = mann_kendall_sen_test(grp["Area (ha)"].values, grp["Anio"].values, alpha=alpha_stat)
        valores = grp["Rendimiento (ton/ha)"].to_numpy()
        tiempos = grp["Anio"].to_numpy()
        residuo = valores - res_r["Sen_Slope (unid/anio)"] * (tiempos - tiempos.min())
        cv_res = float(np.nanstd(residuo, ddof=1) / np.nanmean(valores) * 100)
        mk_rows.append({
            "Departamento": depto,
            "Cluster_DTW": grp["Cluster_DTW_Principal"].iloc[0],
            "Primer_Anio_Valido": int(grp.loc[grp["Rendimiento (ton/ha)"].notna(), "Anio"].iloc[0]),
            "Ultimo_Anio_Valido": int(grp.loc[grp["Rendimiento (ton/ha)"].notna(), "Anio"].iloc[-1]),
            "Primer_Rendimiento_Observado (ton/ha)": round(grp["Rendimiento (ton/ha)"].dropna().iloc[0], 2),
            "Ultimo_Rendimiento_Observado (ton/ha)": round(grp["Rendimiento (ton/ha)"].dropna().iloc[-1], 2),
            "Rend_Media (ton/ha)": round(grp["Rendimiento (ton/ha)"].mean(), 2),
            "Area_Media (ha)": float(grp["Area (ha)"].mean()),
            "Rend_CV (%)": round(grp["Rendimiento (ton/ha)"].std() / max(grp["Rendimiento (ton/ha)"].mean(), 1e-6) * 100, 1),
            "CV_Residual_Destendenciado (%)": round(cv_res, 2),
            "MK_Tau_Rend": res_r["MK_Tau"],
            "MK_Z_Rend": res_r["MK_Z"],
            "MK_p_value_Rend": res_r["MK_p_value"],
            "Sen_Slope_Rend (ton/ha/anio)": res_r["Sen_Slope (unid/anio)"],
            "Tendencia_Rendimiento": res_r["Clasificacion_Tendencia"],
            "Sen_Slope_Area (ha/anio)": res_a["Sen_Slope (unid/anio)"],
            "Tendencia_Area": res_a["Clasificacion_Tendencia"],
        })
    df_mk = pd.DataFrame(mk_rows).sort_values("Sen_Slope_Rend (ton/ha/anio)", ascending=False)
    from statsmodels.stats.multitest import multipletests
    df_mk["MK_p_value_Rend_Holm"] = multipletests(df_mk["MK_p_value_Rend"], method="holm")[1]
    df_mk["Perfil_Tendencia_Holm"] = np.where(
        df_mk["MK_p_value_Rend_Holm"] < alpha_stat,
        np.where(df_mk["Sen_Slope_Rend (ton/ha/anio)"] > 0, "Crecimiento detectado", "Descenso detectado"),
        "Sin tendencia detectada; no demuestra estancamiento")
    df_perfiles = df_mk.merge(df_cobertura, on="Departamento", validate="one_to_one")
    limite_volatilidad = df_perfiles["CV_Residual_Destendenciado (%)"].quantile(0.75)
    df_perfiles["Volatilidad_Relativa"] = np.where(
        df_perfiles["CV_Residual_Destendenciado (%)"] >= limite_volatilidad,
        "Cuartil superior de variabilidad residual", "Por debajo del cuartil superior")
    df_perfiles["Umbral_P75_CV_Residual (%)"] = limite_volatilidad
    df_perfiles.to_csv(os.path.join(dir_tablas, "T12_Perfiles_Transicion_Productiva.csv"), index=False, encoding="utf-8-sig")
    comparacion["asignaciones"] = comparacion["asignaciones"].merge(
        df_mk[["Departamento", "Sen_Slope_Rend (ton/ha/anio)", "MK_p_value_Rend_Holm", "Perfil_Tendencia_Holm"]],
        on="Departamento", validate="one_to_one")
    comparacion["asignaciones"].to_csv(os.path.join(dir_tablas, "T20_Asignaciones_Escala_y_Forma.csv"), index=False, encoding="utf-8-sig")
    df_mk.to_csv(os.path.join(dir_tablas, "T02_MannKendall_SenSlope_Departamentos.csv"), index=False, encoding="utf-8-sig")

    G_full = nx.Graph()
    sigma_dtw = float(np.median(D_multi[D_multi > 0])) if np.any(D_multi > 0) else 1.0
    for i, d_i in enumerate(deptos_bal):
        cl_i = clusters_depto.loc[clusters_depto["Departamento"] == d_i, "Cluster_DTW_Principal"].values[0]
        G_full.add_node(d_i, cluster=cl_i, rend_end=float(piv_rend.iloc[i, -1]))
        for j in range(i + 1, n_d):
            d_j = deptos_bal[j]
            dist_ij = float(D_multi[i, j])
            G_full.add_edge(d_i, d_j, weight=dist_ij, similarity=float(np.exp(-(dist_ij ** 2) / (2 * sigma_dtw ** 2))))

    mst_edges = set(nx.minimum_spanning_edges(G_full, weight="weight", data=False))
    G_sparse = nx.Graph()
    for node, attr in G_full.nodes(data=True):
        G_sparse.add_node(node, **attr)
    for u, v in mst_edges:
        G_sparse.add_edge(u, v, **G_full[u][v], tipo_arista="MST")
    k_vecinos = params["VECINOS_KNN_GRAFO"]
    for i, d_i in enumerate(deptos_bal):
        vecinos = [j for j in np.argsort(D_multi[i]) if j != i][:k_vecinos]
        for j in vecinos:
            d_j = deptos_bal[j]
            if not G_sparse.has_edge(d_i, d_j):
                G_sparse.add_edge(d_i, d_j, **G_full[d_i][d_j], tipo_arista="kNN")

    deg_cent = nx.degree_centrality(G_sparse)
    bet_cent = nx.betweenness_centrality(G_sparse, weight="weight")
    clo_cent = nx.closeness_centrality(G_sparse, distance="weight")
    eig_cent = nx.eigenvector_centrality_numpy(G_sparse, weight="similarity")

    df_grafo = pd.DataFrame({
        "Departamento": deptos_bal,
        "Cluster_DTW_Principal": [G_sparse.nodes[d]["cluster"] for d in deptos_bal],
        "Grado_Conexiones": [G_sparse.degree(d) for d in deptos_bal],
        "Degree_Centrality": [round(deg_cent[d], 4) for d in deptos_bal],
        "Betweenness_Centrality (Puente)": [round(bet_cent[d], 4) for d in deptos_bal],
        "Closeness_Centrality": [round(clo_cent[d], 4) for d in deptos_bal],
        "Eigenvector_Centrality": [round(eig_cent[d], 4) for d in deptos_bal],
    }).sort_values("Betweenness_Centrality (Puente)", ascending=False)
    df_grafo.to_csv(os.path.join(dir_tablas, "T03_Metricas_Topologicas_Grafo_DTW.csv"), index=False, encoding="utf-8-sig")

    # Resúmenes departamentales: evitar tratar los años repetidos como réplicas independientes.
    df_valid_clusters = (df[df["Cluster_DTW_Principal"].isin(orden_clusters)]
                         .groupby(["Departamento", "Cluster_DTW_Principal"], as_index=False).mean(numeric_only=True))
    vars_evaluar_clusters = [
        "Rendimiento (ton/ha)",
        "Area_Lag1 (ha)",
        "Radiacion_Solar_Media (MJ/m2/dia)",
        "Temp_Maxima_Media_Anual (C)",
        "Humedad_Relativa_Media (%)",
        "Precipitacion_Acumulada_Anual (mm)",
        "Precipitacion_CV_Intraanual",
        "Brecha_Rendimiento_Depto (ton/ha)",
    ]
    kw_rows = []
    for var in vars_evaluar_clusters:
        grupos = [grp[var].dropna().values for _, grp in df_valid_clusters.groupby("Cluster_DTW_Principal") if len(grp) > 0]
        f_stat, p_anova = stats.f_oneway(*grupos)
        h_stat, p_kw = stats.kruskal(*grupos)
        N_tot = sum(len(g) for g in grupos)
        eps_sq = h_stat / ((N_tot ** 2 - 1) / (N_tot + 1))
        medias_c = df_valid_clusters.groupby("Cluster_DTW_Principal")[var].mean()
        kw_rows.append({
            "Variable": var,
            **{f"Media_C{i}": round(medias_c.get(key, np.nan), 2) for i, key in enumerate(orden_clusters, 1)},
            "Kruskal_Wallis_H": round(float(h_stat), 2),
            "p_value_KW": float(f"{p_kw:.2e}"),
            "ANOVA_F": round(float(f_stat), 2),
            "p_value_ANOVA": float(f"{p_anova:.2e}"),
            "Tamano_Efecto_Epsilon2": round(float(eps_sq), 4),
            "Significancia": "*** (p<0.001)" if p_kw < 0.001 else ("** (p<0.01)" if p_kw < 0.01 else ("* (p<0.05)" if p_kw < alpha_stat else "ns")),
            "Alcance": "Exploratorio sobre medias departamentales; clusters definidos con estas mismas series",
        })
    df_kw_clusters = pd.DataFrame(kw_rows)
    df_kw_clusters.to_csv(os.path.join(dir_tablas, "T04_Validacion_Estadistica_Clusters_Kruskal_ANOVA.csv"), index=False, encoding="utf-8-sig")

    # Figuras de la Fase 1 (Fig1, Fig2, Fig4) - Estándar Q1 Journal (Academic English, No Titles, Zero Overlaps)
    configurar_estilo_publicacion(plt)

    def _save(fig_obj, name):
        guardar_figura_segura(fig_obj, dir_figuras, name, dpi, plt)

    # --- FIGURA 1: Dendrograma UPGMA y Trayectorias K-Medoids (PAM) ---
    fig1 = plt.figure(figsize=(14.4, 6.2))
    gs1 = fig1.add_gridspec(1, 2, width_ratios=[1.05, 1.38], wspace=0.24)
    ax1a = fig1.add_subplot(gs1[0])
    labels_deptos_en = [formatear_depto_titulo(d) for d in deptos_bal]
    dendrogram(
        Z_avg_multi,
        labels=labels_deptos_en,
        orientation="right",
        leaf_font_size=8.2,
        color_threshold=0,
        above_threshold_color="#8D99AE",
        ax=ax1a,
    )
    colores_hojas = {formatear_depto_titulo(d): paleta_c[relabel_kmed[c]]
                     for d, c in zip(deptos_bal, labels_kmed)}
    for etiqueta in ax1a.get_yticklabels():
        etiqueta.set_color(colores_hojas[etiqueta.get_text()])
    ax1a.set_xlabel(f"Normalized DTW: {'Trajectory Shape' if geometria_principal == 'forma' else 'Productive Scale'} (w = {w_dtw} yr)")
    ax1a.text(-0.14, 1.02, "(a)", transform=ax1a.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    ax1a.text(
        0.97, 0.04,
        f"UPGMA Linkage (Ward-free)\nCophenetic $r_{{coph}} = {coph_avg_multi:.3f}$",
        transform=ax1a.transAxes, ha="right", va="bottom", fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CED4DA", alpha=0.94),
    )

    ax1b = fig1.add_subplot(gs1[1])
    anios_arr = piv_rend.columns.values.astype(int)
    max_upper_band = 0.0
    piv_trayectorias = piv_rend_z if geometria_principal == "forma" else piv_rend
    for cl_key in orden_clusters:
        depts_c = clusters_depto.loc[clusters_depto["Cluster_DTW_Principal"] == cl_key, "Departamento"]
        piv_c = piv_trayectorias.loc[piv_trayectorias.index.isin(depts_c)]
        if len(piv_c) == 0:
            continue
        mean_c = piv_c.mean(axis=0).values
        sem_c = piv_c.sem(axis=0).fillna(0).values
        upper_c = mean_c + 1.96 * sem_c
        max_upper_band = max(max_upper_band, float(np.nanmax(upper_c)), float(np.nanmax(piv_c.values)))
        color_c = paleta_c[cl_key]
        medoid_name = formatear_depto_titulo(medoides_nombres.get(cl_key, ""))
        for _, row_s in piv_c.iterrows():
            ax1b.plot(anios_arr, row_s.values, color=color_c, alpha=0.15, linewidth=0.85)
        ax1b.plot(
            anios_arr, mean_c, color=color_c, linewidth=2.5, marker="o", markersize=4.2,
            label=f"{nombres_c[cl_key]} [Medoid: {medoid_name}, n={len(piv_c)}]",
        )
        ax1b.fill_between(anios_arr, mean_c - 1.96 * sem_c, upper_c, color=color_c, alpha=0.14)

    ax1b.set_xlabel("Year")
    ax1b.set_ylabel("Within-Department Standardized Yield (Z)" if geometria_principal == "forma" else "Crop Yield (t ha$^{-1}$)")
    ax1b.set_xticks(anios_arr[::max(1, len(anios_arr) // 8)])
    ax1b.set_ylim(bottom=float(np.nanmin(piv_trayectorias.values)) - 0.3 if geometria_principal == "forma" else max(0.0, float(np.nanmin(piv_rend.values)) - 1.2), top=max_upper_band * 1.22)
    ax1b.grid(True, linestyle="--", alpha=0.32)
    ax1b.text(-0.08, 1.02, "(b)", transform=ax1b.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    ax1b.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=8.0)
    if not adaptativo["decision"]["cumple_criterios"]:
        ax1b.text(0.98, 0.02, "Exploratory partition: stability criteria not met", transform=ax1b.transAxes,
                 ha="right", va="bottom", fontsize=7.6,
                 bbox=dict(facecolor="white", edgecolor="#CED4DA", alpha=.95))
    _save(fig1, "Fig1_DTW_Dendrograma_y_Trayectorias")

    # --- FIGURA 2: Grafo Topológico de Similitud DTW (MST + k-NN) con Repulsión de Etiquetas 2D ---
    fig2, ax2 = plt.subplots(figsize=(12.2, 8.2))
    pos = nx.kamada_kawai_layout(G_sparse, weight="weight")
    edges_mst = [(u, v) for u, v, d in G_sparse.edges(data=True) if d.get("tipo_arista") == "MST"]
    edges_knn = [(u, v) for u, v, d in G_sparse.edges(data=True) if d.get("tipo_arista") == "kNN"]
    nx.draw_networkx_edges(G_sparse, pos, edgelist=edges_knn, ax=ax2, style="dashed", alpha=0.32, edge_color="#ADB5BD", width=1.1)
    nx.draw_networkx_edges(G_sparse, pos, edgelist=edges_mst, ax=ax2, style="solid", alpha=0.72, edge_color="#343A40", width=1.8)

    for cl_key in orden_clusters:
        nodos_c = [n for n, d in G_sparse.nodes(data=True) if d["cluster"] == cl_key]
        sizes_c = [G_sparse.nodes[n]["rend_end"] * 34 + 115 for n in nodos_c]
        nx.draw_networkx_nodes(
            G_sparse, pos, nodelist=nodos_c, node_size=sizes_c, node_color=paleta_c[cl_key],
            edgecolors="black", linewidths=1.0, alpha=0.92, ax=ax2,
        )

    nodos_orden = list(G_sparse.nodes())
    xs_n = np.array([pos[n][0] for n in nodos_orden])
    ys_n = np.array([pos[n][1] for n in nodos_orden])
    textos_n = [formatear_depto_titulo(n) for n in nodos_orden]

    pad_x = (xs_n.max() - xs_n.min()) * 0.16
    pad_y = (ys_n.max() - ys_n.min()) * 0.16
    xlim_g = (float(xs_n.min() - pad_x), float(xs_n.max() + pad_x))
    ylim_g = (float(ys_n.min() - pad_y), float(ys_n.max() + pad_y))
    ax2.set_xlim(xlim_g)
    ax2.set_ylim(ylim_g)

    lx_n, ly_n = repel_labels_2d(
        xs_n, ys_n, textos_n, xlim_g, ylim_g,
        char_w_frac=0.0105, box_h_frac=0.038, node_r_frac=0.036, n_iter=380, seed=seed,
    )
    span_xg = xlim_g[1] - xlim_g[0]
    span_yg = ylim_g[1] - ylim_g[0]
    for i, txt in enumerate(textos_n):
        dist_norm = np.hypot((lx_n[i] - xs_n[i]) / span_xg, (ly_n[i] - ys_n[i]) / span_yg)
        if dist_norm > 0.042:
            ax2.annotate(
                "", xy=(xs_n[i], ys_n[i]), xytext=(lx_n[i], ly_n[i]),
                arrowprops=dict(arrowstyle="-", color="#495057", lw=0.65, alpha=0.65, shrinkA=4, shrinkB=5),
                zorder=4,
            )
        t_obj = ax2.text(
            lx_n[i], ly_n[i], txt, fontsize=8.2, fontweight="bold", ha="center", va="center", color="#1B1E23", zorder=5,
        )
        t_obj.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    ax2.axis("off")
    handles_g = [
        Line2D([0], [0], marker="o", color="w", label=nombres_c[ck], markerfacecolor=paleta_c[ck], markeredgecolor="black", markersize=8.5)
        for ck in orden_clusters
    ] + [
        Line2D([0], [0], color="#343A40", lw=1.8, linestyle="-", label="Minimum Spanning Tree (MST) Backbone"),
        Line2D([0], [0], color="#ADB5BD", lw=1.2, linestyle="--", label=f"{k_vecinos}-NN Local DTW Similarity Link"),
    ]
    ax2.legend(
        handles=handles_g, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
        frameon=True, facecolor="white", edgecolor="#CED4DA",
        framealpha=0.95, title=f"{'Adaptive' if adaptativo['decision']['cumple_criterios'] else 'Exploratory'} DTW {nombre_algoritmo}, k={k_main} (Node size $\\propto$ {anio_max} Yield)", title_fontsize=8.6, fontsize=8.0,
    )
    _save(fig2, "Fig2_Grafo_Topologico_DTW_MST")

    # --- FIGURA 4: Pendientes de Theil-Sen y Significancia de Mann-Kendall ---
    fig4, ax4 = plt.subplots(figsize=(11.2, 8.4))
    df_mk_sorted = df_mk.sort_values("Sen_Slope_Rend (ton/ha/anio)", ascending=True)
    colores_bar = [paleta_c.get(c, "#8D99AE") for c in df_mk_sorted["Cluster_DTW"]]
    y_idx = np.arange(len(df_mk_sorted))
    ax4.barh(y_idx, df_mk_sorted["Sen_Slope_Rend (ton/ha/anio)"], color=colores_bar, edgecolor="black", height=0.72, alpha=0.92)
    ax4.axvline(0, color="black", linewidth=1.0, linestyle="-")
    for i, (_, r_mk) in enumerate(df_mk_sorted.iterrows()):
        val = float(r_mk["Sen_Slope_Rend (ton/ha/anio)"])
        pval = float(r_mk["MK_p_value_Rend"])
        star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < alpha_stat else ""))
        label_val = f"{val:+.2f}{(' ' + star) if star else ''}"
        if abs(val) < 0.015:
            pos_x = max(val, 0.0) + 0.018
            ha = "left"
        elif val >= 0:
            pos_x = val + 0.018
            ha = "left"
        else:
            pos_x = val - 0.018
            ha = "right"
        ax4.text(pos_x, i, label_val, va="center", ha=ha, fontsize=7.9, fontweight="bold" if star else "normal")

    ax4.set_yticks(y_idx)
    ax4.set_yticklabels([formatear_depto_titulo(d) for d in df_mk_sorted["Departamento"]], fontsize=8.4)
    ax4.set_xlabel(f"Theil–Sen Annual Yield Trend Slope (t ha$^{{-1}}$ yr$^{{-1}}$, {anio_min}–{anio_max})")
    min_s, max_s = float(df_mk_sorted["Sen_Slope_Rend (ton/ha/anio)"].min()), float(df_mk_sorted["Sen_Slope_Rend (ton/ha/anio)"].max())
    ax4.set_xlim(min_s - 0.24, max_s + 0.26)
    ax4.grid(axis="x", linestyle="--", alpha=0.32)

    handles_mk = [
        Patch(facecolor=paleta_c[ck], edgecolor="black", label=nombres_c[ck])
        for ck in orden_clusters
    ]
    if any(c not in paleta_c for c in df_mk_sorted["Cluster_DTW"]):
        handles_mk.append(Patch(facecolor="#8D99AE", edgecolor="black", label="Incomplete Series (Excluded from DTW)"))
    handles_mk.append(Line2D([0], [0], color="none", label="Mann–Kendall: * p < 0.05, ** p < 0.01, *** p < 0.001"))
    ax4.legend(handles=handles_mk, loc="lower right", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=8.0)
    _save(fig4, "Fig4_Tendencias_MannKendall_SenSlope")

    # Cruce de asignaciones: los criterios internos de geometrías distintas no se comparan como un ranking.
    cruce = pd.crosstab(comparacion["asignaciones"]["Perfil_Escala"], comparacion["asignaciones"]["Perfil_Forma"])
    fig9, ax9 = plt.subplots(figsize=(max(6, len(cruce.columns) * 1.3), max(4.5, len(cruce) * 0.7)))
    ax9.imshow(cruce.to_numpy(), cmap="Blues", aspect="auto")
    for row in range(len(cruce)):
        for col in range(len(cruce.columns)):
            ax9.text(col, row, str(cruce.iloc[row, col]), ha="center", va="center",
                     color="white" if cruce.iloc[row, col] > cruce.to_numpy().max() / 2 else "black")
    ax9.set_xticks(np.arange(len(cruce.columns)), cruce.columns)
    ax9.set_yticks(np.arange(len(cruce)), cruce.index)
    ax9.set_xlabel("Temporal Trajectory Profiles (Within-Department Z)")
    ax9.set_ylabel("Productive Scale Profiles (Yield + log-Area)")
    _save(fig9, "Fig9_Comparacion_Escala_y_Forma")

    ctx.update({
        "df": df,
        "piv_rend": piv_rend,
        "piv_rend_z": piv_rend_z,
        "piv_area_idx": piv_area_idx,
        "deptos_bal": deptos_bal,
        "D_multi": D_multi,
        "clusters_depto": clusters_depto,
        "df_grid_dtw": df_grid_dtw,
        "df_mk": df_mk,
        "df_grafo": df_grafo,
        "df_kw_clusters": df_kw_clusters,
        "df_cobertura": df_cobertura,
        "df_perfiles": df_perfiles,
        "N_compartidos": N_compartidos,
        "seleccion_adaptativa": adaptativo["decision"],
        "df_seleccion_adaptativa": adaptativo["seleccion"],
        "df_perturbaciones": adaptativo["perturbaciones"],
        "df_estabilidad_grupos": adaptativo["grupos"],
        "algoritmo_principal": algoritmo_principal,
        "comparacion_geometrias": comparacion,
        "D_escala": D_escala,
        "D_forma": D_forma,
    })
    return ctx
