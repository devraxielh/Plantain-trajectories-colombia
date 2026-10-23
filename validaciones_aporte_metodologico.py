"""Comparaciones controladas: clima incremental y PAM 2/3/4 con/sin alineamiento.

No identifica causalidad, no retunea hiperparámetros ni selecciona la métrica
por comparar silhouettes calculados en espacios diferentes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from statsmodels.stats.multitest import multipletests

from fase_0_ingesta_y_clima import guardar_figura_segura
from fase_3_validacion_rolling_y_xai_shap import construir_catalogo_modelos, preparar_fold
from clustering_adaptativo import matriz_bivariada_observada, seleccionar_clustering_adaptativo, ajustar_particion


def comparar_aporte_clima(df, params, modelos=None):
    """Diez ablaciones pareadas: cinco modelos por dos bases productivas."""
    target = "Rendimiento (ton/ha)"
    area, lag = "Area_Lag1 (ha)", "Rendimiento_Lag1 (ton/ha)"
    clima = [c for c in params["VARIABLES_PREDICTORAS_XAI"] if c != area]
    if not clima or area not in params["VARIABLES_PREDICTORAS_XAI"]:
        raise ValueError("La ablación requiere área rezagada y predictores climáticos explícitos.")
    required = list(dict.fromkeys([target, area, lag] + clima))
    valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    if valid.duplicated(["Departamento", "Anio"]).any():
        raise ValueError("Observaciones duplicadas en ablación.")
    years = np.sort(valid.Anio.unique())
    n = params["ANIOS_EVALUACION_ROLLING_ORIGIN"]
    if not 3 <= n <= len(years) - 2:
        raise ValueError("La ablación requiere tres años de evaluación y dos años previos.")
    eval_years = years[-n:]
    modelos = construir_catalogo_modelos(params) if modelos is None else modelos
    predictions, annual, rows = [], [], []
    rng = np.random.default_rng(params["SEMILLA_ALEATORIA"])
    # Mismos sorteos de bloques anuales para las diez comparaciones; mantiene
    # todos los departamentos de cada bloque y no considera filas independientes.
    draws = rng.integers(n, size=(params.get("REPLICAS_BOOTSTRAP_ABLACION", 2000), n))
    for family, base in [("Area", [area]), ("Area_y_rendimiento_previo", [area, lag])]:
        for model_name, constructor in modelos.items():
            print(f"      -> Ablación climática: {family} / {model_name}", flush=True)
            pair_records, folds = [], []
            for year in eval_years:
                train, test = preparar_fold(valid[valid.Anio < year], valid[valid.Anio == year])
                if train.empty or test.empty or train.Anio.max() >= year:
                    raise ValueError("Partición temporal no válida en ablación.")
                without, with_climate = constructor(), constructor()
                without.fit(train[base], train[target])
                with_climate.fit(train[base + clima], train[target])
                p0, p1 = without.predict(test[base]), with_climate.predict(test[base + clima])
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    raise ValueError("Predicción no finita en ablación.")
                y = test[target].to_numpy()
                for position, (_, observation) in enumerate(test.iterrows()):
                    pair_records.append(dict(Modelo=model_name, Base_Productiva=family,
                        Departamento=observation.Departamento, Anio=int(year), Observado=y[position],
                        Predicho_Sin_Clima=p0[position], Predicho_Con_Clima=p1[position],
                        Max_Anio_Entrenamiento=int(train.Anio.max()), N_Entrenamiento=len(train),
                        Predictores_Sin_Clima=" | ".join(base), Predictores_Con_Clima=" | ".join(base + clima)))
                folds.append(dict(Modelo=model_name, Base_Productiva=family, Anio=int(year), N=len(y),
                    MAE_Sin_Clima=mean_absolute_error(y, p0), MAE_Con_Clima=mean_absolute_error(y, p1),
                    MSE_Sin_Clima=mean_squared_error(y, p0), MSE_Con_Clima=mean_squared_error(y, p1)))
            pred, fold = pd.DataFrame(pair_records), pd.DataFrame(folds)
            pred["Predicho_Persistencia"] = valid.set_index(["Departamento", "Anio"])[lag].reindex(
                pd.MultiIndex.from_frame(pred[["Departamento", "Anio"]])).to_numpy()
            pred["Error_Absoluto_Sin_Clima"] = abs(pred.Observado - pred.Predicho_Sin_Clima)
            pred["Error_Absoluto_Con_Clima"] = abs(pred.Observado - pred.Predicho_Con_Clima)
            fold["Ganancia_MAE"] = fold.MAE_Sin_Clima - fold.MAE_Con_Clima
            fold["Ganancia_RMSE"] = np.sqrt(fold.MSE_Sin_Clima) - np.sqrt(fold.MSE_Con_Clima)
            size = fold.N.to_numpy()[draws]
            boot_gain = np.sqrt((size * fold.MSE_Sin_Clima.to_numpy()[draws]).sum(axis=1) / size.sum(axis=1)) - np.sqrt(
                (size * fold.MSE_Con_Clima.to_numpy()[draws]).sum(axis=1) / size.sum(axis=1))
            low, high = np.quantile(boot_gain, [.025, .975])
            y = pred.Observado.to_numpy()
            rmse0 = np.sqrt(mean_squared_error(y, pred.Predicho_Sin_Clima))
            rmse1 = np.sqrt(mean_squared_error(y, pred.Predicho_Con_Clima))
            gain_mae = fold.Ganancia_MAE.to_numpy()
            pvalue = 1. if np.allclose(gain_mae, 0) else float(wilcoxon(gain_mae, method="auto").pvalue)
            rows.append(dict(Modelo=model_name, Base_Productiva=family,
                N_Observaciones=len(pred), N_Anios=n, Primer_Anio=int(eval_years.min()), Ultimo_Anio=int(eval_years.max()),
                RMSE_Sin_Clima=rmse0, RMSE_Con_Clima=rmse1, Ganancia_RMSE=rmse0-rmse1,
                Ganancia_RMSE_IC95_Inferior=low, Ganancia_RMSE_IC95_Superior=high,
                Reduccion_RMSE_Porcentaje=100*(rmse0-rmse1)/rmse0 if rmse0 else np.nan,
                MAE_Sin_Clima=mean_absolute_error(y, pred.Predicho_Sin_Clima),
                MAE_Con_Clima=mean_absolute_error(y, pred.Predicho_Con_Clima),
                R2_Sin_Clima=r2_score(y, pred.Predicho_Sin_Clima), R2_Con_Clima=r2_score(y, pred.Predicho_Con_Clima),
                RMSE_Persistencia=np.sqrt(mean_squared_error(y, pred.Predicho_Persistencia)),
                Anios_Mejora_MAE=int((gain_mae>0).sum()), Wilcoxon_p=pvalue,
                Alcance="Ganancia positiva favorece clima observado; mismos modelos/hiperparámetros/filas/folds. IC por bloques anuales OOT, modelos fijos; pocos años; sin causalidad ni pronóstico ex ante."))
            predictions.extend(pred.to_dict("records"))
            annual.extend(fold.to_dict("records"))
    summary = pd.DataFrame(rows)
    summary["Wilcoxon_p_Holm"] = multipletests(summary.Wilcoxon_p, method="holm")[1]
    return summary, pd.DataFrame(predictions), pd.DataFrame(annual)


def comparar_dtw_y_sin_alineamiento(ctx, params):
    """Mismos datos y normalización; w=0 es RMS euclídeo sobre años comunes."""
    rend = ctx["piv_rend"]
    areas = ctx["df"].pivot(index="Departamento", columns="Anio", values="Area (ha)").reindex(
        index=rend.index, columns=rend.columns)
    tables, assignments, perturbations, groups = [], [], [], []
    for geometry in ("escala", "forma"):
        options = dict(params, GEOMETRIA_ADAPTATIVA=geometry, ALGORITMOS_ADAPTATIVOS=["pam"],
                       GRILLA_CLUSTERS_K=[2, 3, 4], K_CLUSTERS_FIJO=4, ALGORITMO_CLUSTER_FIJO="pam",
                       VENTANA_SAKOE_CHIBA=0)
        sync, _ = matriz_bivariada_observada(rend, areas, options)
        print(f"      -> Comparación DTW / sin alineamiento: {geometry}, PAM k=2/3/4", flush=True)
        baseline = seleccionar_clustering_adaptativo(rend, areas, options, matriz_base=sync, informar=False)
        dtw_result = ctx["comparacion_geometrias"]["resultados"][geometry]
        dtw = ctx["comparacion_geometrias"]["matrices"][geometry]
        # La diagonal temporal siempre es una ruta DTW admisible.
        if np.any(dtw > sync + 1e-10):
            raise ValueError("DTW supera la ruta diagonal: normalización o soporte inconsistentes.")
        labels_by_k = {k: {"DTW": ajustar_particion(dtw, "pam", k, params),
                          "Sin_alineamiento": ajustar_particion(sync, "pam", k, params)} for k in (2,3,4)}
        for metric, result in [("DTW", dtw_result), ("Sin_alineamiento", baseline)]:
            table = result["seleccion"].query("Algoritmo == 'pam' and k in [2,3,4]").copy()
            table["Geometria"], table["Distancia"] = geometry, metric
            table["Elegida_Para_Paper"] = (metric == "DTW") & table.k.eq(4) & (geometry == params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"])
            table = table.drop(columns=["Seleccionada"])
            for index, row in table.iterrows():
                k = int(row.k)
                labels = labels_by_k[k][metric]
                table.loc[index, "ARI_DTW_vs_SinAlineamiento"] = adjusted_rand_score(labels_by_k[k]["DTW"], labels_by_k[k]["Sin_alineamiento"])
                table.loc[index, "Silhouette_Sobre_DTW"] = silhouette_score(dtw, labels, metric="precomputed")
                table.loc[index, "Silhouette_Sobre_SinAlineamiento"] = silhouette_score(sync, labels, metric="precomputed")
                order = pd.DataFrame({"c": labels, "mean": rend.mean(axis=1).to_numpy()}).groupby("c")["mean"].mean().sort_values().index
                names = {old:f"C{new}" for new,old in enumerate(order,1)}
                for department, label in zip(rend.index, labels):
                    assignments.append(dict(Geometria=geometry, Distancia=metric, k=k,
                        Departamento=department, Grupo=names[label]))
            tables.append(table)
            for key, target in [("perturbaciones", perturbations), ("grupos", groups)]:
                sub = result[key].query("Algoritmo == 'pam' and k in [2,3,4]").copy()
                sub["Geometria"], sub["Distancia"] = geometry, metric
                target.append(sub)
    return (pd.concat(tables, ignore_index=True), pd.DataFrame(assignments),
            pd.concat(perturbations, ignore_index=True), pd.concat(groups, ignore_index=True))


def ejecutar_validaciones_adicionales(ctx, params):
    import matplotlib.pyplot as plt
    print("[VALIDACIONES] Aporte incremental del clima; PAM 2/3/4 y referencia sin alineamiento...", flush=True)
    climate, predictions, annual = comparar_aporte_clima(ctx["df"], params)
    partitions, assignments, perturbations, groups = comparar_dtw_y_sin_alineamiento(ctx, params)
    outputs = {
        "T23_Aporte_Incremental_Clima": climate, "T24_Predicciones_Ablacion_Clima": predictions,
        "T25_Errores_Anuales_Ablacion": annual, "T26_PAM_234_DTW_vs_SinAlinear": partitions,
        "T27_Asignaciones_Comparativas": assignments, "T28_Perturbaciones_Comparativas": perturbations,
        "T29_Estabilidad_Grupos_Comparada": groups}
    for name, table in outputs.items():
        table.to_csv(Path(ctx["dir_tablas"])/f"{name}.csv", index=False, encoding="utf-8-sig")
    ctx["tablas_validacion_adicional"] = outputs
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in climate.reset_index(drop=True).iterrows():
        color = "#2A9D8F" if row.Base_Productiva == "Area" else "#1D3557"
        ax.plot([row.Ganancia_RMSE_IC95_Inferior, row.Ganancia_RMSE_IC95_Superior], [i,i], color=color, lw=2)
        ax.plot(row.Ganancia_RMSE, i, "o", color=color)
    ax.set_yticks(range(len(climate)), [f"{r.Modelo.replace('(Lineal)', '(Linear)')} | {'Lagged area only' if r.Base_Productiva=='Area' else 'Lagged area + yield'}" for r in climate.itertuples()])
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("RMSE without climate − RMSE with observed climate (t ha$^{-1}$)\nPositive values favor adding climate; conditional 95% year-block bootstrap intervals")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.25)
    guardar_figura_segura(fig, ctx["dir_figuras"], "Fig10_Aporte_Incremental_Clima", params["DPI_FIGURAS"], plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, geom in zip(axes, ("escala", "forma")):
        for metric, color in [("DTW", "#1D3557"), ("Sin_alineamiento", "#2A9D8F")]:
            sub = partitions.query("Geometria == @geom and Distancia == @metric").sort_values("k")
            name = "DTW" if metric == "DTW" else "No temporal alignment"
            ax.plot(sub.k, sub.ARI_Temporal_P10, "o-", color=color, label=name + " | ARI P10")
            ax.plot(sub.k, sub.Jaccard_Min_Grupo_P10, "s--", color=color, label=name + " | min. Jaccard P10")
        ax.axhline(params["ARI_P10_MINIMO"], color="#888888", linestyle=":", label="Configured ARI threshold")
        ax.set_title("Production scale" if geom == "escala" else "Trajectory shape")
        ax.set_xticks([2,3,4]); ax.set_xlabel("Number of PAM clusters (k)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Stability under temporal perturbations (P10)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=7)
    guardar_figura_segura(fig, ctx["dir_figuras"], "Fig11_PAM_234_DTW_vs_SinAlineamiento", params["DPI_FIGURAS"], plt)
    principal = partitions[partitions.Geometria.eq(params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"])]
    area_only = climate[climate.Base_Productiva.eq("Area")]
    with_lag = climate[climate.Base_Productiva.eq("Area_y_rendimiento_previo")]
    best_climate = float(climate.RMSE_Con_Clima.min())
    persistence = float(climate.RMSE_Persistencia.iloc[0])
    findings = (
        f"Sin rendimiento previo, clima mejora el RMSE observado en {int((area_only.Ganancia_RMSE>0).sum())}/{len(area_only)} modelos; "
        f"IC de ganancia completamente positivo en {int((area_only.Ganancia_RMSE_IC95_Inferior>0).sum())}.\n"
        f"Con rendimiento previo, mejora el RMSE en {int((with_lag.Ganancia_RMSE>0).sum())}/{len(with_lag)} modelos; "
        f"IC completamente positivo en {int((with_lag.Ganancia_RMSE_IC95_Inferior>0).sum())}.\n"
        f"Contrastes de MAE con p Holm <0.05: {int((climate.Wilcoxon_p_Holm<.05).sum())}/{len(climate)}.\n"
        f"RMSE del predictor ingenuo Y(t-1): {persistence:.5f} t/ha; "
        f"menor RMSE con clima: {best_climate:.5f} t/ha.\n"
        + ("Ningún modelo con clima supera la persistencia en RMSE agrupado observado.\n" if best_climate>=persistence else
           "Hay al menos un modelo con clima con menor RMSE agrupado observado que persistencia; no implica superioridad inferencial.\n"))
    summary = (
        "VALIDACIONES ADICIONALES PARA EL PAPER\n"
        + findings + "\n"
        "1. CLIMA INCREMENTAL. Mismos algoritmos, hiperparámetros, años y observaciones.\n"
        "Ganancia RMSE positiva significa menor error al añadir clima observado.\n"
        + climate[["Modelo","Base_Productiva","RMSE_Sin_Clima","RMSE_Con_Clima","Ganancia_RMSE",
                   "Ganancia_RMSE_IC95_Inferior","Ganancia_RMSE_IC95_Superior","Wilcoxon_p_Holm"]].round(5).to_string(index=False) + "\n\n"
        "Los p-valores comparan MAE anual pareado; Holm se aplica a las diez ablaciones.\n"
        "IC de ganancia RMSE: bloques anuales OOT (todos los departamentos del año),\n"
        "condicionados a predicciones fijas; seis años limitan inferencia y cobertura.\n"
        "No atribuye causalidad ni evalúa pronóstico con clima futuro desconocido.\n\n"
        "2. PAM k=2/3/4 Y DISTANCIA SIN ALINEAMIENTO.\n"
        + principal[["Distancia","k","Silhouette","ARI_Temporal_P10","Jaccard_Min_Grupo_P10","ARI_Cobertura_Comun","Cumple_Criterios","ARI_DTW_vs_SinAlineamiento"]].round(5).to_string(index=False)
        + "\n\nT26 contiene ambas geometrías; T27 las asignaciones; T28/T29 las perturbaciones y estabilidad por grupo.\n"
        "La referencia sin alineamiento es RMS euclídeo entre los mismos años compartidos;\n"
        "se reajusta la misma normalización en cada perturbación. DTW admite la ruta diagonal.\n"
        "Los silhouettes internos de distintas distancias no se interpretan como ranking común.\n"
        "T26 evalúa adicionalmente ambas particiones sobre cada matriz de referencia compartida.\n"
        "Las diferencias de estabilidad y asignación describen el aporte del alineamiento,\n"
        "sin una verdad externa de clusters. No garantizan ventaja predictiva o causal.\n"
        "k=4 se fijó tras explorar los datos; T26 conserva k=2/3, sin ocultar soluciones más estables.\n")
    Path(ctx["dir_result"], "VALIDACIONES_ADICIONALES.txt").write_text(summary, encoding="utf-8")
    ctx["resumen_validaciones_adicionales"] = summary
    return ctx
