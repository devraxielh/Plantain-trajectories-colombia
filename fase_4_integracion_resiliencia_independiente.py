"""
====================================================================================================
FASE 4 (INTEGRACIÓN): INDICADORES INDEPENDIENTES DE RESILIENCIA CLIMÁTICA (RESISTENCIA Rs,
                      RECUPERACIÓN Rc Y ESTABILIDAD PRODUCTIVA St) Y CONSOLIDACIÓN DESCRIPTIVA
====================================================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from fase_0_ingesta_y_clima import TRADUCCION_ENSO_EN, configurar_estilo_publicacion, formatear_depto_titulo, guardar_figura_segura, repel_labels_2d
from fase_1_clustering_dtw_y_topologia import mann_kendall_sen_test


def calcular_indicadores_independientes_resiliencia(df_full_shap: pd.DataFrame, params: dict) -> pd.DataFrame:
    u_z = params["UMBRAL_CHOQUE_CLIMATICO_Z"]
    u_oni = params["UMBRAL_CHOQUE_ONI_ABS"]
    w_rs = params["PESO_INDICE_RESISTENCIA_RS"]
    w_rc = params["PESO_INDICE_RECUPERACION_RC"]
    w_st = params["PESO_INDICE_ESTABILIDAD_ST"]
    if min(w_rs, w_rc, w_st) < 0 or not np.isclose(w_rs + w_rc + w_st, 1):
        raise ValueError("Los pesos de resiliencia deben ser no negativos y sumar 1.")
    alpha_stat = params["ALPHA_SIGNIFICANCIA_ESTADISTICA"]

    rows_res = []
    for depto, grp in df_full_shap.groupby("Departamento"):
        grp = grp.sort_values("Anio").reset_index(drop=True)
        y_vals = grp["Rendimiento (ton/ha)"].values
        t_vals = grp["Anio"].values
        n_y = len(y_vals)

        mk_info = mann_kendall_sen_test(y_vals, t_vals, alpha=alpha_stat)
        slope_sen = mk_info["Sen_Slope (unid/anio)"]
        intercept_sen = np.nanmedian(y_vals - slope_sen * (t_vals - t_vals[0]))
        y_trend = intercept_sen + slope_sen * (t_vals - t_vals[0])
        residuos = y_vals - y_trend
        std_res = float(np.nanstd(residuos, ddof=1)) if np.isfinite(y_vals).sum() >= 3 else np.nan
        media_y = float(np.nanmean(y_vals))
        estabilidad_st = media_y / std_res if std_res > 0 else np.nan
        cv_destendenciado_pct = std_res / media_y * 100 if media_y > 0 else np.nan

        mask_choque = (
            (grp["Anomalia_Precip_Depto (Z)"].abs() >= u_z)
            | (grp["Anomalia_Temp_Max_Depto (Z)"].abs() >= u_z)
            | (grp["ONI_Media_Anual (C)"].abs() >= u_oni)
        ).values

        rs_list, rc_list = [], []
        for idx_t in range(1, n_y - 1):
            if mask_choque[idx_t] and t_vals[idx_t] - t_vals[idx_t - 1] == 1 and t_vals[idx_t + 1] - t_vals[idx_t] == 1:
                y_pre = y_vals[idx_t - 1]
                y_shock = y_vals[idx_t]
                y_post = y_vals[idx_t + 1]
                if np.isfinite([y_pre, y_shock, y_post]).all() and y_pre > 0 and y_shock > 0:
                    rs_list.append(y_shock / y_pre)
                    rc_list.append(y_post / y_shock)

        resistencia_rs = float(np.median(rs_list)) if rs_list else np.nan
        recuperacion_rc = float(np.median(rc_list)) if rc_list else np.nan
        resiliencia_relativa_rr = float(np.median(np.asarray(rs_list) * np.asarray(rc_list))) if rs_list else np.nan

        rend_nino = grp[grp["Fase_ENSO"] == "El Niño"]["Rendimiento (ton/ha)"].mean()
        rend_nina = grp[grp["Fase_ENSO"] == "La Niña"]["Rendimiento (ton/ha)"].mean()

        rows_res.append({
            "Departamento": depto,
            "Cluster_DTW": grp["Cluster_DTW_Principal"].iloc[0],
            "Resistencia_Observada_Rs (Y_shock/Y_pre)": round(resistencia_rs, 3),
            "Recuperacion_Observada_Rc (Y_post/Y_shock)": round(recuperacion_rc, 3),
            "Resiliencia_Observada_Rr (Y_post/Y_pre)": round(resiliencia_relativa_rr, 3),
            "N_Eventos_Validos": len(rs_list),
            "N_Observaciones_SHAP_OOT": int(grp["SHAP_Efecto_Climatico_Neto (ton/ha)"].notna().sum()),
            "Alcance": "Ratios descriptivos de años adyacentes; no atribución causal; años adyacentes pueden tener choques",
            "Estabilidad_Destendenciada_St (Media/Std_Res)": round(estabilidad_st, 2),
            "CV_Residual_Destendenciado (%)": round(cv_destendenciado_pct, 2),
            "Rend_Medio_El_Nino (ton/ha)": round(float(rend_nino), 2) if pd.notna(rend_nino) else np.nan,
            "Rend_Medio_La_Nina (ton/ha)": round(float(rend_nina), 2) if pd.notna(rend_nina) else np.nan,
            "SHAP_Efecto_Climatico_Neto (ton/ha)": round(float(grp["SHAP_Efecto_Climatico_Neto (ton/ha)"].mean()), 3),
            "SHAP_Sensibilidad_Interanual_Std (ton/ha)": round(float(grp["SHAP_Choque_ENSO_y_Anomalias (ton/ha)"].std()), 3),
        })

    df_res = pd.DataFrame(rows_res)
    for col_z in ["Resistencia_Observada_Rs (Y_shock/Y_pre)", "Recuperacion_Observada_Rc (Y_post/Y_shock)", "Estabilidad_Destendenciada_St (Media/Std_Res)"]:
        mn, mx = df_res[col_z].min(), df_res[col_z].max()
        df_res[f"_norm_{col_z}"] = (df_res[col_z] - mn) / max(mx - mn, 1e-6)

    df_res["Indice_Resiliencia_Independiente (0-100)"] = (
        (
            w_rs * df_res["_norm_Resistencia_Observada_Rs (Y_shock/Y_pre)"]
            + w_rc * df_res["_norm_Recuperacion_Observada_Rc (Y_post/Y_shock)"]
            + w_st * df_res["_norm_Estabilidad_Destendenciada_St (Media/Std_Res)"]
        ) * 100.0
    ).round(1)

    cols_drop = [c for c in df_res.columns if c.startswith("_norm_")]
    return df_res.drop(columns=cols_drop).sort_values("Indice_Resiliencia_Independiente (0-100)", ascending=False)


def ejecutar_fase_4(ctx: dict, params: dict) -> dict:
    print("[FASE 4] Calculando Indicadores Independientes de Resiliencia (Rs, Rc, St) y Consolidando Resultados...")

    df = ctx["df"]
    X_bio = ctx["X_bio"]
    features_xai = ctx["features_xai"]
    shap_values = ctx["shap_values"]
    slug_cultivo = ctx["slug_cultivo"]
    dir_datasets = ctx["dir_datasets"]
    dir_tablas = ctx["dir_tablas"]
    dir_figuras = ctx["dir_figuras"]
    dpi = params["DPI_FIGURAS"]
    paleta_c = params["PALETA_CLUSTERS"]
    nombres_c = params["NOMBRES_CLUSTERS"]
    paleta_enso = params["PALETA_ENSO"]

    df_shap = pd.DataFrame(shap_values, index=X_bio.index, columns=[f"SHAP_{c}" for c in features_xai])
    df_full_shap = pd.concat(
        [df.loc[X_bio.index, ["Anio", "Departamento", "Cluster_DTW_Principal", "Fase_ENSO", "Rendimiento (ton/ha)", "Brecha_Rendimiento_Depto (ton/ha)"]], X_bio, df_shap],
        axis=1,
    )
    cols_shap_clima = [f"SHAP_{c}" for c in features_xai if c != "Area_Lag1 (ha)"]
    df_full_shap["SHAP_Efecto_Climatico_Neto (ton/ha)"] = df_full_shap[cols_shap_clima].sum(axis=1)
    df_full_shap["SHAP_Choque_ENSO_y_Anomalias (ton/ha)"] = df_full_shap[
        ["SHAP_Anomalia_Temp_Max_Depto (Z)", "SHAP_Anomalia_Precip_Depto (Z)", "SHAP_ONI_Lag1 (C)", "SHAP_ONI_Media_Anual (C)"]
    ].sum(axis=1)

    # Indicadores observados: toda la historia original; SHAP sólo existe en los años OOT.
    columnas_efecto = ["SHAP_Efecto_Climatico_Neto (ton/ha)", "SHAP_Choque_ENSO_y_Anomalias (ton/ha)"]
    df_res_input = df.join(df_full_shap[columnas_efecto])
    df_resiliencia = calcular_indicadores_independientes_resiliencia(df_res_input, params)
    df_resiliencia.to_csv(os.path.join(dir_tablas, "T09_Indicadores_Independientes_Resiliencia_Rs_Rc_St.csv"), index=False, encoding="utf-8-sig")

    # Exportar Dataset Training Ready en 'result/datasets/'
    ruta_training_xlsx = os.path.join(dir_datasets, f"{slug_cultivo}_TRAINING_READY.xlsx")
    cols_hoja_xai = (
        ["Anio", "Departamento", "Split_Temporal", "Cluster_DTW_Principal", "Cluster_Jerarquico_Average", "Cluster_Jerarquico_Complete", "Estado_Markov_t", "Estado_Markov_Lag1", "Fase_ENSO"]
        + features_xai
        + ["Rendimiento_Lag1 (ton/ha)", "Rendimiento (ton/ha)", "Brecha_Rendimiento_Depto (ton/ha)"]
    )
    df_xai_export = df[cols_hoja_xai].copy()
    matriz_dtw_dist_df = pd.DataFrame(ctx["D_multi"].round(3), index=ctx["deptos_bal"], columns=ctx["deptos_bal"])

    with pd.ExcelWriter(ruta_training_xlsx, engine="openpyxl") as writer:
        df_xai_export.to_excel(writer, sheet_name="01_Panel_Refs_Descriptivas", index=False)
        ctx["df_shap_global"].to_excel(writer, sheet_name="02_Variables_Seleccionadas", index=False)
        ctx["clusters_depto"].to_excel(writer, sheet_name="03_Clusters_KMed_Avg_Comp", index=False)
        ctx["piv_rend"].round(2).to_excel(writer, sheet_name="04_DTW_Rendimiento_TonHa")
        ctx["piv_rend_z"].round(3).to_excel(writer, sheet_name="05_DTW_Rendimiento_ZScore")
        ctx["piv_area_idx"].round(3).to_excel(writer, sheet_name="06_DTW_Area_IndiceBase")
        matriz_dtw_dist_df.to_excel(writer, sheet_name="07_Matriz_Distancias_DTW")
        ctx["P_global_df"].round(4).to_excel(writer, sheet_name="08_Markov_Matriz_Transicion")
        ctx["df_trans"][["Anio", "Departamento", "Cluster_DTW_Principal", "Fase_ENSO", "ONI_Media_Anual (C)", "Estado_Markov_Lag1", "Estado_Markov_t"]].to_excel(
            writer, sheet_name="09_Markov_Secuencia_Anual", index=False
        )
        ctx["df_cobertura"].to_excel(writer, sheet_name="10_Cobertura_Departamentos", index=False)
        ctx["df_transiciones_depto"].to_excel(writer, sheet_name="11_Perfiles_y_Transiciones", index=False)
        ctx["df_markov_clusters"].to_excel(writer, sheet_name="12_Markov_por_Perfil", index=False)
        pd.DataFrame(ctx["N_compartidos"], index=ctx["deptos_bal"], columns=ctx["deptos_bal"]).to_excel(
            writer, sheet_name="13_DTW_Anios_Compartidos")
        ctx["df_seleccion_adaptativa"].to_excel(writer, sheet_name="14_Seleccion_Adaptativa", index=False)
        ctx["df_estabilidad_grupos"].to_excel(writer, sheet_name="15_Estabilidad_Grupos", index=False)
        ctx["comparacion_geometrias"]["comparacion"].to_excel(writer, sheet_name="16_Escala_vs_Forma", index=False)
        ctx["comparacion_geometrias"]["asignaciones"].to_excel(writer, sheet_name="17_Asignaciones_Escala_Forma", index=False)
        pd.DataFrame(ctx["D_escala"], index=ctx["deptos_bal"], columns=ctx["deptos_bal"]).to_excel(writer, sheet_name="18_DTW_Escala")
        pd.DataFrame(ctx["D_forma"], index=ctx["deptos_bal"], columns=ctx["deptos_bal"]).to_excel(writer, sheet_name="19_DTW_Forma")
        ctx["df_exclusiones"].to_excel(writer, sheet_name="20_Exclusiones", index=False)

    # Exportar Libro Maestro de Resultados y Validaciones Estadísticas en 'result/tablas/'
    ruta_excel_res = os.path.join(dir_tablas, "RESULTADOS_METRICAS_Y_VALIDACIONES.xlsx")
    with pd.ExcelWriter(ruta_excel_res, engine="openpyxl") as writer:
        ctx["df_exclusiones"].to_excel(writer, sheet_name="T00_Exclusiones", index=False)
        ctx["df_grid_dtw"].to_excel(writer, sheet_name="T01_Validacion_DTW_SinWard", index=False)
        ctx["df_mk"].to_excel(writer, sheet_name="T02_MannKendall_SenSlope", index=False)
        ctx["df_grafo"].to_excel(writer, sheet_name="T03_Topologia_Grafo_DTW", index=False)
        ctx["df_kw_clusters"].to_excel(writer, sheet_name="T04_Kruskal_ANOVA_Clusters", index=False)
        ctx["df_markov_ergodico"].to_excel(writer, sheet_name="T05_Markov_Ergodico_IC95", index=False)
        ctx["df_diagnostico_supuestos_markov"].to_excel(writer, sheet_name="T06_Markov_Supuestos_AG_Test", index=False)
        ctx["df_bench"].to_excel(writer, sheet_name="T07_Bench_RollingOrigin_CV", index=False)
        ctx["df_shap_global"].to_excel(writer, sheet_name="T08_SHAP_PuntosCambio_IC95", index=False)
        df_resiliencia.to_excel(writer, sheet_name="T09_Resiliencia_Rs_Rc_St", index=False)
        df_full_shap.round(4).to_excel(writer, sheet_name="T10_Matriz_SHAP_Completa", index=False)
        ctx["df_cobertura"].to_excel(writer, sheet_name="T11_Cobertura_Total", index=False)
        ctx["df_perfiles"].to_excel(writer, sheet_name="T12_Perfiles_Productivos", index=False)
        ctx["df_transiciones_depto"].to_excel(writer, sheet_name="T13_Transiciones_Departamento", index=False)
        ctx["df_markov_clusters"].to_excel(writer, sheet_name="T14_Markov_por_Perfil", index=False)
        pd.DataFrame(ctx["N_compartidos"], index=ctx["deptos_bal"], columns=ctx["deptos_bal"]).to_excel(
            writer, sheet_name="T15_DTW_Anios_Compartidos")
        ctx["df_seleccion_adaptativa"].to_excel(writer, sheet_name="T16_Seleccion_Adaptativa", index=False)
        ctx["df_perturbaciones"].to_excel(writer, sheet_name="T17_Perturbaciones_Temporales", index=False)
        ctx["df_estabilidad_grupos"].to_excel(writer, sheet_name="T18_Estabilidad_por_Grupo", index=False)
        ctx["comparacion_geometrias"]["comparacion"].to_excel(writer, sheet_name="T19_Escala_vs_Forma", index=False)
        ctx["comparacion_geometrias"]["asignaciones"].to_excel(writer, sheet_name="T20_Asignaciones_Escala_Forma", index=False)
        pd.DataFrame(ctx["D_escala"], index=ctx["deptos_bal"], columns=ctx["deptos_bal"]).to_excel(writer, sheet_name="T21_DTW_Escala")
        pd.DataFrame(ctx["D_forma"], index=ctx["deptos_bal"], columns=ctx["deptos_bal"]).to_excel(writer, sheet_name="T22_DTW_Forma")
        for nombre, tabla in ctx.get("tablas_validacion_adicional", {}).items():
            hoja = {"T29_Estabilidad_Grupos_Comparada": "T29_Estabilidad_Comparada"}.get(nombre, nombre)
            tabla.to_excel(writer, sheet_name=hoja, index=False)

    # Figura 8: Indicadores Independientes de Resiliencia (Rs, Rc, St) vs Clusters DTW y SHAP (Academic English, No Titles)
    configurar_estilo_publicacion(plt)
    fig8, axes8 = plt.subplots(1, 3, figsize=(16.8, 5.4), gridspec_kw={"width_ratios": [1.22, 0.92, 1.06], "wspace": 0.28})
    orden_c = list(paleta_c)
    etiquetas_c_cortas = [f"C{i}" for i in range(1, len(orden_c) + 1)]
    df_res_valid = df_resiliencia[df_resiliencia["Cluster_DTW"].isin(orden_c)].dropna(subset=[
        "Resistencia_Observada_Rs (Y_shock/Y_pre)", "Recuperacion_Observada_Rc (Y_post/Y_shock)",
        "Estabilidad_Destendenciada_St (Media/Std_Res)"]).copy().reset_index(drop=True)

    if len(df_res_valid):
        xs_r = df_res_valid["Resistencia_Observada_Rs (Y_shock/Y_pre)"].values.astype(float)
        ys_r = df_res_valid["Recuperacion_Observada_Rc (Y_post/Y_shock)"].values.astype(float)
        st_r = df_res_valid["Estabilidad_Destendenciada_St (Media/Std_Res)"].values.astype(float)
        textos_r = [formatear_depto_titulo(d) for d in df_res_valid["Departamento"]]

        pad_rx = (xs_r.max() - xs_r.min()) * 0.22
        pad_ry = (ys_r.max() - ys_r.min()) * 0.18
        xlim_r = (float(xs_r.min() - pad_rx), float(xs_r.max() + pad_rx))
        ylim_r = (float(ys_r.min() - pad_ry), float(ys_r.max() + pad_ry))
        axes8[0].set_xlim(xlim_r)
        axes8[0].set_ylim(ylim_r)

        for cl_key in orden_c:
            sub_r = df_res_valid[df_res_valid["Cluster_DTW"] == cl_key]
            if len(sub_r) > 0:
                axes8[0].scatter(
                    sub_r["Resistencia_Observada_Rs (Y_shock/Y_pre)"],
                    sub_r["Recuperacion_Observada_Rc (Y_post/Y_shock)"],
                    s=sub_r["Estabilidad_Destendenciada_St (Media/Std_Res)"] * 16 + 42,
                    color=paleta_c[cl_key], edgecolor="black", linewidths=0.9, alpha=0.88, zorder=3,
                )

        lx_r, ly_r = repel_labels_2d(
            xs_r, ys_r, textos_r, xlim_r, ylim_r,
            char_w_frac=0.0135, box_h_frac=0.048, node_r_frac=0.036, n_iter=600, seed=params["SEMILLA_ALEATORIA"],
        )
        span_rx = xlim_r[1] - xlim_r[0]
        span_ry = ylim_r[1] - ylim_r[0]
        for i, txt in enumerate(textos_r):
            dist_n = np.hypot((lx_r[i] - xs_r[i]) / span_rx, (ly_r[i] - ys_r[i]) / span_ry)
            if dist_n > 0.028:
                axes8[0].annotate(
                    "", xy=(xs_r[i], ys_r[i]), xytext=(lx_r[i], ly_r[i]),
                    arrowprops=dict(arrowstyle="-", color="#495057", lw=0.65, alpha=0.70, shrinkA=3, shrinkB=4),
                    zorder=4,
                )
            t_obj = axes8[0].text(
                lx_r[i], ly_r[i], txt, fontsize=7.3, fontweight="bold", ha="center", va="center", color="#1B1E23", zorder=5,
            )
            t_obj.set_path_effects([pe.withStroke(linewidth=2.3, foreground="white")])

    else:
        axes8[0].text(0.5, 0.5, "No valid event windows", ha="center", va="center", transform=axes8[0].transAxes)

    axes8[0].axvline(1.0, color="#6C757D", linestyle="--", linewidth=0.95, zorder=1)
    axes8[0].axhline(1.0, color="#6C757D", linestyle="--", linewidth=0.95, zorder=1)
    axes8[0].set_xlabel("Observed Resistance Index $R_s = Y_{\\mathrm{shock}} / Y_{\\mathrm{pre}}$")
    axes8[0].set_ylabel("Observed Recovery Index $R_c = Y_{\\mathrm{post}} / Y_{\\mathrm{shock}}$")
    axes8[0].grid(True, linestyle="--", alpha=0.30)
    axes8[0].text(-0.15, 1.02, "(a)", transform=axes8[0].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")

    handles_r = [
        Line2D([0], [0], marker="o", color="w", label=nombres_c[ck], markerfacecolor=paleta_c[ck], markeredgecolor="black", markersize=8.0)
        for ck in orden_c
    ]
    fig8.legend(
        handles=handles_r, loc="lower center", bbox_to_anchor=(0.33, 1.01), ncol=2,
        frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.96, fontsize=8.0,
        title="DTW Profiles (In panel a, marker size $\\propto$ Detrended Stability $S_t$)", title_fontsize=8.3,
    )

    sns.boxplot(
        data=df_res_valid, x="Cluster_DTW", y="Indice_Resiliencia_Independiente (0-100)",
        order=orden_c, hue="Cluster_DTW", hue_order=orden_c, legend=False,
        palette=paleta_c, ax=axes8[1], width=0.56, fliersize=0,
    )
    sns.stripplot(
        data=df_res_valid, x="Cluster_DTW", y="Indice_Resiliencia_Independiente (0-100)",
        order=orden_c, color="black", size=5.2, jitter=0.14, ax=axes8[1],
    )
    w_rs = int(params["PESO_INDICE_RESISTENCIA_RS"] * 100)
    w_rc = int(params["PESO_INDICE_RECUPERACION_RC"] * 100)
    w_st = int(params["PESO_INDICE_ESTABILIDAD_ST"] * 100)
    axes8[1].set_xticks(np.arange(len(orden_c)), etiquetas_c_cortas, fontweight="bold")
    axes8[1].set_xlabel("DTW Profile")
    axes8[1].set_ylabel(f"Composite Resilience Index ({w_rs}%$R_s$+{w_rc}%$R_c$+{w_st}%$S_t$)")
    axes8[1].grid(axis="y", linestyle="--", alpha=0.30)
    axes8[1].text(-0.16, 1.02, "(b)", transform=axes8[1].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")

    df_f8 = df_full_shap[df_full_shap["Cluster_DTW_Principal"].isin(orden_c)].copy()
    df_f8["ENSO_Phase_EN"] = df_f8["Fase_ENSO"].map(lambda x: TRADUCCION_ENSO_EN.get(x, x))
    orden_enso_en = ["La Niña", "Neutral", "El Niño"]
    paleta_enso_en = {TRADUCCION_ENSO_EN.get(k, k): v for k, v in paleta_enso.items()}
    sns.boxplot(
        data=df_f8, x="Cluster_DTW_Principal", y="SHAP_Efecto_Climatico_Neto (ton/ha)",
        hue="ENSO_Phase_EN", order=orden_c, hue_order=orden_enso_en, palette=paleta_enso_en, ax=axes8[2], fliersize=2.8, width=0.68,
    )
    axes8[2].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes8[2].set_xticks(np.arange(len(orden_c)), etiquetas_c_cortas, fontweight="bold")
    axes8[2].set_xlabel("DTW Profile")
    axes8[2].set_ylabel("OOT Agroclimatic TreeSHAP Contribution $\\sum \\phi_{\\mathrm{clim}}$ (t ha$^{-1}$)")
    axes8[2].grid(axis="y", linestyle="--", alpha=0.30)
    axes8[2].text(-0.15, 1.02, "(c)", transform=axes8[2].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    axes8[2].legend(title="ENSO Phase", loc="upper left", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=7.8, title_fontsize=8.2)

    guardar_figura_segura(fig8, dir_figuras, "Fig8_Resiliencia_Climatica_SHAP_por_Cluster_y_ENSO", dpi, plt)

    ctx["df_resiliencia"] = df_resiliencia
    return ctx
