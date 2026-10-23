"""
====================================================================================================
FASE 3: VALIDACIÓN TEMPORAL EXPANSIVA (ROLLING-ORIGIN SEPARANDO DEPARTAMENTOS POR AÑO),
        BENCHMARKING ML Y TREESHAP CON INTERVALOS DE INCERTIDUMBRE BOOTSTRAP (IC 95%)
====================================================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from xgboost import XGBRegressor
from fase_0_ingesta_y_clima import TRADUCCION_ENSO_EN, TRADUCCION_VARS_EN, configurar_estilo_publicacion, guardar_figura_segura, indices_bootstrap_panel


def calcular_vif(df_features: pd.DataFrame) -> pd.Series:
    X = df_features.dropna().to_numpy(dtype=float)
    if len(X) < 2:
        return pd.Series(np.nan, index=df_features.columns)
    valores = []
    for j in range(X.shape[1]):
        y = X[:, j]
        total = float(np.sum((y - y.mean()) ** 2))
        if total <= np.finfo(float).eps:
            valores.append(np.inf)
            continue
        otros = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        estimado = otros @ np.linalg.lstsq(otros, y, rcond=None)[0]
        residual = float(np.sum((y - estimado) ** 2))
        valores.append(np.inf if residual <= 1e-12 * total else max(1.0, total / residual))
    return pd.Series(valores, index=df_features.columns)


def preparar_fold(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Ajusta referencias climáticas con train; para departamentos nuevos usa la referencia train agrupada."""
    train, test = df_train.copy(), df_test.copy()
    for original, anomaly in [
        ("Precipitacion_Acumulada_Anual (mm)", "Anomalia_Precip_Depto (Z)"),
        ("Temp_Maxima_Media_Anual (C)", "Anomalia_Temp_Max_Depto (Z)"),
    ]:
        resumen = train.groupby("Departamento")[original].agg(["mean", "std"])
        media_global, std_global = train[original].mean(), train[original].std()
        if not np.isfinite(std_global) or std_global <= 0:
            std_global = 1.0
        for sub in (train, test):
            media = sub["Departamento"].map(resumen["mean"]).fillna(media_global)
            desvio = sub["Departamento"].map(resumen["std"]).replace(0, np.nan).fillna(std_global)
            sub[anomaly] = (sub[original] - media) / desvio
    return train, test


def construir_catalogo_modelos(params: dict) -> dict:
    seed = params["SEMILLA_ALEATORIA"]
    return {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=params["RF_N_ESTIMATORS"],
            max_depth=params["RF_MAX_DEPTH"],
            min_samples_leaf=params["RF_MIN_SAMPLES_LEAF"],
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=params["XGB_N_ESTIMATORS"],
            max_depth=params["XGB_MAX_DEPTH"],
            learning_rate=params["XGB_LEARNING_RATE"],
            subsample=params["XGB_SUBSAMPLE"],
            colsample_bytree=params["XGB_COLSAMPLE_BYTREE"],
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=params["LGBM_N_ESTIMATORS"],
            max_depth=params["LGBM_MAX_DEPTH"],
            num_leaves=params["LGBM_NUM_LEAVES"],
            learning_rate=params["LGBM_LEARNING_RATE"],
            subsample=params["LGBM_SUBSAMPLE"],
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=params["GBM_N_ESTIMATORS"],
            max_depth=params["GBM_MAX_DEPTH"],
            learning_rate=params["GBM_LEARNING_RATE"],
            subsample=params["GBM_SUBSAMPLE"],
            random_state=seed,
        ),
        "ElasticNet (Lineal)": lambda: make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=params["ELASTICNET_ALPHA"], l1_ratio=params["ELASTICNET_L1_RATIO"], random_state=seed),
        ),
    }


def validacion_temporal_expansiva_rolling_origin(df: pd.DataFrame, features_x: list, col_target: str, params: dict) -> tuple:
    features_dyn = list(dict.fromkeys(["Rendimiento_Lag1 (ton/ha)"] + features_x))
    # Evaluar ambas familias sobre exactamente las mismas observaciones válidas, sin imputar y.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_dyn + [col_target]).copy()
    constructores_modelos = construir_catalogo_modelos(params)
    anios_ordenados = sorted(df["Anio"].unique())
    n_folds_temp = params["ANIOS_EVALUACION_ROLLING_ORIGIN"]
    if not 3 <= n_folds_temp <= len(anios_ordenados) - 2:
        raise ValueError("Se requieren al menos tres años de evaluación y dos años previos de entrenamiento.")
    anios_evaluacion = anios_ordenados[-n_folds_temp:]

    resultados_modelos = []
    errores_ro_por_modelo = {}
    predicciones_ultimos_anios = {}

    for nombre_mod, constructor in constructores_modelos.items():
        r2_ro_folds, rmse_ro_folds, mae_ro_folds, mape_ro_folds = [], [], [], []
        y_true_ro_all, y_pred_ro_all, anios_ro_all, deptos_ro_all, clusters_ro_all = [], [], [], [], []

        for t_eval in anios_evaluacion:
            df_tr, df_te = preparar_fold(df[df["Anio"] < t_eval], df[df["Anio"] == t_eval])

            m_ro = constructor()
            m_ro.fit(df_tr[features_x], df_tr[col_target])
            preds_t = m_ro.predict(df_te[features_x])
            y_t = df_te[col_target].values

            r2_ro_folds.append(r2_score(y_t, preds_t))
            rmse_ro_folds.append(np.sqrt(mean_squared_error(y_t, preds_t)))
            mae_ro_folds.append(mean_absolute_error(y_t, preds_t))
            mape_ro_folds.append(mean_absolute_percentage_error(y_t[y_t > 0], preds_t[y_t > 0]) * 100)

            y_true_ro_all.extend(y_t)
            y_pred_ro_all.extend(preds_t)
            anios_ro_all.extend(df_te["Anio"].values)
            deptos_ro_all.extend(df_te["Departamento"].values)
            clusters_ro_all.extend(df_te["Cluster_DTW_Principal"].values)

        y_true_dyn_all, y_pred_dyn_all, r2_dyn_folds = [], [], []
        for t_eval in anios_evaluacion:
            df_tr, df_te = preparar_fold(df[df["Anio"] < t_eval], df[df["Anio"] == t_eval])
            m_dyn = constructor()
            m_dyn.fit(df_tr[features_dyn], df_tr[col_target])
            preds_dyn = m_dyn.predict(df_te[features_dyn])
            r2_dyn_folds.append(r2_score(df_te[col_target], preds_dyn))
            y_true_dyn_all.extend(df_te[col_target].values)
            y_pred_dyn_all.extend(preds_dyn)

        y_true_ro_arr = np.array(y_true_ro_all)
        y_pred_ro_arr = np.array(y_pred_ro_all)
        y_true_dyn_arr = np.array(y_true_dyn_all)
        y_pred_dyn_arr = np.array(y_pred_dyn_all)

        # Bloques anuales, no 32 réplicas independientes de un mismo año climático.
        errores_ro_por_modelo[nombre_mod] = np.asarray(mae_ro_folds)
        predicciones_ultimos_anios[nombre_mod] = pd.DataFrame({
            "Anio": anios_ro_all,
            "Departamento": deptos_ro_all,
            "Cluster_DTW": clusters_ro_all,
            "Observado": y_true_dyn_arr,
            "Predicho_Biofisico": y_pred_ro_arr,
            "Predicho_Dinamico": y_pred_dyn_arr,
        })

        resultados_modelos.append({
            "Modelo_ML": nombre_mod,
            "RollingOrigin_Biofisico_R2_Media": round(float(np.mean(r2_ro_folds)), 4),
            "RollingOrigin_Biofisico_R2_Std": round(float(np.std(r2_ro_folds)), 4),
            "RollingOrigin_Biofisico_RMSE (ton/ha)": round(float(np.mean(rmse_ro_folds)), 3),
            "RollingOrigin_Biofisico_MAE (ton/ha)": round(float(np.mean(mae_ro_folds)), 3),
            "RollingOrigin_Biofisico_MAPE (%)": round(float(np.mean(mape_ro_folds)), 2),
            "RollingOrigin_Dinamico_R2": round(float(r2_score(y_true_dyn_arr, y_pred_dyn_arr)), 4),
            "RollingOrigin_Dinamico_R2_Media": round(float(np.mean(r2_dyn_folds)), 4),
            "RollingOrigin_Dinamico_R2_Std": round(float(np.std(r2_dyn_folds)), 4),
            "RollingOrigin_Dinamico_RMSE (ton/ha)": round(float(np.sqrt(mean_squared_error(y_true_dyn_arr, y_pred_dyn_arr))), 3),
            "RollingOrigin_Dinamico_MAE (ton/ha)": round(float(mean_absolute_error(y_true_dyn_arr, y_pred_dyn_arr)), 3),
            "RollingOrigin_Dinamico_MAPE (%)": round(float(mean_absolute_percentage_error(y_true_dyn_arr[y_true_dyn_arr > 0], y_pred_dyn_arr[y_true_dyn_arr > 0]) * 100), 2),
            "N_Anios_Evaluados": len(anios_evaluacion),
            "N_Observaciones_Evaluadas": len(y_true_ro_arr),
            "Alcance": "Complemento supervisado del clustering/Markov; validación temporal en departamentos conocidos, clima observado; no transferencia espacial",
        })

    df_bench = pd.DataFrame(resultados_modelos).sort_values("RollingOrigin_Biofisico_R2_Media", ascending=False)
    stat_friedman, p_friedman = stats.friedmanchisquare(*errores_ro_por_modelo.values())
    mejor_mod = df_bench.iloc[0]["Modelo_ML"]
    err_mejor = errores_ro_por_modelo[mejor_mod]

    wilcoxon_pvals = []
    for mod_name in df_bench["Modelo_ML"]:
        if mod_name == mejor_mod:
            wilcoxon_pvals.append(1.0)
        else:
            diferencias = err_mejor - errores_ro_por_modelo[mod_name]
            p_w = 1.0 if np.allclose(diferencias, 0) else stats.wilcoxon(diferencias).pvalue
            wilcoxon_pvals.append(float(p_w))

    df_bench["Wilcoxon_p_vs_Mejor (RollingOrigin)"] = wilcoxon_pvals
    from statsmodels.stats.multitest import multipletests
    comparaciones = df_bench["Modelo_ML"].ne(mejor_mod)
    df_bench["Wilcoxon_p_Holm"] = 1.0
    df_bench.loc[comparaciones, "Wilcoxon_p_Holm"] = multipletests(
        df_bench.loc[comparaciones, "Wilcoxon_p_vs_Mejor (RollingOrigin)"], method="holm")[1]
    df_bench["Friedman_Chi2_RollingOrigin"] = round(float(stat_friedman), 2)
    df_bench["Friedman_p_value"] = float(f"{p_friedman:.2e}")
    return df_bench, predicciones_ultimos_anios, anios_evaluacion, stat_friedman, p_friedman


def calcular_puntos_cambio_shap_con_incertidumbre(X_bio: pd.DataFrame, shap_values: np.ndarray, features_xai: list, params: dict, panel: pd.DataFrame) -> tuple:
    rng = np.random.default_rng(params["SEMILLA_ALEATORIA"])
    n_obs = len(X_bio)
    n_boot = params["REPLICAS_BOOTSTRAP_SHAP"]
    frac_lowess = params["FRACCION_SUAVIZADO_LOWESS"]
    n_grid = params["NODOS_GRILLA_LOWESS"]
    min_est = params["FRACCION_MINIMA_ESTABILIDAD_CRUCE"]

    shap_rows = []
    curvas_bootstrap = {}

    for i, col in enumerate(features_xai):
        sv_col = shap_values[:, i]
        x_col = X_bio[col].values
        mean_abs_shap = float(np.mean(np.abs(sv_col)))
        total_shap = np.sum(np.mean(np.abs(shap_values), axis=0))
        pct_shap = mean_abs_shap / total_shap * 100 if total_shap > 0 else 0.0
        corr_dir = float(np.corrcoef(x_col, sv_col)[0, 1]) if np.std(x_col) > 0 and np.std(sv_col) > 0 else np.nan

        def suavizar(x, y):
            # Media por nivel del predictor para no interpolar abscisas duplicadas.
            levels = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
            if len(levels) < 3:
                return levels.to_numpy()
            return lowess(levels["y"], levels["x"], frac=max(frac_lowess, 3 / len(levels)), return_sorted=True)

        x_grid = np.linspace(np.percentile(x_col, 2), np.percentile(x_col, 98), n_grid)
        sm_main = suavizar(x_col, sv_col)
        y_main_interp = np.interp(x_grid, sm_main[:, 0], sm_main[:, 1])

        y_boot_curves = np.zeros((n_boot, len(x_grid)))
        cruces_primarios_boot = []

        for b in range(n_boot):
            b_idx = indices_bootstrap_panel(panel, rng)
            sm_b = suavizar(x_col[b_idx], sv_col[b_idx])
            y_b_interp = np.interp(x_grid, sm_b[:, 0], sm_b[:, 1])
            y_boot_curves[b, :] = y_b_interp
            cambios_b = np.where(y_b_interp[:-1] * y_b_interp[1:] < 0)[0]
            if len(cambios_b) > 0:
                idx_cercano = cambios_b[np.argmin(np.abs(x_grid[cambios_b] - np.median(x_col)))]
                cruces_primarios_boot.append(float(x_grid[idx_cercano]))

        estabilidad_cruce_pct = (len(cruces_primarios_boot) / n_boot) * 100.0
        if len(cruces_primarios_boot) >= min_est * n_boot:
            med_cruce = float(np.median(cruces_primarios_boot))
            p25_cruce = float(np.percentile(cruces_primarios_boot, 2.5))
            p975_cruce = float(np.percentile(cruces_primarios_boot, 97.5))
            str_punto_cambio = f"{med_cruce:.2f} [IC95%: {p25_cruce:.2f}, {p975_cruce:.2f}]"
        else:
            med_cruce, p25_cruce, p975_cruce = np.nan, np.nan, np.nan
            str_punto_cambio = "Sin cruce estable; no implica monotonía"

        curvas_bootstrap[col] = {
            "x_grid": x_grid,
            "y_mean": y_main_interp,
            "y_low": np.percentile(y_boot_curves, 2.5, axis=0),
            "y_high": np.percentile(y_boot_curves, 97.5, axis=0),
            "med_cruce": med_cruce,
            "p25_cruce": p25_cruce,
            "p975_cruce": p975_cruce,
            "estabilidad_pct": estabilidad_cruce_pct,
        }

        shap_rows.append({
            "Variable_Predictora": col,
            "Mean_Abs_SHAP (ton/ha)": round(mean_abs_shap, 4),
            "Participacion_Explicativa_SHAP (%)": round(pct_shap, 2),
            "Correlacion_Direccional_con_SHAP": round(corr_dir, 3),
            "Punto_Cambio_Contribucion_Modelo (SHAP=0) [IC95%]": str_punto_cambio,
            "Estabilidad_Bootstrap_Punto_Cambio (%)": round(estabilidad_cruce_pct, 1),
            "Min_SHAP (ton/ha)": round(float(np.min(sv_col)), 3),
            "Max_SHAP (ton/ha)": round(float(np.max(sv_col)), 3),
            "Alcance_IC": "Bootstrap cruzado departamento/año, condicional a modelos OOT ajustados; sin incertidumbre de reentrenamiento ni interpretación causal",
        })

    df_shap_global = pd.DataFrame(shap_rows).sort_values("Mean_Abs_SHAP (ton/ha)", ascending=False)
    return df_shap_global, curvas_bootstrap


def ejecutar_fase_3(ctx: dict, params: dict) -> dict:
    print("[FASE 3] Ejecutando Validación Temporal Expansiva (Rolling-Origin) y Puntos de Cambio TreeSHAP con IC95%...")

    df = ctx["df"]
    nombre_cultivo = ctx["nombre_cultivo"]
    dir_tablas, dir_figuras = ctx["dir_tablas"], ctx["dir_figuras"]
    dpi = params["DPI_FIGURAS"]
    seed = params["SEMILLA_ALEATORIA"]
    paleta_c = params["PALETA_CLUSTERS"]
    nombres_c = params["NOMBRES_CLUSTERS"]
    paleta_enso = params["PALETA_ENSO"]
    features_xai = params["VARIABLES_PREDICTORAS_XAI"]

    df_bench, predicciones_ro, anios_ro_eval, stat_friedman, p_friedman = validacion_temporal_expansiva_rolling_origin(
        df, features_xai, "Rendimiento (ton/ha)", params
    )
    df_bench.to_csv(os.path.join(dir_tablas, "T07_Benchmarking_Validacion_Expansiva_RollingOrigin.csv"), index=False, encoding="utf-8-sig")

    validos = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(dict.fromkeys(features_xai + ["Rendimiento_Lag1 (ton/ha)", "Rendimiento (ton/ha)"])))
    xs_oot, shap_oot = [], []
    for anio in anios_ro_eval:
        train, test = preparar_fold(validos[validos["Anio"] < anio], validos[validos["Anio"] == anio])
        modelo_xai = construir_catalogo_modelos(params)["XGBoost"]()
        modelo_xai.fit(train[features_xai], train["Rendimiento (ton/ha)"])
        explainer = shap.TreeExplainer(modelo_xai)
        valores = explainer.shap_values(test[features_xai])
        predicciones = modelo_xai.predict(test[features_xai])
        if not np.allclose(np.sum(valores, axis=1) + explainer.expected_value, predicciones, atol=1e-4):
            raise ValueError("TreeSHAP no reconstruye las predicciones dentro de la tolerancia.")
        xs_oot.append(test[features_xai])
        shap_oot.append(valores)
    X_bio = pd.concat(xs_oot)
    shap_values = np.concatenate(shap_oot)
    df_xai = df.loc[X_bio.index]
    vif_clean = calcular_vif(X_bio)
    mi_clean = pd.Series(mutual_info_regression(X_bio, df_xai["Rendimiento (ton/ha)"], random_state=seed), index=features_xai)
    df_shap_global, curvas_boot_shap = calcular_puntos_cambio_shap_con_incertidumbre(
        X_bio, shap_values, features_xai, params, df_xai[["Departamento", "Anio"]])
    df_shap_global["Info_Mutua (MI)"] = [round(float(mi_clean[c]), 3) for c in df_shap_global["Variable_Predictora"]]
    df_shap_global["VIF (Multicolinealidad)"] = [round(float(vif_clean[c]), 2) for c in df_shap_global["Variable_Predictora"]]
    df_shap_global.to_csv(os.path.join(dir_tablas, "T08_SHAP_PuntosCambio_Contribucion_e_Incertidumbre_IC95.csv"), index=False, encoding="utf-8-sig")

    configurar_estilo_publicacion(plt)

    def _save(fig_obj, name):
        guardar_figura_segura(fig_obj, dir_figuras, name, dpi, plt)

    # --- FIGURA 5: Benchmarking Rolling-Origin (Estándar Q1 Journal - Academic English, No Titles) ---
    fig5, axes5 = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"wspace": 0.24})
    df_b_plot = df_bench.sort_values("RollingOrigin_Biofisico_R2_Media", ascending=True).copy()
    nombres_mod_en = {"ElasticNet (Lineal)": "ElasticNet (Linear)"}
    etiquetas_mod_en = [nombres_mod_en.get(m, m) for m in df_b_plot["Modelo_ML"]]
    y_m = np.arange(len(df_b_plot))
    axes5[0].barh(
        y_m - 0.18, df_b_plot["RollingOrigin_Biofisico_R2_Media"], height=0.34,
        xerr=df_b_plot["RollingOrigin_Biofisico_R2_Std"],
        label=f"Biophysical Rolling-Origin ({anios_ro_eval[0]}–{anios_ro_eval[-1]})", color="#2A9D8F", edgecolor="black", capsize=3,
    )
    axes5[0].barh(
        y_m + 0.18, df_b_plot["RollingOrigin_Dinamico_R2_Media"], height=0.34,
        xerr=df_b_plot["RollingOrigin_Dinamico_R2_Std"], capsize=3,
        label=f"Dynamic Rolling-Origin ({anios_ro_eval[0]}–{anios_ro_eval[-1]})", color="#1D3557", edgecolor="black",
    )
    axes5[0].set_yticks(y_m)
    axes5[0].set_yticklabels(etiquetas_mod_en, fontweight="bold")
    axes5[0].set_xlabel("Retrospective Out-of-Time $R^2$ (Observed Annual Climate)")
    axes5[0].set_xlim(min(0, float((df_b_plot["RollingOrigin_Biofisico_R2_Media"] - df_b_plot["RollingOrigin_Biofisico_R2_Std"]).min()) - 0.05), 1.06)
    axes5[0].grid(axis="x", linestyle="--", alpha=0.32)
    axes5[0].text(-0.18, 1.02, "(a)", transform=axes5[0].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    axes5[0].text(
        0.98, 0.21, f"Friedman $\\chi^2 = {stat_friedman:.2f},\\; p = {p_friedman:.2e}$",
        transform=axes5[0].transAxes, ha="right", va="bottom", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CED4DA", alpha=0.95),
    )
    axes5[0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=8.0)

    df_ro_best = predicciones_ro["XGBoost"]
    for cl_key, color_c in paleta_c.items():
        sub_o = df_ro_best[df_ro_best["Cluster_DTW"] == cl_key]
        if len(sub_o) > 0:
            axes5[1].scatter(
                sub_o["Observado"], sub_o["Predicho_Dinamico"], color=color_c, edgecolor="black",
                s=48, alpha=0.84, label=nombres_c[cl_key],
            )
    max_y_val = float(max(df_ro_best["Observado"].max(), df_ro_best["Predicho_Dinamico"].max()) * 1.12)
    lims = [0, max_y_val]
    axes5[1].plot(lims, lims, "k--", linewidth=1.3, label="1:1 Exact Identity ($y = \\hat{y}$)")
    r2_ro_dyn = r2_score(df_ro_best["Observado"], df_ro_best["Predicho_Dinamico"])
    rmse_ro_dyn = np.sqrt(mean_squared_error(df_ro_best["Observado"], df_ro_best["Predicho_Dinamico"]))
    axes5[1].set_xlim(lims)
    axes5[1].set_ylim(lims)
    axes5[1].set_xlabel(f"Observed Yield in Rolling-Origin Test Years {anios_ro_eval[0]}–{anios_ro_eval[-1]} (t ha$^{{-1}}$)")
    axes5[1].set_ylabel("Rolling-Origin Predicted Yield (t ha$^{-1}$)")
    axes5[1].grid(True, linestyle="--", alpha=0.32)
    axes5[1].text(-0.10, 1.02, "(b)", transform=axes5[1].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    axes5[1].text(
        0.96, 0.06, f"Out-of-Time XGBoost\n$R^2 = {r2_ro_dyn:.4f}$\nRMSE $= {rmse_ro_dyn:.2f}$ t ha$^{{-1}}$",
        transform=axes5[1].transAxes, ha="right", va="bottom", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#CED4DA", alpha=0.94),
    )
    axes5[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=7.8)
    _save(fig5, "Fig5_Benchmarking_Modelos_ML_y_Prediccion_OOT")

    # --- FIGURA 6: SHAP Global Importance + Beeswarm (Alineación Exacta Compartida Sin Solapamiento) ---
    fig6 = plt.figure(figsize=(14.6, 5.9))
    gs6 = fig6.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 0.045], wspace=0.14)
    ax6a = fig6.add_subplot(gs6[0])
    ax6b = fig6.add_subplot(gs6[1], sharey=ax6a)
    cax6 = fig6.add_subplot(gs6[2])

    df_s_plot = df_shap_global.sort_values("Mean_Abs_SHAP (ton/ha)", ascending=True).reset_index(drop=True)
    y_pos_shap = np.arange(len(df_s_plot))
    etiquetas_vars_en = [TRADUCCION_VARS_EN.get(v, v) for v in df_s_plot["Variable_Predictora"]]
    colores_shap = ["#E76F51" if c >= 0 else "#457B9D" for c in df_s_plot["Correlacion_Direccional_con_SHAP"]]

    ax6a.barh(y_pos_shap, df_s_plot["Mean_Abs_SHAP (ton/ha)"], color=colores_shap, edgecolor="black", height=0.66, alpha=0.92)
    for i, r_s in df_s_plot.iterrows():
        ax6a.text(
            float(r_s["Mean_Abs_SHAP (ton/ha)"]) + 0.022, i,
            f"{r_s['Participacion_Explicativa_SHAP (%)']:.1f}%",
            va="center", ha="left", fontsize=8.1, fontweight="bold",
        )
    ax6a.set_yticks(y_pos_shap)
    ax6a.set_yticklabels(etiquetas_vars_en, fontsize=8.8)
    ax6a.set_xlim(0, float(df_s_plot["Mean_Abs_SHAP (ton/ha)"].max()) * 1.24)
    ax6a.set_xlabel("Mean Absolute TreeSHAP Importance $|\\phi_j|$ (t ha$^{-1}$)")
    ax6a.grid(axis="x", linestyle="--", alpha=0.32)
    ax6a.text(-0.28, 1.02, "(a)", transform=ax6a.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    handles_dir = [
        Patch(facecolor="#E76F51", edgecolor="black", label="Positive Directional Association ($r \\geq 0$)"),
        Patch(facecolor="#457B9D", edgecolor="black", label="Negative Directional Association ($r < 0$)"),
    ]
    ax6a.legend(handles=handles_dir, loc="lower right", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=7.6)

    # Beeswarm limpio en ax6b con escala de color oficial de SHAP (#008bfb -> #ff0051)
    cmap_shap = LinearSegmentedColormap.from_list("shap_rb", ["#008bfb", "#7b52ab", "#ff0051"])
    rng_bs = np.random.default_rng(seed)
    ax6b.axvline(0, color="#343A40", linestyle="-", linewidth=1.0, alpha=0.75)

    for i, r_s in df_s_plot.iterrows():
        col_name = r_s["Variable_Predictora"]
        idx_f = features_xai.index(col_name)
        sv_f = shap_values[:, idx_f]
        xv_f = X_bio[col_name].values.astype(float)
        p5, p95 = np.percentile(xv_f, 5), np.percentile(xv_f, 95)
        norm_v = np.clip((xv_f - p5) / max(p95 - p5, 1e-9), 0.0, 1.0)

        # Jitter proporcional a la densidad local del histograma de valores SHAP
        counts, bin_edges = np.histogram(sv_f, bins=28)
        bin_idx = np.clip(np.digitize(sv_f, bin_edges) - 1, 0, len(counts) - 1)
        density_scale = counts[bin_idx] / max(counts.max(), 1)
        jitter_y = rng_bs.uniform(-0.30, 0.30, size=len(sv_f)) * (0.18 + 0.82 * density_scale)

        ax6b.scatter(
            sv_f, np.full_like(sv_f, float(i)) + jitter_y,
            c=norm_v, cmap=cmap_shap, vmin=0.0, vmax=1.0,
            s=14, alpha=0.78, edgecolors="none", rasterized=True,
        )

    ax6b.tick_params(axis="y", left=False, labelleft=False)
    ax6b.grid(axis="y", linestyle=":", alpha=0.35)
    ax6b.grid(axis="x", linestyle="--", alpha=0.32)
    ax6b.set_xlabel("TreeSHAP Marginal Contribution $\\phi_j(x)$ to Crop Yield (t ha$^{-1}$)")
    ax6b.text(-0.04, 1.02, "(b)", transform=ax6b.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")

    sm = plt.cm.ScalarMappable(cmap=cmap_shap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig6.colorbar(sm, cax=cax6)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "Mid", "High"], fontsize=8.2)
    cb.set_label("Normalized Feature Value", fontsize=9.2)
    cb.outline.set_visible(False)
    _save(fig6, "Fig6_SHAP_Summary_Beeswarm_y_Global")

    # --- FIGURA 7: Puntos de Cambio en la Contribución del Modelo con Bandas Bootstrap 95% ---
    fig7, axes7 = plt.subplots(2, 2, figsize=(13.8, 9.6), gridspec_kw={"hspace": 0.28, "wspace": 0.24})
    vars_curvas = [
        ("Radiacion_Solar_Media (MJ/m2/dia)", "(a)", axes7[0, 0]),
        ("Temp_Maxima_Media_Anual (C)", "(b)", axes7[0, 1]),
        ("Precipitacion_Acumulada_Anual (mm)", "(c)", axes7[1, 0]),
        ("ONI_Lag1 (C)", "(d)", axes7[1, 1]),
    ]
    for col_v, panel_tag, ax_v in vars_curvas:
        idx_v = features_xai.index(col_v)
        x_v = X_bio[col_v].values
        sv_v = shap_values[:, idx_v]
        info_b = curvas_boot_shap[col_v]

        for fase_k, col_f in paleta_enso.items():
            m_f = df_xai["Fase_ENSO"].values == fase_k
            ax_v.scatter(
                x_v[m_f], sv_v[m_f], color=col_f, alpha=0.58, s=25, edgecolor="none",
                label=TRADUCCION_ENSO_EN.get(fase_k, fase_k),
            )

        ax_v.plot(info_b["x_grid"], info_b["y_mean"], color="black", linewidth=2.3, label="LOWESS Mean Contribution")
        ax_v.fill_between(info_b["x_grid"], info_b["y_low"], info_b["y_high"], color="black", alpha=0.16, label="95% Panel Bootstrap CI (Fixed OOT Models)")
        ax_v.axhline(0, color="#343A40", linestyle="--", linewidth=1.1)

        if pd.notna(info_b["med_cruce"]):
            mc, p25c, p975c = info_b["med_cruce"], info_b["p25_cruce"], info_b["p975_cruce"]
            est_pct = info_b["estabilidad_pct"]
            ax_v.axvline(mc, color="#E63946", linestyle=":", linewidth=1.7)
            ax_v.axvspan(
                p25c, p975c, color="#E63946", alpha=0.14,
                label=f"$\\phi_j(x)=0$: {mc:.1f} [{p25c:.1f}, {p975c:.1f}] ({est_pct:.0f}% Boot.)",
            )

        ax_v.text(-0.10, 1.02, panel_tag, transform=ax_v.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
        ax_v.set_xlabel(TRADUCCION_VARS_EN.get(col_v, col_v))
        ax_v.set_ylabel("Marginal TreeSHAP Contribution (t ha$^{-1}$)")
        ax_v.grid(True, linestyle="--", alpha=0.30)
        ax_v.legend(loc="best", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.94, fontsize=7.5)
    _save(fig7, "Fig7_SHAP_Curvas_NoLineales_Umbrales_Climaticos")

    ctx.update({
        "features_xai": features_xai,
        "X_bio": X_bio,
        "shap_values": shap_values,
        "df_bench": df_bench,
        "df_shap_global": df_shap_global,
    })
    return ctx
