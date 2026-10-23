"""
====================================================================================================
FASE 2: CADENAS DE MARKOV, COMPROBACIÓN FORMAL DE IRREDUCIBILIDAD, APERIODICIDAD,
        ESTACIONARIEDAD TEMPORAL (ANDERSON-GOODMAN), SUAVIZADO BAYESIANO DIRICHLET E IC 95%
====================================================================================================
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from fase_0_ingesta_y_clima import TRADUCCION_ENSO_EN, configurar_estilo_publicacion, guardar_figura_segura, indices_bootstrap_panel


def auditar_ergodicidad_estacionariedad_markov(df_trans: pd.DataFrame, estados_orden: list, anio_medio: int, params: dict) -> tuple:
    k = len(estados_orden)
    if k < 2 or df_trans.empty:
        raise ValueError("Markov requiere al menos dos estados y transiciones válidas.")
    alpha_dir = params["ALPHA_DIRICHLET_BAYES"]
    alpha_boot = params["ALPHA_DIRICHLET_BOOTSTRAP"]
    umbral_escasa = params["UMBRAL_CELDA_ESCASA"]
    n_boot = params["REPLICAS_BOOTSTRAP_MARKOV"]
    seed = params["SEMILLA_ALEATORIA"]
    alpha_stat = params["ALPHA_SIGNIFICANCIA_ESTADISTICA"]

    conteos_mat = pd.crosstab(df_trans["Estado_Markov_Lag1"], df_trans["Estado_Markov_t"]).reindex(
        index=estados_orden, columns=estados_orden, fill_value=0
    ).values.astype(float)

    row_sums = conteos_mat.sum(axis=1, keepdims=True)
    P_mle = np.divide(conteos_mat, row_sums, out=np.zeros_like(conteos_mat), where=row_sums > 0)
    P_bayes = (conteos_mat + alpha_dir) / (row_sums + k * alpha_dir)

    G_markov = nx.DiGraph()
    for i in range(k):
        G_markov.add_node(estados_orden[i])
        for j in range(k):
            if P_mle[i, j] > 0:
                G_markov.add_edge(estados_orden[i], estados_orden[j], prob=P_mle[i, j])

    es_irreducible = nx.is_strongly_connected(G_markov)
    n_clases_comunicantes = nx.number_strongly_connected_components(G_markov)
    es_aperiodica = nx.is_aperiodic(G_markov) if es_irreducible else False
    tiene_autobucles = bool(np.all(np.diag(P_mle) > 0))

    n_celdas_total = k * k
    n_celdas_cero = int(np.sum(conteos_mat == 0))
    n_celdas_escasas = int(np.sum(conteos_mat < umbral_escasa))

    # Prueba de Estacionariedad Temporal de Anderson-Goodman (Subperiodo 1 vs Subperiodo 2)
    sub1 = df_trans[df_trans["Anio"] <= anio_medio]
    sub2 = df_trans[df_trans["Anio"] > anio_medio]
    C1 = pd.crosstab(sub1["Estado_Markov_Lag1"], sub1["Estado_Markov_t"]).reindex(index=estados_orden, columns=estados_orden, fill_value=0).values
    C2 = pd.crosstab(sub2["Estado_Markov_Lag1"], sub2["Estado_Markov_t"]).reindex(index=estados_orden, columns=estados_orden, fill_value=0).values

    ag_chi2_total, ag_g2_total, ag_dof = 0.0, 0.0, 0
    for i in range(k):
        n1_i, n2_i = C1[i].sum(), C2[i].sum()
        if n1_i > 0 and n2_i > 0:
            p_pool_i = (C1[i] + C2[i]) / (n1_i + n2_i)
            cols_activas = p_pool_i > 0
            m_act = int(np.sum(cols_activas))
            if m_act > 1:
                ag_dof += (m_act - 1)
                for j in np.where(cols_activas)[0]:
                    e1 = n1_i * p_pool_i[j]
                    e2 = n2_i * p_pool_i[j]
                    ag_chi2_total += ((C1[i, j] - e1) ** 2) / e1 + ((C2[i, j] - e2) ** 2) / e2
                    if C1[i, j] > 0:
                        ag_g2_total += 2.0 * C1[i, j] * np.log(C1[i, j] / e1)
                    if C2[i, j] > 0:
                        ag_g2_total += 2.0 * C2[i, j] * np.log(C2[i, j] / e2)

    p_val_ag_chi2 = float(stats.chi2.sf(ag_chi2_total, ag_dof)) if ag_dof else np.nan
    p_val_ag_g2 = float(stats.chi2.sf(ag_g2_total, ag_dof)) if ag_dof else np.nan

    def _calc_ergodico(P_mat):
        grafo = nx.from_numpy_array(P_mat > 0, create_using=nx.DiGraph)
        if not np.allclose(P_mat.sum(axis=1), 1) or not nx.is_strongly_connected(grafo):
            # No seleccionar arbitrariamente un autovector de una cadena reducible.
            return np.full(k, np.nan), np.nan, np.nan, np.full((k, k), np.nan)
        eigvals, eigvecs = np.linalg.eig(P_mat.T)
        idx_u = np.argmin(np.abs(eigvals - 1.0))
        pi_v = np.real(eigvecs[:, idx_u])
        pi_v = pi_v / np.sum(pi_v)
        eig_s = np.sort(np.abs(eigvals))[::-1]
        gap = 1.0 - (float(eig_s[1]) if len(eig_s) > 1 else 0.0)
        shorr = float((k - np.trace(P_mat)) / (k - 1))
        W = np.tile(pi_v, (k, 1))
        try:
            Z = np.linalg.inv(np.eye(k) - P_mat + W)
            M = np.zeros((k, k))
            for r_i in range(k):
                for c_j in range(k):
                    M[r_i, c_j] = (1.0 / pi_v[c_j]) if r_i == c_j else ((Z[c_j, c_j] - Z[r_i, c_j]) / pi_v[c_j])
        except np.linalg.LinAlgError:
            M = np.full((k, k), np.nan)
        return pi_v, shorr, gap, M

    pi_mle, shorrocks_mle, gap_mle, mfpt_mle = _calc_ergodico(P_mle)
    pi_bayes, _, _, mfpt_bayes = _calc_ergodico(P_bayes)

    rng = np.random.default_rng(seed)
    pi_boot_samples, mfpt_e4_boot, P_boot_samples = [], [], []
    n_trans = len(df_trans)
    idx_t1_arr = df_trans["Estado_Markov_Lag1"].map({e: i for i, e in enumerate(estados_orden)}).values
    idx_t_arr = df_trans["Estado_Markov_t"].map({e: i for i, e in enumerate(estados_orden)}).values

    for _ in range(n_boot):
        b_idx = indices_bootstrap_panel(df_trans, rng)
        C_b = np.zeros((k, k))
        np.add.at(C_b, (idx_t1_arr[b_idx], idx_t_arr[b_idx]), 1.0)
        rs_b = C_b.sum(axis=1, keepdims=True)
        P_b = (C_b + alpha_boot) / (rs_b + k * alpha_boot)
        pi_b, _, _, mfpt_b = _calc_ergodico(P_b)
        pi_boot_samples.append(pi_b * 100)
        mfpt_e4_boot.append(mfpt_b[:, -1])
        P_boot_samples.append(P_b * 100)

    pi_boot_arr = np.array(pi_boot_samples)
    mfpt_boot_arr = np.array(mfpt_e4_boot)
    P_boot_std = np.std(np.array(P_boot_samples), axis=0)
    prop_obs = df_trans["Estado_Markov_t"].value_counts(normalize=True).reindex(estados_orden, fill_value=0).values * 100

    df_markov_ergodico = pd.DataFrame({
        "Estado_Productivo": estados_orden,
        "Transiciones_Salientes_Totales (N_i.)": row_sums.flatten().astype(int),
        f"Celdas_Escasas_En_Fila (n_ij<{umbral_escasa})": np.sum(conteos_mat < umbral_escasa, axis=1).astype(int),
        "Proporcion_Observada (%)": prop_obs.round(2),
        "Pi_Ergodica_MLE (%)": (pi_mle * 100).round(2),
        f"Pi_Ergodica_IC95_Bootstrap_Panel_Suavizado_a{alpha_boot} (%)": [
            f"{np.percentile(pi_boot_arr[:, i], 2.5):.1f} - {np.percentile(pi_boot_arr[:, i], 97.5):.1f}" for i in range(k)
        ],
        f"Pi_Ergodica_Bayes_Dirichlet_a{alpha_dir} (%)": (pi_bayes * 100).round(2),
        "Sensibilidad_Delta_Pi_MLE_vs_Bayes (pp)": np.abs(pi_mle * 100 - pi_bayes * 100).round(2),
        "MFPT_Hacia_E4_MLE (anios)": mfpt_mle[:, -1].round(2),
        "MFPT_Hacia_E4_Bayes (anios)": mfpt_bayes[:, -1].round(2),
        f"MFPT_Hacia_E4_IC95_Bootstrap_Panel_Suavizado_a{alpha_boot}": [
            f"{np.nanpercentile(mfpt_boot_arr[:, i], 2.5):.1f} - {np.nanpercentile(mfpt_boot_arr[:, i], 97.5):.1f}" for i in range(k)
        ],
    })

    df_diagnostico_supuestos_markov = pd.DataFrame([
        {
            "Propiedad_o_Supuesto_Markoviano": "1. Irreducibilidad (Fuerte Conexidad del Grafo)",
            "Resultado_Empirico": f"Irreducible = {es_irreducible} (Clases comunicantes = {n_clases_comunicantes})",
            "Estadistico_Valor": float(n_clases_comunicantes),
            "p_value_o_Criterio": "Cumple (1 sola clase cerrada)" if es_irreducible else "Reducible",
            "Interpretacion_Metodologica": "Todos los estados son mutuamente alcanzables." if es_irreducible else "No todos los estados son mutuamente alcanzables; no se informa una pi ergódica MLE única.",
        },
        {
            "Propiedad_o_Supuesto_Markoviano": "2. Aperiodicidad (MCD de ciclos = 1 y Autobucles P_ii > 0)",
            "Resultado_Empirico": f"Aperiódica = {es_aperiodica} | P_ii > 0 en todos los estados = {tiene_autobucles}",
            "Estadistico_Valor": float(1.0 - gap_mle),
            "p_value_o_Criterio": f"Gap espectral = {gap_mle:.4f}; irreducible={es_irreducible}; aperiódica={es_aperiodica}",
            "Interpretacion_Metodologica": "Existe una distribución estacionaria única y convergencia a ella." if es_irreducible and es_aperiodica else "No se verificaron todas las condiciones de convergencia ergódica.",
        },
        {
            "Propiedad_o_Supuesto_Markoviano": f"3. Estacionariedad Temporal de Anderson-Goodman (Subperiodo <= {anio_medio} vs > {anio_medio})",
            "Resultado_Empirico": "Homogeneidad temporal contrastada vía Chi² y Razón de Verosimilitud G²",
            "Estadistico_Valor": round(ag_chi2_total, 3),
            "p_value_o_Criterio": f"Chi²={ag_chi2_total:.2f} (p={p_val_ag_chi2:.4f}) | G²={ag_g2_total:.2f} (p={p_val_ag_g2:.4f}), gl={ag_dof}",
            "Interpretacion_Metodologica": ("No evaluable: cero grados de libertad." if not ag_dof else
                ("No se rechaza homogeneidad; no prueba estacionariedad." if p_val_ag_chi2 >= alpha_stat else "Se rechaza homogeneidad temporal.")
                + " Contraste asintótico exploratorio; dependencia del panel/celdas escasas limitan el p-valor."),
        },
        {
            "Propiedad_o_Supuesto_Markoviano": "4. Esparsidad / Transiciones Escasas y Sensibilidad Bayesiana",
            "Resultado_Empirico": f"{n_celdas_cero}/{n_celdas_total} celdas en cero; {n_celdas_escasas}/{n_celdas_total} celdas con n_ij < {umbral_escasa}",
            "Estadistico_Valor": round(float(np.max(np.abs(pi_mle * 100 - pi_bayes * 100))), 2),
            "p_value_o_Criterio": f"Máx |π*_MLE - π*_Dirichlet(α={alpha_dir})| = {np.max(np.abs(pi_mle * 100 - pi_bayes * 100)):.2f} puntos porcentuales",
            "Interpretacion_Metodologica": "Diferencia descriptiva MLE/Dirichlet; no demuestra estabilidad. IC: bootstrap cruzado departamento/año con suavizado explícito.",
        },
    ])

    P_mle_df = pd.DataFrame(P_mle, index=estados_orden, columns=estados_orden)
    P_std_df = pd.DataFrame(P_boot_std, index=estados_orden, columns=estados_orden)
    return P_mle_df, P_std_df, df_markov_ergodico, df_diagnostico_supuestos_markov, pi_mle, pi_bayes, shorrocks_mle, gap_mle


def ejecutar_fase_2(ctx: dict, params: dict) -> dict:
    print("[FASE 2] Auditando Irreducibilidad, Periodicidad, Homogeneidad Anderson-Goodman y Sensibilidad Dirichlet...")

    df = ctx["df"]
    anio_min, anio_max = ctx["anio_min"], ctx["anio_max"]
    anio_medio_markov = ctx["anio_medio_markov"]
    dir_tablas, dir_figuras = ctx["dir_tablas"], ctx["dir_figuras"]
    dpi = params["DPI_FIGURAS"]

    cuantiles = params["CUANTILES_ESTADOS_MARKOV"]
    # Estados descriptivos de toda la historia, independientes del split del ML complementario.
    q25, q50, q75 = np.percentile(df["Rendimiento (ton/ha)"].dropna(), cuantiles)
    if not q25 < q50 < q75:
        raise ValueError("Los cuantiles de Markov deben definir cuatro estados distintos.")
    # Las etiquetas deben reflejar los límites usados para clasificar, no otros
    # límites redondeados a una décima (p. ej. 6.0875 no es 6.1).
    limites_texto = [np.format_float_positional(q, precision=10, trim="-") for q in (q25, q50, q75)]
    t25, t50, t75 = limites_texto
    Path(ctx["dir_datasets"], "umbrales_markov.json").write_text(json.dumps({
        "cuantiles_porcentaje": cuantiles,
        "limites_t_ha": [float(q25), float(q50), float(q75)],
        "limites_mostrados": limites_texto,
        "intervalos": "S1: y<q25; S2: q25<=y<q50; S3: q50<=y<q75; S4: y>=q75",
        "referencia": "Todos los rendimientos válidos del panel analizado; no son clústeres DTW",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    def clasificar_estado_markov(r: float) -> str:
        if not np.isfinite(r):
            return None
        if r < q25:
            return f"E1_Bajo (<{t25})"
        elif r < q50:
            return f"E2_Medio_Bajo ([{t25},{t50}))"
        elif r < q75:
            return f"E3_Medio_Alto ([{t50},{t75}))"
        else:
            return f"E4_Alto (>={t75})"

    df["Estado_Markov_t"] = df["Rendimiento (ton/ha)"].apply(clasificar_estado_markov)
    df["Estado_Markov_Lag1"] = df.groupby("Departamento")["Estado_Markov_t"].shift(1)
    df.loc[df.groupby("Departamento")["Anio"].diff().ne(1), "Estado_Markov_Lag1"] = None

    df_trans = df.dropna(subset=["Estado_Markov_Lag1", "Estado_Markov_t"]).copy()
    estados_orden = [clasificar_estado_markov(v) for v in [q25 - 1, q25, q50, q75]]

    (
        P_global_df,
        P_boot_std_df,
        df_markov_ergodico,
        df_diagnostico_supuestos_markov,
        pi_mle,
        pi_bayes,
        shorrocks_glob,
        gap_glob,
    ) = auditar_ergodicidad_estacionariedad_markov(df_trans, estados_orden, anio_medio_markov, params)

    df_markov_ergodico["Alcance"] = "Análisis complementario de largo plazo, condicionado a probabilidades y supuestos del modelo; no calendario de convergencia observado"
    df_markov_ergodico.to_csv(os.path.join(dir_tablas, "T05_Markov_Ergodicidad_Sensibilidad_e_IC95.csv"), index=False, encoding="utf-8-sig")
    df_diagnostico_supuestos_markov.to_csv(os.path.join(dir_tablas, "T06_Markov_Pruebas_Irreducibilidad_y_Estacionariedad.csv"), index=False, encoding="utf-8-sig")

    mapa_idx_estado = {e: i for i, e in enumerate(estados_orden)}
    df_trans["Idx_t_1"] = df_trans["Estado_Markov_Lag1"].map(mapa_idx_estado)
    df_trans["Idx_t"] = df_trans["Estado_Markov_t"].map(mapa_idx_estado)
    df_trans["Tipo_Transicion"] = np.where(
        df_trans["Idx_t"] > df_trans["Idx_t_1"],
        "Ascenso_Productivo",
        np.where(df_trans["Idx_t"] < df_trans["Idx_t_1"], "Retroceso_Productivo", "Persistencia_Estado"),
    )
    tipos = ["Retroceso_Productivo", "Persistencia_Estado", "Ascenso_Productivo"]
    resumen_transiciones = pd.crosstab(df_trans["Departamento"], df_trans["Tipo_Transicion"]).reindex(
        index=sorted(df["Departamento"].unique()), columns=tipos, fill_value=0)
    resumen_transiciones["N_Transiciones_Validas"] = resumen_transiciones[tipos].sum(axis=1)
    for tipo in tipos:
        resumen_transiciones[tipo + " (%)"] = 100 * resumen_transiciones[tipo] / resumen_transiciones["N_Transiciones_Validas"].replace(0, np.nan)
    resumen_transiciones = ctx["df_perfiles"].merge(resumen_transiciones.reset_index(), on="Departamento", validate="one_to_one")
    resumen_transiciones["Referencia_Estados"] = "Cuantiles de todos los rendimientos observados del periodo completo"
    resumen_transiciones.to_csv(os.path.join(dir_tablas, "T13_Transiciones_Observadas_por_Departamento.csv"), index=False, encoding="utf-8-sig")
    filas_cluster = []
    for cluster, grupo in df_trans.groupby("Cluster_DTW_Principal"):
        cuentas = pd.crosstab(grupo["Estado_Markov_Lag1"], grupo["Estado_Markov_t"]).reindex(
            index=estados_orden, columns=estados_orden, fill_value=0)
        for estado_i in estados_orden:
            total_i = int(cuentas.loc[estado_i].sum())
            for estado_j in estados_orden:
                n_ij = int(cuentas.loc[estado_i, estado_j])
                filas_cluster.append({"Cluster_DTW": cluster, "Estado_Origen": estado_i,
                    "Estado_Destino": estado_j, "N_Transiciones": n_ij, "N_Salientes_Origen": total_i,
                    "Probabilidad_MLE (%)": 100 * n_ij / total_i if total_i else np.nan,
                    "Alcance": "Descriptivo; clusters retrospectivos, misma referencia de estados para todos"})
    df_markov_clusters = pd.DataFrame(filas_cluster)
    df_markov_clusters.to_csv(os.path.join(dir_tablas, "T14_Markov_por_Perfil_DTW.csv"), index=False, encoding="utf-8-sig")
    df_enso_plot = pd.crosstab(df_trans["Fase_ENSO"], df_trans["Tipo_Transicion"], normalize="index") * 100
    conteos_enso_markov = pd.crosstab(df_trans["Fase_ENSO"], df_trans["Tipo_Transicion"])
    # ENSO es compartido por año: no usar departamentos como réplicas climáticas independientes.

    # Figura 3: Cadenas de Markov (Estándar Q1 Journal - Academic English, No Titles, Zero Overlaps)
    configurar_estilo_publicacion(plt)
    fig3, axes3 = plt.subplots(1, 3, figsize=(16.6, 5.1), gridspec_kw={"width_ratios": [1.18, 0.98, 1.12], "wspace": 0.34})
    etiquetas_cortas = [
        f"S1: Low\n(<{t25})",
        f"S2: Low-Mid\n[{t25}, {t50})",
        f"S3: High-Mid\n[{t50}, {t75})",
        f"S4: High\n($\\geq${t75})",
    ]

    conteos_globales = pd.crosstab(df_trans["Estado_Markov_Lag1"], df_trans["Estado_Markov_t"]).reindex(
        index=estados_orden, columns=estados_orden, fill_value=0)
    annot_mat = np.empty((4, 4), dtype=object)
    for r_i in range(4):
        for c_j in range(4):
            annot_mat[r_i, c_j] = f"{P_global_df.values[r_i, c_j]*100:.1f}%\n(n={conteos_globales.iloc[r_i, c_j]})"

    sns.heatmap(
        P_global_df.values * 100, annot=annot_mat, fmt="", cmap="YlOrRd", cbar=False,
        xticklabels=etiquetas_cortas, yticklabels=etiquetas_cortas, ax=axes3[0],
        linewidths=0.9, linecolor="white", annot_kws={"size": 8.0},
    )
    axes3[0].tick_params(axis="y", rotation=0, labelsize=8.3)
    axes3[0].tick_params(axis="x", rotation=0, labelsize=8.3)
    axes3[0].set_xlabel("Productive State at Year $t$ (t ha$^{-1}$)")
    axes3[0].set_ylabel("Productive State at Year $t-1$ (t ha$^{-1}$)")
    axes3[0].text(-0.18, 1.03, "(a)", transform=axes3[0].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")

    cols_tr = ["Retroceso_Productivo", "Persistencia_Estado", "Ascenso_Productivo"]
    nombres_tr_en = {
        "Retroceso_Productivo": "Downward Transition",
        "Persistencia_Estado": "State Persistence",
        "Ascenso_Productivo": "Upward Transition",
    }
    trans_perfil = pd.crosstab(df_trans["Cluster_DTW_Principal"], df_trans["Tipo_Transicion"], normalize="index").reindex(
        index=list(params["PALETA_CLUSTERS"]), columns=cols_tr, fill_value=0) * 100
    trans_perfil.index = [label.split("_")[0] for label in trans_perfil.index]
    trans_perfil.columns = [nombres_tr_en[c] for c in cols_tr]
    trans_perfil.plot(kind="bar", stacked=True, ax=axes3[1], color=["#E63946", "#CED4DA", "#2A9D8F"], edgecolor="black", width=0.58)
    axes3[1].set_ylabel("Annual State Transitions (%)")
    axes3[1].set_xlabel("Productive Scale Profile" if params["GEOMETRIA_PRINCIPAL_ADAPTATIVA"] == "escala" else "Temporal Trajectory Profile")
    axes3[1].set_ylim(0, 124)
    axes3[1].tick_params(axis="x", rotation=0, labelsize=8.8)
    axes3[1].text(-0.14, 1.03, "(b)", transform=axes3[1].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    axes3[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CED4DA", framealpha=0.95, fontsize=7.6)

    shares = pd.crosstab(df["Anio"], df["Estado_Markov_t"], normalize="index").reindex(columns=estados_orden, fill_value=0) * 100
    axes3[2].stackplot(shares.index.to_numpy(), shares.to_numpy().T,
        labels=["S1: Low", "S2: Low-Mid", "S3: High-Mid", "S4: High"],
        colors=["#457B9D", "#8ECAE6", "#E9C46A", "#E76F51"], alpha=.9)
    axes3[2].set_xticks(shares.index.to_numpy()[::max(1, len(shares) // 6)])
    axes3[2].set_ylabel("Observed State Share (%)")
    axes3[2].set_xlabel("Observed Year")
    axes3[2].set_ylim(0, 123)
    axes3[2].set_xlim(shares.index.min(), shares.index.max())
    axes3[2].text(-0.14, 1.03, "(c)", transform=axes3[2].transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    axes3[2].legend(loc="upper center", ncol=2, frameon=True, fontsize=7.6)
    # El nombre de archivo se conserva por compatibilidad; el contenido es observado, no una proyeccion.

    guardar_figura_segura(fig3, dir_figuras, "Fig3_Markov_Matrices_y_Distribucion_Estacionaria", dpi, plt)

    ctx.update({
        "df": df,
        "df_trans": df_trans,
        "P_global_df": P_global_df,
        "df_markov_ergodico": df_markov_ergodico,
        "df_diagnostico_supuestos_markov": df_diagnostico_supuestos_markov,
        "df_transiciones_depto": resumen_transiciones,
        "df_markov_clusters": df_markov_clusters,
    })
    return ctx
