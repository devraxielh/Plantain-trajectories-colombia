"""Reanalysis of the current workbook; preserves the manuscript's existing outputs.

Run with the project's scientific Python environment: python reanalizar_platano.py
No weather downloads, imputations, or changes to the source workbook.
"""
from pathlib import Path
import hashlib
import json
import platform
import unicodedata
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.metrics import adjusted_rand_score
from fase_0_ingesta_y_clima import cargar_parametros_planos
from clustering_adaptativo import (
    matriz_bivariada_observada, ajustar_particion, seleccionar_clustering_adaptativo,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'result' / 'reanalisis_2026-10-22'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'PLATANO.xlsx'
before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
COLS = ['Anio', 'Departamento', 'Producto', 'Area_ha', 'Produccion_t', 'Rendimiento_reportado']


def load(path):
    frame = pd.read_excel(path).iloc[:, :6].copy()
    frame.columns = COLS
    frame['Fila_Excel'] = np.arange(2, len(frame) + 2)
    for c in ['Departamento', 'Producto']:
        frame[c] = frame[c].str.strip().str.upper()
    for c in ['Anio', 'Area_ha', 'Produccion_t', 'Rendimiento_reportado']:
        frame[c] = pd.to_numeric(frame[c], errors='raise')
    assert not frame.duplicated(['Anio', 'Departamento', 'Producto']).any()
    assert frame[COLS].notna().all().all()
    assert frame[['Area_ha', 'Produccion_t', 'Rendimiento_reportado']].ge(0).all().all()
    frame['Rendimiento_t_ha'] = frame.Rendimiento_reportado.where(frame.Area_ha.gt(0))
    return frame


def save(frame, name):
    frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def markov(frame, name):
    f = frame.sort_values(['Departamento', 'Anio']).copy()
    q = f.Rendimiento_t_ha.quantile([.25, .5, .75]).to_numpy()
    f['Estado'] = np.where(f.Rendimiento_t_ha.notna(),
                           np.searchsorted(q, f.Rendimiento_t_ha, side='right') + 1, np.nan)
    g = f.groupby('Departamento')
    f['Anio_anterior'] = g.Anio.shift()
    f['Estado_anterior'] = g.Estado.shift()
    t = f.loc[f.Anio.sub(f.Anio_anterior).eq(1) & f.Estado.notna() & f.Estado_anterior.notna()]
    counts = pd.crosstab(t.Estado_anterior, t.Estado).reindex(index=range(1, 5), columns=range(1, 5), fill_value=0)
    save(counts.rename_axis('Origen').reset_index(), 'markov_conteos_' + name)
    save(counts.div(counts.sum(axis=1), axis=0).rename_axis('Origen').reset_index(), 'markov_probabilidades_' + name)
    return dict(cuantiles=q.tolist(), transiciones=len(t),
                retencion_pct=100 * np.trace(counts) / len(t),
                ascenso_pct=100 * np.triu(counts, 1).sum() / len(t),
                descenso_pct=100 * np.tril(counts, -1).sum() / len(t))


raw = load(SOURCE)
old = load(OUT / 'PLATANO_anterior_2007_2022.xlsx')
continental = raw.loc[raw.Departamento.ne('SAN ANDRES Y PROVIDENCIA')].copy()
save(raw, 'datos_actuales_con_rendimiento_valido')
coverage = raw.groupby('Departamento').agg(Primer_anio=('Anio', 'min'), Ultimo_anio=('Anio', 'max'),
    N_filas=('Anio', 'size'), N_rendimientos_validos=('Rendimiento_t_ha', 'count')).reset_index()
save(coverage, 'cobertura')
keys = ['Anio', 'Departamento', 'Producto']
paired = raw.merge(old, on=keys, suffixes=('_actual', '_anterior'), how='outer', indicator=True)
shared = paired.loc[paired['_merge'].eq('both')].copy()
revision_counts, changed = {}, np.zeros(len(shared), dtype=bool)
for c in ['Area_ha', 'Produccion_t', 'Rendimiento_reportado']:
    different = ~np.isclose(shared[c + '_actual'], shared[c + '_anterior'], rtol=0, atol=1e-9)
    revision_counts[c] = int(different.sum())
    changed |= different
save(shared.loc[changed], 'revisiones_historicas')
save(paired.loc[paired['_merge'].eq('left_only')], 'registros_nuevos')
annual = continental.groupby('Anio').agg(Area_ha=('Area_ha', 'sum'), Produccion_t=('Produccion_t', 'sum'),
    Media_departamental_t_ha=('Rendimiento_t_ha', 'mean')).reset_index()
annual['Rendimiento_agregado_t_ha'] = annual.Produccion_t / annual.Area_ha
save(annual, 'totales_anuales_continentales')
profiles = []
for dep, f in continental.sort_values('Anio').groupby('Departamento'):
    valid = f.dropna(subset=['Rendimiento_t_ha'])
    profiles.append(dict(Departamento=dep, Media_rendimiento_t_ha=valid.Rendimiento_t_ha.mean(),
        Media_area_ha=f.Area_ha.mean(), Rendimiento_inicial=valid.Rendimiento_t_ha.iloc[0],
        Rendimiento_final=valid.Rendimiento_t_ha.iloc[-1],
        Pendiente_Sen_t_ha_anio=theilslopes(valid.Rendimiento_t_ha, valid.Anio).slope))
profiles = pd.DataFrame(profiles)
save(profiles, 'perfiles_departamentales')
latest = continental.loc[continental.Anio.eq(raw.Anio.max())].sort_values('Produccion_t', ascending=False).copy()
latest['Participacion_produccion_pct'] = latest.Produccion_t / latest.Produccion_t.sum() * 100
save(latest, 'departamentos_ultimo_anio')

summary = dict(sha256=before, python=platform.python_version(), filas=len(raw), columnas_originales=6,
    periodo=[int(raw.Anio.min()), int(raw.Anio.max())], departamentos=int(raw.Departamento.nunique()),
    filas_continentales=len(continental), rendimientos_validos_continentales=int(continental.Rendimiento_t_ha.notna().sum()),
    nuevos_registros=int(paired['_merge'].eq('left_only').sum()), filas_historicas_modificadas=int(changed.sum()),
    valores_historicos_modificados=revision_counts,
    error_maximo_produccion_sobre_area=float((raw.Rendimiento_reportado - raw.Produccion_t / raw.Area_ha.replace(0, np.nan)).abs().max()),
    markov_historico_actual=markov(continental.loc[continental.Anio.le(2022)], '2007_2022_actual'),
    markov_actual=markov(continental, '2007_2025'),
    markov_archivo_completo=markov(raw, 'incluido_san_andres'))

params = cargar_parametros_planos(str(ROOT / 'parametros_experimento.txt'))
params.update(ALGORITMOS_ADAPTATIVOS=['pam'], GRILLA_CLUSTERS_K=[2, 3, 4], K_CLUSTERS_FIJO=4)
r = continental.pivot(index='Departamento', columns='Anio', values='Rendimiento_t_ha').sort_index().sort_index(axis=1)
a = continental.pivot(index='Departamento', columns='Anio', values='Area_ha').reindex(index=r.index, columns=r.columns)
tables, labels = [], {}
for geometry in ['escala', 'forma']:
    for window in [2, 0]:
        p = dict(params, GEOMETRIA_ADAPTATIVA=geometry, VENTANA_SAKOE_CHIBA=window)
        print(f'RECALCULANDO {geometry}, ventana={window}, {len(r.columns)} anios', flush=True)
        result = seleccionar_clustering_adaptativo(r, a, p, informar=True)
        table = result['seleccion'].assign(Geometria=geometry, Ventana=window)
        tables.append(table)
        labels[(geometry, window)] = result['labels']
        save(result['perturbaciones'], f'perturbaciones_{geometry}_w{window}')
        save(result['grupos'], f'estabilidad_grupos_{geometry}_w{window}')
save(pd.concat(tables, ignore_index=True), 'clustering_comparacion_234')
assignment = pd.DataFrame({'Departamento': r.index})
for (geometry, window), lab in labels.items():
    means = pd.Series(continental.groupby('Departamento').Rendimiento_t_ha.mean().reindex(r.index).to_numpy()).groupby(lab).mean()
    mapping = {label: i + 1 for i, label in enumerate(means.sort_values().index)}
    assignment[f'{geometry}_w{window}_k4'] = [mapping[x] for x in lab]
save(assignment, 'asignaciones_k4')
summary['ari_escala_forma_k4'] = adjusted_rand_score(labels[('escala', 2)], labels[('forma', 2)])
old_assign = pd.read_csv(OUT / 'asignaciones_anteriores.csv')
check = assignment.merge(old_assign[['Departamento', 'Cluster_Principal']], on='Departamento', validate='one_to_one')
assert len(check) == len(assignment)
summary['ari_k4_vs_resultados_anteriores'] = adjusted_rand_score(check.escala_w2_k4, check.Cluster_Principal)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before, 'El archivo original cambio durante el analisis'
(OUT / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
print(annual.loc[annual.Anio.isin([2007, 2022, 2025])].to_string(index=False), flush=True)
print(pd.concat(tables)[['Geometria', 'Ventana', 'k', 'Silhouette', 'ARI_Temporal_P10', 'Jaccard_Min_Grupo_P10', 'Cumple_Criterios']].to_string(index=False), flush=True)
