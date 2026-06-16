# %%
""" RBlJose — SISTEMA NBO
Semana 7 — Evaluación Operativa de Calibradores v1.1
======================================================
Evalúa si la recalibración v1.1 (Semana 6) genera mejores decisiones
de negocio comparada con v1.0, usando datos reales del generador.

Diseño experimental:
  Fase referencia (meses 22-23): período donde el trigger se detectó.
    Ambas versiones se evalúan sobre los mismos datos con etiqueta real.
    Se espera que v1.1 corrija el sesgo identificado.

  Fase proyección (mes 25): período sin etiqueta.
    Se usa score_propension del generador como ground truth para simular
    conversiones. Ambas versiones compiten bajo las mismas restricciones
    de campaña (presupuesto, límites de originación).

Mecanismo de impacto de la calibración:
  Una mejor calibración cambia el RANKING relativo de los clientes.
  Bajo restricción de top-N, esto determina quién queda por encima del
  corte. Si v1.1 rankea mejor a los clientes con mayor propensión real,
  el mismo presupuesto produce más conversiones.

Prerequisitos:
  nbo_calibradores_v11.pkl          — calibradores v1.1 (Semana 6)
  nbo_scores_historicos_s5.csv      — scores XGBoost reales (Semana 5)
  nbo_ofertas.csv                   — ground truth del generador
  nbo_features.csv                  — features mes 25

Outputs:
  nbo_s7_decisiones_comparativas.csv  — selección v1.0 vs v1.1 por mes
  nbo_s7_profit_comparativo.csv       — profit por mes, producto y versión
  nbo_s7_resumen_ejecutivo.csv        — KPIs consolidados
  nbo_s7_visualizaciones.png          — dashboard comparativo
"""


# %%
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import os
import warnings
from datetime import date
from scipy.special import expit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')


# %%
# ──────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────
def separador(titulo):
    print(f"\n{'='*70}")
    print(f"  {titulo}")
    print(f"{'='*70}")

def subseccion(titulo):
    print(f"\n  ── {titulo}")

def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece

# %%
# ──────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ──────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
DATA_DIR   = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'models')

PRODUCTOS_DRIFT = ['tarjeta', 'seguro_vida']
TODOS_PRODUCTOS = ['tarjeta', 'prestamo', 'microcredito',
                   'seguro_vida', 'seguro_salud', 'inversion']

PARAMS_NEGOCIO = {
    'tarjeta'     : {'ticket_anual': 465.0,  'costo_contacto': 2.5,
                     'costo_originacion': 45.0,  'tasa_organica': 0.018},
    'prestamo'    : {'ticket_anual': 675.0,  'costo_contacto': 3.0,
                     'costo_originacion': 60.0,  'tasa_organica': 0.025},
    'microcredito': {'ticket_anual': 300.0,  'costo_contacto': 4.5,
                     'costo_originacion': 85.0,  'tasa_organica': 0.012},
    'seguro_vida' : {'ticket_anual': 31.5,   'costo_contacto': 2.0,
                     'costo_originacion': 15.0,  'tasa_organica': 0.015},
    'seguro_salud': {'ticket_anual': 36.0,   'costo_contacto': 2.0,
                     'costo_originacion': 15.0,  'tasa_organica': 0.014},
    'inversion'   : {'ticket_anual': 120.0,  'costo_contacto': 1.5,
                     'costo_originacion': 10.0,  'tasa_organica': 0.020},
}

# Restricción de campaña: top N por producto
CONTACTOS_MAX = {
    'tarjeta': 800, 'prestamo': 600, 'microcredito': 300,
    'seguro_vida': 1000, 'seguro_salud': 1000, 'inversion': 700,
}

# Meses evaluados
# 22-23: con etiqueta real → evaluación sobre resultados reales
# 25: sin etiqueta → evaluación con score_propension como ground truth
MESES_EVALUACION = {
    22: {'fase': 'referencia', 'fuente_gt': 'etiqueta_real'},
    23: {'fase': 'referencia', 'fuente_gt': 'etiqueta_real'},
    25: {'fase': 'proyeccion', 'fuente_gt': 'score_propension'},
}

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA DE INSUMOS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA DE INSUMOS")

# Calibradores v1.1 (Semana 6)
with open(f'{DATA_DIR}/nbo_calibradores_v11.pkl', 'rb') as f:
    calibradores_v11 = pickle.load(f)
print(f"  ✅ Calibradores v1.1 cargados: "
      f"{[p for p, c in calibradores_v11.items() if c['version']=='v1.1']} recalibrados")

# Scores históricos de Semana 5
scores_s5 = pd.read_csv(f'{DATA_DIR}/nbo_scores_historicos_s5.csv')
print(f"  ✅ Scores históricos S5: {len(scores_s5):,} filas | "
      f"meses {scores_s5['mes_scoring'].min()}–{scores_s5['mes_scoring'].max()}")

# Ofertas reales (ground truth del generador)
ofertas = pd.read_csv(f'{DATA_DIR}/nbo_ofertas.csv')
ofertas['id_producto_lower'] = ofertas['id_producto'].str.lower()
print(f"  ✅ nbo_ofertas: {len(ofertas):,} filas")

# Construir lookup de ground truth
gt_lookup = (
    ofertas[ofertas['grupo'] == 'Tratamiento']
    [['id_cliente', 'mes', 'id_producto_lower',
      'score_propension', 'convirtio_30d', 'etiqueta_completa']]
    .rename(columns={'id_producto_lower': 'producto_nbo'})
    .set_index(['id_cliente', 'mes', 'producto_nbo'])
)
print(f"  ✅ Lookup GT construido: {len(gt_lookup):,} entradas")

# Verificar que el mes 25 tiene score_propension
mes25_check = ofertas[ofertas['mes'] == 25]
if len(mes25_check) == 0:
    raise ValueError("Sin datos en nbo_ofertas.csv para mes 25. "
                     "Verifica que el generador cubre 25 meses.")
print(f"  ✅ Mes 25 disponible: {len(mes25_check):,} registros en nbo_ofertas.csv")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — GENERACIÓN DE PROBABILIDADES v1.0 Y v1.1
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — GENERACIÓN DE PROBABILIDADES v1.0 Y v1.1")

print("""
  Para cada cliente/mes/producto calculamos:
    p_v10  : probabilidad del calibrador original (parámetros A,B de /models/)
    p_v11  : probabilidad del calibrador recalibrado (Semana 6)

  El score_xgb_raw es el mismo para ambas versiones — es el output del
  modelo XGBoost sin procesar. Lo que cambia es la función de escala
  que transforma ese score en probabilidad.

  Diferencia esperada:
    Productos sin recalibrar : p_v10 ≈ p_v11 (mismo A, B)
    Productos recalibrados   : p_v11 corrige el sesgo de p_v10
""")

meses_eval_list = sorted(MESES_EVALUACION.keys())
scores_eval = scores_s5[scores_s5['mes_scoring'].isin(meses_eval_list)].copy()
scores_eval = scores_eval.rename(columns={'mes_scoring': 'mes'})

# Calcular p_v11 para todos los registros
p_v11_list = []
for _, row in scores_eval.iterrows():
    prod = row['producto_nbo']
    cal  = calibradores_v11[prod]
    p    = expit(cal['A'] * row['score_xgb_raw'] + cal['B'])
    p_v11_list.append(float(p))
scores_eval['p_v11'] = p_v11_list

# Verificar diferencias entre versiones para productos recalibrados
subseccion("Diferencia media de probabilidades por producto")
print(f"\n  {'Producto':<20} {'p_v10 media':>13} {'p_v11 media':>13} "
      f"{'Δ media':>10} {'Versión cal':>12}")
print(f"  {'─'*72}")
for prod in sorted(TODOS_PRODUCTOS):
    sub = scores_eval[scores_eval['producto_nbo'] == prod]
    if len(sub) == 0:
        continue
    cal_ver = calibradores_v11[prod]['version']
    delta = sub['p_v11'].mean() - sub['p_calibrada'].mean()
    print(f"  {prod:<20} {sub['p_calibrada'].mean():>13.4f} "
          f"{sub['p_v11'].mean():>13.4f} {delta:>+9.4f} {cal_ver:>12}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — SIMULACIÓN DE DECISIONES BAJO RESTRICCIÓN DE CAMPAÑA
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — SIMULACIÓN DE DECISIONES BAJO RESTRICCIÓN")

print("""
  Para cada mes y producto, ambas versiones rankan el universo disponible
  por su probabilidad calibrada y seleccionan los top N clientes.

  El ground truth de conversión es idéntico para ambas versiones:
    Meses 22-23 : convirtio_30d real de nbo_ofertas.csv
    Mes 25      : Bernoulli(score_propension) — probabilidad real del generador

  Misma semilla de simulación para ambas versiones → la única fuente
  de diferencia en resultados es la selección de clientes, no el ruido.
""")

rng = np.random.default_rng(SEED)
resultados_todas = []

for mes, config in MESES_EVALUACION.items():
    scores_mes = scores_eval[scores_eval['mes'] == mes].copy()

    if len(scores_mes) == 0:
        print(f"  ⚠️  Sin scores para mes {mes}")
        continue

    for prod in sorted(TODOS_PRODUCTOS):
        scores_prod = scores_mes[scores_mes['producto_nbo'] == prod].copy()
        if len(scores_prod) == 0:
            continue

        n_max     = CONTACTOS_MAX[prod]
        p_negocio = PARAMS_NEGOCIO[prod]
        ingreso_neto = p_negocio['ticket_anual'] - p_negocio['costo_originacion']

        for version in ['v10', 'v11']:
            col_prob = 'p_calibrada' if version == 'v10' else 'p_v11'

            # Selección top-N por probabilidad de esta versión
            seleccionados = scores_prod.nlargest(n_max, col_prob).copy()
            n_sel = len(seleccionados)

            # Ground truth de conversión — mismo para ambas versiones
            convirtio_list = []
            for _, row in seleccionados.iterrows():
                cid = row['id_cliente']
                key = (cid, mes, prod)

                if key in gt_lookup.index:
                    row_gt = gt_lookup.loc[key]
                    if isinstance(row_gt, pd.DataFrame):
                        row_gt = row_gt.iloc[0]

                    if config['fuente_gt'] == 'etiqueta_real' and \
                       not pd.isna(row_gt['convirtio_30d']):
                        conv = int(row_gt['convirtio_30d'])
                    else:
                        # Mes 25: simular desde score_propension real
                        p_real = float(row_gt['score_propension'])
                        conv = int(rng.random() < p_real)
                else:
                    conv = int(rng.random() < p_negocio['tasa_organica'])

                convirtio_list.append(conv)

            seleccionados['convirtio'] = convirtio_list

            n_conv      = sum(convirtio_list)
            tasa_conv   = n_conv / n_sel if n_sel > 0 else 0
            ingreso_tot = n_conv * ingreso_neto
            costo_tot   = n_sel * p_negocio['costo_contacto']
            profit      = ingreso_tot - costo_tot
            costo_x_conv = costo_tot / max(n_conv, 1)

            # ECE de la selección: calibración dentro del grupo elegido
            y_true = np.array(convirtio_list, dtype=float)
            y_pred = seleccionados[col_prob].values
            ece    = ece_score(y_true, y_pred) if len(y_true) >= 10 else None

            resultados_todas.append({
                'mes'             : mes,
                'fase'            : config['fase'],
                'fuente_gt'       : config['fuente_gt'],
                'producto'        : prod,
                'version'         : 'v1.0' if version == 'v10' else 'v1.1',
                'n_contactos'     : n_sel,
                'n_conversiones'  : n_conv,
                'tasa_conversion' : round(tasa_conv, 4),
                'ingreso_total'   : round(ingreso_tot, 2),
                'costo_total'     : round(costo_tot, 2),
                'profit'          : round(profit, 2),
                'costo_por_conv'  : round(costo_x_conv, 2),
                'ece_seleccion'   : round(ece, 4) if ece else None,
                'ids_sel'         : set(seleccionados['id_cliente'].values),
            })

df_resultados = pd.DataFrame(resultados_todas)
print(f"  ✅ Simulación completada: {len(df_resultados)} combinaciones")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — ANÁLISIS COMPARATIVO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — ANÁLISIS COMPARATIVO v1.0 vs v1.1")

subseccion("4.1 — Overlap de selección (productos con drift)")
print(f"""
  Overlap = fracción de clientes seleccionados idénticos entre v1.0 y v1.1.
  Overlap < 90% → la recalibración cambia el ranking significativamente.
  Overlap ≈ 100% → misma selección, impacto en probabilidades absolutas únicamente.
""")
print(f"  {'Mes':<6} {'Fase':<12} {'Producto':<20} {'Overlap':>10}")
print(f"  {'─'*52}")

for mes in meses_eval_list:
    for prod in PRODUCTOS_DRIFT:
        sub = df_resultados[
            (df_resultados['mes'] == mes) &
            (df_resultados['producto'] == prod)
        ]
        if len(sub) < 2:
            continue
        ids_v10 = sub[sub['version'] == 'v1.0']['ids_sel'].values[0]
        ids_v11 = sub[sub['version'] == 'v1.1']['ids_sel'].values[0]
        overlap = len(ids_v10 & ids_v11) / max(len(ids_v10), 1)
        nota    = "⚠️ Cambio ranking" if overlap < 0.90 else "≈ Estable"
        print(f"  {mes:<6} {MESES_EVALUACION[mes]['fase']:<12} "
              f"{prod:<20} {overlap:>9.1%}  {nota}")

subseccion("4.2 — Profit total por mes")
resumen_mes = []
print(f"\n  {'Mes':<6} {'Fase':<12} {'Fuente GT':<16} {'Versión':<8} "
      f"{'Conv':>6} {'Tasa%':>7} {'Profit':>12}")
print(f"  {'─'*72}")

for mes in meses_eval_list:
    for ver in ['v1.0', 'v1.1']:
        sub = df_resultados[
            (df_resultados['mes'] == mes) &
            (df_resultados['version'] == ver)
        ]
        tot_conv    = sub['n_conversiones'].sum()
        tot_cont    = sub['n_contactos'].sum()
        tasa        = tot_conv / tot_cont if tot_cont > 0 else 0
        tot_profit  = sub['profit'].sum()
        fase        = MESES_EVALUACION[mes]['fase']
        fuente      = MESES_EVALUACION[mes]['fuente_gt']

        print(f"  {mes:<6} {fase:<12} {fuente:<16} {ver:<8} "
              f"{tot_conv:>6,} {tasa:>6.2%} ${tot_profit:>10,.0f}")
        resumen_mes.append({
            'mes': mes, 'fase': fase, 'fuente_gt': fuente,
            'version': ver, 'conversiones': tot_conv,
            'tasa_conversion': round(tasa, 4),
            'profit': round(tot_profit, 2),
        })
    print()

subseccion("4.3 — Delta v1.1 vs v1.0 por mes")
print(f"\n  {'Mes':<6} {'Fase':<12} {'ΔConv':>8} {'ΔTasa pp':>10} "
      f"{'ΔProfit':>12} {'Mejora%':>10} {'Evaluación':>15}")
print(f"  {'─'*76}")

deltas = []
for mes in meses_eval_list:
    r_v10 = next((r for r in resumen_mes if r['mes']==mes and r['version']=='v1.0'), None)
    r_v11 = next((r for r in resumen_mes if r['mes']==mes and r['version']=='v1.1'), None)
    if not r_v10 or not r_v11:
        continue

    dc  = r_v11['conversiones'] - r_v10['conversiones']
    dt  = (r_v11['tasa_conversion'] - r_v10['tasa_conversion']) * 100
    dp  = r_v11['profit'] - r_v10['profit']
    pct = dp / abs(r_v10['profit']) * 100 if r_v10['profit'] != 0 else 0

    fase = MESES_EVALUACION[mes]['fase']
    if fase == 'referencia':
        eval_str = "✅ Corrección OK" if dp >= 0 else "⚠️ Revisar"
    else:
        eval_str = "✅ Mejora proy." if dp > 0 else "─ Sin mejora"

    print(f"  {mes:<6} {fase:<12} {dc:>+8} {dt:>+9.2f}pp "
          f"${dp:>10,.0f} {pct:>+9.1f}% {eval_str:>15}")

    deltas.append({
        'mes': mes, 'fase': fase,
        'delta_conversiones': dc, 'delta_tasa_pp': round(dt, 3),
        'delta_profit': round(dp, 2), 'mejora_pct': round(pct, 2),
    })

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — VISUALIZACIONES EJECUTIVAS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — VISUALIZACIONES EJECUTIVAS")

C_V10   = '#003366'
C_V11   = '#E8500A'
C_REF   = '#2E8B57'
C_PROJ  = '#CC6600'
C_BG    = '#F8F9FA'
C_GRID  = '#DEE2E6'

fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

x_labels = [f"M{m}\n({MESES_EVALUACION[m]['fase'][:3].upper()})"
            for m in meses_eval_list]
x        = np.arange(len(meses_eval_list))
width    = 0.35

profit_v10 = [r['profit'] for r in resumen_mes if r['version']=='v1.0']
profit_v11 = [r['profit'] for r in resumen_mes if r['version']=='v1.1']

# ── Gráfico 1: Profit total ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(C_BG)

b1 = ax1.bar(x - width/2, profit_v10, width, label='v1.0 (original)',
             color=C_V10, alpha=0.85)
b2 = ax1.bar(x + width/2, profit_v11, width, label='v1.1 (recalibrado)',
             color=C_V11, alpha=0.85)

for i, (pv10, pv11) in enumerate(zip(profit_v10, profit_v11)):
    delta = pv11 - pv10
    y_max = max(pv10, pv11)
    color = C_V11 if delta >= 0 else '#CC2200'
    ax1.annotate(f'Δ ${delta:+,.0f}',
                 xy=(x[i], y_max * 1.01),
                 ha='center', fontsize=9, fontweight='bold', color=color)

for i, mes in enumerate(meses_eval_list):
    color_bg = '#E8F5E9' if MESES_EVALUACION[mes]['fase'] == 'referencia' else '#FFF3E0'
    ax1.axvspan(x[i] - 0.5, x[i] + 0.5, alpha=0.2, color=color_bg, zorder=0)

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_title('Profit por Mes — v1.0 vs v1.1 (datos reales del generador)',
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Profit ($)')
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax1.grid(axis='y', color=C_GRID, linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Gráfico 2: Tasa de conversión ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(C_BG)

tasa_v10 = [r['tasa_conversion']*100 for r in resumen_mes if r['version']=='v1.0']
tasa_v11 = [r['tasa_conversion']*100 for r in resumen_mes if r['version']=='v1.1']

ax2.plot(meses_eval_list, tasa_v10, 'o-', color=C_V10, lw=2.5, ms=8, label='v1.0')
ax2.plot(meses_eval_list, tasa_v11, 's--', color=C_V11, lw=2.5, ms=8, label='v1.1')

for mes in meses_eval_list:
    if MESES_EVALUACION[mes]['fase'] == 'proyeccion':
        ax2.axvspan(mes - 0.5, mes + 0.5, alpha=0.12, color=C_PROJ)
        ax2.text(mes, max(tasa_v10+tasa_v11)*0.97, 'Proyección\n(M25)',
                 ha='center', fontsize=7, color=C_PROJ)

ax2.set_xticks(meses_eval_list)
ax2.set_xticklabels([f'M{m}' for m in meses_eval_list])
ax2.set_title('Tasa de Conversión (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Tasa de conversión (%)')
ax2.legend(fontsize=9)
ax2.grid(color=C_GRID, linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Gráfico 3: ECE por mes — productos con drift ─────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(C_BG)

for prod in PRODUCTOS_DRIFT:
    ece_v10_list, ece_v11_list = [], []
    for mes in meses_eval_list:
        sub = df_resultados[
            (df_resultados['mes'] == mes) &
            (df_resultados['producto'] == prod)
        ]
        ece_v10_list.append(
            sub[sub['version']=='v1.0']['ece_seleccion'].values[0]
            if len(sub[sub['version']=='v1.0']) > 0 else None
        )
        ece_v11_list.append(
            sub[sub['version']=='v1.1']['ece_seleccion'].values[0]
            if len(sub[sub['version']=='v1.1']) > 0 else None
        )

    color_v10 = C_V10 if prod == 'tarjeta' else '#5B84C4'
    color_v11 = C_V11 if prod == 'tarjeta' else '#F4A460'
    ax3.plot(meses_eval_list, ece_v10_list, 'o-', color=color_v10,
             lw=2, ms=7, label=f'{prod} v1.0')
    ax3.plot(meses_eval_list, ece_v11_list, 's--', color=color_v11,
             lw=2, ms=7, label=f'{prod} v1.1')

ax3.set_xticks(meses_eval_list)
ax3.set_xticklabels([f'M{m}' for m in meses_eval_list])
ax3.set_title('ECE de Selección — Productos con Drift', fontsize=11, fontweight='bold')
ax3.set_ylabel('Error de calibración')
ax3.legend(fontsize=8, ncol=2)
ax3.grid(color=C_GRID, linewidth=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Gráfico 4: Profit por producto — fase referencia ─────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(C_BG)

mes_ref = [m for m in meses_eval_list if MESES_EVALUACION[m]['fase'] == 'referencia']
prods_plot = sorted(TODOS_PRODUCTOS)
x_prod = np.arange(len(prods_plot))

profit_prod_v10, profit_prod_v11 = [], []
for prod in prods_plot:
    pv10 = sum(r['profit'] for r in resultados_todas
               if r['mes'] in mes_ref and r['producto']==prod and r['version']=='v1.0')
    pv11 = sum(r['profit'] for r in resultados_todas
               if r['mes'] in mes_ref and r['producto']==prod and r['version']=='v1.1')
    profit_prod_v10.append(pv10)
    profit_prod_v11.append(pv11)

ax4.bar(x_prod - width/2, profit_prod_v10, width, color=C_V10, alpha=0.85, label='v1.0')
ax4.bar(x_prod + width/2, profit_prod_v11, width, color=C_V11, alpha=0.85, label='v1.1')
ax4.set_xticks(x_prod)
ax4.set_xticklabels([p[:6] for p in prods_plot], rotation=30, ha='right', fontsize=8)
ax4.set_title('Profit por Producto — Fase Referencia (M22-23)',
              fontsize=10, fontweight='bold')
ax4.set_ylabel('Profit ($)')
ax4.legend(fontsize=8)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax4.grid(axis='y', color=C_GRID, linewidth=0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Gráfico 5: Dashboard KPIs ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#1A1A2E')
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

total_p_v10 = sum(profit_v10)
total_p_v11 = sum(profit_v11)
total_delta  = total_p_v11 - total_p_v10
total_conv_v10 = sum(r['conversiones'] for r in resumen_mes if r['version']=='v1.0')
total_conv_v11 = sum(r['conversiones'] for r in resumen_mes if r['version']=='v1.1')

kpis = [
    ('Profit Total v1.0',      f'${total_p_v10:,.0f}',          C_V10),
    ('Profit Total v1.1',      f'${total_p_v11:,.0f}',          C_V11),
    ('Ganancia v1.1',          f'${total_delta:+,.0f}',          '#00CC66' if total_delta>=0 else '#CC2200'),
    ('Conv. v1.0',             f'{total_conv_v10:,}',            C_V10),
    ('Conv. v1.1',             f'{total_conv_v11:,}',            C_V11),
    ('ΔConversiones',          f'{total_conv_v11-total_conv_v10:+,}', '#FFD700'),
]

ax5.text(0.5, 0.93, 'RESUMEN EJECUTIVO — v1.0 vs v1.1',
         ha='center', fontsize=12, fontweight='bold',
         color='white', transform=ax5.transAxes)

for idx, (nombre, valor, color) in enumerate(kpis):
    col_x = (idx % 3) * 0.33 + 0.165
    row_y = 0.56 if idx < 3 else 0.22
    rect  = FancyBboxPatch((col_x - 0.12, row_y - 0.12), 0.24, 0.28,
                            boxstyle="round,pad=0.01",
                            facecolor='#2D2D4E', edgecolor=color,
                            linewidth=2, transform=ax5.transAxes)
    ax5.add_patch(rect)
    ax5.text(col_x, row_y + 0.10, nombre, ha='center',
             fontsize=8, color='#AAAACC', transform=ax5.transAxes)
    ax5.text(col_x, row_y - 0.02, valor, ha='center',
             fontsize=13, fontweight='bold', color=color, transform=ax5.transAxes)

fig.suptitle(
    'RBlJose — NBO — Semana 7\nEvaluación Operativa Calibradores v1.0 vs v1.1\n'
    '(Basada en datos reales del generador sintético)',
    fontsize=14, fontweight='bold', color='#1A1A2E', y=0.98
)
plt.savefig(f'{DATA_DIR}/nbo_s7_visualizaciones.png',
            dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print(f"  ✅ Visualizaciones guardadas: nbo_s7_visualizaciones.png")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — CONCLUSIÓN EJECUTIVA
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 6 — CONCLUSIÓN EJECUTIVA")

delta_ref  = sum(d['delta_profit'] for d in deltas if d['fase']=='referencia')
delta_proj = sum(d['delta_profit'] for d in deltas if d['fase']=='proyeccion')
pct_ref    = delta_ref  / abs(sum(r['profit'] for r in resumen_mes
                if r['version']=='v1.0' and r['fase']=='referencia')) * 100
pct_proj   = delta_proj / abs(sum(r['profit'] for r in resumen_mes
                if r['version']=='v1.0' and r['fase']=='proyeccion')) * 100 \
             if any(r['fase']=='proyeccion' for r in resumen_mes) else 0

print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║        CONCLUSIÓN EJECUTIVA — SEMANA 7                          ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  Fase referencia (M22-23 — datos con etiqueta real):            ║
  ║    ΔProfit v1.1 vs v1.0  : ${delta_ref:>10,.0f}  ({pct_ref:>+.1f}%)        ║
  ║                                                                  ║
  ║  Fase proyección (M25 — score_propension como GT):              ║
  ║    ΔProfit v1.1 vs v1.0  : ${delta_proj:>10,.0f}  ({pct_proj:>+.1f}%)        ║
  ║                                                                  ║
  ║  Ganancia total (3 meses): ${total_delta:>10,.0f}                  ║
  ║                                                                  ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  IMPLICACIÓN:                                                    ║
  ║  La mejora en ECE no opera directamente sobre el profit —       ║
  ║  opera a través del ranking. Si el overlap de selección es      ║
  ║  alto (> 90%), ambas versiones eligen casi los mismos clientes  ║
  ║  y el impacto financiero es marginal. Esto es esperable en un   ║
  ║  dataset donde el sesgo de calibración es moderado.             ║
  ║                                                                  ║
  ║  El valor principal de v1.1 es la corrección de sesgos en       ║
  ║  el reporte financiero proyectado: el score_nbo calculado        ║
  ║  con p_v11 es una estimación más honesta del retorno esperado.  ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  RECOMENDACIÓN:                                                  ║
  ║    Desplegar v1.1 para tarjeta y seguro_vida.                   ║
  ║    Monitorear impacto en profit real en próximas 2 campañas.    ║
  ║    Si mejora < 1% sostenida → evaluar si el sesgo original      ║
  ║    era suficientemente grave para justificar la recalibración.  ║
  ╚══════════════════════════════════════════════════════════════════╝
""")

print("""
  NOTA METODOLÓGICA para presentación ante Dirección:
  ────────────────────────────────────────────────────
  Esta evaluación usa datos del generador sintético como proxy de la
  realidad. Los resultados son indicativos de la dirección del efecto,
  no de su magnitud exacta en producción real. La validación definitiva
  requiere un Champion-Challenger A/B en producción con clientes reales.
""")


# %%
# ──────────────────────────────────────────────────────────────────────
# OUTPUTS
# ──────────────────────────────────────────────────────────────────────
separador("OUTPUTS FINALES")

df_resultados.drop(columns=['ids_sel']).to_csv(
    f'{DATA_DIR}/nbo_s7_decisiones_comparativas.csv', index=False)
print(f"  ✅ nbo_s7_decisiones_comparativas.csv")

pd.DataFrame(resumen_mes).to_csv(
    f'{DATA_DIR}/nbo_s7_profit_comparativo.csv', index=False)
print(f"  ✅ nbo_s7_profit_comparativo.csv")

pd.DataFrame([
    {'metrica': 'profit_total_v10',    'valor': round(total_p_v10, 2)},
    {'metrica': 'profit_total_v11',    'valor': round(total_p_v11, 2)},
    {'metrica': 'delta_profit_total',  'valor': round(total_delta, 2)},
    {'metrica': 'delta_ref',           'valor': round(delta_ref, 2)},
    {'metrica': 'delta_proj',          'valor': round(delta_proj, 2)},
    {'metrica': 'conv_v10',            'valor': total_conv_v10},
    {'metrica': 'conv_v11',            'valor': total_conv_v11},
    {'metrica': 'delta_conversiones',  'valor': total_conv_v11 - total_conv_v10},
]).to_csv(f'{DATA_DIR}/nbo_s7_resumen_ejecutivo.csv', index=False)
print(f"  ✅ nbo_s7_resumen_ejecutivo.csv")

separador("SEMANA 7 COMPLETADA")
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     RBlJose — NBO — SEMANA 7 COMPLETADA             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 1 : Calibradores v1.1 + datos reales cargados     ║
  ║  ✓ Bloque 2 : p_v10 y p_v11 calculadas sobre scores reales  ║
  ║  ✓ Bloque 3 : Decisiones bajo restricción con GT real       ║
  ║  ✓ Bloque 4 : Análisis comparativo overlap + profit         ║
  ║  ✓ Bloque 5 : Visualizaciones ejecutivas                    ║
  ║  ✓ Bloque 6 : Conclusión y recomendación de despliegue      ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  DIFERENCIA VS VERSIÓN ANTERIOR:                             ║
  ║    Ya NO usa datos sintéticos inventados                     ║
  ║    Calibradores v1.1 vienen de datos reales (S6 corregida)  ║
  ║    GT de conversión = score_propension del generador DAG    ║
  ╚══════════════════════════════════════════════════════════════╝
""")

# %%
import pickle
with open('nbo_calibradores_v11.pkl', 'rb') as f:
    cal = pickle.load(f)
for p, v in cal.items():
    print(p, v['version'], v['A'], v['B'])


