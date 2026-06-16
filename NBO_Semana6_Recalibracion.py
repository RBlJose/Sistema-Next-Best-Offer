# %%
"""
RBlJose — SISTEMA NBO
Semana 6 — Recalibración Selectiva de Platt Scaling  (versión corregida)
=========================================================================
CORRECCIÓN RESPECTO A LA VERSIÓN ANTERIOR:

  La versión anterior generaba datos sintéticos nuevos (rng_data.beta(2,8,n))
  que no tenían ninguna relación con los 20,000 clientes del generador sintético
  ni con los modelos XGBoost reales. Los calibradores v1.1 resultantes eran
  ajustados sobre distribuciones inventadas.

  Corrección: esta versión usa exclusivamente datos reales:
    - nbo_scores_historicos_s5.csv  → scores XGBoost reales (output Semana 5)
    - nbo_ofertas.csv               → convirtio_30d y score_propension reales
    - El cruce de ambos construye el dataset de recalibración correcto:
      (score_xgb_real, p_calibrada_real, convirtio_30d_real) por cliente/mes/producto

  Trigger de recalibración confirmado por Semana 5:
    - tarjeta    : desviación > 25% en meses 23-24
    - seguro_vida: desviación > 25% en meses 23-24

  Ventana de recalibración: meses 22-23 (únicos con etiqueta real completa
  en el período de backtesting). Mes 24-25 no tienen convirtio_30d real.

  Hold-out: no disponible con etiqueta real post-drift. Se reporta la
  mejora in-sample y se documenta que la validación prospectiva real
  ocurrirá en Semana 7 con el mes 25.

Prerequisitos:
  nbo_scores_historicos_s5.csv  — output de Semana 5 corregida
  nbo_ofertas.csv               — datos originales Semana 2
  /models/                      — modelos y calibradores v1.0

Outputs:
  nbo_calibradores_v11.pkl         — calibradores v1.1 serializados
  nbo_scores_recalibrados_s6.csv   — comparativa p_v10 vs p_v11
  nbo_metricas_recalibracion_s6.csv
  nbo_log_version_calibradores_s6.csv
"""

# %%
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import os
import warnings
from datetime import date, datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

# %%
# ──────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────
def separador(titulo):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print(f"{'='*65}")

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

def platt_scaling_weighted(scores, labels, weights):
    def neg_log_likelihood(params):
        A, B = params
        p    = expit(A * scores + B)
        p    = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.sum(weights * (labels * np.log(p) + (1-labels) * np.log(1-p)))
    result = minimize(neg_log_likelihood, x0=[-1.0, 0.0], method='L-BFGS-B')
    return result.x

# %%
# ──────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ──────────────────────────────────────────────────────────────────────
SEED     = 42
DATA_DIR = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'models')
np.random.seed(SEED)

_triggers_path = f'{DATA_DIR}/nbo_triggers_activos.json'
if os.path.exists(_triggers_path):
    with open(_triggers_path) as f:
        _triggers = json.load(f)
    PRODUCTOS_RECALIBRAR = _triggers['productos_recalibrar']
    print(f"  Triggers cargados desde nbo_triggers_activos.json")
    print(f"  Productos a recalibrar: {PRODUCTOS_RECALIBRAR}")
    print(f"  Fecha detección       : {_triggers['fecha_deteccion']}")
    print(f"  Ciclos evaluados      : {_triggers['ciclos_evaluados']}")
else:
    raise FileNotFoundError(
        "nbo_triggers_activos.json no encontrado. "
        "Ejecuta Semana 5 antes de Semana 6."
    )
TODOS_PRODUCTOS = ['tarjeta', 'prestamo', 'microcredito',
                   'seguro_vida', 'seguro_salud', 'inversion']

# Ventana de recalibración: meses con etiqueta real en el período de drift
# Solo meses 22-23 tienen convirtio_30d completo y corresponden al período
# donde el trigger se activó (Semana 5)
MESES_RECALIB = [22, 23]
EWMA_LAMBDA   = 0.5   # mes 23 pesa más que mes 22

PARAMS_NEGOCIO = {
    'tarjeta'     : {'ticket_anual': 465.0,  'costo_contacto': 2.5,
                     'costo_originacion': 45.0},
    'prestamo'    : {'ticket_anual': 675.0,  'costo_contacto': 3.0,
                     'costo_originacion': 60.0},
    'microcredito': {'ticket_anual': 300.0,  'costo_contacto': 4.5,
                     'costo_originacion': 85.0},
    'seguro_vida' : {'ticket_anual': 31.5,   'costo_contacto': 2.0,
                     'costo_originacion': 15.0},
    'seguro_salud': {'ticket_anual': 36.0,   'costo_contacto': 2.0,
                     'costo_originacion': 15.0},
    'inversion'   : {'ticket_anual': 120.0,  'costo_contacto': 1.5,
                     'costo_originacion': 10.0},
}

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA E INTEGRACIÓN DE DATOS REALES
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA E INTEGRACIÓN DE DATOS REALES")

print("""
  Insumos:
    nbo_scores_historicos_s5.csv → scores XGBoost reales por cliente/mes/producto
    nbo_ofertas.csv              → etiquetas reales (convirtio_30d) y score_propension

  La recalibración necesita el par:
    score_xgb_raw  → input al calibrador (output crudo del modelo)
    convirtio_30d  → ground truth real (del generador DAG)

  Este cruce solo es posible porque Semana 5 ahora exporta score_xgb_raw
  por cliente. En la versión anterior esa columna no existía.
""")

# Cargar scores reales de Semana 5
scores_s5 = pd.read_csv(f'{DATA_DIR}/nbo_scores_historicos_s5.csv')
print(f"  scores_historicos_s5 : {len(scores_s5):,} filas | "
      f"meses {scores_s5['mes_scoring'].min()}–{scores_s5['mes_scoring'].max()}")

# Cargar ofertas reales de Semana 2
ofertas = pd.read_csv(f'{DATA_DIR}/nbo_ofertas.csv')
ofertas['id_producto_lower'] = ofertas['id_producto'].str.lower()
print(f"  nbo_ofertas          : {len(ofertas):,} filas | "
      f"meses {ofertas['mes'].min()}–{ofertas['mes'].max()}")

# Verificar meses con etiqueta
meses_con_etiqueta = sorted(
    ofertas[ofertas['etiqueta_completa'] == True]['mes'].unique()
)
meses_sin_etiqueta = sorted(
    ofertas[ofertas['etiqueta_completa'] == False]['mes'].unique()
)
print(f"\n  Meses CON etiqueta real : {meses_con_etiqueta}")
print(f"  Meses SIN etiqueta      : {meses_sin_etiqueta}")

# Verificar que los meses de recalibración tienen etiqueta
for mes in MESES_RECALIB:
    if mes not in meses_con_etiqueta:
        raise ValueError(
            f"Mes {mes} no tiene etiqueta real. "
            f"Verifica MESES_RECALIB. Meses disponibles: {meses_con_etiqueta}"
        )
print(f"\n  ✅ Meses de recalibración {MESES_RECALIB} tienen etiqueta real")

# ── Construir dataset de recalibración ──────────────────────────────
# Cruce: scores_s5 (scores XGBoost) × ofertas (etiquetas reales)
# Clave: id_cliente + mes + producto

# Preparar scores
scores_recalib = scores_s5[
    scores_s5['mes_scoring'].isin(MESES_RECALIB)
].copy()
scores_recalib = scores_recalib.rename(columns={'mes_scoring': 'mes'})

# Preparar etiquetas — solo Tratamiento con etiqueta completa
etiquetas = ofertas[
    (ofertas['mes'].isin(MESES_RECALIB)) &
    (ofertas['grupo'] == 'Tratamiento') &
    (ofertas['etiqueta_completa'] == True) &
    (ofertas['convirtio_30d'].notna())
][['id_cliente', 'mes', 'id_producto_lower', 'convirtio_30d', 'score_propension']].copy()
etiquetas = etiquetas.rename(columns={'id_producto_lower': 'producto_nbo'})

# Merge
df_recalib = scores_recalib.merge(
    etiquetas[['id_cliente', 'mes', 'producto_nbo', 'convirtio_30d']],
    on=['id_cliente', 'mes', 'producto_nbo'],
    how='inner'
)

print(f"\n  Dataset de recalibración construido:")
print(f"  Scores disponibles   : {len(scores_recalib):,}")
print(f"  Etiquetas disponibles: {len(etiquetas):,}")
print(f"  Cruce exitoso        : {len(df_recalib):,} registros")
print(f"  Tasa de match        : {len(df_recalib)/len(scores_recalib):.1%}")

if len(df_recalib) < 100:
    raise ValueError(
        f"Dataset de recalibración muy pequeño ({len(df_recalib)} registros).\n"
        f"Verifica que nbo_scores_historicos_s5.csv tiene columna 'producto_nbo' "
        f"y que nbo_ofertas.csv tiene datos en meses {MESES_RECALIB}."
    )

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — DIAGNÓSTICO DE CALIBRACIÓN v1.0
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — DIAGNÓSTICO DE CALIBRACIÓN v1.0 (SOBRE DATOS REALES)")

print(f"\n  {'Producto':<20} {'Brier v1.0':>12} {'ECE v1.0':>10} "
      f"{'Tasa pred':>11} {'Tasa real':>11} {'Sesgo':>8} {'Trigger':>10}")
print(f"  {'─'*86}")

metricas_v10 = {}
for prod in sorted(TODOS_PRODUCTOS):
    sub = df_recalib[df_recalib['producto_nbo'] == prod]
    if len(sub) == 0:
        print(f"  {prod:<20} {'(sin datos)':>86}")
        continue

    y     = sub['convirtio_30d'].values.astype(float)
    y_hat = sub['p_calibrada'].values

    brier = brier_score_loss(y, y_hat)
    ece   = ece_score(y, y_hat)
    sesgo = y_hat.mean() - y.mean()
    trigger = "❌ SÍ" if prod in PRODUCTOS_RECALIBRAR else "─ NO"

    print(f"  {prod:<20} {brier:>12.5f} {ece:>10.5f} "
          f"{y_hat.mean():>11.4f} {y.mean():>11.4f} "
          f"{sesgo:>+7.4f} {trigger:>10}")

    metricas_v10[prod] = {
        'brier_v10': brier, 'ece_v10': ece,
        'tasa_pred': y_hat.mean(), 'tasa_real': y.mean(), 'sesgo': sesgo,
    }

print(f"""
  Interpretación:
    Sesgo positivo (pred > real): el modelo sobreestima la probabilidad.
    Sesgo negativo (pred < real): el modelo subestima.
    Ambos casos justifican recalibración. Lo importante es la magnitud.
""")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — PESOS EWMA PARA LA VENTANA DE RECALIBRACIÓN
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — VENTANA DE RECALIBRACIÓN CON PESOS EWMA")

T_max = max(MESES_RECALIB)
df_recalib = df_recalib.copy()

df_recalib['peso_ewma'] = df_recalib['mes'].apply(
    lambda m: EWMA_LAMBDA ** (T_max - m)
)
for prod in df_recalib['producto_nbo'].unique():
    mask = df_recalib['producto_nbo'] == prod
    total = df_recalib.loc[mask, 'peso_ewma'].sum()
    df_recalib.loc[mask, 'peso_ewma_norm'] = df_recalib.loc[mask, 'peso_ewma'] / total

print(f"  λ = {EWMA_LAMBDA} → distribución de pesos por mes:")
for mes in sorted(MESES_RECALIB):
    peso = EWMA_LAMBDA ** (T_max - mes)
    suma = sum(EWMA_LAMBDA ** (T_max - m) for m in MESES_RECALIB)
    print(f"    Mes {mes}: {peso/suma:.1%}")

subseccion("Estadísticas de la ventana de recalibración (datos reales)")
print(f"\n  {'Producto':<20} {'N obs':>8} {'N positivos':>12} "
      f"{'Tasa real':>11} {'Score XGB medio':>16}")
print(f"  {'─'*72}")
for prod in sorted(TODOS_PRODUCTOS):
    sub = df_recalib[df_recalib['producto_nbo'] == prod]
    if len(sub) == 0:
        continue
    print(f"  {prod:<20} {len(sub):>8,} "
          f"{int(sub['convirtio_30d'].sum()):>12,} "
          f"{sub['convirtio_30d'].mean():>11.4f} "
          f"{sub['score_xgb_raw'].mean():>16.4f}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — AJUSTE DE CALIBRADORES v1.1
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — AJUSTE DE CALIBRADORES v1.1 (PLATT SCALING SOBRE DATOS REALES)")

print(f"""
  Recalibración SELECTIVA sobre datos reales:
    Productos recalibrados : {PRODUCTOS_RECALIBRAR}
    Productos conservados  : productos sin trigger activo

  Para los productos recalibrados se ajustan A y B en:
    p = sigmoid(A × score_xgb_raw + B)
  minimizando la log-loss ponderada (EWMA) sobre meses {MESES_RECALIB}.

  Para el resto de productos: los calibradores originales de /models/
  se cargan y se mantienen exactamente igual.
""")

# Cargar parámetros v1.0 de los calibradores originales
# Necesitamos extraer A y B del objeto LogisticRegression guardado
calibradores_v10 = {}
for prod in sorted(TODOS_PRODUCTOS):
    cal = joblib.load(os.path.join(MODELS_DIR, f'nbo_calibrador_{prod}_v1.joblib'))
    # LogisticRegression de sklearn: A = coef_[0][0], B = intercept_[0]
    A_v10 = float(cal.coef_[0][0])
    B_v10 = float(cal.intercept_[0])
    calibradores_v10[prod] = {'objeto': cal, 'A': A_v10, 'B': B_v10}
    print(f"  {prod:<20} A={A_v10:+.4f}  B={B_v10:+.4f}  (v1.0 cargado)")

calibradores_v11 = {}

print(f"\n  {'Producto':<20} {'A v1.0':>10} {'B v1.0':>10} "
      f"{'A v1.1':>10} {'B v1.1':>10} {'ΔB':>10} {'Estado':>15}")
print(f"  {'─'*85}")

for prod in sorted(TODOS_PRODUCTOS):
    sub = df_recalib[df_recalib['producto_nbo'] == prod]
    A_v10 = calibradores_v10[prod]['A']
    B_v10 = calibradores_v10[prod]['B']

    if prod in PRODUCTOS_RECALIBRAR and len(sub) >= 20:
        scores  = sub['score_xgb_raw'].values
        labels  = sub['convirtio_30d'].values.astype(float)
        weights = sub['peso_ewma_norm'].values

        A_new, B_new = platt_scaling_weighted(scores, labels, weights)

        calibradores_v11[prod] = {
            'version'     : 'v1.1',
            'A'           : float(A_new),
            'B'           : float(B_new),
            'A_v10'       : A_v10,
            'B_v10'       : B_v10,
            'fecha_fit'   : str(date.today()),
            'ventana_fit' : f'meses_{min(MESES_RECALIB)}_a_{max(MESES_RECALIB)}',
            'ewma_lambda' : EWMA_LAMBDA,
            'n_obs'       : len(sub),
            'n_positivos' : int(labels.sum()),
            'fuente_datos': 'datos_reales_nbo_ofertas',
        }
        delta_b = B_new - B_v10
        estado = "✅ RECALIBRADO"
        print(f"  {prod:<20} {A_v10:>10.4f} {B_v10:>10.4f} "
              f"{A_new:>10.4f} {B_new:>10.4f} {delta_b:>+9.4f} {estado:>15}")
    else:
        # Conservar v1.0 exactamente
        calibradores_v11[prod] = {
            'version'     : 'v1.0',
            'A'           : A_v10,
            'B'           : B_v10,
            'A_v10'       : A_v10,
            'B_v10'       : B_v10,
            'fecha_fit'   : 'semana3_original',
            'ventana_fit' : 'meses_1_a_18',
            'ewma_lambda' : None,
            'n_obs'       : None if prod not in PRODUCTOS_RECALIBRAR else len(sub),
            'n_positivos' : None,
            'fuente_datos': 'original',
        }
        motivo = "─ sin trigger" if prod not in PRODUCTOS_RECALIBRAR else "⚠️  insuf. datos"
        print(f"  {prod:<20} {A_v10:>10.4f} {B_v10:>10.4f} "
              f"{'─':>10} {'─':>10} {'─':>10} {motivo:>15}")

# Serializar calibradores v1.1
ruta_pkl = f'{DATA_DIR}/nbo_calibradores_v11.pkl'
with open(ruta_pkl, 'wb') as f:
    pickle.dump(calibradores_v11, f)
print(f"\n  ✅ Calibradores v1.1 serializados: nbo_calibradores_v11.pkl")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — EVALUACIÓN COMPARATIVA v1.0 vs v1.1
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — EVALUACIÓN COMPARATIVA v1.0 vs v1.1")

print("""
  Evaluación in-sample sobre meses 22-23 (ventana de recalibración).

  Nota: no hay hold-out con etiqueta real post-drift en este dataset
  porque los meses 24-25 no tienen convirtio_30d. La validación
  prospectiva real ocurrirá en Semana 7 con datos del mes 25
  simulados desde score_propension del generador.
""")

# Calcular p_v11 para todo el dataset de recalibración
df_recalib_eval = df_recalib.copy()
p_v11_list = []
for _, row in df_recalib_eval.iterrows():
    prod = row['producto_nbo']
    cal  = calibradores_v11[prod]
    p    = expit(cal['A'] * row['score_xgb_raw'] + cal['B'])
    p_v11_list.append(float(p))
df_recalib_eval['p_v11'] = p_v11_list

resultados_comparacion = []
print(f"\n  {'Producto':<20} {'Brier v1.0':>12} {'Brier v1.1':>12} {'Δ Brier':>10} "
      f"{'ECE v1.0':>10} {'ECE v1.1':>10} {'Δ ECE':>8} {'Resultado':>12}")
print(f"  {'─'*98}")

for prod in sorted(TODOS_PRODUCTOS):
    sub = df_recalib_eval[df_recalib_eval['producto_nbo'] == prod]
    if len(sub) == 0:
        continue

    y     = sub['convirtio_30d'].values.astype(float)
    y_v10 = sub['p_calibrada'].values
    y_v11 = sub['p_v11'].values

    brier_v10 = brier_score_loss(y, y_v10)
    brier_v11 = brier_score_loss(y, y_v11)
    ece_v10   = ece_score(y, y_v10)
    ece_v11   = ece_score(y, y_v11)

    delta_brier = brier_v11 - brier_v10
    delta_ece   = ece_v11   - ece_v10

    if prod in PRODUCTOS_RECALIBRAR:
        if delta_brier < 0 and delta_ece < 0:
            resultado = "✅ MEJORA"
        elif delta_ece < 0:
            resultado = "⚠️ ECE ok"
        else:
            resultado = "❌ REVISAR"
    else:
        resultado = "─ sin cambio"

    print(f"  {prod:<20} {brier_v10:>12.5f} {brier_v11:>12.5f} {delta_brier:>+9.5f} "
          f"{ece_v10:>10.5f} {ece_v11:>10.5f} {delta_ece:>+7.5f} {resultado:>12}")

    resultados_comparacion.append({
        'producto': prod, 'n_obs': len(sub),
        'brier_v10': round(brier_v10, 6), 'brier_v11': round(brier_v11, 6),
        'delta_brier': round(delta_brier, 6),
        'ece_v10': round(ece_v10, 6), 'ece_v11': round(ece_v11, 6),
        'delta_ece': round(delta_ece, 6),
        'tasa_real': round(y.mean(), 4),
        'tasa_pred_v10': round(y_v10.mean(), 4),
        'tasa_pred_v11': round(y_v11.mean(), 4),
        'sesgo_v10': round(y_v10.mean() - y.mean(), 4),
        'sesgo_v11': round(y_v11.mean() - y.mean(), 4),
        'producto_recalibrado': prod in PRODUCTOS_RECALIBRAR,
    })

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — RELIABILITY DIAGRAMS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 6 — RELIABILITY DIAGRAMS (DATOS REALES)")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

productos_orden = sorted(TODOS_PRODUCTOS)

for idx, prod in enumerate(productos_orden):
    row = idx // 3
    col = idx % 3
    ax  = fig.add_subplot(gs[row, col])
    ax.set_facecolor('#1a1d27')

    sub = df_recalib_eval[df_recalib_eval['producto_nbo'] == prod]
    if len(sub) < 10:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', color='white',
                transform=ax.transAxes)
        continue

    y     = sub['convirtio_30d'].values.astype(float)
    y_v10 = sub['p_calibrada'].values
    y_v11 = sub['p_v11'].values

    n_bins = min(8, max(3, len(sub) // 20))

    try:
        fp_v10, mp_v10 = calibration_curve(y, y_v10, n_bins=n_bins, strategy='uniform')
        ax.plot(mp_v10, fp_v10, 'o-', color='#ff6b6b', lw=2, ms=5, label='v1.0')
    except Exception:
        pass

    if prod in PRODUCTOS_RECALIBRAR:
        try:
            fp_v11, mp_v11 = calibration_curve(y, y_v11, n_bins=n_bins, strategy='uniform')
            ax.plot(mp_v11, fp_v11, 's--', color='#4ecdc4', lw=2, ms=5, label='v1.1')
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], 'w--', alpha=0.4, lw=1)

    ece_v10 = ece_score(y, y_v10)
    ece_v11 = ece_score(y, y_v11)

    titulo = f'{prod}' + (' ❌→✅' if prod in PRODUCTOS_RECALIBRAR else '')
    ax.set_title(titulo, color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('Prob. predicha', color='#888', fontsize=7)
    ax.set_ylabel('Frec. real', color='#888', fontsize=7)
    ax.tick_params(colors='#888', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333')

    info = f'ECE v1.0={ece_v10:.4f}\nN={len(sub):,}'
    if prod in PRODUCTOS_RECALIBRAR:
        info += f'\nECE v1.1={ece_v11:.4f}'
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va='top', fontsize=6.5,
            color='#4ecdc4' if prod in PRODUCTOS_RECALIBRAR else '#aaaaaa',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d27',
                      edgecolor='#444', alpha=0.8))

    ax.legend(fontsize=6.5, facecolor='#1a1d27', labelcolor='white',
              edgecolor='#444', loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.15, color='white')

fig.suptitle(
    'RBlJose NBO — Reliability Diagrams: v1.0 vs v1.1\n'
    '(Datos reales nbo_ofertas.csv, meses 22-23)',
    color='white', fontsize=11, fontweight='bold', y=0.98
)
plt.savefig(f'{DATA_DIR}/nbo_calibration_curves_s6.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print(f"  ✅ Reliability diagrams guardados: nbo_calibration_curves_s6.png")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 7 — LOG DE VERSIONES Y AUDITORÍA
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 7 — LOG DE VERSIONES Y AUDITORÍA")

log_versiones = []
for prod in sorted(TODOS_PRODUCTOS):
    cal  = calibradores_v11[prod]
    m_v10 = metricas_v10.get(prod, {})

    sub = df_recalib_eval[df_recalib_eval['producto_nbo'] == prod]
    if len(sub) > 0:
        y     = sub['convirtio_30d'].values.astype(float)
        y_v11 = sub['p_v11'].values
        brier_v11 = brier_score_loss(y, y_v11)
        ece_v11   = ece_score(y, y_v11)
    else:
        brier_v11 = None
        ece_v11   = None

    aprobado = True
    if prod in PRODUCTOS_RECALIBRAR and ece_v11 is not None:
        aprobado = ece_v11 < m_v10.get('ece_v10', 9999)

    log_versiones.append({
        'producto'              : prod,
        'version_anterior'      : 'v1.0',
        'version_nueva'         : cal['version'],
        'A_v10'                 : round(cal['A_v10'], 6),
        'B_v10'                 : round(cal['B_v10'], 6),
        'A_v11'                 : round(cal['A'], 6),
        'B_v11'                 : round(cal['B'], 6),
        'fecha_fit'             : cal['fecha_fit'],
        'ventana_fit'           : cal['ventana_fit'],
        'fuente_datos'          : cal['fuente_datos'],
        'ewma_lambda'           : cal.get('ewma_lambda'),
        'n_obs_fit'             : cal.get('n_obs'),
        'n_positivos_fit'       : cal.get('n_positivos'),
        'ece_v10_recalib'       : round(m_v10.get('ece_v10', 0), 6),
        'ece_v11_recalib'       : round(ece_v11, 6) if ece_v11 else None,
        'brier_v11_recalib'     : round(brier_v11, 6) if brier_v11 else None,
        'aprobado'              : aprobado,
        'motivo_recalibracion'  : (
            'Trigger desviación > 25% meses 22-23 (Semana 5)'
            if prod in PRODUCTOS_RECALIBRAR else 'N/A'
        ),
        'fecha_registro'        : str(datetime.now())[:19],
    })

df_log = pd.DataFrame(log_versiones)

print(f"\n  {'Producto':<20} {'Versión':>8} {'ECE v1.0':>10} "
      f"{'ECE v1.1':>10} {'Δ ECE':>8} {'Aprobado':>10}")
print(f"  {'─'*68}")
for reg in log_versiones:
    ece_v10 = reg['ece_v10_recalib'] or 0
    ece_v11 = reg['ece_v11_recalib'] or 0
    delta   = (ece_v11 - ece_v10) if reg['ece_v11_recalib'] else 0
    flag    = "✅ SÍ" if reg['aprobado'] else "❌ NO"
    recalib = "← RECALIBRADO" if reg['version_nueva'] == 'v1.1' else ""
    print(f"  {reg['producto']:<20} {reg['version_nueva']:>8} "
          f"{ece_v10:>10.5f} {ece_v11:>10.5f} "
          f"{delta:>+7.5f} {flag:>10}  {recalib}")

# %%
# ──────────────────────────────────────────────────────────────────────
# OUTPUTS
# ──────────────────────────────────────────────────────────────────────
separador("OUTPUTS FINALES")

df_log.to_csv(f'{DATA_DIR}/nbo_log_version_calibradores_s6.csv', index=False)
print(f"  ✅ nbo_log_version_calibradores_s6.csv")

pd.DataFrame(resultados_comparacion).to_csv(
    f'{DATA_DIR}/nbo_metricas_recalibracion_s6.csv', index=False)
print(f"  ✅ nbo_metricas_recalibracion_s6.csv")

df_recalib_eval[['id_cliente', 'mes', 'producto_nbo',
                  'score_xgb_raw', 'p_calibrada', 'p_v11',
                  'convirtio_30d']].to_csv(
    f'{DATA_DIR}/nbo_scores_recalibrados_s6.csv', index=False)
print(f"  ✅ nbo_scores_recalibrados_s6.csv")

separador("SEMANA 6 COMPLETADA")
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     RBlJose — NBO — SEMANA 6 COMPLETADA (v2)        ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 1 : Datos reales de nbo_ofertas.csv + S5          ║
  ║  ✓ Bloque 2 : Diagnóstico v1.0 sobre datos reales           ║
  ║  ✓ Bloque 3 : Ventana EWMA sobre meses 22-23                ║
  ║  ✓ Bloque 4 : Platt v1.1 ajustado sobre datos reales        ║
  ║  ✓ Bloque 5 : Comparativa v1.0 vs v1.1 (in-sample)         ║
  ║  ✓ Bloque 6 : Reliability diagrams (datos reales)           ║
  ║  ✓ Bloque 7 : Log de versiones y auditoría                  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  CORRECCIÓN APLICADA:                                        ║
  ║    Ya NO usa datos sintéticos inventados                     ║
  ║    Usa score_xgb_raw real + convirtio_30d real              ║
  ║    Los calibradores v1.1 son defensibles ante auditoría     ║
  ╚══════════════════════════════════════════════════════════════╝
""")


