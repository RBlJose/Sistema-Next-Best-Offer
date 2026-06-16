# %%
"""
RBlJose — SISTEMA NBO
Semana 9 — Champion-Challenger: v1.0 vs v2.0
==============================================

DEFINICIÓN OPERATIVA DEL EXPERIMENTO
──────────────────────────────────────
  Champion  = sistema completo en producción:
                XGBoost v1.0 + calibradores Platt v1.1 +
                CPE (R1-R8) + greedy optimizer + parámetros de negocio
              INVARIANTE: todo lo que no es el modelo queda congelado.
              Si algo más cambia entre brazos, la comparación está
              contaminada y el resultado no es atribuible al modelo.

  Challenger = mismo sistema pero con:
                XGBoost v2.0 + calibradores Platt v2.0
                (8 features estacionales + variable region)
              El CPE, el optimizer, los params de negocio y el
              presupuesto son idénticos al Champion por construcción.

ASIGNACIÓN
───────────
  Split 50/50 aleatorio estratificado por segmento ANTES del scoring.
  Cada brazo recibe su propio universo de clientes, sin solapamiento.
  Cada brazo opera con $25,000 (50% del presupuesto total).
  Los rankings de ambos modelos son completamente independientes:
  ningún modelo "compite" por los mismos clientes del otro brazo.

CRITERIO DE PROMOCIÓN (jerarquía estricta)
───────────────────────────────────────────
  (1) ΔProfit incremental > +2%                       — criterio primario
  (2) IC 95% bootstrap sobre ΔProfit no cruza cero   — significancia económica
  (3) Default rate Challenger ≤ Default rate Champion — guardia de riesgo

  Tasa de conversión: métrica SECUNDARIA e INFORMATIVA.
  No es criterio de decisión. Un Challenger con más conversiones pero
  peor profit (e.g., por captar perfil de mayor riesgo) NO se promueve.

  Justificación de bootstrap vs z-test de proporciones:
    El z-test con N grande (>10,000) puede dar p<0.05 en diferencias
    de 0.01pp que son económicamente irrelevantes. El bootstrap sobre
    ΔProfit es distribution-free y responde directamente la pregunta
    del CFO: "¿cuánto más gana el banco con v2.0?"

  Justificación del guardia de riesgo:
    Con horizonte de observación T+30, los defaults reales tienen
    lag de 6-12 meses. El proxy utilizado es la PD implícita: ratio
    de clientes originados en segmento de score < 600 sobre total
    originados. Si ese ratio sube en Challenger, el modelo está
    captando conversiones en perfil de mayor riesgo real.

Prerequisitos:
  /models/nbo_model_{producto}_v1.joblib
  /models/nbo_calibrador_{producto}_v1.joblib   (o nbo_calibradores_v11.pkl)
  /models/nbo_model_{producto}_v2.joblib
  /models/nbo_calibrador_{producto}_v2.joblib
  /models/nbo_encoder_{col}_v1.joblib
  /models/nbo_model_metadata_v1.json
  /models/nbo_feature_names_v1.json
  nbo_clientes.csv, nbo_features.csv, nbo_ofertas.csv

Outputs:
  nbo_s9_asignacion_cc.csv          — asignación Champion/Challenger por cliente
  nbo_s9_campana_cc.csv             — universo contactado con grupo asignado
  nbo_s9_resultados_cc.csv          — conversiones T+30 por ciclo y brazo
  nbo_s9_decision_estadistica.csv   — ΔProfit, bootstrap IC, default rate y decisión
  nbo_s9_reporte_ejecutivo.csv      — veredicto final consolidado
  nbo_s9_dashboard.png              — dashboard ejecutivo de comparación
"""

# %%
import numpy as np
import pandas as pd
import joblib
import json
import pickle
import os
import warnings
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from scipy.special import expit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')


# %%
# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════
def separador(titulo):
    print(f"\n{'='*70}")
    print(f"  {titulo}")
    print(f"{'='*70}")

def subseccion(titulo):
    print(f"\n  ── {titulo}")

# %%
# ══════════════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)

DATA_DIR   = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'models')

FECHA_INICIO      = date(2024, 1, 1)
CICLOS_CC         = [22, 23, 24, 25]
PRESUPUESTO_TOTAL = 50_000.0
PRESUPUESTO_BRAZO = PRESUPUESTO_TOTAL * 0.50   # $25,000 por brazo — cada uno opera solo
PCT_CONTROL       = 0.20
SPLIT_CC          = 0.50

# ── Criterios de decisión (jerarquía estricta) ─────────────────────
MIN_PROFIT_LIFT   = 0.02         # ΔProfit > 2%  — criterio primario
N_BOOTSTRAP       = 2_000        # iteraciones bootstrap sobre ΔProfit
ALPHA_BOOTSTRAP   = 0.05         # IC 95% — significancia económica
MAX_DELTA_DEFAULT = 0.00         # guardia riesgo: default rate Challenger ≤ Champion

# Proxy de default: clientes originados con score < umbral
SCORE_UMBRAL_DEFAULT = 600

# ── Guardia de potencia mínima para bootstrap ──────────────────────
# Con n < 50 por brazo el bootstrap no es fiable
GUARDIA_N_BOOTSTRAP = 50

# ── Parámetros de negocio — congelados para ambos brazos ──────────
TASA_ORGANICA = {
    'tarjeta': 0.018, 'prestamo': 0.025, 'microcredito': 0.012,
    'seguro_vida': 0.015, 'seguro_salud': 0.014, 'inversion': 0.020,
}

# Límites de originación por brazo = 50% de los límites totales
# Garantiza que cada brazo opera como si cubriera el 50% del mercado
LIMITES_TOTAL = {
    'tarjeta': 2_000, 'prestamo': 1_500, 'microcredito': 800,
    'seguro_vida': 3_000, 'seguro_salud': 3_000, 'inversion': 2_500,
}
LIMITES_BRAZO = {k: max(1, v // 2) for k, v in LIMITES_TOTAL.items()}

PARAMS_NEGOCIO = {
    'tarjeta':      {'ticket_anual': 465.0,  'costo_contacto': 2.5,  'costo_originacion': 45.0,
                     'pd': 0.055, 'lgd': 0.75, 'rwa': 1.00},
    'prestamo':     {'ticket_anual': 675.0,  'costo_contacto': 3.0,  'costo_originacion': 60.0,
                     'pd': 0.045, 'lgd': 0.78, 'rwa': 0.75},
    'microcredito': {'ticket_anual': 300.0,  'costo_contacto': 4.5,  'costo_originacion': 85.0,
                     'pd': 0.095, 'lgd': 0.82, 'rwa': 0.75},
    'seguro_vida':  {'ticket_anual': 31.5,   'costo_contacto': 2.0,  'costo_originacion': 15.0,
                     'pd': 0.000, 'lgd': 0.00, 'rwa': 0.00},
    'seguro_salud': {'ticket_anual': 36.0,   'costo_contacto': 2.0,  'costo_originacion': 15.0,
                     'pd': 0.000, 'lgd': 0.00, 'rwa': 0.00},
    'inversion':    {'ticket_anual': 120.0,  'costo_contacto': 1.5,  'costo_originacion': 10.0,
                     'pd': 0.000, 'lgd': 0.00, 'rwa': 0.00},
}
PRODUCTOS = list(PARAMS_NEGOCIO.keys())

# Features estacionales del modelo v2.0
FEATURES_ESTACIONALES_V2 = [
    'es_utilidades', 'es_decimo_tercero', 'es_decimo_cuarto',
    'es_inicio_clases', 'es_navidad', 'es_impuesto_renta',
    'mes_calendario', 'trimestre',
]

# Reglas CPE — idénticas para ambos brazos (invariante del experimento)
R7_CICLOS_FATIGA   = 2
R8_CICLOS_SIN_NADA = 3
R8_MESES_SUPRESION = 2

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 0 — VERIFICACIÓN DE INVARIANTES DEL EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 0 — VERIFICACIÓN DE INVARIANTES DEL EXPERIMENTO")

print("""
  Un Champion-Challenger es un experimento controlado.
  La validez causal del resultado depende de que la ÚNICA diferencia
  entre brazos sea el modelo. Cualquier otra asimetría contamina el
  resultado y hace que el ΔProfit no sea atribuible al modelo.

  Invariantes verificados en este bloque:
    [I1] CPE: mismas reglas R1-R8 y parámetros (R7, R8)
    [I2] Optimizer: mismo algoritmo greedy, mismo criterio de orden
    [I3] Parámetros de negocio: mismo ticket, costo_contacto, pd, lgd
    [I4] Presupuesto: $25,000 por brazo (split exacto)
    [I5] Límites de originación: 50% del total por brazo
    [I6] Ground truth T+30: misma fuente (nbo_ofertas.csv)
    [I7] Semilla de simulación T+30: compartida entre brazos
         (la única fuente de variación es qué clientes selecciona
         cada modelo, no el ruido de la simulación)
""")

INVARIANTES = {
    'I1_CPE_R7_ciclos_fatiga'   : R7_CICLOS_FATIGA,
    'I1_CPE_R8_ciclos_sin_nada' : R8_CICLOS_SIN_NADA,
    'I1_CPE_R8_meses_supresion' : R8_MESES_SUPRESION,
    'I2_optimizer'              : 'greedy_ratio_score_costo',
    'I3_params_negocio'         : 'PARAMS_NEGOCIO (congelado)',
    'I4_presupuesto_brazo'      : PRESUPUESTO_BRAZO,
    'I5_limites_brazo'          : LIMITES_BRAZO,
    'I6_ground_truth'           : 'nbo_ofertas.csv (score_propension + convirtio_30d)',
    'I7_seed_t30'               : f'np.random.default_rng({SEED}) — compartida',
}

print(f"  {'Invariante':<35} {'Valor':}")
print(f"  {'─'*65}")
for k, v in INVARIANTES.items():
    print(f"  {k:<35} {v}")
print(f"\n  ✅ Invariantes documentados — cualquier desviación invalida el CC")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA DE MODELOS Champion (v1.0+v1.1) y Challenger (v2.0)")

# ── Feature specs ──────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'nbo_model_metadata_v1.json')) as f:
    metadata_v1 = json.load(f)

with open(os.path.join(MODELS_DIR, 'nbo_feature_names_v1.json')) as f:
    feature_spec_v1 = json.load(f)

FEATURES_FINALES_V1  = feature_spec_v1['features_finales']
FEATURES_CATEGORICAS = feature_spec_v1['features_categoricas']

# Feature spec v2.0: extiende v1.0 con las features estacionales
_path_spec_v2 = os.path.join(MODELS_DIR, 'nbo_feature_names_v2.json')
if os.path.exists(_path_spec_v2):
    with open(_path_spec_v2) as f:
        feature_spec_v2 = json.load(f)
    print("  ✅ nbo_feature_names_v2.json cargado")
else:
    feature_spec_v2 = {
        'features_numericas'  : feature_spec_v1['features_numericas'] + FEATURES_ESTACIONALES_V2,
        'features_categoricas': FEATURES_CATEGORICAS,
        'features_finales'    : (
            feature_spec_v1['features_numericas']
            + FEATURES_ESTACIONALES_V2
            + [c + '_enc' for c in FEATURES_CATEGORICAS]
        ),
    }
    print("  ⚠️  nbo_feature_names_v2.json no encontrado → construido extendiendo v1.0")

FEATURES_FINALES_V2 = feature_spec_v2['features_finales']

# ── Encoders compartidos ───────────────────────────────────────────
encoders = {
    col: joblib.load(os.path.join(MODELS_DIR, f'nbo_encoder_{col}_v1.joblib'))
    for col in FEATURES_CATEGORICAS
}

# ── Champion: XGBoost v1.0 + parámetros Platt v1.1 ─────────────────
# Los calibradores v1.1 están en nbo_calibradores_v11.pkl como dict
# {producto: {'A': float, 'B': float, 'version': str}}
# Se usa sigmoid(A × score_raw + B) directamente.
modelos_v1 = {
    prod: joblib.load(os.path.join(MODELS_DIR, f'nbo_model_{prod}_v1.joblib'))
    for prod in PRODUCTOS
}

_ruta_pkl_v11 = os.path.join(DATA_DIR, 'nbo_calibradores_v11.pkl')
if os.path.exists(_ruta_pkl_v11):
    with open(_ruta_pkl_v11, 'rb') as f:
        cal_params_v11 = pickle.load(f)
    print("  ✅ Calibradores Champion (v1.1) cargados desde nbo_calibradores_v11.pkl")
else:
    # Fallback: extraer A, B de los objetos sklearn v1.0
    cal_params_v11 = {}
    for prod in PRODUCTOS:
        _c = joblib.load(os.path.join(MODELS_DIR, f'nbo_calibrador_{prod}_v1.joblib'))
        cal_params_v11[prod] = {
            'version': 'v1.0', 'A': float(_c.coef_[0][0]), 'B': float(_c.intercept_[0])
        }
    print("  ⚠️  Usando calibradores v1.0 como fallback para Champion")

# ── Challenger: XGBoost v2.0 + calibradores Platt v2.0 ────────────
modelos_v2      = {}
calibradores_v2 = {}
for prod in PRODUCTOS:
    _pm = os.path.join(MODELS_DIR, f'nbo_model_{prod}_v2.joblib')
    _pc = os.path.join(MODELS_DIR, f'nbo_calibrador_{prod}_v2.joblib')
    if not os.path.exists(_pm):
        raise FileNotFoundError(
            f"Modelo Challenger no encontrado: {_pm}\n"
            "Ejecuta nbo_semana8_modelo_v2.py antes de continuar."
        )
    modelos_v2[prod]      = joblib.load(_pm)
    calibradores_v2[prod] = joblib.load(_pc)

print(f"\n  {'Producto':<20} {'Cal Champion':>14} {'Features Ch':>13} {'Features Chl':>14}")
print(f"  {'─'*65}")
for prod in sorted(PRODUCTOS):
    ver = cal_params_v11[prod]['version']
    print(f"  {prod:<20} {ver:>14} {len(FEATURES_FINALES_V1):>13} {len(FEATURES_FINALES_V2):>14}")

print(f"\n  ✅ Champion  : XGBoost v1.0 + Platt {cal_params_v11[PRODUCTOS[0]]['version']}")
print(f"  ✅ Challenger: XGBoost v2.0 + Platt v2.0")
extra_feats = len(FEATURES_FINALES_V2) - len(FEATURES_FINALES_V1)
print(f"  ✅ Features adicionales Challenger: {extra_feats} ({FEATURES_ESTACIONALES_V2})")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — CARGA DE DATOS E INPUTS BASE
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — CARGA DE DATOS")

clientes = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
features = pd.read_csv(f'{DATA_DIR}/nbo_features.csv')
ofertas  = pd.read_csv(f'{DATA_DIR}/nbo_ofertas.csv')

ofertas['fecha_oferta']      = pd.to_datetime(ofertas['fecha_oferta'])
ofertas['id_producto_lower'] = ofertas['id_producto'].str.lower()

COLS_CLIENTES_SLIM = [
    'id_cliente', 'segmento', 'edad', 'ocupacion', 'zona_geografica',
    'canal_principal', 'antiguedad_meses', 'ingreso_mensual', 'score_crediticio',
    'score_buro', 'tiene_atraso_hist', 'max_atraso_dias', 'ratio_deuda_init',
    'hijos', 'estado_civil', 'activo',
]
clientes_slim = clientes[COLS_CLIENTES_SLIM].copy()
# Lookup score para el guardia de riesgo (default rate proxy)
score_por_cliente = clientes.set_index('id_cliente')['score_crediticio']

print(f"  clientes : {len(clientes):>8,}  |  features: {len(features):>8,}  |  ofertas: {len(ofertas):>8,}")

# ── Ground truth lookup ──────────────────────────────────────────────
# Clave: (id_cliente, mes, producto) → score_propension, convirtio_30d
# El mismo lookup alimenta a AMBOS brazos — invariante I6
gt_lookup = (
    ofertas[ofertas['grupo'] == 'Tratamiento']
    [['id_cliente', 'mes', 'id_producto_lower',
      'score_propension', 'convirtio_30d', 'etiqueta_completa']]
    .rename(columns={'id_producto_lower': 'producto_nbo'})
    .set_index(['id_cliente', 'mes', 'producto_nbo'])
)
meses_con_etiqueta = set(ofertas[ofertas['etiqueta_completa'] == True]['mes'].unique())
print(f"  Meses con etiqueta real : {sorted(meses_con_etiqueta)}")
print(f"  Meses sin etiqueta (GT = score_propension): "
      f"{sorted(set(CICLOS_CC) - meses_con_etiqueta)}")

# ── Historial pre-CC ─────────────────────────────────────────────────
MES_INICIO_CC = min(CICLOS_CC)
historial_base = ofertas[
    (ofertas['grupo'] == 'Tratamiento') & (ofertas['mes'] < MES_INICIO_CC)
].copy()
historial_base['id_producto_lower'] = historial_base['id_producto'].str.lower()
print(f"\n  Historial pre-CC (mes < {MES_INICIO_CC}): {len(historial_base):,} contactos reales")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — ASIGNACIÓN ESTRATIFICADA 50/50
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — ASIGNACIÓN ESTRATIFICADA Champion/Challenger")

print(f"""
  División: 50% Champion / 50% Challenger
  Estratificación: por segmento (Masivo / Preferente / Pyme / Premium)

  Por qué estratificar por segmento y no por otra variable:
    El producto NBO lo determina el modelo — no existe antes del scoring.
    El segmento es observable ex-ante y es el predictor más fuerte de
    ingreso, score crediticio y distribución de riesgo.
    Estratificar por él garantiza que ambos brazos tienen la misma
    mezcla de riesgo — condición necesaria para que el ΔProfit sea
    atribuible al modelo y no a heterogeneidad del universo.

  La asignación es FIJA durante todos los ciclos del CC.
  Un cliente que entra al brazo Champion en mes 22 no puede migrar
  al Challenger en mes 24. La migración generaría contaminación
  (el historial de contacto del cliente dependería del modelo).
""")

rng_assign = np.random.default_rng(SEED)
asig_rows  = []

for seg in ['Masivo', 'Preferente', 'Pyme', 'Premium']:
    ids_seg = clientes[clientes['segmento'] == seg]['id_cliente'].values.copy()
    rng_assign.shuffle(ids_seg)
    n_ch = int(len(ids_seg) * SPLIT_CC)
    for i, cid in enumerate(ids_seg):
        asig_rows.append({
            'id_cliente': cid,
            'segmento'  : seg,
            'grupo_cc'  : 'Champion' if i < n_ch else 'Challenger',
        })

df_asignacion = pd.DataFrame(asig_rows)
ids_champion   = set(df_asignacion[df_asignacion['grupo_cc'] == 'Champion']['id_cliente'])
ids_challenger = set(df_asignacion[df_asignacion['grupo_cc'] == 'Challenger']['id_cliente'])

# Verificar balance y comparabilidad entre brazos
print(f"  {'Segmento':<15} {'Champion':>10} {'Challenger':>12} {'Split':>8} "
      f"{'Score medio Ch':>16} {'Score medio Chl':>17}")
print(f"  {'─'*82}")
for seg in ['Masivo', 'Preferente', 'Pyme', 'Premium']:
    sub     = df_asignacion[df_asignacion['segmento'] == seg]
    n_ch    = (sub['grupo_cc'] == 'Champion').sum()
    n_chl   = (sub['grupo_cc'] == 'Challenger').sum()
    ids_ch_seg  = sub[sub['grupo_cc'] == 'Champion']['id_cliente']
    ids_chl_seg = sub[sub['grupo_cc'] == 'Challenger']['id_cliente']
    sc_ch  = clientes[clientes['id_cliente'].isin(ids_ch_seg)]['score_crediticio'].mean()
    sc_chl = clientes[clientes['id_cliente'].isin(ids_chl_seg)]['score_crediticio'].mean()
    delta_score = abs(sc_ch - sc_chl)
    flag = "✅" if delta_score < 5 else "⚠️"
    print(f"  {seg:<15} {n_ch:>10,} {n_chl:>12,} {n_ch/len(sub):>8.1%} "
          f"{sc_ch:>16.1f} {sc_chl:>17.1f}  {flag}")

tot_ch  = (df_asignacion['grupo_cc'] == 'Champion').sum()
tot_chl = (df_asignacion['grupo_cc'] == 'Challenger').sum()
print(f"  {'─'*82}")
print(f"  {'TOTAL':<15} {tot_ch:>10,} {tot_chl:>12,} {tot_ch/len(df_asignacion):>8.1%}")

# Dividir historial pre-CC por brazo
historial_ch  = historial_base[historial_base['id_cliente'].isin(ids_champion)].copy()
historial_chl = historial_base[historial_base['id_cliente'].isin(ids_challenger)].copy()
print(f"\n  Historial Champion  : {len(historial_ch):,}  |  Historial Challenger: {len(historial_chl):,}")
print(f"  ✅ Asignación fija para todos los ciclos del CC")

# %%
# ══════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════

# ── F1. ABT por mes ──────────────────────────────────────────────────
def construir_abt_mes(mes: int) -> pd.DataFrame:
    """
    Construye ABT completa para el mes dado.
    Incluye features base (v1.0) + estacionales si existen en el CSV.
    El subconjunto por brazo se hace FUERA de esta función para
    que ambos modelos trabajen sobre la misma transformación de datos.
    """
    COLS_BASE = [
        'id_cliente', 'mes', 'gasto_3m', 'gasto_supermercado_3m',
        'gasto_farmacia_3m', 'depositos_efectivo_3m', 'tx_digitales_proporcion',
        'saldo_promedio_90d', 'saldo_tendencia', 'variabilidad_saldo',
        'ratio_cuota_ingreso', 'dias_desde_ult_credito',
        'indice_estres_macro', 'estres_x_riesgo',
    ]
    cols_est = [c for c in FEATURES_ESTACIONALES_V2 if c in features.columns]
    features_mes = features[features['mes'] == mes][COLS_BASE + cols_est].copy()
    abt = features_mes.merge(clientes_slim, on='id_cliente', how='left')

    # Feature engineering derivada — idéntica a S3/S5/S7
    abt['bucket_dias_credito'] = pd.cut(
        abt['dias_desde_ult_credito'],
        bins=[-1, 30, 90, 180, 365, 730, 9999],
        labels=['0_30d', '31_90d', '91_180d', '181_365d', '366_730d', 'mas_730d']
    )
    abt['ratio_saldo_ingreso'] = abt['saldo_promedio_90d'] / (abt['ingreso_mensual'] + 1)
    abt['score_muy_alto']      = (abt['score_crediticio'] > 800).astype(int)
    abt['edad_objetivo_sv']    = ((abt['edad'] >= 30) & (abt['edad'] <= 55)).astype(int)
    abt['indep_no_urbano']     = (
        (abt['ocupacion'] == 'Independiente') & (abt['zona_geografica'] != 'Urbana')
    ).astype(int)

    for col in FEATURES_CATEGORICAS:
        abt[col + '_enc'] = encoders[col].transform(abt[col].astype(str))

    abt['canal_principal_val'] = (
        abt['id_cliente'].map(clientes.set_index('id_cliente')['canal_principal'])
        .fillna('Digital')
    )
    return abt


# ── F2. Scoring de un brazo ──────────────────────────────────────────
def score_brazo(abt_brazo: pd.DataFrame,
                modelos_dict: dict,
                cal_fn,
                features_finales: list,
                version_str: str,
                mes: int) -> pd.DataFrame:
    """
    Genera scores para todos los productos sobre los clientes del brazo.
    Si una feature de v2.0 no existe (CSV no patcheado), se imputa a 0
    con advertencia: ausencia de señal, no error.
    Retorna top-2 por cliente por score_nbo descendente.
    """
    fecha_campana = FECHA_INICIO + relativedelta(months=mes - 1)
    scores_list   = []

    for prod in sorted(PRODUCTOS):
        p      = PARAMS_NEGOCIO[prod]
        modelo = modelos_dict[prod]

        abt_work = abt_brazo.copy()
        for feat in features_finales:
            if feat not in abt_work.columns:
                abt_work[feat] = 0.0   # impute conservador

        X     = abt_work[features_finales].copy()
        p_raw = modelo.predict_proba(X)[:, 1]
        p_cal = cal_fn(p_raw, prod)

        ingreso_esp = p_cal * (p['ticket_anual'] - p['costo_originacion'])
        perdida_esp = p['pd'] * p['lgd'] * p['ticket_anual'] * p['rwa']
        score_nbo   = ingreso_esp - perdida_esp - p['costo_contacto']

        scores_list.append(pd.DataFrame({
            'id_cliente'     : abt_brazo['id_cliente'].values,
            'producto_nbo'   : prod,
            'score_xgb_raw'  : p_raw,
            'p_calibrada'    : p_cal,
            'score_nbo'      : score_nbo,
            'costo_contacto' : p['costo_contacto'],
            'ratio_nbo_costo': score_nbo / max(p['costo_contacto'], 0.01),
            'canal_principal': abt_brazo['canal_principal_val'].values,
            'version_modelo' : version_str,
            'mes_scoring'    : mes,
            'fecha_scoring'  : str(fecha_campana),
        }))

    df_scores = pd.concat(scores_list, ignore_index=True)
    df_scores = df_scores.sort_values(
        ['id_cliente', 'score_nbo'], ascending=[True, False]
    )
    df_scores['rank'] = df_scores.groupby('id_cliente').cumcount() + 1
    return df_scores[df_scores['rank'] <= 2].copy()


# ── F3. Calibración ───────────────────────────────────────────────────
def cal_champion(scores_raw: np.ndarray, prod: str) -> np.ndarray:
    """p = sigmoid(A × score_xgb_raw + B)  con parámetros v1.1"""
    A = cal_params_v11[prod]['A']
    B = cal_params_v11[prod]['B']
    return expit(A * scores_raw + B)

def cal_challenger(scores_raw: np.ndarray, prod: str) -> np.ndarray:
    """Calibrador sklearn v2.0"""
    return calibradores_v2[prod].predict_proba(scores_raw.reshape(-1, 1))[:, 1]


# ── F4. CPE — Contact Policy Engine (R1-R8) ──────────────────────────
def construir_estado_clientes(historial, ciclos_anteriores, mes_actual):
    if len(historial) == 0:
        return pd.DataFrame(columns=[
            'id_cliente', 'fecha_ultimo_contacto', 'productos_activos',
            'fatiga_por_producto', 'ciclos_sin_conversion', 'suprimido_hasta_mes',
        ])

    ultimo_contacto = (
        historial.groupby('id_cliente')['fecha_oferta']
        .max().reset_index()
        .rename(columns={'fecha_oferta': 'fecha_ultimo_contacto'})
    )

    conv_col = 'convirtio_30d' if 'convirtio_30d' in historial.columns else 'acepto'
    productos_activos = (
        historial[historial[conv_col] == 1]
        .groupby('id_cliente')['id_producto_lower']
        .apply(lambda x: set(x))
        .reset_index()
        .rename(columns={'id_producto_lower': 'productos_activos'})
    )

    estado = ultimo_contacto.merge(productos_activos, on='id_cliente', how='left')
    estado['productos_activos'] = estado['productos_activos'].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    fatiga_dict   = {}
    sin_nada_dict = {}
    suprimido_dict = {}

    for ciclo_info in ciclos_anteriores:
        df_c  = ciclo_info.get('contactados')
        mes_c = ciclo_info.get('mes')
        if df_c is None or len(df_c) == 0:
            continue
        for _, row in df_c.iterrows():
            cid  = row['id_cliente']
            prod = row['producto_nbo']
            conv = row.get('convirtio_30d', 0)
            if cid not in fatiga_dict:
                fatiga_dict[cid] = {}
            fatiga_dict[cid][prod] = 0 if conv == 1 \
                else fatiga_dict[cid].get(prod, 0) + 1
        if 'convirtio_30d' in df_c.columns:
            conv_ids = set(df_c[df_c['convirtio_30d'] == 1]['id_cliente'])
            for cid in set(df_c['id_cliente']):
                if cid in conv_ids:
                    sin_nada_dict[cid] = 0
                    suprimido_dict.pop(cid, None)
                else:
                    sin_nada_dict[cid] = sin_nada_dict.get(cid, 0) + 1
                    if sin_nada_dict[cid] >= R8_CICLOS_SIN_NADA:
                        suprimido_dict[cid] = mes_c + R8_MESES_SUPRESION
                        sin_nada_dict[cid]  = 0

    estado['fatiga_por_producto']   = estado['id_cliente'].map(lambda x: fatiga_dict.get(x, {}))
    estado['ciclos_sin_conversion'] = estado['id_cliente'].map(lambda x: sin_nada_dict.get(x, 0))
    estado['suprimido_hasta_mes']   = estado['id_cliente'].map(lambda x: suprimido_dict.get(x, 0))
    return estado


def aplicar_cpe(scores_top2, estado, mes_actual, fecha_campana_dt, ciclos_ant):
    """CPE idéntico para ambos brazos — invariante I1."""
    rec_r1 = scores_top2[scores_top2['rank'] == 1].copy()
    rec_r2 = scores_top2[scores_top2['rank'] == 2].copy()

    rec_r1 = rec_r1.merge(estado, on='id_cliente', how='left')
    rec_r1 = rec_r1.merge(
        clientes_slim[['id_cliente', 'score_crediticio', 'max_atraso_dias',
                       'antiguedad_meses', 'activo']],
        on='id_cliente', how='left'
    )
    for col, default in [('productos_activos', set()), ('fatiga_por_producto', {}),
                          ('suprimido_hasta_mes', 0), ('ciclos_sin_conversion', 0)]:
        rec_r1[col] = rec_r1[col].apply(
            lambda x: x if isinstance(x, type(default)) else type(default)()
        )

    rec_r1['dias_desde_contacto'] = (
        fecha_campana_dt - rec_r1['fecha_ultimo_contacto']
    ).dt.days.fillna(9999).astype(int)

    rec_r1['pasa_r1'] = rec_r1['dias_desde_contacto'] >= 30
    rec_r1['pasa_r2'] = rec_r1['dias_desde_contacto'] >= 15
    rec_r1['pasa_r3'] = True
    rec_r1['pasa_r4'] = [
        prod not in act
        for prod, act in zip(rec_r1['producto_nbo'], rec_r1['productos_activos'])
    ]
    rec_r1['pasa_r5'] = (
        (rec_r1['score_crediticio'] >= 550) &
        (rec_r1['max_atraso_dias']  <= 30)  &
        (rec_r1['antiguedad_meses'] >= 3)   &
        (rec_r1['activo'] == True)
    )
    rec_r1['pasa_r8']     = rec_r1['suprimido_hasta_mes'] < mes_actual
    rec_r1['fatigado_r7'] = [
        fat.get(prod, 0) >= R7_CICLOS_FATIGA
        for prod, fat in zip(rec_r1['producto_nbo'], rec_r1['fatiga_por_producto'])
    ]
    rec_r1['elegible'] = (
        rec_r1['pasa_r1'] & rec_r1['pasa_r2'] & rec_r1['pasa_r3'] &
        rec_r1['pasa_r4'] & rec_r1['pasa_r5'] & rec_r1['pasa_r8'] &
        ~rec_r1['fatigado_r7']
    )

    # Fallback R6
    bloq = rec_r1[~rec_r1['elegible'] & rec_r1['pasa_r5'] & rec_r1['pasa_r8']]['id_cliente'].values
    fallbacks = pd.DataFrame()
    if len(bloq) > 0:
        r2e = rec_r2[rec_r2['id_cliente'].isin(bloq)].copy()
        if len(r2e) > 0:
            r2e = r2e.merge(estado, on='id_cliente', how='left')
            r2e = r2e.merge(
                clientes_slim[['id_cliente', 'score_crediticio', 'max_atraso_dias',
                               'antiguedad_meses', 'activo']],
                on='id_cliente', how='left'
            )
            for col, default in [('productos_activos', set()), ('fatiga_por_producto', {})]:
                r2e[col] = r2e[col].apply(lambda x: x if isinstance(x, type(default)) else type(default)())
            r2e['suprimido_hasta_mes'] = r2e['suprimido_hasta_mes'].fillna(0)
            r2e['dias_desde_contacto'] = (
                fecha_campana_dt - r2e['fecha_ultimo_contacto']
            ).dt.days.fillna(9999).astype(int)
            r2e['fatigado_r7'] = [
                fat.get(prod, 0) >= R7_CICLOS_FATIGA
                for prod, fat in zip(r2e['producto_nbo'], r2e['fatiga_por_producto'])
            ]
            pasa_r4_r2 = [
                prod not in act
                for prod, act in zip(r2e['producto_nbo'], r2e['productos_activos'])
            ]
            r2e['elegible'] = (
                (r2e['dias_desde_contacto'] >= 30) &
                (r2e['score_crediticio'] >= 550)   &
                (r2e['max_atraso_dias']  <= 30)    &
                (r2e['antiguedad_meses'] >= 3)     &
                pasa_r4_r2 & ~r2e['fatigado_r7']
            )
            r2e['es_fallback'] = True
            fallbacks = r2e[
                r2e['elegible'] &
                ~r2e['id_cliente'].isin(rec_r1[rec_r1['elegible']]['id_cliente'])
            ].copy()

    elegibles = rec_r1[rec_r1['elegible']].copy()
    elegibles['es_fallback'] = False
    universo = pd.concat([elegibles, fallbacks], ignore_index=True) \
               if len(fallbacks) > 0 else elegibles.copy()

    return universo, {
        'n_inicial'    : rec_r1['id_cliente'].nunique(),
        'bloq_cooling' : (~rec_r1['pasa_r1']).sum(),
        'bloq_activo'  : (~rec_r1['pasa_r4']).sum(),
        'bloq_elegib'  : (~rec_r1['pasa_r5']).sum(),
        'fallbacks'    : len(fallbacks),
        'universo_fin' : len(universo),
    }


# ── F5. Optimizador greedy — invariante I2 ───────────────────────────
def optimizador_greedy(universo, presupuesto, limites):
    ordenado  = universo.sort_values('ratio_nbo_costo', ascending=False)
    pres_rest = presupuesto
    contadores = {prod: 0 for prod in PRODUCTOS}
    sel        = []
    for _, cl in ordenado.iterrows():
        prod  = cl['producto_nbo']
        costo = cl['costo_contacto']
        if cl['score_nbo'] < 0:            continue
        if pres_rest < costo:              continue
        if contadores[prod] >= limites.get(prod, np.inf): continue
        sel.append(cl.to_dict())
        pres_rest        -= costo
        contadores[prod] += 1
    return pd.DataFrame(sel), presupuesto - pres_rest


# ── F6. Simulación T+30 — rng compartida entre brazos (invariante I7) ─
def simular_t30(universo_combinado, mes, rng_t30):
    """
    INVARIANTE I7: la misma rng se usa para ambos brazos dentro del mismo ciclo.
    El orden de iteración es determinístico (sort por id_cliente).
    Garantía: la única fuente de diferencia entre brazos es la selección
    de clientes por el modelo, no la realización del proceso aleatorio.
    """
    universo_sorted = universo_combinado.sort_values('id_cliente').reset_index(drop=True)
    tiene_etiqueta  = mes in meses_con_etiqueta
    conversiones    = []

    for _, cl in universo_sorted.iterrows():
        prod = cl['producto_nbo']
        cid  = cl['id_cliente']
        p    = PARAMS_NEGOCIO[prod]
        key  = (cid, mes, prod)

        if cl['grupo_t'] == 'Tratamiento':
            if key in gt_lookup.index:
                row_gt = gt_lookup.loc[key]
                if isinstance(row_gt, pd.DataFrame):
                    row_gt = row_gt.iloc[0]
                if tiene_etiqueta and not pd.isna(row_gt['convirtio_30d']):
                    conv = int(row_gt['convirtio_30d'])
                else:
                    p_real = float(row_gt['score_propension'])
                    conv   = int(rng_t30.random() < p_real)
            else:
                conv = int(rng_t30.random() < TASA_ORGANICA.get(prod, 0.02))
        else:
            conv = int(rng_t30.random() < TASA_ORGANICA.get(prod, 0.02))

        conversiones.append({
            'id_cliente'     : cid,
            'producto_nbo'   : prod,
            'grupo_cc'       : cl['grupo_cc'],
            'grupo_t'        : cl['grupo_t'],
            'p_calibrada'    : cl['p_calibrada'],
            'score_crediticio': score_por_cliente.get(cid, 650),
            'convirtio_30d'  : conv,
            'ingreso_real'   : (p['ticket_anual'] - p['costo_originacion']) if conv else 0.0,
            'costo_contacto' : p['costo_contacto'] if cl['grupo_t'] == 'Tratamiento' else 0.0,
            'mes'            : mes,
            'fuente_gt'      : 'etiqueta_real' if tiene_etiqueta else 'score_propension',
            'version_modelo' : cl.get('version_modelo', ''),
        })

    return pd.DataFrame(conversiones)


# ── F7. Profit incremental de un brazo ───────────────────────────────
def calcular_profit_incremental(df_conv, brazo, producto=None):
    """
    Profit incremental = ingreso causal - costo de contacto.
    Ingreso causal: solo las conversiones por encima de la tasa orgánica.
    Si producto=None, agrega sobre todos los productos del brazo.
    """
    filtro = df_conv['grupo_cc'] == brazo
    if producto:
        filtro = filtro & (df_conv['producto_nbo'] == producto)

    trat = df_conv[filtro & (df_conv['grupo_t'] == 'Tratamiento')]
    ctrl = df_conv[filtro & (df_conv['grupo_t'] == 'Control')]

    if len(trat) == 0:
        return 0.0, 0.0, 0.0

    if producto:
        p = PARAMS_NEGOCIO[producto]
        tasa_t = trat['convirtio_30d'].mean()
        tasa_c = ctrl['convirtio_30d'].mean() if len(ctrl) > 0 else 0.0
        inc    = max(tasa_t - tasa_c, 0)
        ingreso = inc * len(trat) * (p['ticket_anual'] - p['costo_originacion'])
        costo   = trat['costo_contacto'].sum()
        return ingreso - costo, ingreso, costo
    else:
        total_profit = 0.0
        total_ingreso = 0.0
        total_costo   = 0.0
        for prod in df_conv[filtro]['producto_nbo'].unique():
            pf, ing, cost = calcular_profit_incremental(df_conv, brazo, prod)
            total_profit  += pf
            total_ingreso += ing
            total_costo   += cost
        return total_profit, total_ingreso, total_costo


# ── F8. Bootstrap sobre ΔProfit ──────────────────────────────────────
def bootstrap_delta_profit(df_conv, producto=None,
                            n_boot=N_BOOTSTRAP, seed=SEED):
    """
    IC 95% del ΔProfit (Challenger - Champion) por bootstrap.

    Por qué bootstrap sobre profit y no z-test sobre tasas:
      El profit es función no lineal de las conversiones y el costo
      (que varía por producto). Su distribución no es binomial.
      El bootstrap es el estimador natural para funciones de estadísticas
      complejas sin supuestos distribucionales adicionales.

    Procedimiento:
      En cada iteración se muestrea con reemplazo el universo de
      observaciones de cada brazo INDEPENDIENTEMENTE, preservando
      la estructura de grupos (Champion trat/ctrl, Challenger trat/ctrl).
      Se calcula el profit incremental en cada muestra y se obtiene
      la distribución empírica del delta.
    """
    rng_boot = np.random.default_rng(seed)

    filtro = pd.Series([True] * len(df_conv), index=df_conv.index)
    if producto:
        filtro = df_conv['producto_nbo'] == producto

    df_ch  = df_conv[filtro & (df_conv['grupo_cc'] == 'Champion')].reset_index(drop=True)
    df_chl = df_conv[filtro & (df_conv['grupo_cc'] == 'Challenger')].reset_index(drop=True)

    if len(df_ch) < GUARDIA_N_BOOTSTRAP or len(df_chl) < GUARDIA_N_BOOTSTRAP:
        return np.nan, np.nan, np.nan, np.array([])

    deltas = []
    for _ in range(n_boot):
        idx_ch  = rng_boot.integers(0, len(df_ch),  len(df_ch))
        idx_chl = rng_boot.integers(0, len(df_chl), len(df_chl))
        samp_ch  = df_ch.iloc[idx_ch]
        samp_chl = df_chl.iloc[idx_chl]

        # Recalcular profit en la muestra
        def _profit_muestra(df_samp):
            total = 0.0
            for prod_b in df_samp['producto_nbo'].unique():
                sub  = df_samp[df_samp['producto_nbo'] == prod_b]
                p    = PARAMS_NEGOCIO[prod_b]
                trat = sub[sub['grupo_t'] == 'Tratamiento']
                ctrl = sub[sub['grupo_t'] == 'Control']
                if len(trat) == 0:
                    continue
                tasa_t = trat['convirtio_30d'].mean()
                tasa_c = ctrl['convirtio_30d'].mean() if len(ctrl) > 0 else 0.0
                inc    = max(tasa_t - tasa_c, 0)
                ing    = inc * len(trat) * (p['ticket_anual'] - p['costo_originacion'])
                costo  = trat['costo_contacto'].sum()
                total += ing - costo
            return total

        p_ch  = _profit_muestra(samp_ch)
        p_chl = _profit_muestra(samp_chl)
        deltas.append(p_chl - p_ch)

    deltas = np.array(deltas)
    ic_lo  = np.percentile(deltas, 2.5)
    ic_hi  = np.percentile(deltas, 97.5)
    delta_obs = deltas.mean()
    return delta_obs, ic_lo, ic_hi, deltas


# ── F9. Proxy de default rate ─────────────────────────────────────────
def calcular_default_rate_proxy(df_conv, brazo, producto=None):
    """
    Proxy de riesgo: ratio de clientes ORIGINADOS (convirtieron)
    con score < SCORE_UMBRAL_DEFAULT sobre total originados.

    Por qué es un proxy y no el default real:
      El default real tiene lag de 6-12 meses — no observable en T+30.
      Este proxy captura si el modelo está originando en segmentos de
      mayor riesgo crediticio, lo cual es el precursor del default futuro.

    Umbral 600: puntos debajo del scoring mínimo estándar para tarjeta
    (650) pero por encima del mínimo absoluto (550). Clientes en este
    rango tienen una PD estructuralmente mayor.
    """
    filtro = (df_conv['grupo_cc'] == brazo) & (df_conv['grupo_t'] == 'Tratamiento')
    if producto:
        filtro = filtro & (df_conv['producto_nbo'] == producto)

    originados = df_conv[filtro & (df_conv['convirtio_30d'] == 1)]
    if len(originados) == 0:
        return np.nan

    n_alto_riesgo = (originados['score_crediticio'] < SCORE_UMBRAL_DEFAULT).sum()
    return n_alto_riesgo / len(originados)

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — LOOP CHAMPION-CHALLENGER
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — LOOP CHAMPION-CHALLENGER (meses 22–25)")

print(f"""
  Por ciclo, el flujo es paralelo e independiente por brazo:
    ABT completa del mes
      ├── Subconjunto Champion  → scoring v1.0 → CPE → greedy ($25K)
      └── Subconjunto Challenger→ scoring v2.0 → CPE → greedy ($25K)
    Unión de ambos universos contactados
    T+30 con rng compartida (I7): único punto de contacto entre brazos
""")

# Contenedores acumulativos
resultados_cc   = []    # df_conv de cada ciclo
campanas_all    = []    # universos contactados
metricas_ciclo  = []    # KPIs por ciclo y brazo (informativo)
ciclos_ch_ant   = []
ciclos_chl_ant  = []

# rng compartida para T+30 — invariante I7
rng_global = np.random.default_rng(SEED)

for mes in CICLOS_CC:

    separador(f"CICLO MES {mes} — "
              f"{(FECHA_INICIO + relativedelta(months=mes-1)).strftime('%B %Y').upper()}")

    fecha_campana    = FECHA_INICIO + relativedelta(months=mes - 1)
    fecha_campana_dt = pd.Timestamp(fecha_campana)

    # ── 4.1 ABT del mes ─────────────────────────────────────────────
    abt_mes = construir_abt_mes(mes)
    abt_ch  = abt_mes[abt_mes['id_cliente'].isin(ids_champion)].copy()
    abt_chl = abt_mes[abt_mes['id_cliente'].isin(ids_challenger)].copy()
    print(f"\n  ABT  Champion: {len(abt_ch):,}  |  Challenger: {len(abt_chl):,}")

    # ── 4.2 Scoring independiente por brazo ─────────────────────────
    scores_ch  = score_brazo(abt_ch,  modelos_v1, cal_champion,
                              FEATURES_FINALES_V1, 'champion',  mes)
    scores_chl = score_brazo(abt_chl, modelos_v2, cal_challenger,
                              FEATURES_FINALES_V2, 'challenger', mes)

    subseccion(f"Scoring mes {mes}")
    print(f"  Score NBO medio rank-1  Champion: "
          f"{scores_ch[scores_ch['rank']==1]['score_nbo'].mean():.2f}")
    print(f"  Score NBO medio rank-1 Challenger: "
          f"{scores_chl[scores_chl['rank']==1]['score_nbo'].mean():.2f}")

    # ── 4.3 CPE — mismo algoritmo, historiales separados ───────────
    est_ch  = construir_estado_clientes(historial_ch,  ciclos_ch_ant,  mes)
    est_chl = construir_estado_clientes(historial_chl, ciclos_chl_ant, mes)

    univ_ch,  s_ch  = aplicar_cpe(scores_ch,  est_ch,  mes, fecha_campana_dt, ciclos_ch_ant)
    univ_chl, s_chl = aplicar_cpe(scores_chl, est_chl, mes, fecha_campana_dt, ciclos_chl_ant)

    subseccion(f"CPE mes {mes}")
    print(f"  {'':5} {'Champion':>12} {'Challenger':>14}")
    print(f"  {'Scoring inicial':<22} {s_ch['n_inicial']:>12,} {s_chl['n_inicial']:>14,}")
    print(f"  {'Universo elegible':<22} {s_ch['universo_fin']:>12,} {s_chl['universo_fin']:>14,}")
    print(f"  {'Bloq. cooling':<22} {s_ch['bloq_cooling']:>12,} {s_chl['bloq_cooling']:>14,}")

    if len(univ_ch) == 0 and len(univ_chl) == 0:
        ciclos_ch_ant.append( {'mes': mes, 'contactados': None})
        ciclos_chl_ant.append({'mes': mes, 'contactados': None})
        continue

    # ── 4.4 Greedy optimizer — mismo algoritmo, presupuesto separado
    optim_ch,  pres_ch  = optimizador_greedy(univ_ch,  PRESUPUESTO_BRAZO, LIMITES_BRAZO)
    optim_chl, pres_chl = optimizador_greedy(univ_chl, PRESUPUESTO_BRAZO, LIMITES_BRAZO)

    subseccion(f"Optimizer mes {mes}")
    print(f"  Champion   seleccionados: {len(optim_ch):,}  "
          f"(${pres_ch:,.2f} / ${PRESUPUESTO_BRAZO:,.0f})")
    print(f"  Challenger seleccionados: {len(optim_chl):,}  "
          f"(${pres_chl:,.2f} / ${PRESUPUESTO_BRAZO:,.0f})")

    if len(optim_ch) == 0 and len(optim_chl) == 0:
        ciclos_ch_ant.append( {'mes': mes, 'contactados': None})
        ciclos_chl_ant.append({'mes': mes, 'contactados': None})
        continue

    # ── 4.5 Asignación Tratamiento/Control 80/20 dentro de cada brazo
    for df_opt, brazo_id in [(optim_ch, 'Champion'), (optim_chl, 'Challenger')]:
        df_opt['grupo_t']      = np.where(
            rng_global.random(len(df_opt)) > PCT_CONTROL,
            'Tratamiento', 'Control'
        )
        df_opt['grupo_cc']       = brazo_id
        df_opt['mes_campana']    = mes
        df_opt['fecha_campana']  = str(fecha_campana)
        df_opt['id_campana']     = f"CC_{mes:02d}_{brazo_id[:3].upper()}"

    campanas_all.extend(optim_ch.to_dict('records'))
    campanas_all.extend(optim_chl.to_dict('records'))

    # ── 4.6 T+30 con rng compartida ─────────────────────────────────
    universo_combinado = pd.concat([optim_ch, optim_chl], ignore_index=True)
    df_conv = simular_t30(universo_combinado, mes, rng_global)
    resultados_cc.append(df_conv)

    # KPIs informativos por ciclo (no son el criterio de decisión)
    subseccion(f"T+30 mes {mes} — resumen informativo")
    for brazo in ['Champion', 'Challenger']:
        sub_trat = df_conv[
            (df_conv['grupo_cc'] == brazo) & (df_conv['grupo_t'] == 'Tratamiento')
        ]
        if len(sub_trat) == 0:
            continue
        tasa = sub_trat['convirtio_30d'].mean()
        pf, _, _ = calcular_profit_incremental(df_conv, brazo)
        print(f"  {brazo:<13}  conv.rate={tasa:.4f}  n_trat={len(sub_trat):,}"
              f"  profit_inc=${pf:,.2f}  [INFORMATIVO]")
        metricas_ciclo.append({
            'mes': mes, 'brazo': brazo,
            'n_tratamiento': len(sub_trat),
            'tasa_conv_trat': round(tasa, 4),
            'profit_incremental': round(pf, 2),
            'fuente_gt': df_conv['fuente_gt'].iloc[0],
        })

    # ── 4.7 Actualizar historiales por brazo ─────────────────────────
    for df_opt, hist_ref, ciclos_ref, brazo_id in [
        (optim_ch,  historial_ch,  ciclos_ch_ant,  'Champion'),
        (optim_chl, historial_chl, ciclos_chl_ant, 'Challenger'),
    ]:
        trat = df_opt[df_opt['grupo_t'] == 'Tratamiento'].copy()
        if len(trat) > 0:
            conv_lookup = df_conv[
                (df_conv['grupo_cc'] == brazo_id) &
                (df_conv['grupo_t']  == 'Tratamiento')
            ][['id_cliente', 'producto_nbo', 'convirtio_30d']]
            trat = trat.merge(conv_lookup, on=['id_cliente', 'producto_nbo'], how='left')
            nuevo_hist = pd.DataFrame({
                'id_cliente'       : trat['id_cliente'],
                'id_producto_lower': trat['producto_nbo'],
                'fecha_oferta'     : pd.to_datetime(trat['fecha_campana']),
                'mes'              : mes,
                'grupo'            : 'Tratamiento',
                'convirtio_30d'    : trat['convirtio_30d'].fillna(0),
            })
            if brazo_id == 'Champion':
                historial_ch = pd.concat([historial_ch, nuevo_hist], ignore_index=True)
            else:
                historial_chl = pd.concat([historial_chl, nuevo_hist], ignore_index=True)

        ciclos_ref.append({
            'mes'         : mes,
            'contactados' : df_conv[df_conv['grupo_cc'] == brazo_id].copy(),
        })

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — CRITERIO DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — CRITERIO DE DECISIÓN: ΔProfit + Bootstrap IC + Default Rate")

print(f"""
  Criterios de promoción (jerarquía estricta):
    (1) ΔProfit incremental > {MIN_PROFIT_LIFT:.0%}        — criterio primario
    (2) IC 95% bootstrap sobre ΔProfit no cruza 0  — significancia económica
    (3) Default rate Challenger ≤ Champion         — guardia de riesgo

  Tasa de conversión: MÉTRICA INFORMATIVA, no criterio de decisión.

  Por qué (1) y (2) juntos y no solo (2):
    Un IC 95% que no cruza cero pero con lift de 0.5% no justifica
    el costo operativo de un reentrenamiento: retraining pipeline,
    validación regulatoria, documentación, rollback plan.
    El MIN_PROFIT_LIFT del {MIN_PROFIT_LIFT:.0%} es el umbral de materialidad económica.

  Por qué (3) es un guardia y no un criterio:
    Con T+30 no se ven defaults reales (lag 6-12 meses).
    El proxy de score < {SCORE_UMBRAL_DEFAULT} captura si el Challenger origina
    en el segmento de mayor riesgo latente. Si ese ratio sube,
    la ganancia de profit en el corto plazo es espuria.
""")

if not resultados_cc:
    print("  ⚠️  Sin resultados. El loop no produjo conversiones.")
else:
    df_all_conv = pd.concat(resultados_cc, ignore_index=True)

    # ── Profit acumulado por brazo (todos los ciclos) ────────────────
    profit_ch_total,  ing_ch,  cost_ch  = calcular_profit_incremental(df_all_conv, 'Champion')
    profit_chl_total, ing_chl, cost_chl = calcular_profit_incremental(df_all_conv, 'Challenger')
    delta_profit_global = profit_chl_total - profit_ch_total
    lift_global = delta_profit_global / abs(profit_ch_total) if abs(profit_ch_total) > 0 else 0.0

    subseccion("5.1 — Profit incremental acumulado (todos los ciclos)")
    print(f"\n  {'':35} {'Champion':>14} {'Challenger':>14} {'Delta':>14}")
    print(f"  {'─'*80}")
    print(f"  {'Ingreso incremental':<35} ${ing_ch:>12,.2f}  ${ing_chl:>12,.2f}  "
          f"${ing_chl-ing_ch:>+12,.2f}")
    print(f"  {'Costo contactación':<35} ${cost_ch:>12,.2f}  ${cost_chl:>12,.2f}  "
          f"${cost_chl-cost_ch:>+12,.2f}")
    print(f"  {'Profit incremental':<35} ${profit_ch_total:>12,.2f}  "
          f"${profit_chl_total:>12,.2f}  ${delta_profit_global:>+12,.2f}")
    print(f"  {'Lift Challenger vs Champion':<35} {'':>14} {'':>14} {lift_global:>+13.2%}")

    # ── Bootstrap sobre ΔProfit ──────────────────────────────────────
    subseccion("5.2 — Bootstrap IC 95% sobre ΔProfit")
    print(f"\n  {N_BOOTSTRAP:,} iteraciones | Semilla {SEED}")
    print(f"\n  {'Producto':<20} {'ΔProfit obs':>13} {'IC 95% lo':>12} {'IC 95% hi':>12} "
          f"{'IC cruza 0':>12} {'Crit. (1)':>11} {'Crit. (2)':>11}")
    print(f"  {'─'*96}")

    decisiones  = []
    delta_global_obs, ic_lo_global, ic_hi_global, boot_global = \
        bootstrap_delta_profit(df_all_conv, n_boot=N_BOOTSTRAP, seed=SEED)

    # Por producto
    for prod in sorted(PRODUCTOS):
        d_obs, ic_lo, ic_hi, _ = bootstrap_delta_profit(
            df_all_conv, producto=prod, n_boot=N_BOOTSTRAP, seed=SEED
        )

        profit_ch_p,  _, _ = calcular_profit_incremental(df_all_conv, 'Champion',   prod)
        profit_chl_p, _, _ = calcular_profit_incremental(df_all_conv, 'Challenger', prod)
        lift_p = (profit_chl_p - profit_ch_p) / abs(profit_ch_p) \
                 if abs(profit_ch_p) > 0.01 else 0.0

        dr_ch  = calcular_default_rate_proxy(df_all_conv, 'Champion',   prod)
        dr_chl = calcular_default_rate_proxy(df_all_conv, 'Challenger', prod)
        delta_dr = (dr_chl - dr_ch) if (not np.isnan(dr_chl) and not np.isnan(dr_ch)) else np.nan

        # Evaluar criterios
        n_ch_trat  = len(df_all_conv[
            (df_all_conv['grupo_cc']     == 'Champion') &
            (df_all_conv['producto_nbo'] == prod) &
            (df_all_conv['grupo_t']      == 'Tratamiento')
        ])
        n_chl_trat = len(df_all_conv[
            (df_all_conv['grupo_cc']     == 'Challenger') &
            (df_all_conv['producto_nbo'] == prod) &
            (df_all_conv['grupo_t']      == 'Tratamiento')
        ])
        potencia_ok = min(n_ch_trat, n_chl_trat) >= GUARDIA_N_BOOTSTRAP

        crit1 = potencia_ok and (lift_p > MIN_PROFIT_LIFT)
        crit2 = potencia_ok and (not np.isnan(ic_lo)) and (ic_lo > 0)
        crit3 = np.isnan(delta_dr) or (delta_dr <= MAX_DELTA_DEFAULT)

        if not potencia_ok:
            decision = "⏳ N insuf."
        elif crit1 and crit2 and crit3:
            decision = "✅ PROMOVER"
        elif crit1 and crit2 and not crit3:
            decision = "❌ Riesgo alto"
        elif (crit1 or crit2) and not (crit1 and crit2):
            decision = "⚠️  Parcial"
        else:
            decision = "─ Mantener"

        ic_cruza = "NO" if (not np.isnan(ic_lo) and ic_lo > 0) else "SÍ"
        d_str    = f"${d_obs:+,.2f}"  if not np.isnan(d_obs) else "N/A"
        lo_str   = f"${ic_lo:+,.2f}" if not np.isnan(ic_lo) else "N/A"
        hi_str   = f"${ic_hi:+,.2f}" if not np.isnan(ic_hi) else "N/A"

        print(f"  {prod:<20} {d_str:>13} {lo_str:>12} {hi_str:>12} "
              f"{ic_cruza:>12} {str(crit1):>11} {str(crit2):>11}  {decision}")

        decisiones.append({
            'producto'            : prod,
            'n_ch_trat'           : n_ch_trat,
            'n_chl_trat'          : n_chl_trat,
            # Conversión — métrica informativa
            'tasa_conv_champion'  : round(
                df_all_conv[(df_all_conv['grupo_cc']=='Champion') &
                             (df_all_conv['producto_nbo']==prod) &
                             (df_all_conv['grupo_t']=='Tratamiento')
                            ]['convirtio_30d'].mean(), 4
            ) if n_ch_trat > 0 else None,
            'tasa_conv_challenger': round(
                df_all_conv[(df_all_conv['grupo_cc']=='Challenger') &
                             (df_all_conv['producto_nbo']==prod) &
                             (df_all_conv['grupo_t']=='Tratamiento')
                            ]['convirtio_30d'].mean(), 4
            ) if n_chl_trat > 0 else None,
            # Criterios de decisión
            'profit_champion'     : round(profit_ch_p,  2),
            'profit_challenger'   : round(profit_chl_p, 2),
            'lift_profit'         : round(lift_p,        4),
            'delta_profit_boot'   : round(d_obs,  2) if not np.isnan(d_obs)  else None,
            'ic95_lo'             : round(ic_lo,   2) if not np.isnan(ic_lo)  else None,
            'ic95_hi'             : round(ic_hi,   2) if not np.isnan(ic_hi)  else None,
            'ic_cruza_cero'       : not (not np.isnan(ic_lo) and ic_lo > 0),
            'default_rate_ch'     : round(dr_ch,      4) if not np.isnan(dr_ch)  else None,
            'default_rate_chl'    : round(dr_chl,     4) if not np.isnan(dr_chl) else None,
            'delta_default_rate'  : round(delta_dr,   4) if not np.isnan(delta_dr) else None,
            'crit1_lift'          : crit1,
            'crit2_ic'            : crit2,
            'crit3_riesgo'        : crit3,
            'potencia_ok'         : potencia_ok,
            'decision_producto'   : decision,
        })

    subseccion("5.3 — Bootstrap global (todos los productos)")
    print(f"\n  ΔProfit global observado : ${delta_global_obs:+,.2f}")
    print(f"  IC 95% global            : [${ic_lo_global:+,.2f}, ${ic_hi_global:+,.2f}]")
    ic_global_cruza = ic_lo_global <= 0 if not np.isnan(ic_lo_global) else True
    print(f"  IC cruza cero            : {'SÍ ⚠️' if ic_global_cruza else 'NO ✅'}")
    print(f"  Lift global              : {lift_global:+.2%}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — VEREDICTO FINAL
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 6 — VEREDICTO FINAL")

df_decisiones = pd.DataFrame(decisiones) if decisiones else pd.DataFrame()

if len(df_decisiones) == 0:
    veredicto          = "INDETERMINADO — sin datos suficientes"
    promover_global    = False
    prods_promover     = []
    prods_riesgo       = []
    prods_mantener     = []
    prods_insuf        = []
else:
    prods_promover = df_decisiones[
        df_decisiones['decision_producto'].str.contains('PROMOVER', na=False)
    ]['producto'].tolist()
    prods_riesgo   = df_decisiones[
        df_decisiones['decision_producto'].str.contains('Riesgo', na=False)
    ]['producto'].tolist()
    prods_mantener = df_decisiones[
        df_decisiones['decision_producto'].str.contains('Mantener', na=False)
    ]['producto'].tolist()
    prods_insuf    = df_decisiones[
        df_decisiones['decision_producto'].str.contains('insuf', na=False)
    ]['producto'].tolist()

    n_evaluados     = len(df_decisiones[df_decisiones['potencia_ok'] == True])
    n_promover      = len(prods_promover)
    crit_global_ic  = not ic_global_cruza
    crit_global_lft = lift_global > MIN_PROFIT_LIFT
    crit_global_risk= len(prods_riesgo) == 0

    # Criterio global: IC no cruza cero AND lift > 2% AND sin productos con riesgo alto
    promover_global = crit_global_ic and crit_global_lft and crit_global_risk

    veredicto = ("PROMOVER v2.0 → nuevo Champion"
                 if promover_global else
                 "MANTENER v1.0 como Champion")

print(f"""
  ╔════════════════════════════════════════════════════════════════════╗
  ║          VEREDICTO CHAMPION-CHALLENGER — SEMANA 9                 ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║                                                                    ║
  ║  (1) Lift global ΔProfit > {MIN_PROFIT_LIFT:.0%}?    {str(crit_global_lft if decisiones else '—'):>6}   ({lift_global:>+.2%})     ║
  ║  (2) IC 95% bootstrap no cruza 0?   {str(crit_global_ic if decisiones else '—'):>6}                    ║
  ║  (3) Sin productos con riesgo alto? {str(crit_global_risk if decisiones else '—'):>6}                    ║
  ║                                                                    ║
  ║  Profit Champion   (acum. 4 ciclos): ${profit_ch_total:>10,.2f}               ║
  ║  Profit Challenger (acum. 4 ciclos): ${profit_chl_total:>10,.2f}               ║
  ║  ΔProfit global                    : ${delta_profit_global:>+10,.2f}               ║
  ║  IC 95% bootstrap                  : [${ic_lo_global:>+8,.2f},  ${ic_hi_global:>+8,.2f}]    ║
  ║                                                                    ║
  ║  Productos que aprueban promoción  : {str(prods_promover):<32} ║
  ║  Productos con riesgo alto         : {str(prods_riesgo):<32} ║
  ║  Productos que mantienen Champion  : {str(prods_mantener):<32} ║
  ║  Productos sin datos suficientes   : {str(prods_insuf):<32} ║
  ║                                                                    ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  DECISIÓN: {veredicto:<55}  ║
  ╚════════════════════════════════════════════════════════════════════╝
""")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 7 — DASHBOARD EJECUTIVO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 7 — DASHBOARD EJECUTIVO")

C_CH  = '#003366'
C_CHL = '#E8500A'
C_BG  = '#F8F9FA'
C_GRID = '#DEE2E6'

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.33)

# ── Panel 1: Profit incremental acumulado por producto ───────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(C_BG)

prods_plot = sorted(PRODUCTOS)
x = np.arange(len(prods_plot))
w = 0.35

profit_ch_por_prod  = [
    calcular_profit_incremental(df_all_conv, 'Champion',   p)[0]
    if len(df_all_conv) > 0 else 0.0 for p in prods_plot
] if resultados_cc else [0.0] * len(prods_plot)
profit_chl_por_prod = [
    calcular_profit_incremental(df_all_conv, 'Challenger', p)[0]
    if len(df_all_conv) > 0 else 0.0 for p in prods_plot
] if resultados_cc else [0.0] * len(prods_plot)

ax1.bar(x - w/2, profit_ch_por_prod,  w, color=C_CH,  alpha=0.85, label='Champion v1.0')
ax1.bar(x + w/2, profit_chl_por_prod, w, color=C_CHL, alpha=0.85, label='Challenger v2.0')

for i, (pch, pchl) in enumerate(zip(profit_ch_por_prod, profit_chl_por_prod)):
    delta = pchl - pch
    ymax  = max(pch, pchl, 0)
    col   = C_CHL if delta >= 0 else '#CC2200'
    ax1.annotate(f'Δ${delta:+,.0f}', xy=(x[i], ymax * 1.02 + 10),
                 ha='center', fontsize=8.5, fontweight='bold', color=col)

ax1.set_xticks(x)
ax1.set_xticklabels(prods_plot, fontsize=10)
ax1.set_title('Profit Incremental Acumulado por Producto — meses 22-25\n'
              '(Criterio de decisión primario)',
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Profit Incremental ($)')
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax1.grid(axis='y', color=C_GRID, linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Panel 2: Evolución del profit por ciclo ──────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(C_BG)

if metricas_ciclo:
    df_mc = pd.DataFrame(metricas_ciclo)
    for brazo, col, ls in [('Champion', C_CH, '-'), ('Challenger', C_CHL, '--')]:
        sub = df_mc[df_mc['brazo'] == brazo]
        ax2.plot(sub['mes'], sub['profit_incremental'],
                 'o' + ls, color=col, lw=2.5, ms=8, label=brazo)

ax2.axvline(x=23.5, color='gray', linestyle=':', lw=1.5, alpha=0.6)
ax2.text(23.7, ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else -100,
         '← etiq. real | score_prop. →', fontsize=7, color='gray')
ax2.set_xticks(CICLOS_CC)
ax2.set_xticklabels([f'M{m}' for m in CICLOS_CC])
ax2.set_title('Profit Incremental por Ciclo\n[Informativo — no criterio de decisión]',
              fontsize=11, fontweight='bold')
ax2.set_ylabel('Profit ($)')
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
ax2.grid(color=C_GRID, linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel 3: Bootstrap distribution ΔProfit global ───────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(C_BG)

if resultados_cc and len(boot_global) > 0:
    ax3.hist(boot_global, bins=60, color=C_CHL, alpha=0.7, edgecolor='white', lw=0.3)
    ax3.axvline(x=0,              color='black', lw=2, linestyle='-',  label='Δ=0')
    ax3.axvline(x=delta_global_obs, color=C_CHL, lw=2, linestyle='--', label='Obs.')
    ax3.axvline(x=ic_lo_global,   color='red',   lw=1.5, linestyle=':', label='IC 2.5%')
    ax3.axvline(x=ic_hi_global,   color='green', lw=1.5, linestyle=':', label='IC 97.5%')
    ax3.fill_betweenx(
        [0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 1],
        ic_lo_global, ic_hi_global, alpha=0.10, color=C_CHL
    )
    ax3.legend(fontsize=8)

ax3.set_title(f'Bootstrap ΔProfit Global (n={N_BOOTSTRAP:,})\n'
              f'IC 95%: [${ic_lo_global:+,.2f}, ${ic_hi_global:+,.2f}]',
              fontsize=11, fontweight='bold')
ax3.set_xlabel('ΔProfit Challenger − Champion ($)')
ax3.set_ylabel('Frecuencia bootstrap')
ax3.grid(color=C_GRID, linewidth=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Panel 4: Default rate proxy por producto ─────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(C_BG)

if resultados_cc and len(df_decisiones) > 0 and 'default_rate_ch' in df_decisiones.columns:
    dr_ch_vals  = df_decisiones['default_rate_ch'].fillna(0).values
    dr_chl_vals = df_decisiones['default_rate_chl'].fillna(0).values
    ax4.bar(x - w/2, dr_ch_vals  * 100, w, color=C_CH,  alpha=0.85, label='Champion')
    ax4.bar(x + w/2, dr_chl_vals * 100, w, color=C_CHL, alpha=0.85, label='Challenger')

ax4.set_xticks(x)
ax4.set_xticklabels([p[:6] for p in prods_plot], rotation=30, ha='right', fontsize=8)
ax4.set_title(f'Guardia de Riesgo: % Originados con Score < {SCORE_UMBRAL_DEFAULT}\n'
              f'(Proxy default rate — Challenger ≤ Champion requerido)',
              fontsize=10, fontweight='bold')
ax4.set_ylabel('% originados alto riesgo')
ax4.legend(fontsize=9)
ax4.grid(axis='y', color=C_GRID, linewidth=0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel 5: KPI Summary ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#1A1A2E')
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

_cv = '#00CC66' if promover_global else '#CC4400'
kpis = [
    ('Profit Champion',   f'${profit_ch_total:,.0f}',        C_CH),
    ('Profit Challenger', f'${profit_chl_total:,.0f}',       C_CHL),
    ('ΔProfit',           f'${delta_profit_global:+,.0f}',   _cv),
    ('Lift',              f'{lift_global:+.2%}',             _cv),
    ('IC 95% lo',         f'${ic_lo_global:+,.2f}',          '#FFD700'),
    ('VEREDICTO',         '✅ v2.0' if promover_global else '─ v1.0', _cv),
]
ax5.text(0.5, 0.94, 'RESUMEN EJECUTIVO CC', ha='center',
         fontsize=12, fontweight='bold', color='white', transform=ax5.transAxes)
for idx, (nombre, valor, color) in enumerate(kpis):
    cx = (idx % 3) * 0.33 + 0.165
    cy = 0.60 if idx < 3 else 0.22
    rect = FancyBboxPatch((cx - 0.12, cy - 0.14), 0.24, 0.30,
                           boxstyle="round,pad=0.01", facecolor='#2D2D4E',
                           edgecolor=color, linewidth=2, transform=ax5.transAxes)
    ax5.add_patch(rect)
    ax5.text(cx, cy + 0.10, nombre, ha='center', fontsize=7.5,
             color='#AAAACC', transform=ax5.transAxes)
    ax5.text(cx, cy - 0.02, valor, ha='center', fontsize=12,
             fontweight='bold', color=color, transform=ax5.transAxes)

fig.suptitle(
    'RBlJose — NBO — Semana 9\n'
    'Champion-Challenger: v1.0+Platt-v1.1 (Champion) vs v2.0 (Challenger)\n'
    f'Criterio: ΔProfit > {MIN_PROFIT_LIFT:.0%} AND IC 95% no cruza 0 AND '
    f'default rate Challenger ≤ Champion',
    fontsize=12, fontweight='bold', color='#1A1A2E', y=0.99
)
plt.savefig(f'{DATA_DIR}/nbo_s9_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=C_BG)
plt.close()
print(f"  ✅ nbo_s9_dashboard.png")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 8 — OUTPUTS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 8 — OUTPUTS")

# Asignación CC
df_asignacion.to_csv(f'{DATA_DIR}/nbo_s9_asignacion_cc.csv', index=False)
print(f"  ✅ nbo_s9_asignacion_cc.csv          ({len(df_asignacion):,} clientes)")

# Campaña — universo contactado por ciclo y brazo
if campanas_all:
    pd.DataFrame(campanas_all).drop(
        columns=['productos_activos', 'fatiga_por_producto'], errors='ignore'
    ).to_csv(f'{DATA_DIR}/nbo_s9_campana_cc.csv', index=False)
    print(f"  ✅ nbo_s9_campana_cc.csv             ({len(campanas_all):,} registros)")

# Resultados T+30
if resultados_cc:
    df_all_conv.to_csv(f'{DATA_DIR}/nbo_s9_resultados_cc.csv', index=False)
    print(f"  ✅ nbo_s9_resultados_cc.csv          ({len(df_all_conv):,} registros)")

# Decisión estadística por producto
if len(df_decisiones) > 0:
    df_decisiones.to_csv(f'{DATA_DIR}/nbo_s9_decision_estadistica.csv', index=False)
    print(f"  ✅ nbo_s9_decision_estadistica.csv   ({len(df_decisiones)} productos)")

# Reporte ejecutivo
reporte = [
    {'campo': 'fecha_ejecucion',           'valor': str(date.today())},
    {'campo': 'meses_evaluados',           'valor': str(CICLOS_CC)},
    {'campo': 'presupuesto_por_brazo',     'valor': str(PRESUPUESTO_BRAZO)},
    {'campo': 'min_profit_lift',           'valor': str(MIN_PROFIT_LIFT)},
    {'campo': 'n_bootstrap',               'valor': str(N_BOOTSTRAP)},
    {'campo': 'score_umbral_default',      'valor': str(SCORE_UMBRAL_DEFAULT)},
    {'campo': 'n_champion_clientes',       'valor': str(tot_ch)},
    {'campo': 'n_challenger_clientes',     'valor': str(tot_chl)},
    {'campo': 'profit_champion_total',     'valor': str(round(profit_ch_total,  2))},
    {'campo': 'profit_challenger_total',   'valor': str(round(profit_chl_total, 2))},
    {'campo': 'delta_profit',              'valor': str(round(delta_profit_global, 2))},
    {'campo': 'lift_global',               'valor': str(round(lift_global, 4))},
    {'campo': 'bootstrap_ic95_lo',         'valor': str(round(ic_lo_global, 2))},
    {'campo': 'bootstrap_ic95_hi',         'valor': str(round(ic_hi_global, 2))},
    {'campo': 'ic_cruza_cero',             'valor': str(ic_global_cruza)},
    {'campo': 'productos_promover',        'valor': str(prods_promover)},
    {'campo': 'productos_riesgo_alto',     'valor': str(prods_riesgo)},
    {'campo': 'productos_mantener',        'valor': str(prods_mantener)},
    {'campo': 'productos_datos_insuf',     'valor': str(prods_insuf)},
    {'campo': 'crit1_lift',                'valor': str(crit_global_lft)},
    {'campo': 'crit2_ic_bootstrap',        'valor': str(crit_global_ic)},
    {'campo': 'crit3_default_rate',        'valor': str(crit_global_risk)},
    {'campo': 'decision_global',           'valor': veredicto},
    {'campo': 'modelo_recomendado',        'valor': 'v2.0' if promover_global else 'v1.0'},
    {'campo': 'nota_conversion',           'valor': 'Métrica informativa — no criterio de decisión'},
]
pd.DataFrame(reporte).to_csv(f'{DATA_DIR}/nbo_s9_reporte_ejecutivo.csv', index=False)
print(f"  ✅ nbo_s9_reporte_ejecutivo.csv")

separador("SEMANA 9 COMPLETADA")
print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║       RBlJose — NBO — SEMANA 9 COMPLETADA               ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 0 : Invariantes del experimento documentados          ║
  ║  ✓ Bloque 1 : Champion (v1.0+v1.1) y Challenger (v2.0) cargados║
  ║  ✓ Bloque 2 : Datos + GT lookup (fuente única para ambos brazos)║
  ║  ✓ Bloque 3 : Asignación 50/50 estratificada por segmento       ║
  ║  ✓ Bloque 4 : Loop CC con pipelines paralelos e independientes  ║
  ║  ✓ Bloque 5 : ΔProfit + bootstrap IC 95% + guardia default rate ║
  ║  ✓ Bloque 6 : Veredicto final con jerarquía estricta            ║
  ║  ✓ Bloque 7 : Dashboard ejecutivo (profit / bootstrap / riesgo) ║
  ║  ✓ Bloque 8 : 5 outputs CSV + 1 PNG                             ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  CRITERIO DE DECISIÓN (jerarquía):                              ║
  ║    (1) ΔProfit > 2%               [primario]                    ║
  ║    (2) IC 95% bootstrap no cruza 0 [significancia económica]    ║
  ║    (3) Default rate Chl ≤ Ch       [guardia de riesgo]          ║
  ║    Conversión = métrica informativa, NO criterio                ║
  ╚══════════════════════════════════════════════════════════════════╝
""")



