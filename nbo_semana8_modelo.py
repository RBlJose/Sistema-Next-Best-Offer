# %%
"""
RBlJose — SISTEMA NBO
Semana 8 — Modelos v2.0 con Features Estacionales
====================================================
Objetivo: entrenar modelos challenger (v2.0) que incorporan el calendario
económico ecuatoriano como señal adicional, sin modificar ningún otro
componente del pipeline.

Principios de diseño:
  - Hiperparámetros XGBoost idénticos a Semana 3 (v1.0).
    La mejora en AUC, si existe, debe ser atribuible exclusivamente a
    las features estacionales — no a cambios de configuración.
  - Split temporal estricto meses 1-15/16-18/19-23.
    La validación cruzada está PROHIBIDA en este sistema.
  - Platt Scaling sobre validación (meses 16-18), idéntico a v1.0.

Diseño del experimento (Champion-Challenger — Semana 9):
  Champion  = v1.0 (sin estacionalidad)
  Challenger = v2.0 (con estacionalidad)
  La Semana 9 decidirá si promover v2.0 a champion basándose en
  lift en profit incremental, no solo en AUC.

Prerequisito:
  nbo_generador_sintetico.py ejecutado con features estacionales.
  CSVs base: nbo_clientes.csv, nbo_features.csv, nbo_ofertas.csv
  Modelos v1.0 en /models/ para comparación de AUC.

Outputs:
  /models/nbo_model_{producto}_v2.joblib
  /models/nbo_calibrador_{producto}_v2.joblib
  /models/nbo_encoder_{col}_v2.joblib
  /models/nbo_model_metadata_v2.json
  /models/nbo_feature_names_v2.json
  nbo_recomendaciones_semana8.csv
  nbo_semana8_metricas_comparativas.csv
"""

# %%
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

import os
import json
import joblib

DATA_DIR   = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# %%
# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════
def separador(titulo):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print(f"{'='*65}")

def subseccion(titulo):
    print(f"\n  ── {titulo}")

# %%
# ══════════════════════════════════════════════════════════════════════
# PARÁMETROS DE NEGOCIO
# (Idénticos en todo el proyecto — no se modifican)
# ══════════════════════════════════════════════════════════════════════
PARAMS_NEGOCIO = {
    'tarjeta'     : {'ticket_anual': 465.0,  'costo_contacto': 2.5,
                     'costo_originacion': 45.0,  'pd': 0.055,
                     'lgd': 0.75, 'rwa': 1.00},
    'prestamo'    : {'ticket_anual': 675.0,  'costo_contacto': 3.0,
                     'costo_originacion': 60.0,  'pd': 0.045,
                     'lgd': 0.78, 'rwa': 0.75},
    'microcredito': {'ticket_anual': 300.0,  'costo_contacto': 4.5,
                     'costo_originacion': 85.0,  'pd': 0.095,
                     'lgd': 0.82, 'rwa': 0.75},
    'seguro_vida' : {'ticket_anual': 31.5,   'costo_contacto': 2.0,
                     'costo_originacion': 15.0,  'pd': 0.000,
                     'lgd': 0.00, 'rwa': 0.00},
    'seguro_salud': {'ticket_anual': 36.0,   'costo_contacto': 2.0,
                     'costo_originacion': 15.0,  'pd': 0.000,
                     'lgd': 0.00, 'rwa': 0.00},
    'inversion'   : {'ticket_anual': 120.0,  'costo_contacto': 1.5,
                     'costo_originacion': 10.0,  'pd': 0.000,
                     'lgd': 0.00, 'rwa': 0.00},
}

VERSION_MODELO = 'v2.0_semana8'

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA Y CONSTRUCCIÓN DE LA ABT v2
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA Y CONSTRUCCIÓN DE LA ABT v2")

print("\n  Cargando tablas fuente...")

clientes  = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
productos = pd.read_csv(f'{DATA_DIR}/nbo_productos.csv')
ofertas   = pd.read_csv(f'{DATA_DIR}/nbo_ofertas_actualizada.csv')
features  = pd.read_csv(f'{DATA_DIR}/nbo_features.csv')

print(f"  clientes  : {len(clientes):>8,} filas | {clientes.shape[1]} columnas")
print(f"  productos : {len(productos):>8,} filas | {productos.shape[1]} columnas")
print(f"  ofertas   : {len(ofertas):>8,} filas | {ofertas.shape[1]} columnas")
print(f"  features  : {len(features):>8,} filas | {features.shape[1]} columnas")

# Verificar que las features estacionales están presentes
FEATURES_ESTACIONALES = [
    'mes_calendario', 'es_utilidades', 'es_decimo_tercero', 'es_decimo_cuarto',
    'es_inicio_clases',   # unificada y correcta por región (Sierra/Costa/Amazonia)
    'es_navidad', 'es_impuesto_renta', 'trimestre',
]
faltantes_est = [c for c in FEATURES_ESTACIONALES if c not in features.columns]
if faltantes_est:
    raise ValueError(
        f"Features estacionales faltantes en nbo_features.csv: {faltantes_est}\n"
        f"Ejecuta primero el generador modificado (Semana 8)."
    )
print(f"\n  ✅ Features estacionales presentes: {FEATURES_ESTACIONALES}")

# Normalizar id_producto
ofertas['id_producto_lower'] = ofertas['id_producto'].str.lower()

# ── Filtrar universo de modelado ──────────────────────────────────────
print("\n  Filtrando universo de modelado...")

mask_tratamiento       = ofertas['grupo'] == 'Tratamiento'
mask_etiqueta_completa = ofertas['etiqueta_completa'] == True
mask_etiqueta_notnull  = ofertas['convirtio_30d'].notna()

ofertas_modelo = ofertas[
    mask_tratamiento & mask_etiqueta_completa & mask_etiqueta_notnull
].copy()

print(f"  Ofertas totales              : {len(ofertas):>8,}")
print(f"  Tras filtros (universo modelo): {len(ofertas_modelo):>8,}")

# ── Catálogo de features ──────────────────────────────────────────────
# Features originales (idénticas a v1.0)
FEATURES_NUMERICAS_BASE = [
    'gasto_3m', 'gasto_supermercado_3m', 'gasto_farmacia_3m',
    'depositos_efectivo_3m', 'tx_digitales_proporcion',
    'saldo_promedio_90d', 'saldo_tendencia', 'variabilidad_saldo',
    'ratio_cuota_ingreso', 'dias_desde_ult_credito',
    'indice_estres_macro', 'estres_x_riesgo',
    'edad', 'antiguedad_meses', 'ingreso_mensual',
    'score_crediticio', 'score_buro', 'max_atraso_dias',
    'ratio_deuda_init', 'hijos',
    # Derivadas construidas más abajo
    'ratio_saldo_ingreso', 'score_muy_alto', 'edad_objetivo_sv',
    'indep_no_urbano',
]

# Features estacionales (nuevas en v2.0)
# Se agregan DESPUÉS de las features base para que el contrato de features
# v1.0 y v2.0 sean comparables en las primeras N columnas.
FEATURES_NUMERICAS_V2 = FEATURES_NUMERICAS_BASE + FEATURES_ESTACIONALES

FEATURES_CATEGORICAS = [
    'segmento', 'ocupacion', 'zona_geografica', 'canal_principal',
    'estado_civil', 'tiene_atraso_hist', 'bucket_dias_credito',
    'region',    # ← NUEVO Semana 8
]

COLUMNAS_FEATURES = [
    'id_cliente', 'mes', 'gasto_3m', 'gasto_supermercado_3m',
    'gasto_farmacia_3m', 'depositos_efectivo_3m', 'tx_digitales_proporcion',
    'saldo_promedio_90d', 'saldo_tendencia', 'variabilidad_saldo',
    'ratio_cuota_ingreso', 'dias_desde_ult_credito', 'indice_estres_macro',
    'estres_x_riesgo',
] + FEATURES_ESTACIONALES   # <-- Semana 8: se suman al contrato

features_slim = features[[
    c for c in COLUMNAS_FEATURES if c in features.columns
]].copy()

COLUMNAS_CLIENTES = [
    'id_cliente', 'segmento', 'edad', 'ocupacion', 'zona_geografica',
    'canal_principal', 'antiguedad_meses', 'ingreso_mensual',
    'score_crediticio', 'score_buro', 'tiene_atraso_hist',
    'max_atraso_dias', 'ratio_deuda_init', 'hijos', 'estado_civil',
    'region',    # ← NUEVO Semana 8 — Sierra / Costa / Amazonia
]

# ── Join ofertas × features (anti-leakage: clave id_cliente + mes) ────
print("\n  Construyendo join ofertas × features...")

assert features_slim[['id_cliente','mes']].duplicated().sum() == 0, \
    "Duplicados en clave (id_cliente, mes) de features"

abt = ofertas_modelo.merge(
    features_slim, on=['id_cliente', 'mes'], how='left', validate='many_to_one'
)
abt = abt.merge(
    clientes[COLUMNAS_CLIENTES], on='id_cliente', how='left', validate='many_to_one'
)

n_nulas = abt['saldo_promedio_90d'].isna().sum()
print(f"  Filas en ABT              : {len(abt):>8,}")
print(f"  Filas sin features (NaN)  : {n_nulas:>8,}  ({n_nulas/len(abt):.1%})")

# ── Variables derivadas (idénticas a v1.0 — no se modifican) ─────────
abt['bucket_dias_credito'] = pd.cut(
    abt['dias_desde_ult_credito'],
    bins=[-1, 30, 90, 180, 365, 730, 9999],
    labels=['0_30d', '31_90d', '91_180d', '181_365d', '366_730d', 'mas_730d']
)
abt['ratio_saldo_ingreso'] = abt['saldo_promedio_90d'] / (abt['ingreso_mensual'] + 1)
abt['score_muy_alto']      = (abt['score_crediticio'] > 800).astype(int)
abt['edad_objetivo_sv']    = ((abt['edad'] >= 30) & (abt['edad'] <= 55)).astype(int)
abt['indep_no_urbano']     = (
    (abt['ocupacion'] == 'Independiente') &
    (abt['zona_geografica'] != 'Urbana')
).astype(int)

# ── Resumen ABT ───────────────────────────────────────────────────────
subseccion("Resumen ABT v2")
print(f"\n  Dimensiones: {abt.shape[0]:,} filas × {abt.shape[1]} columnas")
print(f"\n  Distribución por producto:")
print(
    abt.groupby('id_producto_lower').agg(
        n_ofertas   = ('convirtio_30d', 'count'),
        tasa_conv   = ('convirtio_30d', 'mean'),
        n_positivos = ('convirtio_30d', 'sum'),
    ).round(4).to_string()
)

subseccion("Verificación estacionalidad en ABT")
print(f"\n  Tasa conversión media por mes calendario (todos los productos):")
temp = abt.groupby('mes_calendario')['convirtio_30d'].agg(['mean','count'])
temp.columns = ['tasa_conv', 'n_obs']
print(temp.to_string())

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — SPLIT TEMPORAL
# Idéntico a Semana 3 — no se modifica.
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — SPLIT TEMPORAL")

CORTE_TRAIN_FIN = 15
CORTE_VAL_FIN   = 18

train = abt[abt['mes'] <= CORTE_TRAIN_FIN].copy()
val   = abt[(abt['mes'] > CORTE_TRAIN_FIN) & (abt['mes'] <= CORTE_VAL_FIN)].copy()
test  = abt[abt['mes'] > CORTE_VAL_FIN].copy()

print(f"\n  {'Conjunto':<15} {'Meses':<12} {'Filas':>10} {'% total':>8} {'Tasa conv':>10}")
print(f"  {'-'*58}")
for nombre, df, meses in [
    ('Entrenamiento', train, f"1 – {CORTE_TRAIN_FIN}"),
    ('Validación',    val,   f"{CORTE_TRAIN_FIN+1} – {CORTE_VAL_FIN}"),
    ('Test OOT',      test,  f"{CORTE_VAL_FIN+1} – 23"),
]:
    print(f"  {nombre:<15} {meses:<12} {len(df):>10,} "
          f"{len(df)/len(abt):>8.1%} {df['convirtio_30d'].mean():>10.3f}")

assert train['mes'].max() < val['mes'].min(), "❌ Solapamiento train/val"
assert val['mes'].max()   < test['mes'].min(), "❌ Solapamiento val/test"
print(f"\n  ✅ Sin solapamiento temporal")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — ENCODING
# Los encoders v2 se entrenan sobre el mismo universo que v1,
# por tanto las clases son idénticas. Se versionan por separado
# para mantener el contrato de features v2 autocontenido.
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — ENCODING Y CONTRATO DE FEATURES v2")

from sklearn.preprocessing import LabelEncoder

print("\n  Aplicando Label Encoding...")
encoders_v2 = {}

for col in FEATURES_CATEGORICAS:
    le = LabelEncoder()
    le.fit(abt[col].astype(str))
    encoders_v2[col] = le

    for df in [train, val, test, abt]:
        df[col + '_enc'] = le.transform(df[col].astype(str))

    joblib.dump(le, os.path.join(MODELS_DIR, f'nbo_encoder_{col}_v2.joblib'))
    print(f"  {col:<25}: {len(le.classes_)} categorías → guardado como v2")

FEATURES_FINALES_V2 = FEATURES_NUMERICAS_V2 + [c + '_enc' for c in FEATURES_CATEGORICAS]

print(f"\n  Features v1.0 : {len(FEATURES_NUMERICAS_BASE) - 4 + 4 + len(FEATURES_CATEGORICAS)}")
print(f"  Features v2.0 : {len(FEATURES_FINALES_V2)}")
print(f"  Nuevas (estacionales): {len(FEATURES_ESTACIONALES)}")

# Verificar NaN
nans = train[FEATURES_FINALES_V2].isnull().sum()
nans_sig = nans[nans > 0]
if len(nans_sig) == 0:
    print("\n  ✅ Sin NaN en features de entrenamiento")
else:
    print(f"\n  ⚠️  Imputando {len(nans_sig)} features con NaN...")
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median')
    train[FEATURES_FINALES_V2] = imp.fit_transform(train[FEATURES_FINALES_V2])
    val[FEATURES_FINALES_V2]   = imp.transform(val[FEATURES_FINALES_V2])
    test[FEATURES_FINALES_V2]  = imp.transform(test[FEATURES_FINALES_V2])

# Guardar contrato de features v2
feature_spec_v2 = {
    'version'              : 'v2.0',
    'features_finales'     : FEATURES_FINALES_V2,
    'features_numericas'   : FEATURES_NUMERICAS_V2,
    'features_estacionales': FEATURES_ESTACIONALES,
    'features_categoricas' : FEATURES_CATEGORICAS,
    'n_features_v1'        : len(FEATURES_FINALES_V2) - len(FEATURES_ESTACIONALES),
    'n_features_v2'        : len(FEATURES_FINALES_V2),
}
with open(os.path.join(MODELS_DIR, 'nbo_feature_names_v2.json'), 'w') as f:
    json.dump(feature_spec_v2, f, indent=2)
print(f"\n  ✅ Contrato guardado: nbo_feature_names_v2.json")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — ENTRENAMIENTO XGBOOST v2.0
# Hiperparámetros IDÉNTICOS a Semana 3 — cualquier diferencia en AUC
# es atribuible exclusivamente a las features estacionales.
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — ENTRENAMIENTO XGBOOST v2.0")

import xgboost as xgb
from sklearn.metrics import roc_auc_score

print(f"\n  XGBoost versión: {xgb.__version__}")
print(f"  Hiperparámetros: IDÉNTICOS a v1.0 (Semana 3)")
print(f"  Diferencia con v1.0: {len(FEATURES_ESTACIONALES)} features estacionales adicionales\n")

modelos_v2    = {}
metricas_v2   = {}

for producto in sorted(PARAMS_NEGOCIO.keys()):

    subseccion(f"Producto: {producto.upper()}")

    X_train = train[train['id_producto_lower'] == producto][FEATURES_FINALES_V2]
    y_train = train[train['id_producto_lower'] == producto]['convirtio_30d'].astype(int)
    X_val   = val[val['id_producto_lower'] == producto][FEATURES_FINALES_V2]
    y_val   = val[val['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = round(n_neg / max(n_pos, 1), 1)

    print(f"\n  Train: {len(X_train):,} | Positivos: {n_pos:,} ({n_pos/len(y_train):.1%})")
    print(f"  Val:   {len(X_val):,}   | scale_pos_weight = {spw}")

    # ── Hiperparámetros idénticos a v1.0 ─────────────────────────────
    modelo = xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 20,
        scale_pos_weight      = spw,
        objective             = 'binary:logistic',
        eval_metric           = 'auc',
        early_stopping_rounds = 50,
        random_state          = SEED,
        n_jobs                = -1,
        verbosity             = 0,
    )

    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_iter = modelo.best_iteration
    p_train   = modelo.predict_proba(X_train)[:, 1]
    p_val     = modelo.predict_proba(X_val)[:, 1]
    auc_train = roc_auc_score(y_train, p_train)
    auc_val   = roc_auc_score(y_val, p_val)
    ks_val, _ = ks_2samp(p_val[y_val == 1], p_val[y_val == 0])

    print(f"  Mejor iteración  : {best_iter}")
    print(f"  AUC train        : {auc_train:.4f}")
    print(f"  AUC val          : {auc_val:.4f}")
    gap = auc_train - auc_val
    print(f"  Diferencia       : {gap:.4f}  {'⚠️ Posible overfitting' if gap > 0.05 else '✅ OK'}")
    print(f"  KS val           : {ks_val:.4f}")

    modelos_v2[producto] = {
        'model'   : modelo,
        'features': FEATURES_FINALES_V2,
        'best_iter': best_iter,
        'spw'     : spw,
    }
    metricas_v2[producto] = {
        'auc_train': auc_train, 'auc_val': auc_val,
        'ks_val'   : ks_val,    'n_train': len(X_train),
        'n_val'    : len(X_val), 'n_pos_train': int(n_pos),
        'spw'      : spw,       'best_iter': best_iter,
    }

    # Serializar modelo v2
    joblib.dump(
        modelo,
        os.path.join(MODELS_DIR, f'nbo_model_{producto}_v2.joblib')
    )

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — CALIBRACIÓN PLATT SCALING v2
# Mismo protocolo que Semana 3: fit en validación, evaluar en test.
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — CALIBRACIÓN DE PROBABILIDADES v2")

from sklearn.linear_model import LogisticRegression

calibradores_v2      = {}
metricas_calibracion = {}

for producto in sorted(modelos_v2.keys()):

    subseccion(f"Calibración: {producto.upper()}")

    modelo    = modelos_v2[producto]['model']
    feats     = modelos_v2[producto]['features']

    X_val_p  = val[val['id_producto_lower']  == producto][feats]
    y_val_p  = val[val['id_producto_lower']  == producto]['convirtio_30d'].astype(int)
    X_test_p = test[test['id_producto_lower'] == producto][feats]
    y_test_p = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    scores_val  = modelo.predict_proba(X_val_p)[:, 1].reshape(-1, 1)
    scores_test = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)

    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(scores_val, y_val_p)
    calibradores_v2[producto] = platt

    p_cal_test = platt.predict_proba(scores_test)[:, 1]
    p_raw_test = scores_test.ravel()

    auc_raw = roc_auc_score(y_test_p, p_raw_test)
    auc_cal = roc_auc_score(y_test_p, p_cal_test)
    brier_raw = float(np.mean((p_raw_test - y_test_p) ** 2))
    brier_cal = float(np.mean((p_cal_test - y_test_p) ** 2))

    tasa_real   = float(y_test_p.mean())
    media_p_raw = float(p_raw_test.mean())
    media_p_cal = float(p_cal_test.mean())

    print(f"\n  Tasa real en test    : {tasa_real:.4f}")
    print(f"  Media P bruta        : {media_p_raw:.4f}  (error: {abs(media_p_raw - tasa_real):.4f})")
    print(f"  Media P calibrada    : {media_p_cal:.4f}  (error: {abs(media_p_cal - tasa_real):.4f})")
    print(f"  AUC raw / calibrado  : {auc_raw:.4f} / {auc_cal:.4f}")
    print(f"  Brier raw / calibrado: {brier_raw:.5f} / {brier_cal:.5f}")

    metricas_calibracion[producto] = {
        'tasa_real'    : tasa_real,
        'media_p_raw'  : media_p_raw,
        'media_p_cal'  : media_p_cal,
        'auc_test_raw' : float(auc_raw),
        'auc_test_cal' : float(auc_cal),
        'brier_raw'    : brier_raw,
        'brier_cal'    : brier_cal,
        'A_platt'      : float(platt.coef_[0][0]),
        'B_platt'      : float(platt.intercept_[0]),
        'best_iteration': modelos_v2[producto]['best_iter'],
    }

    mask_test_prod = test['id_producto_lower'] == producto
    test.loc[mask_test_prod, 'p_raw_v2']       = p_raw_test
    test.loc[mask_test_prod, 'p_calibrada_v2'] = p_cal_test

    # Serializar calibrador v2
    joblib.dump(
        platt,
        os.path.join(MODELS_DIR, f'nbo_calibrador_{producto}_v2.joblib')
    )

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — MÉTRICAS TÉCNICAS v2.0 EN TEST OOT
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 6 — MÉTRICAS TÉCNICAS v2.0 EN TEST OOT (meses 19-23)")

def calcular_psi(p_ref, p_actual, n_bins=10):
    bins = np.percentile(p_ref, np.linspace(0, 100, n_bins + 1))
    bins[0]  = -np.inf
    bins[-1] = np.inf
    ref_counts = np.histogram(p_ref,    bins=bins)[0]
    act_counts = np.histogram(p_actual, bins=bins)[0]
    ref_pct = (ref_counts + 0.5) / (len(p_ref)    + n_bins * 0.5)
    act_pct = (act_counts + 0.5) / (len(p_actual) + n_bins * 0.5)
    return round(float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))), 4)

metricas_test_v2 = {}

print(f"\n  {'Producto':<20} {'AUC test':>10} {'KS test':>10} "
      f"{'PSI val→test':>14} {'Estado PSI':>12}")
print(f"  {'-'*70}")

for producto in sorted(modelos_v2.keys()):
    modelo = modelos_v2[producto]['model']
    platt  = calibradores_v2[producto]
    feats  = FEATURES_FINALES_V2

    X_val_p  = val[val['id_producto_lower']  == producto][feats]
    X_test_p = test[test['id_producto_lower'] == producto][feats]
    y_test_p = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    p_val_raw  = modelo.predict_proba(X_val_p)[:, 1].reshape(-1, 1)
    p_test_raw = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)
    p_val_cal  = platt.predict_proba(p_val_raw)[:, 1]
    p_cal_test = platt.predict_proba(p_test_raw)[:, 1]

    auc_test = float(roc_auc_score(y_test_p, p_cal_test))
    ks_test, _ = ks_2samp(
        p_cal_test[y_test_p == 1],
        p_cal_test[y_test_p == 0]
    )
    psi = calcular_psi(p_val_cal, p_cal_test)
    estado_psi = "✅ Estable" if psi < 0.10 else ("⚠️ Alerta" if psi < 0.20 else "❌ Drift")

    print(f"  {producto:<20} {auc_test:>10.4f} {float(ks_test):>10.4f} "
          f"{psi:>14.4f} {estado_psi:>12}")

    metricas_test_v2[producto] = {
        'auc_test': auc_test,
        'ks_test' : float(ks_test),
        'psi'     : psi,
    }

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 7 — SHAP v2.0 — IMPORTANCIA DE FEATURES (INCL. ESTACIONALES)
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 7 — SHAP v2.0 — IMPORTANCIA DE FEATURES INCLUYENDO ESTACIONALES")

print("""
  El foco de Semana 8 en SHAP es verificar que las features estacionales
  tienen importancia real y no son ruido. Una feature estacional con
  mean(|SHAP|) > 0 confirma que el modelo la usa — pero lo importante
  es el ranking relativo versus las features base.
  
  Si las estacionales aparecen en el top-10 de algún producto, el modelo
  las está explotando como señal, lo cual es esperado para tarjeta
  (es_navidad, es_decimo_tercero) e inversión (es_utilidades).
""")

import xgboost as _xgb

shap_importancias = {}

for producto in sorted(modelos_v2.keys()):
    subseccion(f"SHAP: {producto.upper()}")

    modelo_xgb = modelos_v2[producto]['model']
    feats      = modelos_v2[producto]['features']
    X_test_p   = test[test['id_producto_lower'] == producto][feats].copy()

    if len(X_test_p) == 0:
        continue

    dmat        = _xgb.DMatrix(X_test_p, feature_names=feats)
    shap_matrix = modelo_xgb.get_booster().predict(dmat, pred_contribs=True)
    shap_df     = pd.DataFrame(shap_matrix[:, :-1], columns=feats)
    importancia = shap_df.abs().mean().sort_values(ascending=False)

    shap_importancias[producto] = importancia

    print(f"\n  Top 12 features por mean(|SHAP|):")
    top_val = importancia.iloc[0]
    for i, (feat, val) in enumerate(importancia.head(12).items()):
        bar  = "█" * int(val / top_val * 20)
        flag = " ← ESTACIONAL" if feat in FEATURES_ESTACIONALES else ""
        print(f"  {i+1:2d}. {feat:<32} {val:.5f}  {bar}{flag}")

# Resumen: ranking de features estacionales por producto
subseccion("Resumen: ranking de features estacionales en cada producto")
print(f"\n  {'Producto':<20} {'Feature estacional':<30} {'Rank':>6} {'mean|SHAP|':>12}")
print(f"  {'─'*74}")
for producto, imp in shap_importancias.items():
    rank_dict = {feat: i+1 for i, feat in enumerate(imp.index)}
    for feat in FEATURES_ESTACIONALES:
        if feat in rank_dict:
            rank = rank_dict[feat]
            val  = imp[feat]
            if rank <= 15:  # solo mostrar si está en top 15
                print(f"  {producto:<20} {feat:<30} {rank:>6} {val:>12.5f}")            

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 8 — COMPARACIÓN v1.0 vs v2.0 EN TEST OOT
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 8 — COMPARACIÓN AUC v1.0 vs v2.0 EN TEST OOT")

# Intentar cargar métricas v1.0 desde metadata
metricas_v10_cargadas = {}
metadata_v1_path = os.path.join(MODELS_DIR, 'nbo_model_metadata_v1.json')
if os.path.exists(metadata_v1_path):
    with open(metadata_v1_path) as f:
        meta_v1 = json.load(f)
    for prod, m in meta_v1.get('metricas_por_producto', {}).items():
        metricas_v10_cargadas[prod] = m.get('auc_test_cal', None)
    print(f"\n  ✅ Metadata v1.0 cargada desde: {metadata_v1_path}")
else:
    print(f"\n  ⚠️  Metadata v1.0 no encontrada en {metadata_v1_path}")
    print(f"     La comparación mostrará solo métricas v2.0.")
    print(f"     Para comparación completa: ejecutar primero nbo_semana3_modelos.py con")
    print(f"     el bloque de serialización de metadata.")

metricas_comparativas = []

print(f"\n  {'Producto':<20} {'AUC v1.0':>10} {'AUC v2.0':>10} "
      f"{'Δ AUC':>10} {'KS v2.0':>10} {'PSI v2.0':>10} {'Veredicto':>14}")
print(f"  {'─'*86}")

for producto in sorted(metricas_test_v2.keys()):
    m_v2   = metricas_test_v2[producto]
    m_cal  = metricas_calibracion[producto]
    auc_v2 = m_v2['auc_test']
    auc_v1 = metricas_v10_cargadas.get(producto, None)

    if auc_v1 is not None:
        delta = auc_v2 - auc_v1
        if delta > 0.005:
            veredicto = "✅ Mejora"
        elif delta > -0.003:
            veredicto = "≈ Neutro"
        else:
            veredicto = "⚠️ Regresión"
        auc_v1_str = f"{auc_v1:.4f}"
        delta_str  = f"{delta:+.4f}"
    else:
        veredicto = "─ sin v1.0"
        auc_v1_str = "N/D"
        delta_str  = "N/D"

    print(f"  {producto:<20} {auc_v1_str:>10} {auc_v2:>10.4f} "
          f"{delta_str:>10} {m_v2['ks_test']:>10.4f} "
          f"{m_v2['psi']:>10.4f} {veredicto:>14}")

    metricas_comparativas.append({
        'producto'      : producto,
        'auc_v10'       : auc_v1,
        'auc_v20'       : auc_v2,
        'delta_auc'     : (auc_v2 - auc_v1) if auc_v1 else None,
        'ks_v20'        : m_v2['ks_test'],
        'psi_v20'       : m_v2['psi'],
        'brier_cal_v20' : m_cal['brier_cal'],
        'tasa_real_test': m_cal['tasa_real'],
        'sesgo_cal_v20' : m_cal['media_p_cal'] - m_cal['tasa_real'],
        'n_features_v20': len(FEATURES_FINALES_V2),
        'n_features_est': len(FEATURES_ESTACIONALES),
    })

print(f"""
  Interpretación:
    Δ AUC > 0.005 → estacionalidad agrega señal real (champion-challenger
                    candidato fuerte en Semana 9).
    Δ AUC ≈ 0    → estacionalidad es neutral; el calendardio ecuatoriano
                    puede no tener variación suficiente en 25 meses.
    Δ AUC < -0.003 → las features estacionales introducen ruido. Revisar
                    si el generador modela estacionalidad explícita.
""")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 9 — RANKING NBO v2.0 (VECTORIAL)
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 9 — RANKING NBO v2.0 (VECTORIAL)")

MES_SCORING   = test['mes'].max()
FECHA_SCORING = f"2025-{(MES_SCORING - 1) % 12 + 1:02d}-01"

test_mes = test[test['mes'] == MES_SCORING].copy()

print(f"\n  Scoring sobre mes {MES_SCORING} — "
      f"{test_mes['id_cliente'].nunique():,} clientes únicos")

# ── Canal principal (una sola vez)
canal_map = clientes.set_index('id_cliente')['canal_principal']
test_mes['canal_principal'] = test_mes['id_cliente'].map(canal_map).fillna('Digital')

# ── Calcular scores por producto (vectorizado)
scores_list = []

for producto in modelos_v2.keys():

    mask = test_mes['id_producto_lower'] == producto
    df_p = test_mes.loc[mask].copy()

    if df_p.empty:
        continue

    modelo = modelos_v2[producto]['model']
    feats  = FEATURES_FINALES_V2
    platt  = calibradores_v2[producto]
    p      = PARAMS_NEGOCIO[producto]

    X = df_p[feats].values

    # Probabilidades
    p_raw = modelo.predict_proba(X)[:, 1]
    p_cal = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    # Score NBO
    ingreso_esp = p_cal * (p['ticket_anual'] - p['costo_originacion'])
    perdida_esp = p['pd'] * p['lgd'] * p['ticket_anual'] * p['rwa']
    score_nbo   = ingreso_esp - perdida_esp - p['costo_contacto']

    df_p['producto_nbo']    = producto
    df_p['p_calibrada']     = p_cal
    df_p['score_xgb_raw']   = p_raw
    df_p['score_nbo']       = score_nbo
    df_p['costo_contacto']  = p['costo_contacto']
    df_p['ratio_nbo_costo'] = score_nbo / max(p['costo_contacto'], 0.01)

    scores_list.append(df_p[
        ['id_cliente', 'producto_nbo', 'score_nbo',
         'p_calibrada', 'score_xgb_raw',
         'ratio_nbo_costo', 'costo_contacto', 'canal_principal']
    ])

# ── Unir todo
df_scores = pd.concat(scores_list, ignore_index=True)

# ── Ranking vectorizado
df_scores = df_scores.sort_values(['id_cliente', 'score_nbo'], ascending=[True, False])
df_scores['rank'] = df_scores.groupby('id_cliente').cumcount() + 1

# ── Top 1 (igual que antes)
df_nbo_top1 = df_scores[df_scores['rank'] == 1].copy()

# ── Metadata
df_scores['mes_scoring']   = MES_SCORING
df_scores['fecha_scoring'] = FECHA_SCORING
df_scores['version_modelo'] = VERSION_MODELO

# ── Output final
df_nbo = df_scores.copy()

# ── Reporte
print(f"\n  Distribución de recomendaciones NBO v2.0 (rank 1):")
print(f"  {'Producto':<20} {'Clientes':>10} {'%':>8}")
print(f"  {'─'*42}")

dist = df_nbo_top1['producto_nbo'].value_counts()

for prod, cnt in dist.items():
    pct = cnt / len(df_nbo_top1) * 100
    bar = '█' * int(pct / 2)
    print(f"  {prod:<20} {cnt:>10,} {pct:>7.1f}%  {bar}")

print(f"\n  Total clientes           : {len(df_nbo_top1):,}")
print(f"  Score NBO promedio (r1)  : {df_nbo_top1['score_nbo'].mean():.2f}")
print(f"  P calibrada promedio (r1): {df_nbo_top1['p_calibrada'].mean():.4f}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 10 — SERIALIZACIÓN METADATA v2 Y OUTPUTS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 10 — SERIALIZACIÓN METADATA v2 Y OUTPUTS")

metadata_v2 = {
    'version'     : VERSION_MODELO,
    'seed'        : SEED,
    'split_train' : f'meses_1_a_{CORTE_TRAIN_FIN}',
    'split_val'   : f'meses_{CORTE_TRAIN_FIN+1}_a_{CORTE_VAL_FIN}',
    'split_test'  : f'meses_{CORTE_VAL_FIN+1}_a_23',
    'n_features_v2'      : len(FEATURES_FINALES_V2),
    'n_features_v1'      : len(FEATURES_FINALES_V2) - len(FEATURES_ESTACIONALES),
    'features_estacionales': FEATURES_ESTACIONALES,
    'hiperparametros'    : {
        'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
        'early_stopping_rounds': 50, 'objective': 'binary:logistic',
    },
    'metricas_por_producto': {
        prod: {
            'auc_test_cal'    : metricas_calibracion[prod]['auc_test_cal'],
            'auc_test_raw'    : metricas_calibracion[prod]['auc_test_raw'],
            'brier_cal'       : metricas_calibracion[prod]['brier_cal'],
            'ks_test'         : metricas_test_v2[prod]['ks_test'],
            'psi'             : metricas_test_v2[prod]['psi'],
            'best_iteration'  : metricas_calibracion[prod]['best_iteration'],
            'A_platt'         : metricas_calibracion[prod]['A_platt'],
            'B_platt'         : metricas_calibracion[prod]['B_platt'],
            'n_train'         : metricas_v2[prod]['n_train'],
            'tasa_real_test'  : metricas_calibracion[prod]['tasa_real'],
        }
        for prod in sorted(PARAMS_NEGOCIO.keys())
    }
}

with open(os.path.join(MODELS_DIR, 'nbo_model_metadata_v2.json'), 'w') as f:
    json.dump(metadata_v2, f, indent=2)

# Guardar recomendaciones (contrato compatible con Semana 4)
df_nbo.to_csv(f'{DATA_DIR}/nbo_recomendaciones_semana8.csv', index=False)

# Guardar métricas comparativas v1.0 vs v2.0
pd.DataFrame(metricas_comparativas).to_csv(
    f'{DATA_DIR}/nbo_semana8_metricas_comparativas.csv', index=False
)

# Resumen de archivos generados
print(f"\n  {'Archivo':<50} {'Tipo'}")
print(f"  {'─'*65}")
archivos_generados = [
    (f'{MODELS_DIR}/nbo_model_{{producto}}_v2.joblib',       '× 6 modelos XGBoost'),
    (f'{MODELS_DIR}/nbo_calibrador_{{producto}}_v2.joblib',  '× 6 calibradores Platt'),
    (f'{MODELS_DIR}/nbo_encoder_{{col}}_v2.joblib',          '× 7 encoders LabelEncoder'),
    (f'{MODELS_DIR}/nbo_model_metadata_v2.json',             'Metadata y métricas'),
    (f'{MODELS_DIR}/nbo_feature_names_v2.json',              'Contrato de features'),
    ('nbo_recomendaciones_semana8.csv',                       'Ranking NBO v2.0'),
    ('nbo_semana8_metricas_comparativas.csv',                 'Comparativa v1.0 vs v2.0'),
]
for arch, tipo in archivos_generados:
    print(f"  {arch:<50} {tipo}")


# ══════════════════════════════════════════════════════════════════════
# REPORTE FINAL
# ══════════════════════════════════════════════════════════════════════
separador("REPORTE FINAL — SEMANA 8")

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     RBlJose — NBO — SEMANA 8 COMPLETADA             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 1  : ABT v2 con {len(FEATURES_FINALES_V2)} features ({len(FEATURES_ESTACIONALES)} estacionales)    ║
  ║  ✓ Bloque 2  : Split temporal 1-15 / 16-18 / 19-23          ║
  ║  ✓ Bloque 3  : Encoders v2 + contrato nbo_feature_names_v2  ║
  ║  ✓ Bloque 4  : XGBoost v2.0 (mismos hiperparámetros v1.0)  ║
  ║  ✓ Bloque 5  : Calibración Platt v2 (fit en val, eval test) ║
  ║  ✓ Bloque 6  : AUC / KS / PSI en test OOT                   ║
  ║  ✓ Bloque 7  : SHAP — verificación importancia estacionales  ║
  ║  ✓ Bloque 8  : Comparativa v1.0 vs v2.0 (input Champion-C.) ║
  ║  ✓ Bloque 9  : Ranking NBO v2.0 (contrato compatible S4)    ║
  ║  ✓ Bloque 10 : Metadata v2 + serialización completa         ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Próximo paso — Semana 9: Champion-Challenger               ║
  ║    Champion  = v1.0 (sin estacionalidad)                    ║
  ║    Challenger = v2.0 (con estacionalidad)                   ║
  ║    Criterio de promoción: lift en profit incremental         ║
  ║    Input: nbo_semana8_metricas_comparativas.csv             ║
  ╚══════════════════════════════════════════════════════════════╝
""")


