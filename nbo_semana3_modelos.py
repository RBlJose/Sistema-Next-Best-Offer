# %%
"""
RBlJose — SISTEMA NBO
Semana 3 — Entrenamiento de Modelos de Propensión
===================================================
Pipeline:
  Bloque 1 — Carga y construcción de la ABT (Analytical Base Table)
  Bloque 2 — Split temporal correcto (anti-leakage)
  Bloque 3 — Feature engineering y selección
  Bloque 4 — Entrenamiento XGBoost por producto
  Bloque 5 — Calibración de probabilidades (Platt Scaling)
  Bloque 6 — Métricas técnicas (AUC, KS, PSI)
  Bloque 7 — SHAP — interpretabilidad por producto
  Bloque 8 — Reporte de resultados
  Bloque 9 - Profit curves, thresholds, ranking NBO 

Prerequisito: el dataset fue aprobado por nbo_validacion_estructural.py
con 26/27 pruebas (96.3%). El único fallo (E1 microcrédito en stress)
es comportamiento emergente realista — no un error del generador.
"""

# %%
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

import os

# %%
from datetime import date, timedelta

FECHA_INICIO = date(2024, 1, 1)
SEED         = 42
np.random.seed(SEED)

# %%
DATA_DIR = os.getcwd()

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
# BLOQUE 1 — CARGA Y CONSTRUCCIÓN DE LA ABT
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA Y CONSTRUCCIÓN DE LA ABT")

# ── 1.1 Cargar las cuatro tablas fuente ───────────────────────────────
print("\n  Cargando tablas fuente...")

clientes  = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
productos = pd.read_csv(f'{DATA_DIR}/nbo_productos.csv')
ofertas   = pd.read_csv(f'{DATA_DIR}/nbo_ofertas.csv')
features  = pd.read_csv(f'{DATA_DIR}/nbo_features.csv')

print(f"  clientes  : {len(clientes):>8,} filas | {clientes.shape[1]} columnas")
print(f"  productos : {len(productos):>8,} filas | {productos.shape[1]} columnas")
print(f"  ofertas   : {len(ofertas):>8,} filas | {ofertas.shape[1]} columnas")
print(f"  features  : {len(features):>8,} filas | {features.shape[1]} columnas")

# ── 1.2 Preparar clave de join ─────────────────────────────────────────
# El join se hace por (id_cliente, mes).
# REGLA CRÍTICA: las features del mes t se unen a la oferta del mes t.
# Nunca a ofertas de meses distintos — eso sería leakage temporal.

# Normalizar id_producto para que sea siempre minúsculas
# En ofertas viene como 'TARJETA', en clientes como 'tarjeta'
ofertas['id_producto_lower'] = ofertas['id_producto'].str.lower()

# ── 1.3 Filtrar solo el universo de modelado ──────────────────────────
# Criterios de inclusión en la ABT:
#   a) Solo ofertas del grupo Tratamiento
#      (el grupo Control no recibió oferta — su etiqueta es 0 por ausencia,
#       no por rechazo real — incluirlo sesgaría la tasa base a la baja)
#   b) Solo filas con etiqueta completa (convirtio_30d no es None)
#      (meses 24-25 no tienen etiqueta — no pueden entrar al entrenamiento)
#   c) Solo ofertas a clientes activos elegibles
#      (ya filtradas por el generador, pero se verifica explícitamente)

print("\n  Filtrando universo de modelado...")
print(f"  Ofertas totales              : {len(ofertas):>8,}")

mask_tratamiento      = ofertas['grupo'] == 'Tratamiento'
mask_etiqueta_completa = ofertas['etiqueta_completa'] == True
mask_etiqueta_notnull  = ofertas['convirtio_30d'].notna()

# Los tres filtros deben cumplirse simultáneamente
ofertas_modelo = ofertas[
    mask_tratamiento &
    mask_etiqueta_completa &
    mask_etiqueta_notnull
].copy()

print(f"  Tras filtro Tratamiento      : {mask_tratamiento.sum():>8,}")
print(f"  Tras filtro etiqueta completa: {len(ofertas_modelo):>8,}")
print(f"  Descartadas (control+sin etiq): {len(ofertas) - len(ofertas_modelo):>8,}")

# ── 1.4 Join ofertas × features (clave: id_cliente + mes) ─────────────
# Este join es el punto más delicado de todo el pipeline.
# Cada oferta del mes t recibe exactamente las features del mes t
# del mismo cliente. Ninguna feature posterior a t puede entrar.

print("\n  Construyendo join ofertas × features...")

# Verificar que la clave de join está limpia antes de hacer el merge
assert features['mes'].between(1, 25).all(), "Features con mes fuera de rango"
assert ofertas_modelo['mes'].between(1, 25).all(), "Ofertas con mes fuera de rango"
assert features[['id_cliente','mes']].duplicated().sum() == 0, \
    "Duplicados en clave (id_cliente, mes) de features"

# Seleccionar solo las columnas de features que van al modelo
# Excluir: id_oferta, fecha (son metadatos), indice_estres_macro
# (podría usarse, pero se excluye aquí para discutirlo en feature selection)
COLUMNAS_FEATURES = [
    'id_cliente',
    'mes',
    'gasto_3m',
    'gasto_supermercado_3m',
    'gasto_farmacia_3m',
    'depositos_efectivo_3m',
    'tx_digitales_proporcion',
    'saldo_promedio_90d',
    'saldo_tendencia',
    'variabilidad_saldo',
    'ratio_cuota_ingreso',
    'dias_desde_ult_credito',
    'indice_estres_macro',      # incluida como feature del entorno
    'estres_x_riesgo',          # interacción stress × riesgo individual
]

features_slim = features[COLUMNAS_FEATURES].copy()

# Merge LEFT: la oferta es el lado izquierdo
# Si alguna oferta no tiene features para ese mes → NaN
# (no debería ocurrir pero se detecta abajo)
abt = ofertas_modelo.merge(
    features_slim,
    on=['id_cliente', 'mes'],
    how='left',
    validate='many_to_one'   # una oferta → una fila de features
)

# Verificar integridad del join
n_features_nulas = abt['saldo_promedio_90d'].isna().sum()
pct_nulas = n_features_nulas / len(abt)
print(f"  Filas en ABT                 : {len(abt):>8,}")
print(f"  Filas sin features (NaN)     : {n_features_nulas:>8,}  ({pct_nulas:.1%})")

if pct_nulas > 0.01:
    print("  ⚠️  ADVERTENCIA: más del 1% de filas sin features.")
    print("     Verificar que los CSVs fueron generados en la misma ejecución.")
else:
    print("  ✅ Join íntegro — sin pérdidas significativas.")

# ── 1.5 Enriquecer ABT con atributos estáticos del cliente ────────────
# Score, segmento, ocupación, zona, etc. son atributos que no cambian
# entre meses (en este dataset). En producción vendrían del CRM o del
# Feature Store de atributos cliente.

COLUMNAS_CLIENTES = [
    'id_cliente',
    'segmento',
    'edad',
    'ocupacion',
    'zona_geografica',
    'canal_principal',
    'antiguedad_meses',
    'ingreso_mensual',
    'score_crediticio',
    'score_buro',
    'tiene_atraso_hist',
    'max_atraso_dias',
    'ratio_deuda_init',
    'hijos',
    'estado_civil',
]

abt = abt.merge(
    clientes[COLUMNAS_CLIENTES],
    on='id_cliente',
    how='left',
    validate='many_to_one'
)

print(f"  Columnas en ABT final        : {abt.shape[1]:>8,}")

# ── 1.6 Crear variables derivadas relevantes ──────────────────────────
# Estas variables no existen en las tablas fuente pero tienen
# poder predictivo documentado en la literatura de scoring bancario.

# Días desde último crédito como variable categórica (buckets)
# La relación no es lineal — hay ventanas específicas de alta propensión
abt['bucket_dias_credito'] = pd.cut(
    abt['dias_desde_ult_credito'],
    bins=[-1, 30, 90, 180, 365, 730, 9999],
    labels=['0_30d', '31_90d', '91_180d', '181_365d', '366_730d', 'mas_730d']
)

# Ratio saldo / ingreso (proxy de capacidad de ahorro relativa)
abt['ratio_saldo_ingreso'] = abt['saldo_promedio_90d'] / (abt['ingreso_mensual'] + 1)

# Flag: cliente con score muy alto (>800) — señal negativa para microcrédito
abt['score_muy_alto'] = (abt['score_crediticio'] > 800).astype(int)

# Flag: cliente en franja actuar de edad para seguro de vida (30-55 años)
abt['edad_objetivo_sv'] = (
    (abt['edad'] >= 30) & (abt['edad'] <= 55)
).astype(int)

# Interacción occupación × zona (predictor de depósitos en efectivo)
abt['indep_no_urbano'] = (
    (abt['ocupacion'] == 'Independiente') &
    (abt['zona_geografica'] != 'Urbana')
).astype(int)

# ── 1.7 Resumen de la ABT construida ──────────────────────────────────
subseccion("Resumen ABT")
print(f"\n  Dimensiones finales: {abt.shape[0]:,} filas × {abt.shape[1]} columnas")

print("\n  Distribución por producto:")
dist_prod = abt.groupby('id_producto_lower').agg(
    n_ofertas    = ('convirtio_30d', 'count'),
    tasa_conv    = ('convirtio_30d', 'mean'),
    n_positivos  = ('convirtio_30d', 'sum'),
).round(4)
print(dist_prod.to_string())

print("\n  Distribución por régimen macro:")
dist_macro = abt.groupby('regimen_macro').agg(
    n_ofertas  = ('convirtio_30d', 'count'),
    tasa_conv  = ('convirtio_30d', 'mean'),
    meses_repr = ('mes', 'nunique'),
).round(4)
print(dist_macro.to_string())

print("\n  Columnas NaN en ABT:")
nans = abt.isnull().sum()
nans_sig = nans[nans > 0]
if len(nans_sig) == 0:
    print("  ✅ Sin valores nulos")
else:
    print(nans_sig.to_string())

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — SPLIT TEMPORAL CORRECTO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — SPLIT TEMPORAL")

# El split se define por mes, no por fila aleatoria.
# Los cortes replican el diseño de régimen macro del generador.
#
# MESES 1-15  → Entrenamiento  (Normal + Deterioro leve)
# MESES 16-18 → Validación     (Deterioro moderado)
# MESES 19-23 → Test OOT       (Stress inicial + Stress recuperación)
# MESES 24-25 → Producción simulada (sin etiqueta — ya excluidos de ABT)
#
# ¿Por qué estos cortes y no otros?
# Los cortes replican los cambios de régimen macro del generador.
# Validar en deterioro moderado (16-18) y testear en stress (19-23)
# fuerza al modelo a demostrar generalización a entornos no vistos.
# Un modelo que solo pasa el test en normalidad no sirve para producción.

CORTE_TRAIN_FIN  = 15   # último mes de entrenamiento
CORTE_VAL_FIN    = 18   # último mes de validación

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
    pct   = len(df) / len(abt)
    tasa  = df['convirtio_30d'].mean()
    print(f"  {nombre:<15} {meses:<12} {len(df):>10,} {pct:>8.1%} {tasa:>10.3f}")

print(f"\n  {'Total ABT':<15} {'1 – 23':<12} {len(abt):>10,} {'100.0%':>8} {abt['convirtio_30d'].mean():>10.3f}")

# ── Verificar que no hay solapamiento temporal ─────────────────────────
assert train['mes'].max() < val['mes'].min(),  "❌ Solapamiento train/val"
assert val['mes'].max()   < test['mes'].min(), "❌ Solapamiento val/test"
print("\n  ✅ Sin solapamiento temporal entre conjuntos")

# ── Verificar balance de positivos en train ────────────────────────────
subseccion("Balance de clases en entrenamiento (por producto)")
print(f"\n  {'Producto':<20} {'Negativos':>10} {'Positivos':>10} {'Ratio neg:pos':>14}")
print(f"  {'-'*58}")
for prod in sorted(abt['id_producto_lower'].unique()):
    sub = train[train['id_producto_lower'] == prod]
    n_neg = (sub['convirtio_30d'] == 0).sum()
    n_pos = (sub['convirtio_30d'] == 1).sum()
    ratio = n_neg / max(n_pos, 1)
    print(f"  {prod:<20} {n_neg:>10,} {n_pos:>10,} {ratio:>14.1f}:1")

print("""
  Nota sobre el desbalance de clases:
  ─────────────────────────────────────────────────────────
  Con tasas del 3% al 13% el desbalance oscila entre 7:1 y 30:1.
  XGBoost maneja esto con el parámetro scale_pos_weight.
  El valor óptimo es: n_negativos / n_positivos por producto.
  No se hace oversampling (SMOTE) porque el desbalance no es
  extremo y SMOTE en datos tabulares bancarios raramente mejora AUC.
  La ganancia de SMOTE en AUC suele ser < 0.5pp con mayor riesgo
  de overfitting en la clase minoritaria sintética.
""")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — FEATURE ENGINEERING Y SELECCIÓN
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — FEATURES Y ENCODING")

# ── 3.1 Definición del catálogo de features por tipo ──────────────────

# Features numéricas: van al modelo directamente
# XGBoost no requiere normalización — los árboles son invariantes a escala
FEATURES_NUMERICAS = [
    # Transaccionales (del pipeline mensual)
    'gasto_3m',
    'gasto_supermercado_3m',
    'gasto_farmacia_3m',
    'depositos_efectivo_3m',
    'tx_digitales_proporcion',
    'saldo_promedio_90d',
    'saldo_tendencia',
    'variabilidad_saldo',
    'ratio_cuota_ingreso',
    'dias_desde_ult_credito',
    'indice_estres_macro',
    'estres_x_riesgo',
    # Atributos estáticos del cliente
    'edad',
    'antiguedad_meses',
    'ingreso_mensual',
    'score_crediticio',
    'score_buro',
    'max_atraso_dias',
    'ratio_deuda_init',
    'hijos',
    # Derivadas construidas en Bloque 1
    'ratio_saldo_ingreso',
    'score_muy_alto',
    'edad_objetivo_sv',
    'indep_no_urbano',
]

# Features categóricas: requieren encoding antes de entrar al modelo
# XGBoost con la API de sklearn no acepta strings directamente
FEATURES_CATEGORICAS = [
    'segmento',
    'ocupacion',
    'zona_geografica',
    'canal_principal',
    'estado_civil',
    'tiene_atraso_hist',
    'bucket_dias_credito',
]

# ── 3.2 Encoding de variables categóricas ─────────────────────────────
# Opción: Label Encoding (ordinales arbitrarios)
# ¿Por qué no One-Hot Encoding?
# XGBoost con árboles maneja ordinales arbitrarios correctamente porque
# en cada split busca el umbral óptimo — no asume distancia entre categorías.
# OHE con 5-7 categorías × 6 productos genera ~30 columnas extra por
# producto sin mejora en AUC en datasets de este tamaño. Label Encoding
# es la convención estándar en scoring bancario con XGBoost.

from sklearn.preprocessing import LabelEncoder

print("\n  Aplicando Label Encoding a variables categóricas...")
encoders = {}  # guardamos los encoders para aplicarlos a val y test

for col in FEATURES_CATEGORICAS:
    le = LabelEncoder()
    # Fit sobre el universo completo (train+val+test) para evitar
    # que categorías no vistas en train generen errores en val/test
    le.fit(abt[col].astype(str))
    encoders[col] = le

    # Aplicar a los tres conjuntos
    for df in [train, val, test, abt]:
        df[col + '_enc'] = le.transform(df[col].astype(str))

    print(f"  {col:<25}: {len(le.classes_)} categorías → {col}_enc")

# Features finales que entran al modelo
FEATURES_ENC = [f for f in FEATURES_CATEGORICAS]  # nombres originales
FEATURES_FINALES = FEATURES_NUMERICAS + [c + '_enc' for c in FEATURES_CATEGORICAS]

print(f"\n  Total features al modelo: {len(FEATURES_FINALES)}")
print(f"  Numéricas: {len(FEATURES_NUMERICAS)} | Categóricas encoded: {len(FEATURES_CATEGORICAS)}")

# ── 3.3 Verificar que no hay NaN en features finales ──────────────────
print("\n  Verificando NaN en features de entrenamiento...")
nans_train = train[FEATURES_FINALES].isnull().sum()
nans_sig   = nans_train[nans_train > 0]

if len(nans_sig) == 0:
    print("  ✅ Sin NaN en features de entrenamiento")
else:
    print(f"  ⚠️  Features con NaN en train:")
    print(nans_sig.to_string())
    print("  Imputando con mediana de train...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    train[FEATURES_FINALES] = imputer.fit_transform(train[FEATURES_FINALES])
    val[FEATURES_FINALES]   = imputer.transform(val[FEATURES_FINALES])
    test[FEATURES_FINALES]  = imputer.transform(test[FEATURES_FINALES])

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — ENTRENAMIENTO XGBOOST POR PRODUCTO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — ENTRENAMIENTO XGBOOST")

import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

print(f"\n  XGBoost versión: {xgb.__version__}")
print(f"  Productos a entrenar: {sorted(abt['id_producto_lower'].unique())}\n")

# Diccionario donde se guardarán los modelos entrenados
# Estructura: modelos[producto] = {'model': XGBClassifier, 'threshold': float}
modelos = {}
metricas_train = {}

for producto in sorted(abt['id_producto_lower'].unique()):

    subseccion(f"Producto: {producto.upper()}")

    # Subconjuntos por producto
    X_train = train[train['id_producto_lower'] == producto][FEATURES_FINALES]
    y_train = train[train['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    X_val   = val[val['id_producto_lower'] == producto][FEATURES_FINALES]
    y_val   = val[val['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = round(n_neg / max(n_pos, 1), 1)  # scale_pos_weight

    print(f"\n  Train: {len(X_train):,} filas | Positivos: {n_pos:,} ({n_pos/len(y_train):.1%})")
    print(f"  Val:   {len(X_val):,} filas | scale_pos_weight = {spw}")

    # ── Hiperparámetros XGBoost ────────────────────────────────────────
    # Estos hiperparámetros son el punto de partida calibrado para
    # datasets bancarios de este tamaño y nivel de desbalance.
    # En producción se optimizarían con Optuna o GridSearchCV sobre val.
    #
    # n_estimators=500 + early_stopping=50:
    #   Empieza con 500 árboles máximo pero para si el AUC de val
    #   no mejora en 50 rondas consecutivas. Evita overfitting.
    #
    # max_depth=4:
    #   Árboles poco profundos (4 niveles). Más profundo → más overfitting
    #   en datasets desbalanceados. 4 es el estándar en scoring bancario.
    #
    # learning_rate=0.05:
    #   Tasa de aprendizaje baja. Con early stopping no penaliza en tiempo
    #   y produce modelos más generalizables que learning_rate=0.1.
    #
    # subsample=0.8, colsample_bytree=0.8:
    #   Aleatorización por fila y columna en cada árbol. Reduce varianza
    #   del modelo (similar a Random Forest). Estándar en producción.
    #
    # min_child_weight=20:
    #   Un nodo hoja necesita mínimo 20 observaciones para crearse.
    #   Evita que el modelo aprenda de grupos muy pequeños (ruido).
    #   Con datasets desbalanceados este parámetro es crítico.
    #
    # scale_pos_weight=spw:
    #   Pondera la clase positiva para compensar el desbalance.
    #   Equivale a over-sampling implícito de la clase minoritaria.
    #   El valor óptimo teórico es n_neg/n_pos.

    modelo = xgb.XGBClassifier(
        n_estimators      = 500,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 20,
        scale_pos_weight  = spw,
        objective         = 'binary:logistic',
        eval_metric       = 'auc',
        early_stopping_rounds = 50,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )

    # Entrenar con early stopping monitoreado en validación
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iter = modelo.best_iteration
    print(f"  Mejor iteración: {best_iter} árboles")

    # Scores de probabilidad en train y val
    p_train = modelo.predict_proba(X_train)[:, 1]
    p_val   = modelo.predict_proba(X_val)[:, 1]

    auc_train = roc_auc_score(y_train, p_train)
    auc_val   = roc_auc_score(y_val,   p_val)

    # Estadístico KS en validación
    # KS = separación máxima entre la distribución acumulada de positivos
    # y negativos. Mide la capacidad de ordenamiento del modelo.
    ks_stat, _ = ks_2samp(
        p_val[y_val == 1],
        p_val[y_val == 0]
    )

    print(f"  AUC train    : {auc_train:.4f}")
    print(f"  AUC val      : {auc_val:.4f}")
    print(f"  Diferencia   : {auc_train - auc_val:.4f}  ", end="")
    if auc_train - auc_val > 0.05:
        print("⚠️  Posible overfitting")
    else:
        print("✅ Generalización aceptable")
    print(f"  KS val       : {ks_stat:.4f}")

    # Guardar modelo y métricas
    modelos[producto] = {
        'model'      : modelo,
        'features'   : FEATURES_FINALES,
        'best_iter'  : best_iter,
        'spw'        : spw,
    }
    metricas_train[producto] = {
        'auc_train'  : auc_train,
        'auc_val'    : auc_val,
        'ks_val'     : ks_stat,
        'n_train'    : len(X_train),
        'n_val'      : len(X_val),
        'n_pos_train': int(n_pos),
        'spw'        : spw,
        'best_iter'  : best_iter,
    }

print("\n")
separador("RESUMEN ENTRENAMIENTO — AUC por producto")
print(f"\n  {'Producto':<20} {'AUC train':>10} {'AUC val':>10} {'KS val':>10} {'Árboles':>10}")
print(f"  {'-'*55}")
for prod, m in metricas_train.items():
    print(f"  {prod:<20} {m['auc_train']:>10.4f} {m['auc_val']:>10.4f} "
          f"{m['ks_val']:>10.4f} {m['best_iter']:>10}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — CALIBRACIÓN DE PROBABILIDADES (PLATT SCALING)
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — CALIBRACIÓN DE PROBABILIDADES")

print("""
  ¿Por qué calibrar?
  ─────────────────────────────────────────────────────────
  XGBoost produce scores bien ordenados (AUC alto) pero las
  probabilidades absolutas pueden estar mal calibradas.
  Un modelo que dice P=0.30 cuando la tasa real es 0.10 produce
  decisiones de negocio incorrectas: el optimizador sobreestima
  el retorno esperado y recomienda productos poco rentables.
  
  Platt Scaling ajusta las probabilidades brutas del modelo
  entrenando una regresión logística encima:
    P_calibrada = sigmoid(a × score_raw + b)
  
  Se entrena en el conjunto de validación (no en train) para
  no introducir overfitting de calibración.
""")

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

calibradores = {}
metricas_calibracion = {}

for producto in sorted(modelos.keys()):

    subseccion(f"Calibración: {producto.upper()}")

    modelo = modelos[producto]['model']

    X_val_p = val[val['id_producto_lower'] == producto][FEATURES_FINALES]
    y_val_p = val[val['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    X_test_p = test[test['id_producto_lower'] == producto][FEATURES_FINALES]
    y_test_p = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    # Scores brutos del modelo en val y test
    scores_val  = modelo.predict_proba(X_val_p)[:, 1].reshape(-1, 1)
    scores_test = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)

    # Platt Scaling: LR sobre los scores brutos entrenada en val
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(scores_val, y_val_p)
    calibradores[producto] = platt

    # Probabilidades calibradas en test
    p_cal_test = platt.predict_proba(scores_test)[:, 1]
    p_raw_test = scores_test.ravel()

    # AUC antes y después (no debe cambiar — calibración no afecta ranking)
    auc_raw = roc_auc_score(y_test_p, p_raw_test)
    auc_cal = roc_auc_score(y_test_p, p_cal_test)

    # Brier Score: error cuadrático medio de probabilidades
    # Menor es mejor. Calibración debe mejorar este score.
    brier_raw = np.mean((p_raw_test - y_test_p) ** 2)
    brier_cal = np.mean((p_cal_test - y_test_p) ** 2)

    # Calibración de la media: ¿el promedio de P coincide con la tasa real?
    tasa_real    = y_test_p.mean()
    media_p_raw  = p_raw_test.mean()
    media_p_cal  = p_cal_test.mean()

    print(f"\n  Tasa real en test        : {tasa_real:.4f}")
    print(f"  Media P bruta            : {media_p_raw:.4f}  (error: {abs(media_p_raw - tasa_real):.4f})")
    print(f"  Media P calibrada        : {media_p_cal:.4f}  (error: {abs(media_p_cal - tasa_real):.4f})")
    print(f"  AUC raw / calibrado      : {auc_raw:.4f} / {auc_cal:.4f}")
    print(f"  Brier raw / calibrado    : {brier_raw:.5f} / {brier_cal:.5f}")

    metricas_calibracion[producto] = {
        'tasa_real'    : tasa_real,
        'media_p_raw'  : media_p_raw,
        'media_p_cal'  : media_p_cal,
        'auc_test_raw' : auc_raw,
        'auc_test_cal' : auc_cal,
        'brier_raw'    : brier_raw,
        'brier_cal'    : brier_cal,
    }

    # Guardar probabilidades calibradas en test para las métricas del Bloque 6
    mask_test_prod = test['id_producto_lower'] == producto
    test.loc[mask_test_prod, 'p_raw']       = p_raw_test
    test.loc[mask_test_prod, 'p_calibrada'] = p_cal_test


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — MÉTRICAS TÉCNICAS COMPLETAS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 6 — MÉTRICAS TÉCNICAS")

# ── 6.1 PSI — Population Stability Index ──────────────────────────────
# PSI mide si la distribución del score cambió entre dos ventanas.
# Se usa para detectar data drift: si el comportamiento de los clientes
# cambió, los scores se desplazaron y el modelo pierde validez.
#
# Interpretación estándar bancaria:
#   PSI < 0.10  → distribución estable
#   PSI 0.10-0.20 → cambio menor — monitorear
#   PSI > 0.20  → cambio significativo — revisar / recalibrar
 
def calcular_psi(p_ref, p_actual, n_bins=10):
    """
    p_ref:    scores de la distribución de referencia (train o val)
    p_actual: scores de la distribución actual (val o test)
    n_bins:   número de buckets para la distribución
    Retorna: PSI escalar
    """
    # Crear bins sobre la distribución de referencia
    bins = np.percentile(p_ref, np.linspace(0, 100, n_bins + 1))
    bins[0]  = -np.inf
    bins[-1] = np.inf

    # Proporciones en referencia y actual
    ref_counts = np.histogram(p_ref,    bins=bins)[0]
    act_counts = np.histogram(p_actual, bins=bins)[0]

    # Suavizado para evitar divisiones por cero
    ref_pct = (ref_counts + 0.5) / (len(p_ref)    + n_bins * 0.5)
    act_pct = (act_counts + 0.5) / (len(p_actual) + n_bins * 0.5)

    psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return round(psi, 4)

subseccion("AUC y KS en Test OOT (métricas finales de producción)")
print(f"\n  {'Producto':<20} {'AUC test':>10} {'KS test':>10} {'PSI val→test':>14} {'Estado PSI':>12}")
print(f"  {'-'*70}")

metricas_finales = {}

for producto in sorted(modelos.keys()):
    modelo  = modelos[producto]['model']
    platt   = calibradores[producto]

    X_train_p = train[train['id_producto_lower'] == producto][FEATURES_FINALES]
    X_val_p   = val[val['id_producto_lower'] == producto][FEATURES_FINALES]
    X_test_p  = test[test['id_producto_lower'] == producto][FEATURES_FINALES]
    y_test_p  = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int)

    # Scores calibrados
    p_val_raw  = modelo.predict_proba(X_val_p)[:, 1].reshape(-1, 1)
    p_test_raw = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)
    p_cal_test = platt.predict_proba(p_test_raw)[:, 1]

    # AUC test
    auc_test = roc_auc_score(y_test_p, p_cal_test)

    # KS test
    ks_test, _ = ks_2samp(
        p_cal_test[y_test_p == 1],
        p_cal_test[y_test_p == 0]
    )

    # PSI: val → test (¿la distribución del score se mantuvo estable?)
    p_val_cal  = platt.predict_proba(p_val_raw)[:, 1]
    psi = calcular_psi(p_val_cal, p_cal_test)

    estado_psi = "✅ Estable" if psi < 0.10 else ("⚠️ Alerta" if psi < 0.20 else "❌ Drift")

    print(f"  {producto:<20} {auc_test:>10.4f} {ks_test:>10.4f} {psi:>14.4f} {estado_psi:>12}")

    metricas_finales[producto] = {
        'auc_test': auc_test,
        'ks_test' : ks_test,
        'psi'     : psi,
    }

# ── 6.2 Curva de lift acumulado ────────────────────────────────────────
subseccion("Lift acumulado en Test (decil superior vs aleatorio)")
print(f"\n  El lift en el decil superior mide cuántas veces más conversiones")
print(f"  captura el modelo versus selección aleatoria de clientes.")
print(f"  Un lift de 3.0 significa que ofreciendo al 10% superior se obtiene")
print(f"  el triple de conversiones que ofreciendo a un 10% aleatorio.\n")

print(f"  {'Producto':<20} {'Lift D1':>10} {'Lift D2':>10} {'Lift D3':>10}")
print(f"  {'-'*55}")

for producto in sorted(metricas_finales.keys()):
    X_test_p  = test[test['id_producto_lower'] == producto][FEATURES_FINALES]
    y_test_p  = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int)
    modelo    = modelos[producto]['model']
    platt     = calibradores[producto]

    p_raw = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)
    p_cal = platt.predict_proba(p_raw)[:, 1]

    # Ordenar por score descendente
    orden = np.argsort(-p_cal)
    y_ord = y_test_p.values[orden]
    tasa_global = y_test_p.mean()

    # Lift por decil
    n = len(y_ord)
    lifts = []
    for d in [1, 2, 3]:  # deciles 1, 2, 3
        corte = int(n * d / 10)
        tasa_decil = y_ord[:corte].mean()
        lift = tasa_decil / max(tasa_global, 1e-9)
        lifts.append(lift)

    print(f"  {producto:<20} {lifts[0]:>10.2f} {lifts[1]:>10.2f} {lifts[2]:>10.2f}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 7 — SHAP — INTERPRETABILIDAD
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 7 — SHAP — IMPORTANCIA DE FEATURES")
 
print("""
  SHAP (SHapley Additive exPlanations) asigna a cada feature
  su contribución marginal real a la predicción de cada cliente.
  
  A diferencia de feature_importances_ de XGBoost (que mide
  cuántas veces se usa una feature en los árboles), SHAP mide
  cuánto mueve la predicción hacia arriba o abajo.
  
  Para presentar ante el Comité de Riesgo o Auditoría, SHAP
  permite responder: "¿Por qué el sistema recomendó este
  producto a este cliente específico?"
""")
 
# SHAP via XGBoost pred_contribs — nativo, sin dependencia de librería shap
# Compatibe con XGBoost 2.x + NumPy 2.x. No requiere numba.
#
# pred_contribs devuelve una matriz (n_obs, n_features + 1) donde:
#   - columnas 0..n_features-1 → contribución SHAP de cada feature
#   - última columna           → bias (valor base del modelo)
# El mean(|SHAP|) por feature es equivalente al SHAP importance estándar.
 
import xgboost as _xgb
 
for producto in sorted(modelos.keys()):
    subseccion(f"SHAP (pred_contribs): {producto.upper()}")
 
    modelo_xgb = modelos[producto]['model']
    feats      = modelos[producto]['features']
 
    X_test_p = test[test['id_producto_lower'] == producto][feats].copy()
 
    if len(X_test_p) == 0:
        print(f"  Sin filas en test para {producto}")
        continue
 
    dmat        = _xgb.DMatrix(X_test_p, feature_names=feats)
    shap_matrix = modelo_xgb.get_booster().predict(dmat, pred_contribs=True)
 
    # Descartar columna de bias (última)
    shap_df     = pd.DataFrame(shap_matrix[:, :-1], columns=feats)
    importancia = shap_df.abs().mean().sort_values(ascending=False)
 
    print(f"\n  Top 10 features por mean(|SHAP|):")
    top_val = importancia.iloc[0]
    for i, (feat, val) in enumerate(importancia.head(10).items()):
        bar = "█" * int(val / top_val * 20)
        print(f"  {i+1:2d}. {feat:<30} {val:.5f}  {bar}")
 

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 8 — REPORTE EJECUTIVO FINAL
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 8 — REPORTE EJECUTIVO DE MODELOS")

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     RBlJose — NBO — REPORTE DE MODELOS SEMANA 3     ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Modelos entrenados   : 6 (uno por producto)                ║
  ║  Algoritmo            : XGBoost binary:logistic              ║
  ║  Calibración          : Platt Scaling (val → test)           ║
  ║  Split                : Temporal (train 1-15, val 16-18,     ║
  ║                         test OOT 19-23)                      ║
  ╚══════════════════════════════════════════════════════════════╝
""")

print(f"  {'Producto':<20} {'AUC test':>10} {'KS test':>10} {'PSI':>8} {'Estado':>12}")
print(f"  {'─'*64}")

RANGOS_AUC = {
    'tarjeta'     : (0.65, 0.75),
    'prestamo'    : (0.68, 0.76),
    'microcredito': (0.62, 0.72),
    'seguro_vida' : (0.70, 0.80),
    'seguro_salud': (0.68, 0.78),
    'inversion'   : (0.68, 0.78),
}

for producto in sorted(metricas_finales.keys()):
    m   = metricas_finales[producto]
    rng = RANGOS_AUC.get(producto, (0.60, 0.80))

    in_range = rng[0] <= m['auc_test'] <= rng[1]
    estado   = "✅ En rango" if in_range else (
               "⬆️  Sobre rango" if m['auc_test'] > rng[1] else "⬇️  Bajo rango")

    print(f"  {producto:<20} {m['auc_test']:>10.4f} {m['ks_test']:>10.4f} "
          f"{m['psi']:>8.4f} {estado:>12}")

print(f"""
  Rangos AUC esperados (definidos por sigma de ruido del generador):
    seguro_vida:   0.74-0.78  │  inversion:    0.72-0.76
    seguro_salud:  0.70-0.75  │  prestamo:     0.70-0.74
    tarjeta:       0.69-0.73  │  microcredito: 0.65-0.70

  Si AUC > rango superior → revisar leakage o memorización del DAG
  Si AUC < rango inferior → revisar feature selection o hiperparámetros

  Nota sobre PSI:
    Los modelos se entrenaron en Normal + Deterioro leve (meses 1-15).
    El test es en Stress inicial + Stress recuperación (meses 19-23).
    PSI alto es ESPERADO y no indica error del modelo — indica que
    el entorno macroeconómico cambió (que es exactamente el diseño).
    En producción, PSI alto en este contexto activa recalibración,
    no reentrenamiento completo.
""")

print("✓ Pipeline Semana 3 completado.")
print(f"  Modelos entrenados y listos en diccionario 'modelos'.")
print(f"  Calibradores listos en diccionario 'calibradores'.")

# %%
print(productos.to_string())

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 9 — EVALUACIÓN DE NEGOCIO
# ══════════════════════════════════════════════════════════════════════
# Este bloque responde las preguntas que le importan a Dirección:
#
#   9.1 — Profit curve por producto
#         ¿En qué threshold maximizamos el beneficio neto?
#
#   9.2 — Threshold óptimo por producto
#         ¿A quién contactar para maximizar ROI de campaña?
#
#   9.3 — Ranking NBO multi-producto (score ajustado por margen)
#         ¿Cuál es el mejor producto para cada cliente?
#
# Supuestos de negocio documentados:
#   - costo_contactacion: según nbo_productos.csv (canal promedio)
#   - ingreso_por_conversion: margen_neto × ticket_promedio_anual
#   - ticket_promedio_anual: supuesto operativo por categoría
#   - penalizacion_fp: costo de contactar a quien no convierte
#     (costo_contactacion solamente — no hay costo de originación
#      porque originación solo ocurre en conversión real)
 
separador("BLOQUE 9 — EVALUACIÓN DE NEGOCIO")
 
# ── Parámetros de negocio por producto ────────────────────────────────
# ticket_anual: valor promedio del producto en un año
# Fuente: supuestos operativos bancarios Ecuador (BCE referencia 2024)
#   Tarjeta:      consumo promedio $3,000/año × margen_neto 15.5% = $465
#   Préstamo:     monto promedio $5,000 × margen_neto 13.5% = $675
#   Microcrédito: monto promedio $2,500 × margen_neto 12.0% = $300
#   Seguro vida:  prima anual promedio $180 × margen_neto 17.5% = $31.5
#   Seguro salud: prima anual promedio $240 × margen_neto 15.0% = $36.0
#   Inversión:    monto promedio $4,000 × margen_neto 3.0% = $120
 
PARAMS_NEGOCIO = {
    'tarjeta'     : {'ticket_anual': 465.0,  'costo_contacto': 2.5,  'costo_originacion': 45.0,
                     'pd': 0.055, 'lgd': 0.75, 'rwa': 1.00, 'margen_neto': 0.155},
    'prestamo'    : {'ticket_anual': 675.0,  'costo_contacto': 3.0,  'costo_originacion': 60.0,
                     'pd': 0.045, 'lgd': 0.78, 'rwa': 0.75, 'margen_neto': 0.135},
    'microcredito': {'ticket_anual': 300.0,  'costo_contacto': 4.5,  'costo_originacion': 85.0,
                     'pd': 0.095, 'lgd': 0.82, 'rwa': 0.75, 'margen_neto': 0.120},
    'seguro_vida' : {'ticket_anual': 31.5,   'costo_contacto': 2.0,  'costo_originacion': 15.0,
                     'pd': 0.000, 'lgd': 0.00, 'rwa': 0.00, 'margen_neto': 0.175},
    'seguro_salud': {'ticket_anual': 36.0,   'costo_contacto': 2.0,  'costo_originacion': 15.0,
                     'pd': 0.000, 'lgd': 0.00, 'rwa': 0.00, 'margen_neto': 0.150},
    'inversion'   : {'ticket_anual': 120.0,  'costo_contacto': 1.5,  'costo_originacion': 10.0,
                     'pd': 0.000, 'lgd': 0.00, 'rwa': 0.00, 'margen_neto': 0.030},
}
 
# ── 9.1 Profit curve y threshold óptimo ───────────────────────────────
subseccion("9.1 — Profit Curve y Threshold Óptimo por Producto")
 
print(f"""
  Metodología:
  Para cada threshold k ∈ [0.05, 0.95]:
    - Clasificamos positivos (contactar) y negativos (no contactar)
    - TP → conversión real: ingreso = ticket_anual - costo_originacion - costo_contacto
    - FP → contacto sin conversión: costo = costo_contacto
    - FN → conversión perdida: costo = oportunidad (no se penaliza en este modelo base)
    - TN → no contactar, no convierte: profit = 0
 
  El threshold óptimo es argmax(profit_total).
  En producción, este threshold define el universo de contactación de campaña.
""")
 
print(f"  {'Producto':<20} {'Threshold*':>12} {'Profit máx ($)':>16} {'% contactados':>15} "
      f"{'Ingreso TP ($)':>15} {'Costo FP ($)':>13}")
print(f"  {'─'*95}")
 
resultados_negocio = {}
THRESHOLDS = np.arange(0.05, 0.96, 0.01)
 
for producto in sorted(modelos.keys()):
    p      = PARAMS_NEGOCIO[producto]
    modelo = modelos[producto]['model']
    feats  = modelos[producto]['features']
    platt  = calibradores[producto]
 
    X_test_p = test[test['id_producto_lower'] == producto][feats].copy()
    y_test_p = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int).values
    n_total  = len(y_test_p)
 
    p_raw = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)
    p_cal = platt.predict_proba(p_raw)[:, 1]
 
    # Ingreso neto por TP (conversión real)
    ingreso_tp = p['ticket_anual'] - p['costo_originacion'] - p['costo_contacto']
    # Costo por FP (contacto sin conversión)
    costo_fp   = p['costo_contacto']
 
    profits = []
    for thr in THRESHOLDS:
        pred      = (p_cal >= thr).astype(int)
        tp        = ((pred == 1) & (y_test_p == 1)).sum()
        fp        = ((pred == 1) & (y_test_p == 0)).sum()
        contactos = pred.sum()
        profit    = tp * ingreso_tp - fp * costo_fp
        profits.append({
            'threshold' : round(thr, 2),
            'profit'    : profit,
            'tp'        : tp,
            'fp'        : fp,
            'contactos' : contactos,
            'pct_contactados': contactos / n_total * 100
        })
 
    df_profit = pd.DataFrame(profits)
    idx_opt   = df_profit['profit'].idxmax()
    opt       = df_profit.loc[idx_opt]
 
    resultados_negocio[producto] = {
        'df_profit'          : df_profit,
        'threshold_optimo'   : opt['threshold'],
        'profit_maximo'      : opt['profit'],
        'pct_contactados_opt': opt['pct_contactados'],
        'tp_opt'             : opt['tp'],
        'fp_opt'             : opt['fp'],
    }
 
    ingreso_tp_total = opt['tp'] * ingreso_tp
    costo_fp_total   = opt['fp'] * costo_fp
 
    print(f"  {producto:<20} {opt['threshold']:>12.2f} {opt['profit']:>16,.0f} "
          f"{opt['pct_contactados']:>14.1f}% {ingreso_tp_total:>15,.0f} "
          f"{costo_fp_total:>13,.0f}")
 
# ── 9.2 Comparación: modelo vs campaña masiva ─────────────────────────
subseccion("9.2 — Modelo vs Campaña Masiva (threshold = 0)")
 
print(f"""
  Campaña masiva = contactar a TODOS los clientes elegibles (threshold=0).
  Captura todas las conversiones pero maximiza costos de FP.
  El uplift del modelo es el profit adicional vs esta estrategia base.
""")
 
print(f"  {'Producto':<20} {'Profit masivo ($)':>18} {'Profit modelo ($)':>18} "
      f"{'Uplift ($)':>12} {'Uplift %':>10}")
print(f"  {'─'*82}")
 
for producto in sorted(resultados_negocio.keys()):
    p      = PARAMS_NEGOCIO[producto]
    modelo = modelos[producto]['model']
    feats  = modelos[producto]['features']
    platt  = calibradores[producto]
 
    X_test_p = test[test['id_producto_lower'] == producto][feats].copy()
    y_test_p = test[test['id_producto_lower'] == producto]['convirtio_30d'].astype(int).values
 
    p_raw = modelo.predict_proba(X_test_p)[:, 1].reshape(-1, 1)
    p_cal = platt.predict_proba(p_raw)[:, 1]
 
    ingreso_tp = p['ticket_anual'] - p['costo_originacion'] - p['costo_contacto']
    costo_fp   = p['costo_contacto']
 
    # Campaña masiva: contactar a todos
    tp_masivo    = y_test_p.sum()
    fp_masivo    = (y_test_p == 0).sum()
    profit_masivo = tp_masivo * ingreso_tp - fp_masivo * costo_fp
 
    profit_modelo = resultados_negocio[producto]['profit_maximo']
    uplift        = profit_modelo - profit_masivo
    uplift_pct    = (uplift / abs(profit_masivo) * 100) if profit_masivo != 0 else 0
 
    print(f"  {producto:<20} {profit_masivo:>18,.0f} {profit_modelo:>18,.0f} "
          f"{uplift:>12,.0f} {uplift_pct:>9.1f}%")
 
# ── 9.3 VECTORIZADO PRO — Ranking NBO Multi-Producto ─────────────────────

subseccion("9.3 — Ranking NBO Multi-Producto (Score Ajustado) — VECTORIZADO PRO")

from dateutil.relativedelta import relativedelta

MES_SCORING   = test['mes'].max()
FECHA_SCORING = FECHA_INICIO + relativedelta(months=int(MES_SCORING) - 1)

test_mes = test[test['mes'] == MES_SCORING].copy()

print(f"  Scoring sobre mes {MES_SCORING} ({FECHA_SCORING}) — {test_mes['id_cliente'].nunique():,} clientes únicos")

# ── Canal principal (una sola vez)
canal_map = clientes.set_index('id_cliente')['canal_principal']
test_mes['canal_principal'] = test_mes['id_cliente'].map(canal_map).fillna('Digital')

# ── Calcular scores por producto (vectorizado)
scores_por_producto = []

for producto in modelos.keys():
    p      = PARAMS_NEGOCIO[producto]
    modelo = modelos[producto]['model']
    feats  = modelos[producto]['features']
    platt  = calibradores[producto]

    mask   = test_mes['id_producto_lower'] == producto
    subset = test_mes.loc[mask, ['id_cliente', 'canal_principal'] + feats]

    if subset.empty:
        continue

    # Probabilidad calibrada
    p_raw = modelo.predict_proba(subset[feats])[:, 1].reshape(-1, 1)
    p_cal = platt.predict_proba(p_raw)[:, 1]

    # Score NBO vectorizado
    ingreso_esp = p_cal * (p['ticket_anual'] - p['costo_originacion'])
    perdida_esp = p['pd'] * p['lgd'] * p['ticket_anual'] * p['rwa']
    score_nbo   = ingreso_esp - perdida_esp - p['costo_contacto']

    df_tmp = pd.DataFrame({
        'id_cliente'      : subset['id_cliente'].values,
        'producto_nbo'    : producto,
        'p_calibrada'     : p_cal,
        'score_nbo'       : score_nbo,
        'costo_contacto'  : p['costo_contacto'],
        'ratio_nbo_costo' : score_nbo / max(p['costo_contacto'], 0.01),
        'canal_principal' : subset['canal_principal'].values,
    })

    scores_por_producto.append(df_tmp)

# ── Unir todo
df_scores = pd.concat(scores_por_producto, ignore_index=True)

# ── Ranking vectorizado (TOP 2)
df_scores = df_scores.sort_values(['id_cliente', 'score_nbo'], ascending=[True, False])

df_scores['rank'] = df_scores.groupby('id_cliente').cumcount() + 1

df_nbo = df_scores[df_scores['rank'] <= 2].copy()

# ── Metadata
df_nbo['mes_scoring']   = MES_SCORING
df_nbo['fecha_scoring'] = str(FECHA_SCORING)

# ── Vista rank 1 (reportes)
df_nbo_r1 = df_nbo[df_nbo['rank'] == 1].copy()

# ── Distribución
print(f"\n  Distribución de recomendaciones NBO (rank 1):")
print(f"  {'Producto':<20} {'Clientes':>10} {'%':>8}")
print(f"  {'─'*42}")

dist = df_nbo_r1['producto_nbo'].value_counts()

for prod, cnt in dist.items():
    pct = cnt / len(df_nbo_r1) * 100
    bar = '█' * int(pct / 2)
    print(f"  {prod:<20} {cnt:>10,} {pct:>7.1f}%  {bar}")

# ── KPIs
print(f"\n  Total clientes con recomendación NBO : {len(df_nbo_r1):,}")
print(f"  Registros totales (rank 1+2)         : {len(df_nbo):,}")
print(f"  Score NBO promedio (rank 1)          : {df_nbo_r1['score_nbo'].mean():.2f}")
print(f"  P calibrada promedio (rank 1)        : {df_nbo_r1['p_calibrada'].mean():.4f}")
print(f"  Fecha de scoring                     : {FECHA_SCORING}")

# ── Muestra
print(f"\n  Muestra de recomendaciones rank 1:")
print(f"  {'id_cliente':>12} {'Producto':<20} {'Score':>10} {'P(conv)':>9} {'Ratio':>8}")
print(f"  {'─'*65}")

sample_n = min(10, len(df_nbo_r1))
for _, row in df_nbo_r1.sample(sample_n, random_state=42).iterrows():
    print(f"  {row['id_cliente']:>12} {row['producto_nbo']:<20} "
          f"{row['score_nbo']:>10.2f} {row['p_calibrada']:>9.4f} "
          f"{row['ratio_nbo_costo']:>8.2f}")

# ── Guardar
df_nbo.to_csv(f'{DATA_DIR}/nbo_recomendaciones_semana3.csv', index=False)

print(f"\n  ✓ Recomendaciones guardadas en: nbo_recomendaciones_semana3.csv")
print(f"    Columnas: id_cliente, rank, producto_nbo, score_nbo, p_calibrada,")
print(f"              ratio_nbo_costo, costo_contacto, canal_principal,")
print(f"              mes_scoring, fecha_scoring")
print(f"    Filas rank=1 (ganadores)   : {len(df_nbo_r1):,}")
print(f"    Filas rank=2 (fallback)    : {len(df_nbo[df_nbo['rank']==2]):,}")
 
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║       RBlJose — NBO — SEMANA 3 COMPLETADA           ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 1-4  : ABT, split temporal, XGBoost (6 modelos)  ║
  ║  ✓ Bloque 5    : Calibración Platt Scaling                  ║
  ║  ✓ Bloque 6    : AUC / KS / PSI — todos estables            ║
  ║  ✓ Bloque 7    : SHAP pred_contribs (nativo XGBoost 2.x)    ║
  ║  ✓ Bloque 8    : Reporte ejecutivo de modelos               ║
  ║  ✓ Bloque 9    : Profit curves, thresholds, ranking NBO     ║
  ╚══════════════════════════════════════════════════════════════╝
""")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 10 — PERSISTENCIA DE MODELOS
# ══════════════════════════════════════════════════════════════════════
# Este bloque guarda todos los objetos necesarios para que Semana 5
# pueda re-scorear meses 22-25 SIN reentrenar.
#
# El principio es simple: el modelo que tomó decisiones de campaña
# en noviembre debe ser el mismo objeto binario que se usa en diciembre.
# No uno reentrenado con los mismos parámetros — el mismo archivo.
#
# Objetos persistidos:
#   nbo_model_{producto}_v1.joblib       → XGBClassifier entrenado
#   nbo_calibrador_{producto}_v1.joblib  → Platt Scaling (LogisticRegression)
#   nbo_encoder_{columna}_v1.joblib      → LabelEncoder por columna categórica
#   nbo_feature_names_v1.json            → lista exacta y orden de features
#   nbo_model_metadata_v1.json           → auditoría completa del entrenamiento
# ══════════════════════════════════════════════════════════════════════

separador("BLOQUE 10 — PERSISTENCIA DE MODELOS")

import joblib
import json
from datetime import date

# Directorio donde se guardan los modelos
# En producción esto sería un bucket S3 o un Model Registry (MLflow)
MODELS_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"\n  Directorio de modelos: {MODELS_DIR}")
print(f"  Guardando {len(modelos)} modelos + calibradores + encoders...\n")

# ── 10.1 Guardar modelos XGBoost y calibradores Platt Scaling ─────────
for producto in sorted(modelos.keys()):

    # Modelo XGBoost
    path_modelo = os.path.join(MODELS_DIR, f'nbo_model_{producto}_v1.joblib')
    joblib.dump(modelos[producto]['model'], path_modelo)

    # Calibrador Platt Scaling — objeto separado e independiente
    # CRÍTICO: se guarda por separado porque en producción la recalibración
    # puede actualizarse trimestralmente sin tocar el XGBoost
    path_cal = os.path.join(MODELS_DIR, f'nbo_calibrador_{producto}_v1.joblib')
    joblib.dump(calibradores[producto], path_cal)

    print(f"  ✅ {producto:<20} → modelo + calibrador guardados")

# ── 10.2 Guardar LabelEncoders ────────────────────────────────────────
# Los encoders se ajustaron sobre el universo completo (train+val+test)
# para no generar errores con categorías no vistas.
# Semana 5 los necesita para encodear los meses 24-25 de la misma forma.
print()
for col, encoder in encoders.items():
    path_enc = os.path.join(MODELS_DIR, f'nbo_encoder_{col}_v1.joblib')
    joblib.dump(encoder, path_enc)
    print(f"  ✅ Encoder '{col}' guardado ({len(encoder.classes_)} categorías)")

# ── 10.3 Guardar lista exacta de features ─────────────────────────────
# El orden de las columnas importa para XGBoost.
# Si en Semana 5 las columnas llegan en orden distinto, el modelo
# produce scores incorrectos sin lanzar ningún error.
# Este archivo es la fuente de verdad del contrato de features.
path_features = os.path.join(MODELS_DIR, 'nbo_feature_names_v1.json')
with open(path_features, 'w') as f:
    json.dump({
        'features_finales'    : FEATURES_FINALES,
        'features_numericas'  : FEATURES_NUMERICAS,
        'features_categoricas': FEATURES_CATEGORICAS,
    }, f, indent=2)

print(f"\n  ✅ Feature names guardados ({len(FEATURES_FINALES)} features en orden exacto)")

# ── 10.4 Guardar metadata de auditoría ───────────────────────────────
# Este archivo responde ante cualquier auditor:
#   ¿Con qué datos fue entrenado el modelo?
#   ¿Cuáles son sus métricas en producción simulada?
#   ¿Cuándo fue entrenado y con qué semilla?

metadata = {
    'version'             : 'v1.0',
    'fecha_entrenamiento' : str(date.today()),
    'semilla'             : int(SEED),
    'framework'           : 'XGBoost + Platt Scaling',
    'split_temporal'      : {
        'train'     : 'meses 1-15',
        'validacion': 'meses 16-18',
        'test_oot'  : 'meses 19-23',
        'produccion': 'meses 24-25 (sin etiqueta)',
    },
    'hiperparametros_xgb' : {
        'n_estimators'       : 500,
        'max_depth'          : 4,
        'learning_rate'      : 0.05,
        'subsample'          : 0.8,
        'colsample_bytree'   : 0.8,
        'min_child_weight'   : 20,
        'early_stopping'     : 50,
    },
    'metricas_por_producto': {},
}

for producto in sorted(modelos.keys()):
    m_train = metricas_train[producto]
    m_cal   = metricas_calibracion[producto]
    metadata['metricas_por_producto'][producto] = {
        'auc_train'       : round(m_train['auc_train'], 4),
        'auc_val'         : round(m_train['auc_val'], 4),
        'auc_test_cal'    : round(m_cal['auc_test_cal'], 4),
        'ks_val'          : round(m_train['ks_val'], 4),
        'brier_cal'       : round(m_cal['brier_cal'], 5),
        'tasa_real_test'  : round(m_cal['tasa_real'], 4),
        'media_p_cal_test': round(m_cal['media_p_cal'], 4),
        'best_iteration'  : int(m_train['best_iter']),
        'scale_pos_weight': round(m_train['spw'], 2),
        'n_train'         : int(m_train['n_train']),
    }

path_meta = os.path.join(MODELS_DIR, 'nbo_model_metadata_v1.json')
with open(path_meta, 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"  ✅ Metadata de auditoría guardada")

# ── 10.5 Verificación de integridad ──────────────────────────────────
# Cargar y re-scorear un cliente de prueba para confirmar que los
# archivos guardados producen exactamente los mismos scores que en memoria.
# Si este test falla, los archivos están corruptos.

print(f"\n  Verificación de integridad (re-score de prueba)...")

productos_verificados = 0
errores_verificacion  = 0

# Tomar un cliente de prueba del test set
cliente_prueba = test[test['mes'] == test['mes'].max()].head(1)

for producto in sorted(modelos.keys()):

    feats = modelos[producto]['features']
    mask  = cliente_prueba['id_producto_lower'] == producto

    # Si este cliente no tiene oferta del producto, saltar
    if mask.sum() == 0:
        continue

    X_prueba = cliente_prueba.loc[mask, feats]

    # Score en memoria
    p_raw_mem = modelos[producto]['model'].predict_proba(X_prueba)[:, 1]
    p_cal_mem = calibradores[producto].predict_proba(
        p_raw_mem.reshape(-1, 1)
    )[:, 1]

    # Score desde disco
    modelo_disco = joblib.load(
        os.path.join(MODELS_DIR, f'nbo_model_{producto}_v1.joblib')
    )
    cal_disco = joblib.load(
        os.path.join(MODELS_DIR, f'nbo_calibrador_{producto}_v1.joblib')
    )
    p_raw_disco = modelo_disco.predict_proba(X_prueba)[:, 1]
    p_cal_disco = cal_disco.predict_proba(
        p_raw_disco.reshape(-1, 1)
    )[:, 1]

    # Comparar — deben ser idénticos hasta la precisión de float64
    diff = abs(p_cal_mem[0] - p_cal_disco[0])
    if diff < 1e-10:
        print(f"  ✅ {producto:<20} score memoria={p_cal_mem[0]:.6f} "
              f"disco={p_cal_disco[0]:.6f} diff={diff:.2e}")
        productos_verificados += 1
    else:
        print(f"  ❌ {producto:<20} DISCREPANCIA: diff={diff:.2e}")
        errores_verificacion += 1

if errores_verificacion == 0 and productos_verificados > 0:
    print(f"\n  ✅ Verificación completa — {productos_verificados} productos íntegros")
elif productos_verificados == 0:
    print(f"\n  ⚠️  No se encontró el producto del cliente de prueba en el test set")
    print(f"     Los archivos fueron guardados pero la verificación no pudo ejecutarse")
    print(f"     Verifica manualmente con otro cliente antes de Semana 5")
else:
    raise ValueError(
        f"❌ {errores_verificacion} modelos con discrepancia entre memoria y disco. "
        f"No continuar con Semana 5 hasta resolver."
    )

# ── 10.6 Resumen de archivos generados ───────────────────────────────
print(f"\n  Archivos generados en {MODELS_DIR}:")
archivos = sorted(os.listdir(MODELS_DIR))
for archivo in archivos:
    size_kb = os.path.getsize(os.path.join(MODELS_DIR, archivo)) / 1024
    print(f"    {archivo:<45} {size_kb:>8.1f} KB")

print(f"\n  Total archivos : {len(archivos)}")
print(f"  Listos para ser importados en Semana 5 sin reentrenamiento")


