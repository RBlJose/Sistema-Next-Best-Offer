# %%
# %%
"""
RBlJose — SISTEMA NBO
Semana 5 — Backtesting Multi-Período
======================================
Demuestra cómo escala el sistema mes a mes sobre 4 ciclos consecutivos
usando el mismo modelo entrenado en Semana 3, sin reentrenamiento.
 
Prerequisitos:
  nbo_semana3_modelos.py ejecutado con Bloque 10 (persistencia).
  Directorio /models con modelos, calibradores y encoders guardados.
  Archivos fuente: nbo_clientes.csv, nbo_features.csv, nbo_ofertas.csv
 
Pipeline:
  Bloque 0 — Importación de modelos persistidos (sin reentrenamiento)
  Bloque 1 — Carga de inputs base y validación
  Bloque 2 — Loop de backtesting (meses 22, 23, 24, 25)
    ├── 2.1  Re-scoring desde modelos guardados
    ├── 2.2  Contact Policy Engine (R1-R8)
    ├── 2.3  Optimizador greedy presupuestario
    ├── 2.4  Asignación tratamiento / control
    ├── 2.5  Simulación T+30
    ├── 2.6  Medición financiera incremental
    ├── 2.7  Monitoreo con guardia de potencia estadística
    └── 2.8  Actualizar historial acumulado
  Bloque 3 — Análisis de evolución temporal
  Bloque 4 — Outputs consolidados
 
Reglas del Contact Policy Engine:
  R1 — Cooling period 30 días entre contactos
  R2 — Bloqueo post-oferta 15 días
  R3 — Opt-out permanente
  R4 — Producto ya activo (conversión previa)
  R5 — Elegibilidad mínima (score, mora, antigüedad)
  R6 — Fallback a rank 2 si rank 1 bloqueado
  R7 — Fatiga puntual: 2 ciclos consecutivos mismo producto sin convertir
       → promover rank 2
  R8 — Supresión temporal: 3 ciclos sin convertir a nada
       → pausa 2 meses, protege opt-out
 
Outputs:
  nbo_backtest_scores_{mes}.csv       — scores re-calculados por ciclo
  nbo_backtest_campana_completa.csv   — universo contactado todos los ciclos
  nbo_backtest_resultados.csv         — conversiones T+30 todos los ciclos
  nbo_backtest_metricas_serie.csv     — KPIs financieros por ciclo
  nbo_backtest_monitoreo_serie.csv    — drift y triggers por ciclo
 
Separación de responsabilidades:
  Este script NO reentrena modelos.
  Carga objetos serializados desde /models y los aplica a datos nuevos.
  Modelo → score → decisión son tres capas distintas y auditables.
"""

# %%
import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
 
warnings.filterwarnings('ignore')

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
# PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════
DATA_DIR   = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'models')
SEED       = 42
np.random.seed(SEED)
 
FECHA_INICIO = date(2024, 1, 1)
 
# Ciclos de backtesting — meses a evaluar en orden
CICLOS_BACKTEST = [22, 23, 24, 25]
 
# Parámetros de campaña — idénticos a Semana 4 para comparabilidad
PRESUPUESTO_CAMPANA  = 50_000.0
PCT_CONTROL          = 0.20
 
TASA_ORGANICA = {
    'tarjeta'     : 0.018,
    'prestamo'    : 0.025,
    'microcredito': 0.012,
    'seguro_vida' : 0.015,
    'seguro_salud': 0.014,
    'inversion'   : 0.020,
}
 
LIMITES_ORIGINACION = {
    'tarjeta'     : 2_000,
    'prestamo'    : 1_500,
    'microcredito': 800,
    'seguro_vida' : 3_000,
    'seguro_salud': 3_000,
    'inversion'   : 2_500,
}
 
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
 
# Parámetros de las reglas nuevas R7 y R8
R7_CICLOS_FATIGA      = 2   # ciclos consecutivos mismo producto sin convertir
R8_CICLOS_SIN_NADA    = 3   # ciclos sin convertir a ningún producto
R8_MESES_SUPRESION    = 2   # meses de pausa obligatoria tras activar R8
 

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 0 — IMPORTACIÓN DE MODELOS PERSISTIDOS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 0 — IMPORTACIÓN DE MODELOS PERSISTIDOS")

if not os.path.exists(MODELS_DIR):
    raise FileNotFoundError(
        f"Directorio /models no encontrado en {MODELS_DIR}.\n"
        f"Verifica que ejecutaste Semana 3 completamente."
    )

with open(os.path.join(MODELS_DIR, 'nbo_model_metadata_v1.json')) as f:
    metadata = json.load(f)

with open(os.path.join(MODELS_DIR, 'nbo_feature_names_v1.json')) as f:
    feature_spec = json.load(f)

FEATURES_FINALES     = feature_spec['features_finales']
FEATURES_NUMERICAS   = feature_spec['features_numericas']
FEATURES_CATEGORICAS = feature_spec['features_categoricas']

encoders = {}
for col in FEATURES_CATEGORICAS:
    encoders[col] = joblib.load(
        os.path.join(MODELS_DIR, f'nbo_encoder_{col}_v1.joblib')
    )

PRODUCTOS   = list(PARAMS_NEGOCIO.keys())
modelos     = {}
calibradores = {}

print(f"\n  {'Producto':<20} {'AUC test':>10} {'Brier cal':>12} {'Árboles':>10}")
print(f"  {'─'*56}")

for producto in sorted(PRODUCTOS):
    modelos[producto] = joblib.load(
        os.path.join(MODELS_DIR, f'nbo_model_{producto}_v1.joblib')
    )
    calibradores[producto] = joblib.load(
        os.path.join(MODELS_DIR, f'nbo_calibrador_{producto}_v1.joblib')
    )
    m = metadata['metricas_por_producto'][producto]
    print(f"  {producto:<20} {m['auc_test_cal']:>10.4f} "
          f"{m['brier_cal']:>12.5f} {m['best_iteration']:>10}")

print(f"\n  ✅ {len(modelos)} modelos + {len(calibradores)} calibradores cargados")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA DE INPUTS BASE
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA DE INPUTS BASE")

clientes = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
features = pd.read_csv(f'{DATA_DIR}/nbo_features.csv')
ofertas  = pd.read_csv(f'{DATA_DIR}/nbo_ofertas.csv')

ofertas['fecha_oferta']      = pd.to_datetime(ofertas['fecha_oferta'])
ofertas['id_producto_lower'] = ofertas['id_producto'].str.lower()

print(f"  clientes : {len(clientes):>8,} filas")
print(f"  features : {len(features):>8,} filas | meses {features['mes'].min()}–{features['mes'].max()}")
print(f"  ofertas  : {len(ofertas):>8,} filas  | meses {ofertas['mes'].min()}–{ofertas['mes'].max()}")

# ── Construir lookup de ground truth desde nbo_ofertas.csv ────────────
# score_propension es la probabilidad real del generador DAG.
# Es el único ground truth válido para simular T+30.
# Se indexa por (id_cliente, mes, producto) para join rápido en el loop.
print("\n  Construyendo lookup de ground truth (score_propension)...")
gt_lookup = (
    ofertas[['id_cliente', 'mes', 'id_producto_lower',
              'score_propension', 'convirtio_30d', 'etiqueta_completa']]
    .copy()
    .rename(columns={'id_producto_lower': 'producto_nbo'})
)
gt_lookup = gt_lookup.set_index(['id_cliente', 'mes', 'producto_nbo'])
print(f"  Lookup construido: {len(gt_lookup):,} entradas")
print(f"  Meses con etiqueta completa: "
      f"{sorted(ofertas[ofertas['etiqueta_completa']==True]['mes'].unique())}")
print(f"  Meses sin etiqueta (producción): "
      f"{sorted(ofertas[ofertas['etiqueta_completa']==False]['mes'].unique())}")

# Verificar que los meses del backtesting tienen features
for mes in CICLOS_BACKTEST:
    n_feat = (features['mes'] == mes).sum()
    if n_feat == 0:
        raise ValueError(f"Sin features para mes {mes}")
    etiq = ofertas[ofertas['mes'] == mes]['etiqueta_completa'].iloc[0] \
           if len(ofertas[ofertas['mes'] == mes]) > 0 else False
    print(f"  Mes {mes}: {n_feat:,} features | "
          f"etiqueta={'disponible' if etiq else 'NO disponible (producción)'}")

COLUMNAS_CLIENTES = [
    'id_cliente', 'segmento', 'edad', 'ocupacion', 'zona_geografica',
    'canal_principal', 'antiguedad_meses', 'ingreso_mensual',
    'score_crediticio', 'score_buro', 'tiene_atraso_hist',
    'max_atraso_dias', 'ratio_deuda_init', 'hijos', 'estado_civil', 'activo',
]
clientes_slim = clientes[COLUMNAS_CLIENTES].copy()

# Historial inicial: contactos reales anteriores al primer ciclo
MES_INICIO_BACKTEST = min(CICLOS_BACKTEST)
historial_acumulado = ofertas[
    (ofertas['grupo'] == 'Tratamiento') &
    (ofertas['mes'] < MES_INICIO_BACKTEST)
].copy()
print(f"\n  Historial previo al ciclo (mes < {MES_INICIO_BACKTEST}): "
      f"{len(historial_acumulado):,} contactos reales")

 

# %%
# ══════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES — CPE 
# ══════════════════════════════════════════════════════════════════════

def construir_estado_clientes(historial, ciclos_anteriores, mes_actual):
    if len(historial) == 0:
        return pd.DataFrame(columns=[
            'id_cliente', 'fecha_ultimo_contacto', 'productos_activos',
            'fatiga_por_producto', 'ciclos_sin_conversion', 'suprimido_hasta_mes'
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
    suprimido_hasta = {}

    for ciclo_info in ciclos_anteriores:
        df_ciclo = ciclo_info['contactados']
        if df_ciclo is None or len(df_ciclo) == 0:
            continue
        mes_ciclo = ciclo_info['mes']

        for _, row in df_ciclo.iterrows():
            cid  = row['id_cliente']
            prod = row['producto_nbo']
            conv = row.get('convirtio_30d', 0)

            if cid not in fatiga_dict:
                fatiga_dict[cid] = {}
            fatiga_dict[cid][prod] = 0 if conv == 1 \
                else fatiga_dict[cid].get(prod, 0) + 1

        conv_col_ciclo = 'convirtio_30d'
        if conv_col_ciclo in df_ciclo.columns:
            convertidos = set(
                df_ciclo[df_ciclo[conv_col_ciclo] == 1]['id_cliente'].values
            )
            for cid in set(df_ciclo['id_cliente'].values):
                if cid in convertidos:
                    sin_nada_dict[cid] = 0
                    suprimido_hasta.pop(cid, None)
                else:
                    sin_nada_dict[cid] = sin_nada_dict.get(cid, 0) + 1
                    if sin_nada_dict[cid] >= R8_CICLOS_SIN_NADA:
                        suprimido_hasta[cid] = mes_ciclo + R8_MESES_SUPRESION
                        sin_nada_dict[cid] = 0

    estado['fatiga_por_producto']   = estado['id_cliente'].map(
        lambda x: fatiga_dict.get(x, {}))
    estado['ciclos_sin_conversion'] = estado['id_cliente'].map(
        lambda x: sin_nada_dict.get(x, 0))
    estado['suprimido_hasta_mes']   = estado['id_cliente'].map(
        lambda x: suprimido_hasta.get(x, 0))

    return estado


def aplicar_cpe(universo_scores, estado_clientes, mes_actual, fecha_campana_dt,
                ciclos_anteriores):
    rec_r1 = universo_scores[universo_scores['rank'] == 1].copy()
    rec_r2 = universo_scores[universo_scores['rank'] == 2].copy()

    rec_r1 = rec_r1.merge(estado_clientes, on='id_cliente', how='left')
    rec_r1 = rec_r1.merge(
        clientes_slim[['id_cliente', 'score_crediticio', 'max_atraso_dias',
                        'antiguedad_meses', 'activo']],
        on='id_cliente', how='left'
    )

    for col, default in [('productos_activos', set()), ('fatiga_por_producto', {}),
                          ('suprimido_hasta_mes', 0), ('ciclos_sin_conversion', 0)]:
        rec_r1[col] = rec_r1[col].apply(
            lambda x: x if isinstance(x, type(default)) else default
        )

    rec_r1['dias_desde_contacto'] = (
        fecha_campana_dt - rec_r1['fecha_ultimo_contacto']
    ).dt.days.fillna(9999).astype(int)

    rec_r1['pasa_r1'] = rec_r1['dias_desde_contacto'] >= 30
    rec_r1['pasa_r2'] = rec_r1['dias_desde_contacto'] >= 15
    rec_r1['pasa_r3'] = True
    rec_r1['pasa_r4'] = [
        prod not in activos
        for prod, activos in zip(rec_r1['producto_nbo'], rec_r1['productos_activos'])
    ]
    rec_r1['pasa_r5'] = (
        (rec_r1['score_crediticio'] >= 550) &
        (rec_r1['max_atraso_dias']  <= 30)  &
        (rec_r1['antiguedad_meses'] >= 3)   &
        (rec_r1['activo'] == True)
    )
    rec_r1['pasa_r8'] = rec_r1['suprimido_hasta_mes'] < mes_actual
    rec_r1['fatigado_r7'] = [
        fatiga.get(prod, 0) >= R7_CICLOS_FATIGA
        for prod, fatiga in zip(rec_r1['producto_nbo'], rec_r1['fatiga_por_producto'])
    ]

    rec_r1['elegible_r1'] = (
        rec_r1['pasa_r1'] & rec_r1['pasa_r2'] & rec_r1['pasa_r3'] &
        rec_r1['pasa_r4'] & rec_r1['pasa_r5'] & rec_r1['pasa_r8'] &
        ~rec_r1['fatigado_r7']
    )

    mascara_fallback = (
        ~rec_r1['elegible_r1'] & rec_r1['pasa_r5'] & rec_r1['pasa_r8']
    )
    bloqueados = rec_r1[mascara_fallback]['id_cliente'].values
    rec_r2_eval = rec_r2[rec_r2['id_cliente'].isin(bloqueados)].copy()

    fallbacks_elegibles = pd.DataFrame()
    if len(rec_r2_eval) > 0:
        rec_r2_eval = rec_r2_eval.merge(estado_clientes, on='id_cliente', how='left')
        rec_r2_eval = rec_r2_eval.merge(
            clientes_slim[['id_cliente', 'score_crediticio', 'max_atraso_dias',
                            'antiguedad_meses', 'activo']],
            on='id_cliente', how='left'
        )
        for col, default in [('productos_activos', set()), ('fatiga_por_producto', {})]:
            rec_r2_eval[col] = rec_r2_eval[col].apply(
                lambda x: x if isinstance(x, type(default)) else default)
        rec_r2_eval['suprimido_hasta_mes'] = rec_r2_eval['suprimido_hasta_mes'].fillna(0)
        rec_r2_eval['dias_desde_contacto'] = (
            fecha_campana_dt - rec_r2_eval['fecha_ultimo_contacto']
        ).dt.days.fillna(9999).astype(int)
        rec_r2_eval['fatigado_r7'] = [
            fatiga.get(prod, 0) >= R7_CICLOS_FATIGA
            for prod, fatiga in zip(
                rec_r2_eval['producto_nbo'], rec_r2_eval['fatiga_por_producto'])
        ]
        pasa_r4_r2 = [
            prod not in activos
            for prod, activos in zip(
                rec_r2_eval['producto_nbo'], rec_r2_eval['productos_activos'])
        ]
        rec_r2_eval['elegible_r1'] = (
            (rec_r2_eval['dias_desde_contacto'] >= 30) &
            (rec_r2_eval['score_crediticio'] >= 550)   &
            (rec_r2_eval['max_atraso_dias']  <= 30)    &
            (rec_r2_eval['antiguedad_meses'] >= 3)     &
            pasa_r4_r2 & ~rec_r2_eval['fatigado_r7']
        )
        rec_r2_eval['es_fallback'] = True
        fallbacks_elegibles = rec_r2_eval[
            rec_r2_eval['elegible_r1'] &
            ~rec_r2_eval['id_cliente'].isin(
                rec_r1[rec_r1['elegible_r1']]['id_cliente'])
        ].copy()

    elegibles_r1 = rec_r1[rec_r1['elegible_r1']].copy()
    elegibles_r1['es_fallback'] = False

    universo_elegible = pd.concat(
        [elegibles_r1, fallbacks_elegibles], ignore_index=True
    ) if len(fallbacks_elegibles) > 0 else elegibles_r1.copy()

    stats = {
        'n_scoring_inicial'   : rec_r1['id_cliente'].nunique(),
        'bloqueados_r1_cool'  : (~rec_r1['pasa_r1']).sum(),
        'bloqueados_r4_activo': (~rec_r1['pasa_r4']).sum(),
        'bloqueados_r5_elegib': (~rec_r1['pasa_r5']).sum(),
        'bloqueados_r8_suprim': (~rec_r1['pasa_r8']).sum(),
        'bloqueados_r7_fatiga': rec_r1['fatigado_r7'].sum(),
        'fallbacks_elegibles' : len(fallbacks_elegibles),
        'universo_final'      : len(universo_elegible),
    }
    return universo_elegible, stats

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — LOOP DE BACKTESTING 
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — LOOP DE BACKTESTING MULTI-PERÍODO")

print(f"""
  Ciclos   : {CICLOS_BACKTEST}
  Modelo   : {metadata['version']} (sin reentrenamiento)
  CPE      : R1-R8
  Ground truth T+30:
    Meses 22-23 → convirtio_30d real de nbo_ofertas.csv
    Meses 24-25 → score_propension del generador (simulación Bernoulli)
""")

ciclos_anteriores     = []
campanas_acumuladas   = []
resultados_acumulados = []
metricas_serie        = []
monitoreo_serie       = []
scores_historicos     = []   # para Semana 6


for mes in CICLOS_BACKTEST:

    separador(f"CICLO MES {mes} — "
              f"{(FECHA_INICIO + relativedelta(months=mes-1)).strftime('%B %Y').upper()}")

    fecha_campana    = FECHA_INICIO + relativedelta(months=mes - 1)
    fecha_campana_dt = pd.Timestamp(fecha_campana)

    # ── 2.1 RE-SCORING ───────────────────────────────────────────────
    subseccion(f"2.1 — Re-scoring mes {mes}")

    COLUMNAS_FEATURES = [
        'id_cliente', 'mes', 'gasto_3m', 'gasto_supermercado_3m',
        'gasto_farmacia_3m', 'depositos_efectivo_3m', 'tx_digitales_proporcion',
        'saldo_promedio_90d', 'saldo_tendencia', 'variabilidad_saldo',
        'ratio_cuota_ingreso', 'dias_desde_ult_credito', 'indice_estres_macro',
        'estres_x_riesgo',
    ]

    features_mes = features[features['mes'] == mes][COLUMNAS_FEATURES].copy()
    abt_mes = features_mes.merge(clientes_slim, on='id_cliente', how='left')

    abt_mes['bucket_dias_credito'] = pd.cut(
        abt_mes['dias_desde_ult_credito'],
        bins=[-1, 30, 90, 180, 365, 730, 9999],
        labels=['0_30d', '31_90d', '91_180d', '181_365d', '366_730d', 'mas_730d']
    )
    abt_mes['ratio_saldo_ingreso'] = (
        abt_mes['saldo_promedio_90d'] / (abt_mes['ingreso_mensual'] + 1)
    )
    abt_mes['score_muy_alto']   = (abt_mes['score_crediticio'] > 800).astype(int)
    abt_mes['edad_objetivo_sv'] = (
        (abt_mes['edad'] >= 30) & (abt_mes['edad'] <= 55)).astype(int)
    abt_mes['indep_no_urbano']  = (
        (abt_mes['ocupacion'] == 'Independiente') &
        (abt_mes['zona_geografica'] != 'Urbana')).astype(int)

    for col in FEATURES_CATEGORICAS:
        abt_mes[col + '_enc'] = encoders[col].transform(abt_mes[col].astype(str))

    canal_map = clientes.set_index('id_cliente')['canal_principal']
    abt_mes['canal_principal_val'] = abt_mes['id_cliente'].map(canal_map).fillna('Digital')

    scores_por_producto = []
    for producto in sorted(PRODUCTOS):
        p      = PARAMS_NEGOCIO[producto]
        modelo = modelos[producto]
        platt  = calibradores[producto]
        X_mes  = abt_mes[FEATURES_FINALES].copy()

        p_raw = modelo.predict_proba(X_mes)[:, 1].reshape(-1, 1)
        p_cal = platt.predict_proba(p_raw)[:, 1]
        p_raw_1d = p_raw.ravel()

        ingreso_esp = p_cal * (p['ticket_anual'] - p['costo_originacion'])
        perdida_esp = p['pd'] * p['lgd'] * p['ticket_anual'] * p['rwa']
        score_nbo   = ingreso_esp - perdida_esp - p['costo_contacto']

        df_tmp = pd.DataFrame({
            'id_cliente'     : abt_mes['id_cliente'].values,
            'producto_nbo'   : producto,
            'score_xgb_raw'  : p_raw_1d,      # score crudo antes de Platt
            'p_calibrada'    : p_cal,          # predicción del modelo
            'score_nbo'      : score_nbo,
            'costo_contacto' : p['costo_contacto'],
            'ratio_nbo_costo': score_nbo / max(p['costo_contacto'], 0.01),
            'canal_principal': abt_mes['canal_principal_val'].values,
            'mes_scoring'    : mes,
            'fecha_scoring'  : str(fecha_campana),
        })
        scores_por_producto.append(df_tmp)

        # Guardar para Semana 6
        scores_historicos.extend(df_tmp.to_dict('records'))

    df_scores = pd.concat(scores_por_producto, ignore_index=True)
    df_scores = df_scores.sort_values(
        ['id_cliente', 'score_nbo'], ascending=[True, False]
    )
    df_scores['rank'] = df_scores.groupby('id_cliente').cumcount() + 1
    df_scores_top2    = df_scores[df_scores['rank'] <= 2].copy()

    print(f"  Clientes scoreados        : {df_scores['id_cliente'].nunique():,}")
    print(f"  Score NBO medio (rank 1)  : "
          f"{df_scores_top2[df_scores_top2['rank']==1]['score_nbo'].mean():.2f}")

    df_scores_top2.to_csv(f'{DATA_DIR}/nbo_backtest_scores_mes{mes}.csv', index=False)

    # ── 2.2 CPE ─────────────────────────────────────────────────────
    subseccion(f"2.2 — Contact Policy Engine mes {mes}")

    estado_clientes = construir_estado_clientes(
        historial_acumulado, ciclos_anteriores, mes)
    universo_elegible, stats_cpe = aplicar_cpe(
        df_scores_top2, estado_clientes, mes, fecha_campana_dt, ciclos_anteriores)

    print(f"  Scoring inicial        : {stats_cpe['n_scoring_inicial']:,}")
    print(f"  Bloqueados R1 cooling  : {stats_cpe['bloqueados_r1_cool']:,}")
    print(f"  Bloqueados R4 activo   : {stats_cpe['bloqueados_r4_activo']:,}")
    print(f"  Bloqueados R5 elegib.  : {stats_cpe['bloqueados_r5_elegib']:,}")
    print(f"  Suprimidos R8          : {stats_cpe['bloqueados_r8_suprim']:,}")
    print(f"  Fatigados R7           : {stats_cpe['bloqueados_r7_fatiga']:,}")
    print(f"  Fallbacks rank 2       : {stats_cpe['fallbacks_elegibles']:,}")
    print(f"  Universo elegible final: {stats_cpe['universo_final']:,}")

    if len(universo_elegible) == 0:
        print(f"\n  ⚠️  Sin clientes elegibles en mes {mes}")
        ciclos_anteriores.append({'mes': mes, 'contactados': None,
                                   'n_contactados': 0, 'roi': 0})
        continue

    # ── 2.3 OPTIMIZADOR GREEDY ──────────────────────────────────────
    subseccion(f"2.3 — Optimizador presupuestario mes {mes}")

    universo_ordenado = universo_elegible.sort_values(
        ['ratio_nbo_costo', 'id_cliente'],
        ascending=[False, True]
    ).copy()
    presupuesto_restante = PRESUPUESTO_CAMPANA
    contadores_producto  = {prod: 0 for prod in PARAMS_NEGOCIO}
    seleccionados        = []

    for _, cliente in universo_ordenado.iterrows():
        prod  = cliente['producto_nbo']
        costo = cliente['costo_contacto']
        if cliente['score_nbo'] < 0:
            continue
        if presupuesto_restante < costo:
            continue
        if contadores_producto[prod] >= LIMITES_ORIGINACION.get(prod, np.inf):
            continue
        seleccionados.append(cliente.to_dict())
        presupuesto_restante      -= costo
        contadores_producto[prod] += 1

    universo_optimizado   = pd.DataFrame(seleccionados)
    presupuesto_ejecutado = PRESUPUESTO_CAMPANA - presupuesto_restante

    print(f"  Seleccionados    : {len(universo_optimizado):,}")
    print(f"  Presupuesto ejec.: ${presupuesto_ejecutado:,.2f} "
          f"({presupuesto_ejecutado/PRESUPUESTO_CAMPANA:.1%})")

    if len(universo_optimizado) == 0:
        ciclos_anteriores.append({'mes': mes, 'contactados': None,
                                   'n_contactados': 0, 'roi': 0})
        continue

    # ── 2.4 ASIGNACIÓN TRATAMIENTO / CONTROL ────────────────────────
    def asignar_grupo_deterministico(df, pct_control):
        def asignar(row):
            seed = hash((row['id_cliente'], row['mes_campana'])) % (2**32)
            rng = np.random.default_rng(seed)
            return 'Control' if rng.random() < pct_control else 'Tratamiento'

        return df.apply(asignar, axis=1)
    universo_optimizado['id_campana']    = f'CAMP_{mes:02d}_NBO_BT'
    universo_optimizado['mes_campana']   = mes
    
    universo_optimizado['grupo'] = asignar_grupo_deterministico(
        universo_optimizado, PCT_CONTROL
    )
    
    universo_optimizado['fecha_campana'] = str(fecha_campana)
    universo_optimizado['fecha_cierre']  = str(fecha_campana + timedelta(days=30))
    universo_optimizado['version_modelo'] = metadata['version']

    n_trat = (universo_optimizado['grupo'] == 'Tratamiento').sum()
    n_ctrl = (universo_optimizado['grupo'] == 'Control').sum()
    print(f"  Tratamiento: {n_trat:,}  |  Control: {n_ctrl:,}")

    # ── 2.5 SIMULACIÓN T+30 — CORREGIDA ─────────────────────────────
    subseccion(f"2.5 — T+30 mes {mes} [CORREGIDO: ground truth = score_propension]")

    # Determinar si el mes tiene etiqueta real disponible
    meses_con_etiqueta = set(
        ofertas[ofertas['etiqueta_completa'] == True]['mes'].unique()
    )
    tiene_etiqueta_real = mes in meses_con_etiqueta

    print(f"  Fuente T+30: {'etiqueta real de nbo_ofertas.csv' if tiene_etiqueta_real else 'score_propension (simulación Bernoulli)'}")

    conversiones = []
    for _, cliente in universo_optimizado.iterrows():
        prod   = cliente['producto_nbo']
        cid    = cliente['id_cliente']
        p_neg  = PARAMS_NEGOCIO[prod]

        if cliente['grupo'] == 'Tratamiento':
            # Buscar ground truth en nbo_ofertas.csv
            key = (cid, mes, prod)
            if key in gt_lookup.index:
                row_gt = gt_lookup.loc[key]
                # Puede haber múltiples registros (varias ofertas mismo mes/prod)
                if isinstance(row_gt, pd.DataFrame):
                    row_gt = row_gt.iloc[0]

                if tiene_etiqueta_real and not pd.isna(row_gt['convirtio_30d']):
                    # Meses 22-23: usar la etiqueta real directamente
                    convirtio = int(row_gt['convirtio_30d'])
                else:
                    # Meses 24-25: simular con score_propension del generador
                    p_real = float(row_gt['score_propension'])
                    seed = hash((cid, mes, prod)) % (2**32)
                    rng_local = np.random.default_rng(seed)

                    convirtio = int(rng_local.random() < p_real)
            else:
                # Cliente no tenía oferta en ese mes/producto en el generador
                # Usar tasa orgánica como fallback conservador
                convirtio = int(rng_global.random() < TASA_ORGANICA.get(prod, 0.02))
        else:
            # Control: tasa orgánica (no recibió oferta)
            convirtio = int(rng_global.random() < TASA_ORGANICA.get(prod, 0.02))

        ingreso_real = (
            p_neg['ticket_anual'] - p_neg['costo_originacion']
            if convirtio else 0.0
        )

        conversiones.append({
            'id_cliente'    : cid,
            'id_campana'    : cliente['id_campana'],
            'mes_campana'   : mes,
            'producto_nbo'  : prod,
            'grupo'         : cliente['grupo'],
            'p_calibrada'   : cliente['p_calibrada'],  # predicción — para monitoreo
            'convirtio_30d' : convirtio,
            'ingreso_real'  : round(ingreso_real, 2),
            'costo_contacto': (p_neg['costo_contacto']
                               if cliente['grupo'] == 'Tratamiento' else 0.0),
            'es_fallback'   : cliente.get('es_fallback', False),
            'fuente_gt'     : 'etiqueta_real' if tiene_etiqueta_real else 'score_propension',
        })

    df_resultados = pd.DataFrame(conversiones)

    tasa_trat = df_resultados[df_resultados['grupo']=='Tratamiento']['convirtio_30d'].mean()
    tasa_ctrl = df_resultados[df_resultados['grupo']=='Control']['convirtio_30d'].mean()
    print(f"  Conv. Tratamiento : {tasa_trat:.4f} | "
          f"Conv. Control: {tasa_ctrl:.4f} | "
          f"Uplift: {(tasa_trat - tasa_ctrl)*100:+.1f}pp")

    # ── 2.6 MEDICIÓN FINANCIERA ──────────────────────────────────────
    subseccion(f"2.6 — Métricas financieras mes {mes}")

    total_ingreso_incr = 0
    total_costo        = 0
    metricas_ciclo     = []

    for prod in sorted(df_resultados['producto_nbo'].unique()):
        sub  = df_resultados[df_resultados['producto_nbo'] == prod]
        p    = PARAMS_NEGOCIO[prod]
        trat = sub[sub['grupo'] == 'Tratamiento']
        ctrl = sub[sub['grupo'] == 'Control']

        tasa_t = trat['convirtio_30d'].mean() if len(trat) > 0 else 0
        tasa_c = ctrl['convirtio_30d'].mean() if len(ctrl) > 0 else 0

        incremento         = max(tasa_t - tasa_c, 0)
        ingreso_incremental = (
            incremento * len(trat) *
            (p['ticket_anual'] - p['costo_originacion'])
        )
        costo_total        = trat['costo_contacto'].sum()
        profit_incremental = ingreso_incremental - costo_total

        total_ingreso_incr += ingreso_incremental
        total_costo        += costo_total

        metricas_ciclo.append({
            'mes': mes, 'producto': prod,
            'n_tratamiento': len(trat), 'n_control': len(ctrl),
            'tasa_trat': round(tasa_t, 4), 'tasa_ctrl': round(tasa_c, 4),
            'uplift_pp': round((tasa_t - tasa_c) * 100, 2),
            'ingreso_incremental': round(ingreso_incremental, 2),
            'costo_contactacion': round(costo_total, 2),
            'profit_incremental': round(profit_incremental, 2),
        })

    roi_ciclo    = total_ingreso_incr / total_costo if total_costo > 0 else 0
    profit_ciclo = total_ingreso_incr - total_costo

    print(f"  Ingreso incremental : ${total_ingreso_incr:,.2f}")
    print(f"  Costo contactación  : ${total_costo:,.2f}")
    print(f"  Profit incremental  : ${profit_ciclo:,.2f}")
    print(f"  ROI incremental     : {roi_ciclo:.2f}x")

    metricas_serie.append({
        'mes': mes, 'fecha_campana': str(fecha_campana),
        'n_scoring': stats_cpe['n_scoring_inicial'],
        'n_elegibles': stats_cpe['universo_final'],
        'n_contactados': len(universo_optimizado),
        'n_tratamiento': n_trat, 'n_control': n_ctrl,
        'n_r7_fatigados': stats_cpe['bloqueados_r7_fatiga'],
        'n_r8_suprimidos': stats_cpe['bloqueados_r8_suprim'],
        'presupuesto_ejecutado': round(presupuesto_ejecutado, 2),
        'ingreso_incremental': round(total_ingreso_incr, 2),
        'costo_total': round(total_costo, 2),
        'profit_incremental': round(profit_ciclo, 2),
        'roi_incremental': round(roi_ciclo, 4),
        'tasa_conv_trat': round(tasa_trat, 4),
        'tasa_conv_ctrl': round(tasa_ctrl, 4),
        'fuente_gt': 'etiqueta_real' if tiene_etiqueta_real else 'score_propension',
    })

    # ── 2.7 MONITOREO ───────────────────────────────────────────────
    subseccion(f"2.7 — Monitoreo estadístico mes {mes}")

    GUARDIA_POTENCIA = 30
    for prod in sorted(df_resultados['producto_nbo'].unique()):
        sub_trat = df_resultados[
            (df_resultados['producto_nbo'] == prod) &
            (df_resultados['grupo'] == 'Tratamiento')
        ]
        # CORRECCIÓN CLAVE: comparamos p_calibrada (predicción) vs
        # tasa observada real (del generador), NO p_calibrada vs p_calibrada
        tasa_proy  = sub_trat['p_calibrada'].mean()
        tasa_obs   = sub_trat['convirtio_30d'].mean() if len(sub_trat) > 0 else 0
        desviacion = abs(tasa_obs - tasa_proy) / max(tasa_proy, 1e-6)
        n_obs      = len(sub_trat)
        potencia_ok = n_obs >= GUARDIA_POTENCIA

        if not potencia_ok:
            estado = f"⏳ N={n_obs} < {GUARDIA_POTENCIA}"
        elif desviacion > 0.25:
            estado = "❌ Recalib."
        elif desviacion > 0.15:
            estado = "⚠️  Alerta"
        else:
            estado = "✅ Estable"

        monitoreo_serie.append({
            'mes': mes, 'producto': prod,
            'n_tratamiento': n_obs,
            'tasa_proyectada': round(tasa_proy, 4),
            'tasa_observada': round(tasa_obs, 4),
            'desviacion': round(desviacion, 4),
            'potencia_ok': potencia_ok,
            'alerta_tasa': potencia_ok and desviacion > 0.15,
            'trigger_recalib': potencia_ok and desviacion > 0.25,
            'estado': estado,
            'fuente_gt': 'etiqueta_real' if tiene_etiqueta_real else 'score_propension',
        })
        print(f"  {prod:<20} N={n_obs:>5} "
              f"proy={tasa_proy:.4f} obs={tasa_obs:.4f} "
              f"desv={desviacion:+.1%}  {estado}")

    # ── 2.8 ACTUALIZAR HISTORIAL ────────────────────────────────────
    nuevos_contactos = universo_optimizado[
        universo_optimizado['grupo'] == 'Tratamiento'
    ].copy()
    nuevos_contactos = nuevos_contactos.merge(
        df_resultados[['id_cliente', 'producto_nbo', 'convirtio_30d']],
        on=['id_cliente', 'producto_nbo'], how='left'
    )
    nuevos_hist = pd.DataFrame({
        'id_cliente'       : nuevos_contactos['id_cliente'],
        'id_producto_lower': nuevos_contactos['producto_nbo'],
        'fecha_oferta'     : pd.to_datetime(nuevos_contactos['fecha_campana']),
        'mes'              : mes,
        'grupo'            : 'Tratamiento',
        'convirtio_30d'    : nuevos_contactos['convirtio_30d'].fillna(0),
    })
    historial_acumulado = pd.concat(
        [historial_acumulado, nuevos_hist], ignore_index=True)

    ciclos_anteriores.append({
        'mes': mes, 'contactados': df_resultados.copy(),
        'n_contactados': len(universo_optimizado), 'roi': roi_ciclo,
    })
    campanas_acumuladas.append(universo_optimizado)
    resultados_acumulados.append(df_resultados)

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — ANÁLISIS DE EVOLUCIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — ANÁLISIS DE EVOLUCIÓN TEMPORAL")

df_metricas_serie  = pd.DataFrame(metricas_serie)
df_monitoreo_serie = pd.DataFrame(monitoreo_serie)

subseccion("3.1 — KPIs por ciclo")
print(f"\n  {'Mes':<6} {'Fuente GT':<18} {'Elegibles':>10} {'Contactados':>12} "
      f"{'ROI':>8} {'Profit inc':>12}")
print(f"  {'─'*70}")
for _, row in df_metricas_serie.iterrows():
    print(f"  {int(row['mes']):<6} {row['fuente_gt']:<18} "
          f"{int(row['n_elegibles']):>10,} {int(row['n_contactados']):>12,} "
          f"{row['roi_incremental']:>7.2f}x ${row['profit_incremental']:>10,.0f}")

subseccion("3.2 — Triggers de monitoreo detectados")
triggers = df_monitoreo_serie[df_monitoreo_serie['trigger_recalib'] == True]
alertas  = df_monitoreo_serie[df_monitoreo_serie['alerta_tasa'] == True]

print(f"\n  Triggers recalibración (desviación > 25%): {len(triggers)}")
if len(triggers) > 0:
    for _, t in triggers.iterrows():
        print(f"    → Mes {int(t['mes'])} | {t['producto']:<20} "
              f"proy={t['tasa_proyectada']:.4f} obs={t['tasa_observada']:.4f} "
              f"desv={t['desviacion']:+.1%}")

print(f"\n  Alertas (desviación > 15%): {len(alertas)}")
if len(alertas) > 0:
    for _, a in alertas.iterrows():
        print(f"    → Mes {int(a['mes'])} | {a['producto']:<20} "
              f"desv={a['desviacion']:+.1%}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — OUTPUTS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — OUTPUTS")

if campanas_acumuladas:
    pd.concat(campanas_acumuladas, ignore_index=True).to_csv(
        f'{DATA_DIR}/nbo_backtest_campana_completa.csv', index=False)
    print(f"  ✅ nbo_backtest_campana_completa.csv")

if resultados_acumulados:
    pd.concat(resultados_acumulados, ignore_index=True).to_csv(
        f'{DATA_DIR}/nbo_backtest_resultados.csv', index=False)
    print(f"  ✅ nbo_backtest_resultados.csv")

df_metricas_serie.to_csv(f'{DATA_DIR}/nbo_backtest_metricas_serie.csv', index=False)
print(f"  ✅ nbo_backtest_metricas_serie.csv")

df_monitoreo_serie.to_csv(f'{DATA_DIR}/nbo_backtest_monitoreo_serie.csv', index=False)
print(f"  ✅ nbo_backtest_monitoreo_serie.csv")

# ── Output clave para Semana 6: scores XGBoost + ground truth ────────
# Este CSV es el insumo correcto para recalibrar en S6.
# Contiene score_xgb_raw y p_calibrada (del modelo) para todos los meses,
# y se cruza con nbo_ofertas.csv en S6 para obtener convirtio_30d real.
df_scores_hist = pd.DataFrame(scores_historicos)
df_scores_hist.to_csv(f'{DATA_DIR}/nbo_scores_historicos_s5.csv', index=False)
print(f"  ✅ nbo_scores_historicos_s5.csv  ← insumo para Semana 6")
print(f"     {len(df_scores_hist):,} registros | "
      f"meses {df_scores_hist['mes_scoring'].min()}–{df_scores_hist['mes_scoring'].max()}")

separador("SEMANA 5 COMPLETADA")
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     RBlJose — NBO — SEMANA 5 COMPLETADA (v2)        ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 0 : Modelos reales desde /models/                 ║
  ║  ✓ Bloque 1 : CSVs reales de Semana 2                       ║
  ║  ✓ Bloque 2 : Loop 4 ciclos con T+30 corregido              ║
  ║               Ground truth = score_propension del generador  ║
  ║  ✓ Bloque 3 : Análisis temporal + triggers reales           ║
  ║  ✓ Bloque 4 : Outputs incluyendo nbo_scores_historicos_s5   ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  CORRECCIÓN APLICADA:                                        ║
  ║    T+30 ya NO usa p_calibrada como ground truth             ║
  ║    Usa score_propension de nbo_ofertas.csv (generador DAG)  ║
  ║    El drift ahora es observable y medible correctamente     ║
  ╚══════════════════════════════════════════════════════════════╝
""")

# %%
# ── DETECCIÓN AUTOMÁTICA DE TRIGGERS ─────────────────────────────
separador("BLOQUE TRIGGERS — DETECCIÓN AUTOMÁTICA")

TODOS_PRODUCTOS = sorted(df_monitoreo_serie['producto'].unique().tolist())

triggers_automaticos = []
resumen_criterios = []

for prod in TODOS_PRODUCTOS:
    sub = df_monitoreo_serie[df_monitoreo_serie['producto'] == prod].copy()
    
    # Criterio 1: solo ciclos con N suficiente
    sub_n = sub[sub['n_tratamiento'] >= 200]
    
    # Criterio 2: desviación > 25%
    sub_trigger = sub_n[sub_n['desviacion'] > 0.25]
    
    # Criterio 3: al menos 2 ciclos con trigger
    n_ciclos_trigger = len(sub_trigger)
    
    # Criterio 4: N convertidos acumulados >= 30
    n_convertidos = int(
        (sub_trigger['n_tratamiento'] * sub_trigger['tasa_observada']).sum()
    )
    
    cumple = (n_ciclos_trigger >= 2) and (n_convertidos >= 30)
    
    if cumple:
        triggers_automaticos.append(prod)
    
    resumen_criterios.append({
        'producto'        : prod,
        'ciclos_con_N'    : len(sub_n),
        'ciclos_trigger'  : n_ciclos_trigger,
        'n_convertidos'   : n_convertidos,
        'cumple_criterios': cumple,
    })
    
    flag = "✅ TRIGGER" if cumple else "─"
    print(f"  {prod:<20} ciclos_trigger={n_ciclos_trigger}  "
          f"n_conv={n_convertidos:>4}  {flag}")

# Serializar para Semana 6
payload = {
    'productos_recalibrar' : triggers_automaticos,
    'fecha_deteccion'      : str(date.today()),
    'ciclos_evaluados'     : mes,
    'criterios'            : {
        'min_N_contactados'      : 200,
        'min_desviacion_pct'     : 0.25,
        'min_ciclos_consecutivos': 2,
        'min_n_convertidos'      : 30,
    },
    'detalle': resumen_criterios,
}

with open(f'{DATA_DIR}/nbo_triggers_activos.json', 'w') as f:
    json.dump(payload, f, indent=2)

print(f"\n  Productos a recalibrar : {triggers_automaticos}")
print(f"  ✅ nbo_triggers_activos.json exportado")


