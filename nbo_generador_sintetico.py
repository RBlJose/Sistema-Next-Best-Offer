# %%
"""
RBlJose — SISTEMA NBO
Generador de Datos Sintéticos — Semana 2
================================================
Arquitectura: Señal Causal DAG + Ruido Estructural con componente persistente
Enfoque 3 consolidado: z = señal_dag + ε_cliente + ε_mes + ε_externo

Tablas generadas:
  1. CLIENTES       — 20,000 clientes con variables raíz y derivadas
  2. PRODUCTOS      — 6 productos con parámetros de margen, PD, LGD, RWA
  3. TRANSACCIONES  — historial transaccional 25 meses por cliente
  4. FEATURES       — tabla derivada construida por pipeline (input directo del modelo)
  5. OFERTAS        — registro de ofertas con variable objetivo convirtio_30d

Período:  Enero 2024 — Enero 2026 (25 meses)
Régimen macro:
  Meses  1-12 → Normal
  Meses 13-15 → Deterioro leve
  Meses 16-18 → Deterioro moderado
  Meses 19-21 → Stress inicial      (Validación)
  Meses 22-23 → Stress+recuperación (Test)
  Meses 24-25 → Recuperación        (Producción sin etiqueta)
"""

# %%
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid  # función sigmoid estable
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# %%
# ══════════════════════════════════════════════════════════════════════
# SEMILLA Y CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)

N_CLIENTS   = 20_000
N_MESES     = 25
FECHA_INICIO = date(2024, 1, 1)

# Productos en scope Fase 1
PRODUCTOS = ['tarjeta', 'prestamo', 'microcredito',
             'seguro_vida', 'seguro_salud', 'inversion']

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — RÉGIMEN MACROECONÓMICO
# Índice de estrés macro por mes (0.0=normal, 1.0=crisis severa)
# Afecta señal de propensión, PD realizada y comportamiento transaccional
# ══════════════════════════════════════════════════════════════════════
REGIMEN_MACRO = {
    # mes: {'stress': float, 'delta_pd': float, 'nombre': str}
    **{m: {'stress': 0.05, 'delta_pd': 0.00, 'nombre': 'Normal'}
       for m in range(1, 13)},
    **{m: {'stress': 0.20, 'delta_pd': 0.02, 'nombre': 'Deterioro_leve'}
       for m in range(13, 16)},
    **{m: {'stress': 0.40, 'delta_pd': 0.04, 'nombre': 'Deterioro_moderado'}
       for m in range(16, 19)},
    **{m: {'stress': 0.65, 'delta_pd': 0.07, 'nombre': 'Stress_inicial'}
       for m in range(19, 22)},
    **{m: {'stress': 0.55, 'delta_pd': 0.05, 'nombre': 'Stress_recuperacion'}
       for m in range(22, 24)},
    **{m: {'stress': 0.25, 'delta_pd': 0.02, 'nombre': 'Recuperacion'}
       for m in range(24, 26)},
}

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — TABLA PRODUCTOS
# Parámetros fijos del portafolio: margen, PD base, LGD, RWA
# ══════════════════════════════════════════════════════════════════════
def generar_tabla_productos() -> pd.DataFrame:
    """
    Parámetros calibrados con datos reales del mercado ecuatoriano.
    PD base: probabilidad de default en condiciones normales.
    LGD: Loss Given Default — fracción del capital no recuperada.
    factor_rwa: ponderación de riesgo regulatorio Basilea III.
    """
    productos_data = [
        {
            'id_producto':       'TARJETA',
            'nombre':            'Tarjeta de Crédito',
            'categoria':         'Credito',
            'margen_bruto':      0.230,   # 23% EA promedio rotativa
            'margen_neto':       0.155,   # tras costos operativos y provisiones
            'costo_contacto':    2.50,    # USD por contacto digital
            'costo_originacion': 45.00,
            'pd_base_esperada':  0.055,   # 5.5% base en condiciones normales
            'lgd_estimada':      0.75,    # sin garantía real
            'consume_rwa':       True,
            'factor_rwa':        1.00,    # 100% ponderación — activo riesgoso
            'flujo_tipo':        'Volatil',
            'activo':            True
        },
        {
            'id_producto':       'PRESTAMO',
            'nombre':            'Préstamo Personal',
            'categoria':         'Credito',
            'margen_bruto':      0.200,
            'margen_neto':       0.135,
            'costo_contacto':    3.00,
            'costo_originacion': 60.00,
            'pd_base_esperada':  0.045,
            'lgd_estimada':      0.78,
            'consume_rwa':       True,
            'factor_rwa':        0.75,
            'flujo_tipo':        'Estable',
            'activo':            True
        },
        {
            'id_producto':       'MICROCREDITO',
            'nombre':            'Microcrédito',
            'categoria':         'Credito',
            'margen_bruto':      0.290,   # tasa máxima regulada Ecuador
            'margen_neto':       0.120,   # margen neto bajo por costo operativo alto
            'costo_contacto':    4.50,    # canal presencial dominante
            'costo_originacion': 85.00,
            'pd_base_esperada':  0.095,   # mayor PD del portafolio
            'lgd_estimada':      0.82,
            'consume_rwa':       True,
            'factor_rwa':        0.75,
            'flujo_tipo':        'Volatil',
            'activo':            True
        },
        {
            'id_producto':       'SEGURO_VIDA',
            'nombre':            'Seguro de Vida',
            'categoria':         'Seguro',
            'margen_bruto':      0.200,   # comisión 20% sobre prima
            'margen_neto':       0.175,   # costos bajos, sin RWA
            'costo_contacto':    2.00,
            'costo_originacion': 15.00,
            'pd_base_esperada':  0.000,   # no aplica — no es producto crediticio
            'lgd_estimada':      0.000,
            'consume_rwa':       False,   # CRÍTICO: mejor RAROC por capital=0
            'factor_rwa':        0.000,
            'flujo_tipo':        'Renovable',
            'activo':            True
        },
        {
            'id_producto':       'SEGURO_SALUD',
            'nombre':            'Seguro de Salud',
            'categoria':         'Seguro',
            'margen_bruto':      0.175,
            'margen_neto':       0.150,
            'costo_contacto':    2.00,
            'costo_originacion': 15.00,
            'pd_base_esperada':  0.000,
            'lgd_estimada':      0.000,
            'consume_rwa':       False,
            'factor_rwa':        0.000,
            'flujo_tipo':        'Renovable',
            'activo':            True
        },
        {
            'id_producto':       'INVERSION',
            'nombre':            'Inversión Básica (DPF)',
            'categoria':         'Inversion',
            'margen_bruto':      0.040,   # spread 4% entre tasa pasiva/activa
            'margen_neto':       0.030,
            'costo_contacto':    1.50,
            'costo_originacion': 10.00,
            'pd_base_esperada':  0.000,   # pasivo del banco — no consume capital
            'lgd_estimada':      0.000,
            'consume_rwa':       False,
            'factor_rwa':        0.000,
            'flujo_tipo':        'Estable',
            'activo':            True
        },
    ]
    return pd.DataFrame(productos_data)



# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — TABLA CLIENTES
# Variables raíz → derivadas → score crediticio (respetando DAG causal)
# ══════════════════════════════════════════════════════════════════════
def generar_tabla_clientes() -> pd.DataFrame:
    """
    Generación con cadena causal estricta según DAG del Paso 4.
    ORDEN OBLIGATORIO:
      1. segmento (raíz)
      2. ingreso ← segmento
      3. edad, ocupacion, zona ← segmento
      4. hijos ← edad + estado_civil
      5. canal_principal ← zona
      6. antiguedad ← segmento (clientes premium tienen más historia)
      7. score ← ingreso + antiguedad + atrasos + deuda
    """
    n = N_CLIENTS

    # ── 1. Segmento ───────────────────────────────────────────────────
    segmentos = np.random.choice(
        ['Masivo', 'Preferente', 'Pyme', 'Premium'],
        size=n,
        p=[0.60, 0.25, 0.10, 0.05]
    )

    # ── 2. Ingreso mensual ← segmento (log-normal, siempre positivo) ──
    # Parámetros: (mu_log, sigma_log, min_usd, max_usd)
    INCOME_PARAMS = {
        'Masivo':     (6.37, 0.35,   300,  1_200),
        'Preferente': (7.09, 0.30,   800,  3_500),
        'Pyme':       (7.72, 0.40, 1_200, 12_000),
        'Premium':    (8.29, 0.35, 2_000, 25_000),
    }
    ingreso = np.zeros(n)
    for seg, (mu, sigma, lo, hi) in INCOME_PARAMS.items():
        mask = segmentos == seg
        raw  = np.random.lognormal(mu, sigma, mask.sum())
        ingreso[mask] = np.clip(raw, lo, hi)

    # ── 3. Edad ← segmento ────────────────────────────────────────────
    AGE_PARAMS = {
        'Masivo':     (35, 12, 18, 70),
        'Preferente': (40, 10, 25, 68),
        'Pyme':       (43,  9, 28, 65),
        'Premium':    (47,  8, 30, 65),
    }
    edad = np.zeros(n, dtype=int)
    for seg, (mu, sigma, lo, hi) in AGE_PARAMS.items():
        mask = segmentos == seg
        raw  = np.random.normal(mu, sigma, mask.sum())
        edad[mask] = np.clip(raw, lo, hi).astype(int)

    # ── 4. Ocupación ← segmento ───────────────────────────────────────
    OCC_DIST = {
        'Masivo':     {'Empleado': 0.65, 'Independiente': 0.28, 'Empresario': 0.07},
        'Preferente': {'Empleado': 0.55, 'Independiente': 0.30, 'Empresario': 0.15},
        'Pyme':       {'Empleado': 0.20, 'Independiente': 0.35, 'Empresario': 0.45},
        'Premium':    {'Empleado': 0.25, 'Independiente': 0.25, 'Empresario': 0.50},
    }
    ocupacion = np.empty(n, dtype=object)
    for seg, dist in OCC_DIST.items():
        mask = segmentos == seg
        ocupacion[mask] = np.random.choice(
            list(dist.keys()), size=mask.sum(), p=list(dist.values())
        )

    # ── 5. Zona geográfica ← segmento ────────────────────────────────
    ZONE_DIST = {
        'Masivo':     {'Urbana': 0.55, 'Periurbana': 0.30, 'Rural': 0.15},
        'Preferente': {'Urbana': 0.75, 'Periurbana': 0.20, 'Rural': 0.05},
        'Pyme':       {'Urbana': 0.70, 'Periurbana': 0.25, 'Rural': 0.05},
        'Premium':    {'Urbana': 0.90, 'Periurbana': 0.09, 'Rural': 0.01},
    }
    zona = np.empty(n, dtype=object)
    for seg, dist in ZONE_DIST.items():
        mask = segmentos == seg
        zona[mask] = np.random.choice(
            list(dist.keys()), size=mask.sum(), p=list(dist.values())
        )

    # ── 6. Estado civil y hijos ← edad + segmento ────────────────────
    estado_civil = np.random.choice(
        ['Soltero', 'Casado', 'Divorciado', 'Viudo'],
        size=n, p=[0.35, 0.50, 0.12, 0.03]
    )
    # P(hijos > 0) sube con edad 28-45 y si casado
    p_hijos = (
        0.20
        + 0.35 * ((edad >= 28) & (edad <= 45))
        + 0.20 * (estado_civil == 'Casado')
    )
    tiene_hijos = np.random.binomial(1, np.clip(p_hijos, 0, 1))
    hijos = np.where(
        tiene_hijos,
        np.random.choice([1, 2, 3], size=n, p=[0.45, 0.40, 0.15]),
        0
    )

    # ── 7. Canal principal ← zona geográfica ─────────────────────────
    canal_principal = np.where(
        zona == 'Urbana',
        np.random.choice(['Digital', 'Sucursal'], size=n, p=[0.65, 0.35]),
        np.random.choice(['Sucursal', 'Cajero'], size=n, p=[0.55, 0.45])
    )

    # ── 8. Antigüedad ← segmento (meses de relación con el banco) ────
    ANTIG_PARAMS = {
        'Masivo':     (24, 18,  3, 120),
        'Preferente': (42, 24,  6, 180),
        'Pyme':       (54, 30, 12, 240),
        'Premium':    (72, 36, 18, 300),
    }
    antiguedad = np.zeros(n, dtype=int)
    for seg, (mu, sigma, lo, hi) in ANTIG_PARAMS.items():
        mask = segmentos == seg
        raw  = np.random.normal(mu, sigma, mask.sum())
        antiguedad[mask] = np.clip(raw, lo, hi).astype(int)

    # ── 9. Atrasos históricos (endógeno al perfil de riesgo) ──────────
    p_atraso = np.where(
        segmentos == 'Masivo',   0.18,
        np.where(segmentos == 'Preferente', 0.08,
        np.where(segmentos == 'Pyme',       0.10, 0.04))
    )
    tiene_atraso = np.random.binomial(1, p_atraso)
    max_atraso   = np.where(
        tiene_atraso,
        np.random.choice([15, 30, 60, 90], size=n, p=[0.30, 0.35, 0.25, 0.10]),
        0
    )

    # ── 10. Ratio de deuda ────────────────────────────────────────────
    ratio_deuda = np.clip(
        np.random.beta(2, 5, size=n) * 0.6,  # concentrado entre 0.05-0.40
        0.0, 0.65
    )

    # ── 11. Score crediticio ← DAG causal ────────────────────────────
    # DIRECCIÓN: ingreso → score (NO al revés)
    # Componentes calibrados para distribución realista ecuatoriana:
    # ~15% < 550, ~35% 550-700, ~35% 700-800, ~15% > 800
    # Score base por segmento — ancla la distribución correctamente
    score_base_seg = np.where(
        segmentos == 'Premium',    840,
        np.where(segmentos == 'Pyme',       770,
        np.where(segmentos == 'Preferente', 700, 620))  # Masivo base
    )

    # Componente ingreso: rango 0-120 puntos
    ingreso_norm = np.clip(
        (ingreso - 300) / (25000 - 300), 0, 1
    )
    score_ingreso = ingreso_norm * 120

    # Componente antigüedad: rango 0-80 puntos
    score_antiguedad = np.clip(antiguedad / 60 * 80, 0, 80)

    # Penalizaciones
    score_pen_atraso = np.where(tiene_atraso,
        np.where(max_atraso >= 60, -200, -120), 0)
    score_pen_deuda  = np.clip(-ratio_deuda * 150, -100, 0)

    # Ruido idiosincrático
    ruido_score = np.random.normal(0, 40, size=n)

    score_raw = (score_base_seg + score_ingreso + score_antiguedad
                 + score_pen_atraso + score_pen_deuda + ruido_score)
    score_crediticio = np.clip(score_raw, 0, 1000).astype(int)

    # ── 12. Score buró (correlacionado pero no idéntico) ──────────────
    score_buro = np.clip(
        score_crediticio + np.random.normal(0, 60, size=n),
        0, 999
    ).astype(int)

    # ── 13. Nivel educación ───────────────────────────────────────────
    EDU_DIST = {
        'Masivo':     {'Secundaria': 0.55, 'Universidad': 0.38, 'Postgrado': 0.07},
        'Preferente': {'Secundaria': 0.25, 'Universidad': 0.55, 'Postgrado': 0.20},
        'Pyme':       {'Secundaria': 0.20, 'Universidad': 0.50, 'Postgrado': 0.30},
        'Premium':    {'Secundaria': 0.05, 'Universidad': 0.40, 'Postgrado': 0.55},
    }
    nivel_edu = np.empty(n, dtype=object)
    for seg, dist in EDU_DIST.items():
        mask = segmentos == seg
        nivel_edu[mask] = np.random.choice(
            list(dist.keys()), size=mask.sum(), p=list(dist.values())
        )

    # ── 14. Flag activo: universo elegible ────────────────────────────
    # Criterios del Paso 3 del sistema — pre-computable desde atributos estáticos
    activo = (
        (antiguedad >= 3)
        & (max_atraso <= 30)
    )

    # ── Ensamble DataFrame CLIENTES ───────────────────────────────────
    ids = [f'CLI_{i:06d}' for i in range(n)]

    df = pd.DataFrame({
        'id_cliente':         ids,
        'edad':               edad,
        'ingreso_mensual':    np.round(ingreso, 2),
        'segmento':           segmentos,
        'score_crediticio':   score_crediticio,
        'score_buro':         score_buro,
        'antiguedad_meses':   antiguedad,
        'estado_civil':       estado_civil,
        'hijos':              hijos,
        'nivel_educacion':    nivel_edu,
        'zona_geografica':    zona,
        'canal_principal':    canal_principal,
        'ocupacion':          ocupacion,
        'tiene_atraso_hist':  tiene_atraso,
        'max_atraso_dias':    max_atraso,
        'ratio_deuda_init':   np.round(ratio_deuda, 4),
        'activo':             activo,
        'fecha_actualizacion': str(FECHA_INICIO),
    })

    return df

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — RUIDO PERSISTENTE POR CLIENTE
# Se genera UNA VEZ y es estable en el tiempo.
# Representa heterogeneidad individual no observable:
# aversión al riesgo, preferencias personales, relaciones con otras
# instituciones, factores de comportamiento no medibles.
# ══════════════════════════════════════════════════════════════════════
SIGMA_CLIENTE = {
    'tarjeta':      0.80,
    'prestamo':     0.90,
    'microcredito': 1.00,
    'seguro_vida':  0.60,
    'seguro_salud': 0.60,
    'inversion':    0.70,
}

def generar_ruido_persistente(n: int) -> dict:
    """
    Genera ruido persistente por cliente para cada producto.
    Este array se pasa al generador de propensión en cada mes
    sin ser regenerado — es la personalidad latente del cliente.
    """
    return {
        prod: np.random.normal(0, sigma, size=n)
        for prod, sigma in SIGMA_CLIENTE.items()
    }


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — SEÑAL CAUSAL DAG POR PRODUCTO
# Implementa el Nivel 4 del DAG del Paso 4.
# Cada producto tiene sus propios predictores — NO son intercambiables.
# ══════════════════════════════════════════════════════════════════════
def calcular_señal_dag(clientes: pd.DataFrame,
                       features_mes: dict,
                       producto: str,
                       stress: float) -> np.ndarray:
    """
    Señal causal en escala logit para cada cliente en un mes dado.
    Negativo = menor propensión, positivo = mayor propensión.

    Los coeficientes están en escala logit (log-odds):
      +1.0 ≈ multiplicar odds por e ≈ +35% en probabilidad desde base
      -1.0 ≈ dividir odds por e    ≈ -26% en probabilidad desde base

    Señal intencionalmente débil: el modelo no debe memorizar el DAG.
    """
    n = len(clientes)
    señal = np.zeros(n)

    # Atajos para legibilidad
    score  = clientes['score_crediticio'].values
    ingreso = clientes['ingreso_mensual'].values
    edad   = clientes['edad'].values
    hijos  = clientes['hijos'].values
    ocup   = clientes['ocupacion'].values
    seg    = clientes['segmento'].values

    # Features construidas por el pipeline del mes
    gasto_3m       = features_mes.get('gasto_3m', np.zeros(n))
    gasto_super    = features_mes.get('gasto_supermercado_3m', np.zeros(n))
    gasto_farmacia = features_mes.get('gasto_farmacia_3m', np.zeros(n))
    depositos_ef   = features_mes.get('depositos_efectivo_3m', np.zeros(n))
    tx_digital     = features_mes.get('tx_digitales_proporcion', np.zeros(n))
    saldo_90d      = features_mes.get('saldo_promedio_90d', np.zeros(n))
    saldo_tend     = features_mes.get('saldo_tendencia', np.zeros(n))
    ratio_ci       = features_mes.get('ratio_cuota_ingreso', np.zeros(n))
    dias_cred      = features_mes.get('dias_desde_ult_credito', np.full(n, 999))

    # Percentiles para señales relativas — calculados sobre el mes
    p60_super    = np.percentile(gasto_super[gasto_super > 0], 60) if (gasto_super > 0).any() else 1
    p60_farmacia = np.percentile(gasto_farmacia[gasto_farmacia > 0], 60) if (gasto_farmacia > 0).any() else 1
    p70_saldo    = np.percentile(saldo_90d[saldo_90d > 0], 70) if (saldo_90d > 0).any() else 1
    p70_dep_ef   = np.percentile(depositos_ef[depositos_ef > 0], 70) if (depositos_ef > 0).any() else 1

    if producto == 'tarjeta':
        señal = (
            + 1.4 * (score > 700).astype(float)
            + 1.0 * (tx_digital > 0.50).astype(float)
            + 0.8 * (gasto_super > p60_super).astype(float)
            + 0.6 * (ingreso > 800).astype(float)
            - 1.8 * (ratio_ci > 0.35).astype(float)
            - 1.3 * (clientes['tiene_atraso_hist'].values == 1).astype(float)
            - 0.7 * stress                              # macro deprime propensión
        )

    elif producto == 'prestamo':
        señal = (
            + 1.2 * (saldo_tend > 0).astype(float)
            + 1.0 * (antiguedad_meses_ok := (clientes['antiguedad_meses'].values > 12)).astype(float)
            + 0.8 * (score > 650).astype(float)
            + 0.7 * (ingreso > 600).astype(float)
            # Señal temporal: ciclos naturales de demanda
            + 0.5 * ((dias_cred > 90) & (dias_cred < 365)).astype(float)
            - 1.6 * (ratio_ci > 0.40).astype(float)
            - 0.9 * stress
        )

    elif producto == 'microcredito':
        variab_saldo = features_mes.get('variabilidad_saldo', np.zeros(n))
        señal = (
            + 1.8 * (depositos_ef > p70_dep_ef).astype(float)
            + 1.4 * (variab_saldo > np.percentile(variab_saldo, 60)).astype(float)
            + 1.0 * (ocup == 'Independiente').astype(float)
            - 1.5 * (score > 800).astype(float)        # tiene mejores opciones
            - 1.2 * (ocup == 'Empleado').astype(float) # perfil asalariado no es target
            + 1.1 * stress                              # stress SUBE demanda (peligroso)
        )

    elif producto == 'seguro_vida':
        señal = (
            + 1.6 * ((edad >= 30) & (edad <= 55)).astype(float)
            + 1.3 * (hijos > 0).astype(float)
            + 1.0 * (ingreso > 800).astype(float)
            + 0.8 * (saldo_90d > np.percentile(saldo_90d, 50)).astype(float)
            # Ventana post-préstamo: 30 días de oportunidad
            + 1.4 * ((dias_cred > 0) & (dias_cred <= 30)).astype(float)
            - 1.4 * (hijos == 0).astype(float)
            - 0.9 * (edad < 28).astype(float)
            - 0.3 * stress                              # seguros relativamente anticíclicos
        )

    elif producto == 'seguro_salud':
        señal = (
            + 1.5 * (gasto_farmacia > p60_farmacia).astype(float)
            + 1.2 * (ocup == 'Independiente').astype(float)  # sin cobertura empleador
            + 1.0 * (hijos > 0).astype(float)
            + 0.8 * ((edad >= 28) & (edad <= 55)).astype(float)
            + 0.6 * (ingreso > 600).astype(float)
            - 1.3 * (ocup == 'Empleado').astype(float)       # cobertura por empleador
            - 0.2 * stress
        )

    elif producto == 'inversion':
        señal = (
            + 1.8 * (saldo_90d > p70_saldo).astype(float)
            + 1.2 * (saldo_tend > 0).astype(float)
            + 0.9 * (edad > 35).astype(float)
            # Ventana post-cancelación crédito: flujo liberado
            + 1.6 * ((dias_cred > 30) & (dias_cred <= 90)).astype(float)
            - 1.4 * (ratio_ci > 0.30).astype(float)
            - 1.0 * (saldo_90d < np.percentile(saldo_90d, 30)).astype(float)
            - 0.5 * stress                              # incertidumbre reduce ahorro
        )

    return señal


def ajustar_señal_por_interacciones(señal: np.ndarray,
                                    productos_activos: dict,
                                    producto_target: str) -> np.ndarray:
    """
    Modifica la señal causal según interacciones del Paso 2 del DAG.
    Las interacciones van en la SEÑAL, no en el ruido.
    Magnitudes en logit calibradas desde las % del documento.
    """
    ajuste = np.zeros(len(señal))

    if producto_target == 'seguro_vida':
        if 'prestamo' in productos_activos:
            # +40% probabilidad → ~+0.85 en logit
            ajuste += 0.85 * productos_activos['prestamo'].astype(float)

    elif producto_target == 'seguro_salud':
        if 'tarjeta' in productos_activos:
            # +25% → ~+0.58
            ajuste += 0.58 * productos_activos['tarjeta'].astype(float)
        if 'seguro_vida' in productos_activos:
            # +45% → ~+0.98
            ajuste += 0.98 * productos_activos['seguro_vida'].astype(float)

    elif producto_target == 'microcredito':
        if 'tarjeta' in productos_activos:
            # -60% → ~-1.83
            ajuste -= 1.83 * productos_activos['tarjeta'].astype(float)

    elif producto_target == 'inversion':
        if 'prestamo' in productos_activos:
            # -35% → ~-0.73
            ajuste -= 0.73 * productos_activos['prestamo'].astype(float)

    return señal + ajuste

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — GENERACIÓN DE TRANSACCIONES Y FEATURES
# Pipeline batch mensual: construye features desde transacciones
# respetando el corte temporal (anti-leakage).
# ══════════════════════════════════════════════════════════════════════
def generar_features_mes(clientes: pd.DataFrame,
                         mes: int,
                         stress: float) -> dict:
    """
    Simula directamente las features agregadas por cliente en el mes t.
    En producción estas vendrían del pipeline SQL sobre TRANSACCIONES.

    Las features son función de los atributos del cliente + ruido
    transaccional mensual + efecto macro.
    """
    n = len(clientes)
    ingreso = clientes['ingreso_mensual'].values
    score   = clientes['score_crediticio'].values
    ocup    = clientes['ocupacion'].values
    seg     = clientes['segmento'].values

    # Factor macro: stress reduce gasto discrecional, sube variabilidad
    factor_stress = 1 - 0.3 * stress

    # Gasto total 3 meses (función de ingreso + segmento + stress)
    gasto_base = ingreso * np.random.uniform(0.55, 0.85, n) * factor_stress
    gasto_3m   = np.maximum(gasto_base * 3 + np.random.normal(0, ingreso * 0.1, n), 0)

    # Gasto supermercado (predictor tarjeta)
    gasto_super = gasto_3m * np.random.uniform(0.10, 0.25, n)

    # Gasto farmacia (predictor seguro salud)
    # Independientes gastan más de bolsillo
    base_farmacia = np.where(ocup == 'Independiente', 0.04, 0.02)
    gasto_farmacia = gasto_3m * base_farmacia * np.random.lognormal(0, 0.5, n)

    # Depósitos en efectivo (predictor microcrédito)
    # Concentrado en Independientes y zona periurbana/rural
    p_deposito_ef = np.where(
        (ocup == 'Independiente') & (clientes['zona_geografica'].values != 'Urbana'),
        0.45, 0.10
    )
    depositos_ef = np.where(
        np.random.binomial(1, p_deposito_ef),
        np.random.lognormal(5, 0.8, n),  # montos pequeños irregulares
        0.0
    )

    # Proporción transacciones digitales
    tx_digital_base = np.where(
        clientes['canal_principal'].values == 'Digital', 0.75, 0.30
    )
    tx_digital = np.clip(
        tx_digital_base + np.random.normal(0, 0.10, n),
        0, 1
    )

    # Saldo promedio 90 días (función de ingreso - gasto + ahorro)
    saldo_base = ingreso * np.random.uniform(1.5, 4.0, n)  # múltiplos del ingreso
    saldo_90d  = np.maximum(
        saldo_base * (1 - 0.2 * stress) + np.random.normal(0, ingreso * 0.5, n),
        0
    )

    # Tendencia de saldo (pendiente de regresión lineal simulada)
    # Positiva si ahorra, negativa si gasta más que ingresa
    saldo_tend = np.random.normal(
        np.where(score > 700, 0.02, -0.01),  # score alto → tendencia positiva
        0.05, n
    ) * (1 - 0.5 * stress)

    # Variabilidad de saldo (predictor microcrédito)
    variab_saldo = np.abs(np.random.normal(
        np.where(ocup == 'Independiente', 0.35, 0.12),
        0.10, n
    ))

    # Ratio cuota / ingreso
    ratio_ci = np.clip(
        clientes['ratio_deuda_init'].values
        + np.random.normal(0, 0.03, n)
        + 0.05 * stress,  # stress sube el ratio efectivo
        0, 0.70
    )

    # Días desde último crédito
    dias_cred = np.random.choice(
        [0, 15, 30, 60, 90, 180, 365, 730, 999],
        size=n,
        p=[0.05, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.15, 0.10]
    )

    return {
        'gasto_3m':                np.round(gasto_3m, 2),
        'gasto_supermercado_3m':   np.round(gasto_super, 2),
        'gasto_farmacia_3m':       np.round(gasto_farmacia, 2),
        'depositos_efectivo_3m':   np.round(depositos_ef, 2),
        'tx_digitales_proporcion': np.round(tx_digital, 4),
        'saldo_promedio_90d':      np.round(saldo_90d, 2),
        'saldo_tendencia':         np.round(saldo_tend, 4),
        'variabilidad_saldo':      np.round(variab_saldo, 4),
        'ratio_cuota_ingreso':     np.round(ratio_ci, 4),
        'dias_desde_ult_credito':  dias_cred,
        'indice_estres_macro':     np.full(n, round(stress, 4)),
        'estres_x_riesgo':         np.round(stress * (1 - score / 1000), 4),
    }


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 7 — INTERCEPTOS POR PRODUCTO
# Controlan la tasa base de conversión global.
# Calibrados para que el promedio ponderado del portafolio sea ~8%.
# ══════════════════════════════════════════════════════════════════════
# sigmoid(α) = tasa_base_objetivo cuando señal=0 y ruido=0
# tarjeta:      sigmoid(-2.44) ≈ 0.080 → 8%
# prestamo:     sigmoid(-2.44) ≈ 0.080 → 8%
# microcredito: sigmoid(-2.94) ≈ 0.050 → 5%  (segmento más pequeño)
# seguro_vida:  sigmoid(-2.20) ≈ 0.100 → 10% (señal fuerte, buen predictor)
# seguro_salud: sigmoid(-2.44) ≈ 0.080 → 8%
# inversion:    sigmoid(-3.00) ≈ 0.047 → 5%  (ticket alto, mercado selectivo)

INTERCEPTOS = {
    'tarjeta':      -4.40,
    'prestamo':     -4.20,
    'microcredito': -4.50,
    'seguro_vida':  -3.90,
    'seguro_salud': -4.10,
    'inversion':    -4.50,
}

# Sigma del ruido temporal y externo (por producto)
SIGMA_MES      = {'tarjeta': 0.60, 'prestamo': 0.50, 'microcredito': 0.70,
                  'seguro_vida': 0.40, 'seguro_salud': 0.50, 'inversion': 0.40}
SIGMA_EXTERNO  = {'tarjeta': 0.50, 'prestamo': 0.40, 'microcredito': 0.60,
                  'seguro_vida': 0.30, 'seguro_salud': 0.40, 'inversion': 0.30}

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 8 — PIPELINE PRINCIPAL
# Itera 25 meses, genera ofertas y conversiones por cliente × producto
# ══════════════════════════════════════════════════════════════════════
def generar_ofertas_y_features(clientes: pd.DataFrame,
                                ruido_persistente: dict) -> tuple:
    """
    Genera las tablas OFERTAS y FEATURES para los 25 meses.
    Retorna: (df_ofertas, df_features)
    """
    n = len(clientes)
    ids = clientes['id_cliente'].values

    ofertas_list  = []
    features_list = []

    # Estado de productos activos por cliente (empieza vacío)
    # Se actualiza mes a mes cuando hay conversión
    productos_activos_cliente = {
        prod: np.zeros(n, dtype=bool) for prod in PRODUCTOS
    }

    oferta_counter = 0

    for mes in range(1, N_MESES + 1):

        fecha_corte = FECHA_INICIO + timedelta(days=30 * (mes - 1))
        macro = REGIMEN_MACRO[mes]
        stress = macro['stress']

        # ── Features del mes ──────────────────────────────────────────
        features_mes = generar_features_mes(clientes, mes, stress)

        # ── Elegibilidad base (filtros duros) ─────────────────────────
        elegible_base = (
            clientes['activo'].values
            & (clientes['antiguedad_meses'].values >= 3)
            & (clientes['max_atraso_dias'].values <= 30)
        )

        # ── Registro de features para la tabla FEATURES ───────────────
        for i in range(n):
            features_list.append({
                'id_cliente':              ids[i],
                'fecha_corte':             str(fecha_corte),
                'mes':                     mes,
                'regimen_macro':           macro['nombre'],
                'version_features':        'v1.0',
                **{k: v[i] for k, v in features_mes.items()},
                'score_crediticio':        clientes['score_crediticio'].iloc[i],
                'ingreso_mensual':         clientes['ingreso_mensual'].iloc[i],
                'edad':                    clientes['edad'].iloc[i],
                'hijos':                   clientes['hijos'].iloc[i],
                'ocupacion':               clientes['ocupacion'].iloc[i],
                'segmento':                clientes['segmento'].iloc[i],
            })

        # ── Generación de propensión y conversión por producto ─────────
        for prod in PRODUCTOS:

            α = INTERCEPTOS[prod]
            ε_cliente = ruido_persistente[prod]
            ε_mes     = np.random.normal(0, SIGMA_MES[prod], n)
            ε_ext     = np.random.normal(0, SIGMA_EXTERNO[prod], n)

            # Señal causal desde DAG
            señal = calcular_señal_dag(clientes, features_mes, prod, stress)

            # Ajuste por interacciones entre productos
            señal = ajustar_señal_por_interacciones(
                señal, productos_activos_cliente, prod
            )

            # Propensión latente = señal + ruido estructural
            z = señal + ε_cliente + ε_mes + ε_ext

            # Probabilidad de conversión
            p_conv = sigmoid(α + z)

            # Elegibilidad específica por producto
            if prod in ['tarjeta', 'prestamo', 'microcredito']:
                elegible = (
                    elegible_base
                    & (clientes['score_crediticio'].values >= 550)
                    & (features_mes['ratio_cuota_ingreso'] < 0.45)
                    & ~productos_activos_cliente[prod]  # no tiene ya este producto
                )
                if prod == 'tarjeta':
                    elegible = elegible & (clientes['score_crediticio'].values >= 650)
                elif prod == 'microcredito':
                    elegible = elegible & (clientes['score_crediticio'].values < 820)
            elif prod == 'inversion':
                elegible = (
                    elegible_base
                    & (features_mes['saldo_promedio_90d'] > 500)
                    & ~productos_activos_cliente[prod]
                )
            else:  # seguros
                elegible = elegible_base & ~productos_activos_cliente[prod]

            # Asignar grupo tratamiento/control (80/20)
            grupo = np.where(
                np.random.binomial(1, 0.80, n),
                'Tratamiento', 'Control'
            )

            # Conversión observada (solo en tratamiento — grupo control no recibe oferta)
            convirtio = np.where(
                elegible & (grupo == 'Tratamiento'),
                np.random.binomial(1, np.clip(p_conv, 0, 1)),
                0
            )

            # Etiqueta completa: solo si T + 30 días ≤ 2025-11-30
            fecha_limite_etiqueta = date(2025, 11, 30)
            fecha_mas_30 = fecha_corte + timedelta(days=30)
            etiqueta_completa = fecha_mas_30 <= fecha_limite_etiqueta

            # Actualizar productos activos para próximo mes
            nuevos_activos = elegible & (convirtio == 1)
            productos_activos_cliente[prod] = (
                productos_activos_cliente[prod] | nuevos_activos
            )

            # Registrar solo clientes elegibles (reduce volumen de tabla)
            idx_elegibles = np.where(elegible)[0]

            for i in idx_elegibles:
                oferta_counter += 1
                ofertas_list.append({
                    'id_oferta':        f'OFR_{oferta_counter:010d}',
                    'id_cliente':       ids[i],
                    'id_producto':      prod.upper(),
                    'fecha_oferta':     str(fecha_corte),
                    'mes':              mes,
                    'regimen_macro':    macro['nombre'],
                    'canal_oferta':     clientes['canal_principal'].iloc[i],
                    'score_propension': round(float(p_conv[i]), 5),
                    'version_modelo':   'v1.0_fase1',
                    'grupo':            grupo[i],
                    'acepto':           int(convirtio[i]) if grupo[i] == 'Tratamiento' else None,
                    'convirtio_30d':    int(convirtio[i]) if etiqueta_completa and grupo[i] == 'Tratamiento' else None,
                    'etiqueta_completa': etiqueta_completa,
                    'ingreso_real':     None,  # se calcula post-conversión en producción
                })

        if mes % 5 == 0:
            print(f'  Mes {mes:2d}/{N_MESES} procesado — '
                  f'régimen: {macro["nombre"]} — '
                  f'ofertas acumuladas: {oferta_counter:,}')

    return pd.DataFrame(ofertas_list), pd.DataFrame(features_list)


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 9 — VALIDACIÓN DEL DATASET
# Verifica que las tasas base y distribuciones sean correctas
# antes de guardar los archivos.
# ══════════════════════════════════════════════════════════════════════
def validar_dataset(clientes, ofertas, features):
    print("\n" + "="*60)
    print("VALIDACIÓN DEL DATASET SINTÉTICO")
    print("="*60)

    # 1. Distribución de segmentos
    print("\n[1] Distribución de segmentos:")
    print(clientes['segmento'].value_counts(normalize=True).round(3).to_string())

    # 2. Distribución de score crediticio
    print("\n[2] Score crediticio — percentiles:")
    score = clientes['score_crediticio']
    print(f"  < 550:    {(score < 550).mean():.1%}  (objetivo ~15%)")
    print(f"  550-700:  {((score >= 550) & (score < 700)).mean():.1%}  (objetivo ~35%)")
    print(f"  700-800:  {((score >= 700) & (score < 800)).mean():.1%}  (objetivo ~35%)")
    print(f"  > 800:    {(score >= 800).mean():.1%}  (objetivo ~15%)")

    # 3. Tasa base de conversión por producto
    print("\n[3] Tasa base de conversión por producto (set completo):")
    elegibles = ofertas[
        (ofertas['grupo'] == 'Tratamiento') &
        (ofertas['etiqueta_completa'] == True)
    ]
    for prod in PRODUCTOS:
        sub = elegibles[elegibles['id_producto'] == prod.upper()]
        if len(sub) > 0:
            tasa = sub['convirtio_30d'].mean()
            print(f"  {prod:15s}: {tasa:.3f} ({tasa:.1%})  — n elegibles = {len(sub):,}")

    # 4. Tasa global del portafolio
    tasa_global = elegibles['convirtio_30d'].mean()
    print(f"\n  TASA GLOBAL PORTAFOLIO: {tasa_global:.3f} ({tasa_global:.1%})  (objetivo ~8%)")

    # 5. Clientes elegibles
    print(f"\n[4] Universo elegible: {clientes['activo'].mean():.1%} de clientes")

    # 6. Volumen de ofertas por régimen macro
    print("\n[5] Volumen de ofertas por régimen macro:")
    print(ofertas.groupby('regimen_macro')['id_oferta'].count().to_string())

    # 7. Check anti-leakage básico
    print("\n[6] Check anti-leakage:")
    features_fechas = pd.to_datetime(features['fecha_corte'])
    print(f"  Rango fechas features: {features_fechas.min().date()} → {features_fechas.max().date()}")
    ofertas_fechas = pd.to_datetime(ofertas['fecha_oferta'])
    print(f"  Rango fechas ofertas:  {ofertas_fechas.min().date()} → {ofertas_fechas.max().date()}")

    print("\n" + "="*60)
    print("VALIDACIÓN COMPLETADA")
    print("="*60)

# %%
# ══════════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("="*60)
    print("RBlJose — NBO GENERADOR SINTÉTICO")
    print(f"N_CLIENTS={N_CLIENTS:,} | N_MESES={N_MESES} | SEED={SEED}")
    print("="*60)

    # 1. Tabla PRODUCTOS
    print("\n[1/5] Generando tabla PRODUCTOS...")
    df_productos = generar_tabla_productos()
    print(f"  {len(df_productos)} productos generados")

    # 2. Tabla CLIENTES
    print("\n[2/5] Generando tabla CLIENTES (cadena causal DAG)...")
    df_clientes = generar_tabla_clientes()
    print(f"  {len(df_clientes):,} clientes generados")
    print(f"  Universo elegible inicial: {df_clientes['activo'].sum():,} "
          f"({df_clientes['activo'].mean():.1%})")

    # 3. Ruido persistente (una sola generación)
    print("\n[3/5] Generando ruido persistente por cliente...")
    ruido_persistente = generar_ruido_persistente(N_CLIENTS)
    print(f"  Ruido persistente generado para {len(PRODUCTOS)} productos")

    # 4. Tablas OFERTAS y FEATURES (pipeline mensual)
    print("\n[4/5] Ejecutando pipeline mensual (25 meses)...")
    df_ofertas, df_features = generar_ofertas_y_features(
        df_clientes, ruido_persistente
    )
    print(f"\n  Ofertas generadas: {len(df_ofertas):,}")
    print(f"  Registros features: {len(df_features):,}")

    # 5. Validación
    print("\n[5/5] Validando dataset...")
    validar_dataset(df_clientes, df_ofertas, df_features)

    # 6. Guardar archivos CSV
    print("\nGuardando archivos CSV...")
    import os
    out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    df_clientes.to_csv(f'{out_dir}/nbo_clientes.csv',    index=False)
    df_productos.to_csv(f'{out_dir}/nbo_productos.csv',  index=False)
    df_ofertas.to_csv(f'{out_dir}/nbo_ofertas.csv',      index=False)
    df_features.to_csv(f'{out_dir}/nbo_features.csv',    index=False)

    print("\nArchivos guardados:")
    for f in ['nbo_clientes.csv', 'nbo_productos.csv',
              'nbo_ofertas.csv', 'nbo_features.csv']:
        size = os.path.getsize(f'{out_dir}/{f}') / 1024
        print(f"  {f}: {size:.1f} KB")

    print("\n✓ Generación completa.")


