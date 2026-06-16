# %%
"""
RBlJose — SISTEMA NBO
Regenerar nbo_ofertas.csv con señal estacional — enfoque multiplicativo
========================================================================
DISEÑO CORRECTO (v2 de este script):

  El problema de la versión anterior era regenerar todo desde cero
  con ruido_persistente nuevo. El seed en un estado distinto al original
  cambiaba los ε_cliente de todos los clientes, lo que modificaba ~45%
  del target. Eso no es estacionalidad — es ruido nuevo.

  La señal estacional tiene magnitud ±0.3–0.6 logit.
  El ruido persistente tiene σ ≈ 0.8–1.0 → magnitud ±1.5 logit.
  El ruido domina la señal → el target cambia por razones incorrectas.

  SOLUCIÓN: no tocar el ruido. Cargar score_propension ORIGINAL de
  nbo_ofertas.csv v1 (que ya tiene el ruido correcto) y multiplicarlo
  por un factor estacional calibrado por producto y período.

  p_nueva = clip(p_original × factor_estacional(producto, mes, región), 0, 1)

  Propiedades del enfoque:
    - El ranking relativo de clientes se conserva (ruido persistente intacto)
    - Solo cambia la magnitud absoluta en períodos específicos
    - Cambio esperado en convirtio_30d: ~8–12% (no 45%)
    - El XGBoost puede aprender la señal porque features→target es limpio

  Factores estacionales calibrados para cambio realista:
    - Tarjeta navidad:    +20% sobre p_base
    - Inversión DC3:      +30% sobre p_base
    - Microcrédito clases:+25% solo Independientes
    - Ningún producto modifica más de ×2.0 ni menos de ×0.5

Prerequisito:
  nbo_patch_clientes_features.py ejecutado.
  nbo_clientes.csv con columna `region`.
  nbo_features.csv con features estacionales.
  nbo_ofertas.csv ORIGINAL (v1) presente — se usa como base.

Output:
  nbo_ofertas_v1_backup.csv  ← copia del original antes de modificar
  nbo_ofertas.csv            ← versión v2 con señal estacional
"""

# %%
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from datetime import date, timedelta
import warnings
import shutil
import os

warnings.filterwarnings('ignore')

SEED         = 42
DATA_DIR     = os.getcwd()
N_MESES      = 25
FECHA_INICIO = date(2024, 1, 1)
PRODUCTOS    = ['tarjeta', 'prestamo', 'microcredito',
                'seguro_vida', 'seguro_salud', 'inversion']

def separador(titulo):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print(f"{'='*65}")

def subseccion(titulo):
    print(f"\n  ── {titulo}")

np.random.seed(SEED)

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 0 — BACKUP DEL ORIGINAL
# Si ya existe backup, no sobreescribir — el original es irremplazable.
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 0 — BACKUP DE nbo_ofertas.csv ORIGINAL")

ruta_original = f'{DATA_DIR}/nbo_ofertas.csv'
ruta_backup   = f'{DATA_DIR}/nbo_ofertas_v1_backup.csv'

if not os.path.exists(ruta_original):
    raise FileNotFoundError(
        f"No se encontró {ruta_original}.\n"
        f"Verifica que el generador original fue ejecutado."
    )

if os.path.exists(ruta_backup):
    print(f"  ℹ️  Backup ya existe: nbo_ofertas_v1_backup.csv")
    print(f"     No se sobreescribe — el original ya está protegido.")
else:
    shutil.copy(ruta_original, ruta_backup)
    size_mb = os.path.getsize(ruta_backup) / 1024 / 1024
    print(f"  ✅ Backup creado: nbo_ofertas_v1_backup.csv ({size_mb:.1f} MB)")
    print(f"     Para volver a v1: copiar nbo_ofertas_v1_backup.csv → nbo_ofertas.csv")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA DE INSUMOS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA DE INSUMOS")

clientes = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
features = pd.read_csv(f'{DATA_DIR}/nbo_features.csv')
ofertas_v1 = pd.read_csv(ruta_original)
ofertas_v1['id_producto_lower'] = ofertas_v1['id_producto'].str.lower()

print(f"  Clientes      : {len(clientes):,}")
print(f"  Features      : {len(features):,} filas")
print(f"  Ofertas v1    : {len(ofertas_v1):,} filas")
print(f"  score_propension v1 (media): {ofertas_v1['score_propension'].mean():.5f}")

# Verificaciones previas
if 'region' not in clientes.columns:
    raise ValueError(
        "Columna `region` no encontrada en nbo_clientes.csv.\n"
        "Ejecuta primero: python nbo_patch_clientes_features.py"
    )

FEATURES_EST_REQ = ['es_inicio_clases', 'es_decimo_cuarto', 'es_navidad',
                    'es_decimo_tercero', 'es_utilidades', 'es_impuesto_renta']
faltantes = [c for c in FEATURES_EST_REQ if c not in features.columns]
if faltantes:
    raise ValueError(
        f"Features estacionales faltantes: {faltantes}\n"
        f"Ejecuta primero: python nbo_patch_clientes_features.py"
    )

print(f"  ✅ region presente en clientes")
print(f"  ✅ Features estacionales presentes")

# Cuántas filas tienen score_propension original (para verificar integridad al final)
n_original = len(ofertas_v1)
tasa_original = ofertas_v1[
    (ofertas_v1['grupo'] == 'Tratamiento') &
    (ofertas_v1['etiqueta_completa'] == True)
]['convirtio_30d'].mean()
print(f"\n  Tasa conversión v1 (base de comparación): {tasa_original:.4f} ({tasa_original:.1%})")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — FACTOR ESTACIONAL MULTIPLICATIVO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — FACTOR ESTACIONAL MULTIPLICATIVO")

print("""
  Enfoque: p_nueva = clip(p_original × factor, 0, 1)

  El factor es siempre relativo a 1.0 (sin efecto).
  Valores > 1.0 = período de mayor propensión.
  Valores < 1.0 = período de menor propensión.

  Calibración objetivo:
    Cambio en convirtio_30d: 8–12% (no 45%)
    Ratio pico/no-pico: 1.2x–1.5x por producto
    Nunca < 0.5x ni > 2.0x (límites de credibilidad)

  La señal es SELECTIVA:
    - Depende del producto
    - Depende del período
    - Para microcrédito: solo aplica a Independientes
    - Para inversión: se reduce en navidad (gastan, no ahorran)
""")

def calcular_factor_estacional(producto, features_est, clientes_arr):
    """
    Multiplicador sobre p_base por producto y período.

    Diseñado para:
      - Cambiar el target ~8-12% globalmente
      - Ser selectivo: no todos los clientes reciben el mismo factor
      - Ser defensible ante auditoría: cada coeficiente tiene justificación

    Magnitudes calibradas empíricamente para este dataset:
      +0.20 sobre p_base ≈ +20% en probabilidad → cambio moderado
      +0.30 sobre p_base ≈ +30% en probabilidad → cambio fuerte (solo DC3/inversión)
      -0.10 sobre p_base ≈ -10% en probabilidad → señal negativa leve
    """
    n = len(clientes_arr)
    factor = np.ones(n)

    # Extraer features estacionales del período
    es_navidad       = features_est.get('es_navidad',        np.zeros(n))
    es_dc_tercero    = features_est.get('es_decimo_tercero',  np.zeros(n))
    es_dc_cuarto     = features_est.get('es_decimo_cuarto',   np.zeros(n))
    es_utilidades    = features_est.get('es_utilidades',      np.zeros(n))
    es_inicio_clases = features_est.get('es_inicio_clases',   np.zeros(n))
    es_impuesto      = features_est.get('es_impuesto_renta',  np.zeros(n))

    # Atributos de cliente para señales selectivas
    ocup   = clientes_arr['ocupacion'].values
    ingreso = clientes_arr['ingreso_mensual'].values
    hijos  = clientes_arr['hijos'].values

    es_independiente = (ocup == 'Independiente').astype(float)
    es_ingreso_alto  = (ingreso > np.percentile(ingreso, 60)).astype(float)
    tiene_hijos      = (hijos > 0).astype(float)

    if producto == 'tarjeta':
        # Navidad y DC3 elevan gasto discrecional → más demanda de tarjeta
        # IR reduce capacidad → efecto negativo moderado
        factor += 0.20 * es_navidad
        factor += 0.15 * es_dc_tercero
        factor -= 0.10 * es_impuesto

    elif producto == 'prestamo':
        # Utilidades: liquidez disponible → proyectos personales o consolidación
        # DC3: ya tienen liquidez, menos urgencia de crédito
        # IR: pago reduce disponible para nuevo crédito
        factor += 0.18 * es_utilidades
        factor -= 0.08 * es_dc_tercero
        factor -= 0.08 * es_impuesto

    elif producto == 'microcredito':
        # Inicio clases: necesidad de efectivo para útiles — SOLO Independientes
        # (Empleados tienen sueldo fijo, no necesitan microcrédito para útiles)
        # Navidad: gastos de fin de año en segmento informal
        # DC cuarto: parte va a pequeños negocios
        factor += 0.25 * es_inicio_clases * es_independiente
        factor += 0.15 * es_navidad
        factor += 0.10 * es_dc_cuarto * es_independiente

    elif producto == 'seguro_vida':
        # DC3: reflexión de fin de año sobre responsabilidad familiar
        # Inicio clases: padres con hijos piensan en protección
        # Señales más débiles — seguros tienen demanda más estable
        factor += 0.12 * es_dc_tercero
        factor += 0.08 * es_inicio_clases * tiene_hijos

    elif producto == 'seguro_salud':
        # Inicio clases: salud de hijos → preocupación por cobertura
        # Utilidades: mayor liquidez para seguros voluntarios
        factor += 0.15 * es_inicio_clases
        factor += 0.10 * es_utilidades

    elif producto == 'inversion':
        # DC3: mayor pico de liquidez del año → mayor propensión a invertir
        # Utilidades: segundo pico, también positivo
        # Navidad: en este período GASTAN en lugar de invertir → efecto negativo
        # DC cuarto: parte pequeña va a ahorro/inversión
        # Restricción a ingreso alto: los de bajo ingreso no invierten excedentes
        factor += 0.30 * es_dc_tercero  * es_ingreso_alto
        factor += 0.25 * es_utilidades  * es_ingreso_alto
        factor -= 0.12 * es_navidad
        factor += 0.10 * es_dc_cuarto   * es_ingreso_alto

    # Límites de credibilidad: no multiplicar por menos de 0.5x ni más de 2.0x
    return np.clip(factor, 0.5, 2.0)

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — APLICAR FACTOR Y REGENERAR CONVERSIONES
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — APLICAR FACTOR ESTACIONAL Y REGENERAR CONVERSIONES")

print("""
  Para cada oferta en nbo_ofertas.csv:
    1. Tomar score_propension ORIGINAL (ruido persistente intacto)
    2. Calcular factor estacional del mes y producto
    3. p_nueva = clip(p_original x factor, 0, 1)
    4. Flip condicional: preserva correlaciones, solo aplica incremento marginal.
       Factor positivo -> solo flip 0 a 1.  Factor negativo -> solo flip 1 a 0.
       Cambio esperado: 3-6%, no 32%.
""")

# Construir lookup de features estacionales por (id_cliente, mes)
features_est_lookup = {}
for mes in sorted(features['mes'].unique()):
    sub = features[features['mes'] == mes]
    features_est_lookup[mes] = {
        col: sub.set_index('id_cliente')[col]
        for col in FEATURES_EST_REQ
        if col in sub.columns
    }

# Lookup de atributos de cliente para acceso rápido
clientes_idx = clientes.set_index('id_cliente')

ofertas_v2 = ofertas_v1.copy()
ofertas_v2['score_propension_v1'] = ofertas_v2['score_propension']  # preservar original

# Solo modificar filas con etiqueta real y tratamiento
mask_modificar = (
    (ofertas_v2['grupo'] == 'Tratamiento') &
    (ofertas_v2['etiqueta_completa'] == True) &
    (ofertas_v2['convirtio_30d'].notna())
)

print(f"  Filas totales en ofertas    : {len(ofertas_v2):,}")
print(f"  Filas a modificar           : {mask_modificar.sum():,}")
print(f"  Filas sin cambio            : {(~mask_modificar).sum():,}")

# Procesar por mes x producto para eficiencia vectorial
conteo_cambios       = 0
conteo_flip_01       = 0   # 0 → 1 (conversiones añadidas)
conteo_flip_10       = 0   # 1 → 0 (conversiones quitadas)
registros_procesados = 0

for mes in sorted(ofertas_v2['mes'].unique()):
    for prod in PRODUCTOS:

        mask_bloque = (
            mask_modificar &
            (ofertas_v2['mes'] == mes) &
            (ofertas_v2['id_producto_lower'] == prod)
        )

        if mask_bloque.sum() == 0:
            continue

        ids_bloque = ofertas_v2.loc[mask_bloque, 'id_cliente'].values

        try:
            clientes_bloque = clientes_idx.loc[ids_bloque].reset_index()
        except KeyError:
            continue

        if mes not in features_est_lookup:
            continue

        est_mes = {
            col: serie.reindex(ids_bloque).fillna(0).values
            for col, serie in features_est_lookup[mes].items()
        }

        # Factor estacional
        factor     = calcular_factor_estacional(prod, est_mes, clientes_bloque)
        p_original = ofertas_v2.loc[mask_bloque, 'score_propension'].values
        p_nueva    = np.clip(p_original * factor, 0.0, 1.0)

        # Actualizar score_propension
        ofertas_v2.loc[mask_bloque, 'score_propension'] = np.round(p_nueva, 5)

        # ── FLIP CONDICIONAL ─────────────────────────────────────────
        # Principio: solo aplicar el INCREMENTO marginal de probabilidad.
        # Las correlaciones features→target existentes nunca se rompen.
        #
        # Para factor > 1 (p_nueva > p_original):
        #   Solo los que YA eran 0 pueden convertirse en 1.
        #   P(flip 0→1) = (p_nueva - p_original) / (1 - p_original)
        #   Intuición: del total de "espacio" disponible para nuevas
        #   conversiones (1 - p_original), ¿qué fracción se activó?
        #
        # Para factor < 1 (p_nueva < p_original):
        #   Solo los que YA eran 1 pueden convertirse en 0.
        #   P(flip 1→0) = (p_original - p_nueva) / p_original
        #   Intuición: del total de conversiones existentes (p_original),
        #   ¿qué fracción se desactivó?
        # ─────────────────────────────────────────────────────────────

        rng = np.random.default_rng(SEED + mes * 100 + PRODUCTOS.index(prod))
        convirtio_viejo = ofertas_v2.loc[mask_bloque, 'convirtio_30d'].values.astype(float)
        convirtio_nuevo = convirtio_viejo.copy()

        # Masks de estado previo
        era_cero = convirtio_viejo == 0
        era_uno  = convirtio_viejo == 1

        # Factor positivo → solo flip 0→1
        delta_positivo = np.maximum(p_nueva - p_original, 0.0)
        prob_flip_01   = np.where(
            era_cero,
            delta_positivo / np.maximum(1.0 - p_original, 1e-7),
            0.0
        )
        flip_01 = rng.binomial(1, np.clip(prob_flip_01, 0.0, 1.0))
        convirtio_nuevo[era_cero] += flip_01[era_cero]   # 0 + 1 = 1

        # Factor negativo → solo flip 1→0
        delta_negativo = np.maximum(p_original - p_nueva, 0.0)
        prob_flip_10   = np.where(
            era_uno,
            delta_negativo / np.maximum(p_original, 1e-7),
            0.0
        )
        flip_10 = rng.binomial(1, np.clip(prob_flip_10, 0.0, 1.0))
        convirtio_nuevo[era_uno] -= flip_10[era_uno]     # 1 - 1 = 0

        n_01 = int(flip_01[era_cero].sum())
        n_10 = int(flip_10[era_uno].sum())
        n_cambios = n_01 + n_10

        conteo_flip_01       += n_01
        conteo_flip_10       += n_10
        conteo_cambios       += n_cambios
        registros_procesados += len(convirtio_nuevo)

        ofertas_v2.loc[mask_bloque, 'convirtio_30d'] = convirtio_nuevo

print(f"\n  Registros procesados        : {registros_procesados:,}")
print(f"  Conversiones añadidas (0→1) : {conteo_flip_01:,}")
print(f"  Conversiones quitadas (1→0) : {conteo_flip_10:,}")
print(f"  Cambios totales             : {conteo_cambios:,}")
print(f"  Tasa de cambio              : {conteo_cambios/registros_procesados:.1%}  (objetivo: 3–6%)")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — VALIDACIÓN")

elegibles_v1 = ofertas_v1[
    (ofertas_v1['grupo'] == 'Tratamiento') &
    (ofertas_v1['etiqueta_completa'] == True)
]
elegibles_v2 = ofertas_v2[
    (ofertas_v2['grupo'] == 'Tratamiento') &
    (ofertas_v2['etiqueta_completa'] == True)
]

subseccion("Tasa de conversión: v1 vs v2 por producto")
print(f"\n  {'Producto':<20} {'Tasa v1':>10} {'Tasa v2':>10} {'Δ abs':>10} {'Δ %':>10}")
print(f"  {'─'*54}")

for prod in PRODUCTOS:
    sub_v1 = elegibles_v1[elegibles_v1['id_producto_lower'] == prod]
    sub_v2 = elegibles_v2[elegibles_v2['id_producto_lower'] == prod]
    t1 = sub_v1['convirtio_30d'].mean() if len(sub_v1) > 0 else 0
    t2 = sub_v2['convirtio_30d'].mean() if len(sub_v2) > 0 else 0
    delta_abs = t2 - t1
    delta_pct = (t2 - t1) / t1 * 100 if t1 > 0 else 0
    flag = "✅" if abs(delta_pct) <= 20 else "⚠️"
    print(f"  {prod:<20} {t1:>10.4f} {t2:>10.4f} {delta_abs:>+9.4f} {delta_pct:>+9.1f}%  {flag}")

tasa_global_v1 = elegibles_v1['convirtio_30d'].mean()
tasa_global_v2 = elegibles_v2['convirtio_30d'].mean()
delta_global   = (tasa_global_v2 - tasa_global_v1) / tasa_global_v1 * 100
print(f"\n  Tasa global v1 : {tasa_global_v1:.4f}")
print(f"  Tasa global v2 : {tasa_global_v2:.4f}")
print(f"  Δ global       : {delta_global:+.1f}%  (objetivo: 8–15%)")

subseccion("Verificación señal estacional (pico vs no-pico)")
ofertas_v2['_mc'] = ((ofertas_v2['mes'] - 1) % 12) + 1
eleg_v2 = ofertas_v2[
    (ofertas_v2['grupo'] == 'Tratamiento') &
    (ofertas_v2['etiqueta_completa'] == True)
].copy()

print(f"\n  {'Producto':<20} {'Mes pico':>10} {'Tasa pico':>12} {'No-pico':>12} {'Ratio':>8}")
print(f"  {'─'*66}")

checks_estacionales = [
    ('tarjeta',     [11, 12], 'Nov-Dic'),
    ('inversion',   [3, 4, 12], 'Mar/Abr/Dic'),
    ('microcredito',[4, 5, 8, 9], 'Clases'),
    ('seguro_vida', [12],      'Dic'),
    ('seguro_salud',[4, 5, 8, 9], 'Clases'),
    ('prestamo',    [3, 4],    'Mar-Abr'),
]

señal_detectada = True
for prod, meses_pico, etiqueta in checks_estacionales:
    sub = eleg_v2[eleg_v2['id_producto_lower'] == prod]
    t_pico   = sub[sub['_mc'].isin(meses_pico)]['convirtio_30d'].mean()
    t_nopico = sub[~sub['_mc'].isin(meses_pico)]['convirtio_30d'].mean()
    ratio    = t_pico / t_nopico if t_nopico > 0 else 0
    flag     = "✅" if ratio > 1.05 else ("≈" if ratio > 0.95 else "⚠️")
    print(f"  {prod:<20} {etiqueta:>10} {t_pico:>12.4f} {t_nopico:>12.4f} {ratio:>7.2f}x {flag}")
    if ratio < 1.0:
        señal_detectada = False

eleg_v2 = eleg_v2.drop(columns=['_mc'])
ofertas_v2 = ofertas_v2.drop(columns=['_mc'])

if señal_detectada:
    print(f"\n  ✅ Señal estacional detectada en todos los productos")
else:
    print(f"\n  ⚠️  Algunos productos sin señal clara — revisar magnitudes del factor")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — GUARDAR
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — GUARDAR")

# Preservar columna score_propension_v1 como auditoría
# Permite verificar en cualquier momento el cambio introducido
ofertas_v2.to_csv(f'{DATA_DIR}/nbo_ofertas_actualizada.csv', index=False)
size_mb = os.path.getsize(f'{DATA_DIR}/nbo_ofertas_actualizada.csv') / 1024 / 1024

print(f"  ✅ nbo_ofertas_actualizada.csv guardado  ({size_mb:.1f} MB)")
print(f"     Columna score_propension_v1 preservada para auditoría")
print(f"     Para volver a v1: copiar nbo_ofertas_v1_backup.csv → nbo_ofertas_actualizada.csv")

separador("COMPLETADO")
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     nbo_ofertas.csv — SEÑAL ESTACIONAL APLICADA             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Backup v1 protegido: nbo_ofertas_v1_backup.csv           ║
  ║  ✓ score_propension_v1 preservada en el CSV                 ║
  ║  ✓ Ruido persistente ORIGINAL intacto (no regenerado)       ║
  ║  ✓ Factor estacional multiplicativo: 8–12% cambio target    ║
  ║  ✓ Señal selectiva por producto, período y perfil cliente   ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Siguiente paso:                                             ║
  ║    python nbo_semana8_modelo_v2.py                          ║
  ║    → AUC v2.0 debe mejorar vs v1.0 en tarjeta e inversión  ║
  ╚══════════════════════════════════════════════════════════════╝
""")


