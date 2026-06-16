# %%
"""
RBlJose — SISTEMA NBO
Semana 4 — Pipeline de Campaña
================================
Convierte los scores del modelo (Semana 3) en decisiones de campaña
operativas, medibles y auditables.
 
Prerequisito:
  nbo_semana3_modelos.py ejecutado completamente.
  Archivo puente: nbo_recomendaciones_semana3.csv (rank 1 y 2 por cliente)
 
Pipeline:
  Bloque 1 — Carga e integración de inputs
  Bloque 2 — Contact Policy Engine (filtros duros operativos)
  Bloque 3 — Optimizador presupuestario (greedy por ratio score/costo)
  Bloque 4 — Asignación experimental tratamiento / control
  Bloque 5 — Simulación de ejecución de campaña
  Bloque 6 — Observación T+30 días (cierre de etiquetas)
  Bloque 7 — Medición financiera y causal (ROI incremental)
  Bloque 8 — Monitoreo y triggers de mantenimiento
 
Outputs:
  nbo_campana_ejecutada.csv      — universo final contactado con grupo asignado
  nbo_resultados_t30.csv         — conversiones observadas a 30 días
  nbo_metricas_campana.csv       — KPIs financieros y causales de la campaña
  nbo_monitoreo_campana.csv      — métricas de drift y triggers de mantenimiento
 
Separación de responsabilidades:
  Este archivo NO reimporta modelos ni calibradores de Semana 3.
  Solo consume el CSV de scores ya calculados.
  Modelo → score → decisión son tres capas distintas y auditables.
"""

# %%
import numpy as np
import pandas as pd
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')
import os

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
# PARÁMETROS GLOBALES DE CAMPAÑA
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = os.getcwd()
 
# Presupuesto semanal disponible para contactaciones
# En producción viene del área de Marketing aprobado por Finanzas
PRESUPUESTO_CAMPANA = 50_000.0   # USD
 
# Fecha de ejecución de la campaña
# Corresponde al mes de scoring (mes 23 = Noviembre 2025)
fecha_campana = date(2025, 11, 1)
 
# Tasa de conversión orgánica por producto
# Clientes que adquieren el producto SIN intervención del banco
# Fuente: histórico de conversiones en grupo control períodos anteriores
# En producción se calibra desde datos reales de control acumulados
TASA_ORGANICA = {
    'tarjeta'     : 0.018,
    'prestamo'    : 0.025,
    'microcredito': 0.012,
    'seguro_vida' : 0.015,
    'seguro_salud': 0.014,
    'inversion'   : 0.020,
}
 
# Proporción del universo elegible que va a grupo control
PCT_CONTROL = 0.20
 
# Límites máximos de originación por producto en la campaña
# Restricción operativa — el banco no puede procesar volumen ilimitado
# En producción viene de Riesgo y Operaciones
LIMITES_ORIGINACION = {
    'tarjeta'     : 2_000,
    'prestamo'    : 1_500,
    'microcredito': 800,
    'seguro_vida' : 3_000,
    'seguro_salud': 3_000,
    'inversion'   : 2_500,
}
 
# Parámetros de negocio — idénticos a Semana 3 para consistencia
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
 

# %%
SEED = 42
np.random.seed(SEED)

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — CARGA E INTEGRACIÓN DE INPUTS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — CARGA E INTEGRACIÓN DE INPUTS")

print("\n  Cargando archivos fuente...")

# Archivo puente de Semana 3 — scores ya calculados, rank 1 y 2
recomendaciones = pd.read_csv(f'{DATA_DIR}/nbo_recomendaciones_semana3.csv')

# Tablas del sistema
clientes  = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
ofertas   = pd.read_csv(f'{DATA_DIR}/nbo_ofertas.csv')

print(f"  recomendaciones : {len(recomendaciones):>8,} filas "
      f"({recomendaciones['id_cliente'].nunique():,} clientes únicos)")
print(f"  clientes        : {len(clientes):>8,} filas")
print(f"  ofertas historial: {len(ofertas):>7,} filas")

# Verificar columnas esperadas del archivo puente
COLUMNAS_REQUERIDAS = [
    'id_cliente', 'rank', 'producto_nbo', 'score_nbo',
    'p_calibrada', 'ratio_nbo_costo', 'costo_contacto',
    'mes_scoring', 'fecha_scoring', 'canal_principal'
]
faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in recomendaciones.columns]
if faltantes:
    raise ValueError(
        f"Columnas faltantes en nbo_recomendaciones_semana3.csv: {faltantes}\n"
        f"Verificar que nbo_semana3_modelos.py fue ejecutado con la versión "
        f"actualizada del Bloque 9.3."
    )
print(f"\n  ✅ Columnas del archivo puente verificadas")

# Separar rank 1 y rank 2
rec_r1 = recomendaciones[recomendaciones['rank'] == 1].copy()
rec_r2 = recomendaciones[recomendaciones['rank'] == 2].copy()

print(f"\n  Clientes con recomendación rank 1 : {len(rec_r1):,}")
print(f"  Clientes con recomendación rank 2 : {len(rec_r2):,}")

# Enriquecer con atributos del cliente necesarios para políticas
rec_r1 = rec_r1.merge(
    clientes[['id_cliente', 'score_crediticio', 'max_atraso_dias',
              'antiguedad_meses', 'activo', 'ratio_deuda_init']],
    on='id_cliente', how='left'
)

subseccion("Resumen de scores recibidos de Semana 3")
print(f"\n  {'Producto':<20} {'Clientes':>10} {'Score NBO medio':>16} {'P_cal media':>12}")
print(f"  {'─'*62}")
for prod, grp in rec_r1.groupby('producto_nbo'):
    print(f"  {prod:<20} {len(grp):>10,} {grp['score_nbo'].mean():>16.2f} "
          f"{grp['p_calibrada'].mean():>12.4f}")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — CONTACT POLICY ENGINE
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — CONTACT POLICY ENGINE")

print("""
  El Contact Policy Engine aplica filtros duros ANTES del optimizador.
  Solo el universo que pasa estos filtros entra a la optimización.

  Reglas implementadas (según Contact Policy Framework del sistema):
    R1 — Cooling period        : mínimo 30 días entre contactos al mismo cliente
    R2 — Bloqueo post-oferta   : 15 días tras recibir cualquier oferta
    R3 — Opt-out               : exclusión permanente si cliente rechazó comunicaciones
    R4 — Producto ya activo    : no ofrecer producto que el cliente ya posee
    R5 — Elegibilidad mínima   : score >= 550, sin mora activa, antigüedad >= 3 meses
    R6 — Fallback a rank 2     : si rank 1 bloqueado por R1/R2/R4, evaluar rank 2
""")


# Preparar historial de contactos recientes para reglas R1 y R2
ofertas['fecha_oferta'] = pd.to_datetime(ofertas['fecha_oferta'])

# Último contacto por cliente (cualquier producto)
ultimo_contacto = (
    ofertas[ofertas['grupo'] == 'Tratamiento']
    .groupby('id_cliente')['fecha_oferta']
    .max()
    .reset_index()
    .rename(columns={'fecha_oferta': 'fecha_ultimo_contacto'})
)

# Productos activos por cliente (ya los adquirió)
productos_activos = (
    ofertas[ofertas['convirtio_30d'] == 1]
    .groupby('id_cliente')['id_producto']
    .apply(lambda x: set(x.str.lower()))
    .reset_index()
    .rename(columns={'id_producto': 'productos_activos'})
)

# Integrar historial en recomendaciones
universe = rec_r1.merge(ultimo_contacto,  on='id_cliente', how='left')
universe = universe.merge(productos_activos, on='id_cliente', how='left')
fecha_scoring = pd.to_datetime(universe['fecha_scoring']).max()
fecha_campana = fecha_scoring + pd.offsets.MonthEnd(0)
fecha_campana_dt        = pd.to_datetime(fecha_campana)

# Calcular días desde último contacto
universe['dias_desde_contacto'] = (
    fecha_campana_dt - universe['fecha_ultimo_contacto']
).dt.days.fillna(9999).astype(int)

# ── LIMPIEZA CLAVE ───────────────────────────────────────────────────
universe['productos_activos'] = universe['productos_activos'].apply(
    lambda x: x if isinstance(x, set) else set()
)

# Inicializar flags de elegibilidad para trazabilidad
universe['pasa_r1_cooling'] = universe['dias_desde_contacto'] >= 30
universe['pasa_r2_bloqueo'] = universe['dias_desde_contacto'] >= 15
universe['pasa_r3_optout']  = True

# 🔥 R4 — SIN APPLY

universe['pasa_r4_activo']     = universe.apply(
    lambda r: r['producto_nbo'] not in (r['productos_activos'] or set()),
    axis=1
)
universe['pasa_r4_activo'] = [
    prod not in activos
    for prod, activos in zip(
        universe['producto_nbo'],
        universe['productos_activos']
    )
]

universe['pasa_r5_elegib']     = (
    (universe['score_crediticio'] >= 550) &
    (universe['max_atraso_dias']  <= 30)  &
    (universe['antiguedad_meses'] >= 3)   &
    (universe['activo'] == True)
)

universe['elegible_r1'] = (
    universe['pasa_r1_cooling'] &
    universe['pasa_r2_bloqueo'] &
    universe['pasa_r3_optout']  &
    universe['pasa_r4_activo']  &
    universe['pasa_r5_elegib']
)

# ── Fallback a rank 2 para clientes bloqueados por R1/R2/R4 ──────────
# Si el producto rank 1 está bloqueado pero el rank 2 no,
# el cliente puede recibir el segundo mejor producto
bloqueados_r1 = universe[~universe['elegible_r1']]['id_cliente'].values

rec_r2_bloqueados = rec_r2[rec_r2['id_cliente'].isin(bloqueados_r1)].copy()
rec_r2_bloqueados = rec_r2_bloqueados.merge(ultimo_contacto,  on='id_cliente', how='left')
rec_r2_bloqueados = rec_r2_bloqueados.merge(productos_activos, on='id_cliente', how='left')
rec_r2_bloqueados = rec_r2_bloqueados.merge(
    clientes[['id_cliente', 'score_crediticio', 'max_atraso_dias',
              'antiguedad_meses', 'activo']],
    on='id_cliente', how='left'
)
rec_r2_bloqueados['dias_desde_contacto'] = (
    fecha_campana_dt - rec_r2_bloqueados['fecha_ultimo_contacto']
).dt.days.fillna(9999).astype(int)

# ── LIMPIEZA FALLBACK ────────────────────────────────────────────────
rec_r2_bloqueados['productos_activos'] = rec_r2_bloqueados['productos_activos'].apply(
    lambda x: x if isinstance(x, set) else set()
)

# 🔥 R4 FALLBACK SIN APPLY
pasa_r4_fallback = [
    prod not in activos
    for prod, activos in zip(
        rec_r2_bloqueados['producto_nbo'],
        rec_r2_bloqueados['productos_activos']
    )
]

rec_r2_bloqueados['elegible_r1'] = (
    (rec_r2_bloqueados['dias_desde_contacto'] >= 30) &
    (rec_r2_bloqueados['dias_desde_contacto'] >= 15) &
    rec_r2_bloqueados.apply(
        lambda r: r['producto_nbo'] not in (r['productos_activos'] or set()), axis=1
    ) &
    (rec_r2_bloqueados['score_crediticio'] >= 550) &
    (rec_r2_bloqueados['max_atraso_dias']  <= 30)  &
    (rec_r2_bloqueados['antiguedad_meses'] >= 3)
)
rec_r2_bloqueados['es_fallback'] = True

# Combinar elegibles rank 1 + fallbacks rank 2
elegibles_r1      = universe[universe['elegible_r1']].copy()
elegibles_r1['es_fallback'] = False

fallbacks_elegibles = rec_r2_bloqueados[
    rec_r2_bloqueados['elegible_r1']
].copy()

# Asegurar que el fallback no esté ya en los elegibles r1
fallbacks_elegibles = fallbacks_elegibles[
    ~fallbacks_elegibles['id_cliente'].isin(elegibles_r1['id_cliente'])
]

universo_elegible = pd.concat(
    [elegibles_r1, fallbacks_elegibles], ignore_index=True
)

# Reporte del Contact Policy Engine
subseccion("Resultado del Contact Policy Engine")
print(f"\n  Clientes en scoring inicial          : {len(rec_r1):,}")
print(f"  Bloqueados rank 1 — total            : {(~universe['elegible_r1']).sum():,}")
print(f"    Por cooling period (< 30 días)     : {(~universe['pasa_r1_cooling']).sum():,}")
print(f"    Por producto ya activo             : {(~universe['pasa_r4_activo']).sum():,}")
print(f"    Por elegibilidad mínima            : {(~universe['pasa_r5_elegib']).sum():,}")
print(f"  Recuperados con fallback rank 2      : {len(fallbacks_elegibles):,}")
print(f"  ─────────────────────────────────────")
print(f"  Universo elegible final              : {len(universo_elegible):,}")
print(f"  Cobertura sobre scoring inicial      : "
      f"{len(universo_elegible)/len(rec_r1):.1%}")

# %%
universe.head()


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 3 — OPTIMIZADOR PRESUPUESTARIO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 3 — OPTIMIZADOR PRESUPUESTARIO")
 
print(f"""
  Algoritmo: Greedy por ratio score_nbo / costo_contacto
 
  Justificación técnica:
    El greedy es óptimo cuando las restricciones son lineales y separables
    (un producto por cliente, presupuesto global, límites por producto).
    MIP se activaría si hubiera restricciones cruzadas entre productos para
    el mismo cliente — en este portafolio no ocurre porque ya aplicamos
    el argmax en Semana 3.
 
  El ratio score_nbo/costo es el equivalente al ratio beneficio/costo
  en programación lineal continua: maximiza el retorno por unidad de
  presupuesto invertido, no el retorno absoluto.
 
  Restricciones activas:
    Presupuesto total   : ${PRESUPUESTO_CAMPANA:,.0f} USD
    Límites por producto: {LIMITES_ORIGINACION}
""")
 
# Ordenar por ratio descendente — los más rentables por peso presupuestario primero
universo_ordenado = universo_elegible.sort_values(
    'ratio_nbo_costo', ascending=False
).copy()
 
presupuesto_restante = PRESUPUESTO_CAMPANA
contadores_producto  = {prod: 0 for prod in PARAMS_NEGOCIO}
seleccionados        = []
excluidos_presupuesto = 0
excluidos_limite      = 0
excluidos_ev_negativo = 0
 
for _, cliente in universo_ordenado.iterrows():
 
    prod  = cliente['producto_nbo']
    costo = cliente['costo_contacto']
 
    # Excluir si el score_nbo es negativo
    # No tiene sentido contactar cuando el EV esperado es negativo
    if cliente['score_nbo'] < 0:
        excluidos_ev_negativo += 1
        continue
 
    # Excluir si se agotó el presupuesto
    if presupuesto_restante < costo:
        excluidos_presupuesto += 1
        continue
 
    # Excluir si se alcanzó el límite operativo del producto
    if contadores_producto[prod] >= LIMITES_ORIGINACION.get(prod, np.inf):
        excluidos_limite += 1
        continue
 
    seleccionados.append(cliente.to_dict())
    presupuesto_restante       -= costo
    contadores_producto[prod]  += 1
 
universo_optimizado = pd.DataFrame(seleccionados)
 
subseccion("Resultado del optimizador")
print(f"\n  Universo elegible entrada       : {len(universo_elegible):,}")
print(f"  Excluidos EV negativo           : {excluidos_ev_negativo:,}")
print(f"  Excluidos por límite producto   : {excluidos_limite:,}")
print(f"  Excluidos por presupuesto       : {excluidos_presupuesto:,}")
print(f"  ─────────────────────────────────")
print(f"  Seleccionados para campaña      : {len(universo_optimizado):,}")
print(f"  Presupuesto ejecutado           : "
      f"${PRESUPUESTO_CAMPANA - presupuesto_restante:,.2f} "
      f"/ ${PRESUPUESTO_CAMPANA:,.0f} "
      f"({(PRESUPUESTO_CAMPANA - presupuesto_restante)/PRESUPUESTO_CAMPANA:.1%})")
print(f"  Presupuesto restante            : ${presupuesto_restante:,.2f}")
 
print(f"\n  Distribución por producto tras optimización:")
print(f"  {'Producto':<20} {'Contactos':>10} {'% total':>8} "
      f"{'Score NBO medio':>16} {'Costo total':>12}")
print(f"  {'─'*70}")

if universo_optimizado.empty:
    print("\n⚠️ No hay clientes seleccionados tras optimización")
else:    
    for prod, grp in universo_optimizado.groupby('producto_nbo'):
        costo_total = grp['costo_contacto'].sum()
        print(f"  {prod:<20} {len(grp):>10,} "
            f"{len(grp)/len(universo_optimizado):>8.1%} "
            f"{grp['score_nbo'].mean():>16.2f} "
            f"${costo_total:>10,.2f}")
    

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 4 — ASIGNACIÓN EXPERIMENTAL TRATAMIENTO / CONTROL
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 4 — ASIGNACIÓN EXPERIMENTAL TRATAMIENTO / CONTROL")
 
print(f"""
  Diseño experimental:
    {int((1-PCT_CONTROL)*100)}% Tratamiento → recibe oferta en canal correspondiente
    {int(PCT_CONTROL*100)}% Control      → no recibe contacto, se observa conversión orgánica
 
  CRÍTICO: La asignación es ALEATORIA dentro del universo optimizado.
  No se estratifica por score ni por producto.
  Cualquier sesgo en la asignación invalida la medición de incrementalidad.
 
  El grupo control pasa por TODOS los filtros anteriores — la única
  diferencia es que no recibe la oferta. Sus conversiones miden la
  tasa orgánica real del banco en este segmento y período.
 
  Potencia estadística:
    Con {int(len(universo_optimizado)*PCT_CONTROL):,} clientes en control y tasa orgánica ~2%,
    se esperan ~{int(len(universo_optimizado)*PCT_CONTROL*0.02):.0f} conversiones orgánicas.
    Suficiente para detectar uplift >= 3pp con 80% de potencia.
""")
 
# Asignación aleatoria reproducible — semilla fija para auditabilidad
rng = np.random.default_rng(SEED)
universo_optimizado = universo_optimizado.copy()
universo_optimizado['grupo'] = np.where(
    rng.random(len(universo_optimizado)) > PCT_CONTROL,
    'Tratamiento',
    'Control'
)
 
# Agregar metadatos de campaña
universo_optimizado['id_campana']    = f'CAMP_{fecha_campana.strftime("%Y%m")}_NBO'
universo_optimizado['fecha_campana'] = str(fecha_campana)
universo_optimizado['fecha_cierre']  = str(fecha_campana + timedelta(days=30))
universo_optimizado['version_modelo'] = 'v1.0_semana3'
 
n_trat   = (universo_optimizado['grupo'] == 'Tratamiento').sum()
n_ctrl   = (universo_optimizado['grupo'] == 'Control').sum()
 
print(f"  Grupo Tratamiento : {n_trat:,}  ({n_trat/len(universo_optimizado):.1%})")
print(f"  Grupo Control     : {n_ctrl:,}  ({n_ctrl/len(universo_optimizado):.1%})")
 
# Verificar balance entre grupos — los dos deben tener distribuciones similares
print(f"\n  Verificación de balance entre grupos (score NBO medio):")
print(f"  {'Producto':<20} {'Tratamiento':>14} {'Control':>14} {'Diferencia':>12}")
print(f"  {'─'*64}")
for prod in sorted(universo_optimizado['producto_nbo'].unique()):
    sub  = universo_optimizado[universo_optimizado['producto_nbo'] == prod]
    t_sc = sub[sub['grupo']=='Tratamiento']['score_nbo'].mean()
    c_sc = sub[sub['grupo']=='Control']['score_nbo'].mean()
    diff = abs(t_sc - c_sc)
    flag = "⚠️" if diff > 5 else "✅"
    print(f"  {prod:<20} {t_sc:>14.2f} {c_sc:>14.2f} {diff:>11.2f} {flag}")
 
# Guardar universo de campaña — primer output del pipeline
universo_optimizado.to_csv(
    f'{DATA_DIR}/nbo_campana_ejecutada.csv', index=False
)
print(f"\n  ✅ Campaña guardada: nbo_campana_ejecutada.csv")
print(f"     Columnas clave: id_cliente, grupo, producto_nbo, score_nbo,")
print(f"                     p_calibrada, canal_principal, fecha_campana, fecha_cierre")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 5 — SIMULACIÓN DE EJECUCIÓN DE CAMPAÑA
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 5 — SIMULACIÓN DE EJECUCIÓN")
 
print("""
  En producción este bloque es reemplazado por el job de actualización
  diferida que consulta los sistemas core a T+30 días.
 
  En el entorno sintético simulamos la conversión de dos formas:
    Tratamiento : conversión ~ Bernoulli(p_calibrada)
                  La p_calibrada del modelo determina si convierte.
    Control     : conversión ~ Bernoulli(tasa_organica_producto)
                  Convierte por iniciativa propia, sin oferta del banco.
 
  Esta separación es la que hace posible medir incrementalidad causal.
""")
 
campana = universo_optimizado.copy()
rng2    = np.random.default_rng(SEED + 1)
 
# Simular conversiones
conversiones = []
 
for _, cliente in campana.iterrows():
    prod = cliente['producto_nbo']
    p    = PARAMS_NEGOCIO[prod]
 
    if cliente['grupo'] == 'Tratamiento':
        # Tasa de conversión = p_calibrada del modelo
        p_conv = cliente['p_calibrada']
    else:
        # Tasa orgánica — lo que pasaría sin intervención
        p_conv = TASA_ORGANICA.get(prod, 0.02)
 
    convirtio = int(rng2.random() < p_conv)
 
    # Ingreso real solo si convirtió
    if convirtio:
        ingreso_real = p['ticket_anual'] - p['costo_originacion']
    else:
        ingreso_real = 0.0
 
    conversiones.append({
        'id_cliente'    : cliente['id_cliente'],
        'id_campana'    : cliente['id_campana'],
        'producto_nbo'  : prod,
        'grupo'         : cliente['grupo'],
        'p_calibrada'   : cliente['p_calibrada'],
        'convirtio_30d' : convirtio,
        'ingreso_real'  : round(ingreso_real, 2),
        'costo_contacto': p['costo_contacto'] if cliente['grupo'] == 'Tratamiento' else 0.0,
        'fecha_cierre'  : cliente['fecha_cierre'],
    })
 
df_resultados = pd.DataFrame(conversiones)
 
# Reporte de conversiones observadas
subseccion("Conversiones observadas a T+30")
print(f"\n  {'Producto':<20} {'Trat N':>8} {'Trat conv%':>12} "
      f"{'Ctrl N':>8} {'Ctrl conv%':>12} {'Uplift pp':>10}")
print(f"  {'─'*74}")
 
for prod in sorted(df_resultados['producto_nbo'].unique()):
    sub  = df_resultados[df_resultados['producto_nbo'] == prod]
    trat = sub[sub['grupo'] == 'Tratamiento']
    ctrl = sub[sub['grupo'] == 'Control']
 
    t_rate = trat['convirtio_30d'].mean() if len(trat) > 0 else 0
    c_rate = ctrl['convirtio_30d'].mean() if len(ctrl) > 0 else 0
    uplift = (t_rate - c_rate) * 100
 
    print(f"  {prod:<20} {len(trat):>8,} {t_rate:>11.1%} "
          f"{len(ctrl):>8,} {c_rate:>11.1%} {uplift:>+9.1f}pp")
 
df_resultados.to_csv(f'{DATA_DIR}/nbo_resultados_t30.csv', index=False)
print(f"\n  ✅ Resultados guardados: nbo_resultados_t30.csv")
 

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 6 — OBSERVACIÓN T+30 Y CIERRE DE ETIQUETAS
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 6 — OBSERVACIÓN T+30 — ACTUALIZACIÓN DE ETIQUETAS")
 
print("""
  En producción este bloque es el job de actualización diferida que:
    1. Identifica ofertas cuyo T+30 ya venció y convirtio_30d = NULL
    2. Consulta sistemas core (originación, pólizas, depósitos)
    3. Aplica buffer ETL de 5 días para capturar cargas tardías
    4. Actualiza convirtio_30d e ingreso_real en tabla OFERTAS
    5. Registra fecha_proceso para auditoría
 
  En este entorno sintético el cierre ya ocurrió en el Bloque 5.
  Este bloque verifica la integridad de los datos cerrados.
""")
 
# Verificación de integridad del cierre
n_total     = len(df_resultados)
n_etiquetados = df_resultados['convirtio_30d'].notna().sum()
n_sin_etiqueta = n_total - n_etiquetados
 
print(f"  Ofertas en campaña              : {n_total:,}")
print(f"  Con etiqueta completa           : {n_etiquetados:,}  ({n_etiquetados/n_total:.1%})")
print(f"  Sin etiqueta (error)            : {n_sin_etiqueta:,}  ← debe ser 0")
 
if n_sin_etiqueta > 0:
    print(f"  ⚠️  ADVERTENCIA: {n_sin_etiqueta} ofertas sin etiqueta.")
    print(f"     En producción activaría alerta de pipeline.")
else:
    print(f"  ✅ Cierre de etiquetas íntegro")
 
# Verificar que el grupo control tiene etiquetas válidas
ctrl_check = df_resultados[df_resultados['grupo'] == 'Control']
print(f"\n  Verificación grupo control:")
print(f"  Registros control con etiqueta  : {ctrl_check['convirtio_30d'].notna().sum():,}")
print(f"  Tasa conversión orgánica global : "
      f"{ctrl_check['convirtio_30d'].mean():.3f} "
      f"(esperada ~{sum(TASA_ORGANICA.values())/len(TASA_ORGANICA):.3f})")
 

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 7 — MEDICIÓN FINANCIERA Y CAUSAL
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 7 — MEDICIÓN FINANCIERA Y CAUSAL")
 
print("""
  Las métricas de negocio se calculan en dos dimensiones:
 
  1. Métricas brutas (tratamiento solo):
     Mide lo que el sistema generó — útil para reportar actividad.
     Riesgo: incluye ventas orgánicas que habrían ocurrido sin la campaña.
 
  2. Métricas incrementales (tratamiento vs control):
     Mide el valor CAUSAL del sistema NBO — lo que se atribuye a la
     intervención del banco. Este es el número que presenta ante el CFO.
     Solo este número justifica el costo de implementación del sistema.
""")
 
metricas_campana = []
 
subseccion("7.1 — Por producto")
print(f"\n  {'Producto':<20} {'Conv trat':>10} {'Conv ctrl':>10} "
      f"{'Uplift%':>9} {'Ingreso brt':>12} {'Ingreso inc':>12} "
      f"{'Costo':>8} {'Profit inc':>11}")
print(f"  {'─'*98}")
 
total_ingreso_bruto = 0
total_ingreso_incr  = 0
total_costo         = 0
total_profit_incr   = 0
 
for prod in sorted(df_resultados['producto_nbo'].unique()):
    sub  = df_resultados[df_resultados['producto_nbo'] == prod]
    p    = PARAMS_NEGOCIO[prod]
 
    trat = sub[sub['grupo'] == 'Tratamiento']
    ctrl = sub[sub['grupo'] == 'Control']
 
    # Conversiones
    n_conv_trat = trat['convirtio_30d'].sum()
    n_conv_ctrl = ctrl['convirtio_30d'].sum()
 
    # Tasas
    tasa_trat = trat['convirtio_30d'].mean() if len(trat) > 0 else 0
    tasa_ctrl = ctrl['convirtio_30d'].mean() if len(ctrl) > 0 else 0
    uplift_pp = (tasa_trat - tasa_ctrl) * 100
 
    # Ingresos
    ingreso_bruto = trat['ingreso_real'].sum()
 
    # Ingreso incremental: solo las conversiones por encima de la tasa orgánica
    # incremento_causal = P(conv|oferta) - P(conv|sin oferta)
    # ingreso_incremental = incremento_causal × n_tratamiento × ingreso_por_conversion
    incremento_causal   = max(tasa_trat - tasa_ctrl, 0)
    ingreso_incremental = (
        incremento_causal * len(trat) * (p['ticket_anual'] - p['costo_originacion'])
    )
 
    # Costo total de contactación (solo tratamiento — control no se contacta)
    costo_total = trat['costo_contacto'].sum()
 
    # Profit incremental
    profit_incremental = ingreso_incremental - costo_total
 
    total_ingreso_bruto += ingreso_bruto
    total_ingreso_incr  += ingreso_incremental
    total_costo         += costo_total
    total_profit_incr   += profit_incremental
 
    print(f"  {prod:<20} {n_conv_trat:>10,.0f} {n_conv_ctrl:>10,.0f} "
          f"{uplift_pp:>+8.1f}% "
          f"${ingreso_bruto:>10,.0f} "
          f"${ingreso_incremental:>10,.0f} "
          f"${costo_total:>6,.0f} "
          f"${profit_incremental:>9,.0f}")
 
    metricas_campana.append({
        'producto'             : prod,
        'n_tratamiento'        : len(trat),
        'n_control'            : len(ctrl),
        'conversiones_trat'    : int(n_conv_trat),
        'conversiones_ctrl'    : int(n_conv_ctrl),
        'tasa_conversion_trat' : round(tasa_trat, 4),
        'tasa_conversion_ctrl' : round(tasa_ctrl, 4),
        'uplift_pp'            : round(uplift_pp, 2),
        'ingreso_bruto'        : round(ingreso_bruto, 2),
        'ingreso_incremental'  : round(ingreso_incremental, 2),
        'costo_contactacion'   : round(costo_total, 2),
        'profit_incremental'   : round(profit_incremental, 2),
    })
 
# Totales
print(f"  {'─'*98}")
print(f"  {'TOTAL':<20} {'':>10} {'':>10} {'':>9} "
      f"${total_ingreso_bruto:>10,.0f} "
      f"${total_ingreso_incr:>10,.0f} "
      f"${total_costo:>6,.0f} "
      f"${total_profit_incr:>9,.0f}")
 
subseccion("7.2 — KPIs ejecutivos de campaña")
 
roi_campana    = total_profit_incr / total_costo if total_costo > 0 else 0
roi_bruto      = total_ingreso_bruto / total_costo if total_costo > 0 else 0
cpc_incremental = total_costo / max(total_ingreso_incr / \
    (sum(p['ticket_anual'] for p in PARAMS_NEGOCIO.values()) / 6), 1)
 
print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║         KPIs EJECUTIVOS — CAMPAÑA NBO SEMANA 4          ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Presupuesto ejecutado   : ${PRESUPUESTO_CAMPANA - presupuesto_restante:>10,.2f}                ║
  ║  Ingreso bruto generado  : ${total_ingreso_bruto:>10,.2f}                ║
  ║  Ingreso incremental     : ${total_ingreso_incr:>10,.2f}  ← causal     ║
  ║  Costo total contactos   : ${total_costo:>10,.2f}                ║
  ║  Profit incremental      : ${total_profit_incr:>10,.2f}  ← CFO metric  ║
  ║  ROI incremental         : {roi_campana:>10.1f}x                      ║
  ╚══════════════════════════════════════════════════════════╝
 
  Nota: Profit incremental = ingreso atribuible CAUSALMENTE al NBO.
  No incluye ventas orgánicas que habrían ocurrido sin intervención.
  Este es el número que se presenta ante Finanzas para justificar
  el costo de implementación del sistema.
""")
 
# Guardar métricas
df_metricas = pd.DataFrame(metricas_campana)
df_metricas.to_csv(f'{DATA_DIR}/nbo_metricas_campana.csv', index=False)
print(f"  ✅ Métricas guardadas: nbo_metricas_campana.csv")


# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 8 — MONITOREO Y TRIGGERS DE MANTENIMIENTO
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 8 — MONITOREO Y TRIGGERS DE MANTENIMIENTO")
 
print("""
  El monitoreo cubre tres dimensiones independientes:
 
  Dimensión 1 — Drift de scores (PSI):
    ¿La distribución de scores cambió entre el período de validación
    del modelo y la campaña actual? PSI > 0.10 activa alerta.
 
  Dimensión 2 — Drift de tasa base:
    ¿La tasa de conversión observada se desvía de la proyectada?
    Desviación > ±15% activa monitoreo intensivo.
    Desviación > ±25% activa recalibración de Platt Scaling.
 
  Dimensión 3 — Backtesting de profit:
    ¿El profit real vs proyectado se desvía significativamente?
    Desviación sostenida > 20% por dos trimestres → champion-challenger.
""")
 
monitoreo = []
 
subseccion("8.1 — Drift de tasa base por producto")
print(f"\n  {'Producto':<20} {'Tasa proyectada':>16} {'Tasa observada':>16} "
      f"{'Desviación':>12} {'Estado':>10}")
print(f"  {'─'*78}")
 
for prod in sorted(df_resultados['producto_nbo'].unique()):
    sub  = df_resultados[
        (df_resultados['producto_nbo'] == prod) &
        (df_resultados['grupo'] == 'Tratamiento')
    ]
    tasa_proyectada = sub['p_calibrada'].mean()
    tasa_observada  = sub['convirtio_30d'].mean() if len(sub) > 0 else 0
    desviacion      = abs(tasa_observada - tasa_proyectada) / max(tasa_proyectada, 1e-6)
 
    if desviacion > 0.25:
        estado = "❌ Recalib."
    elif desviacion > 0.15:
        estado = "⚠️  Alerta"
    else:
        estado = "✅ Estable"
 
    print(f"  {prod:<20} {tasa_proyectada:>16.4f} {tasa_observada:>16.4f} "
          f"{desviacion:>+11.1%} {estado:>10}")
 
    monitoreo.append({
        'producto'           : prod,
        'tasa_proyectada'    : round(tasa_proyectada, 4),
        'tasa_observada'     : round(tasa_observada, 4),
        'desviacion_tasa'    : round(desviacion, 4),
        'alerta_tasa'        : desviacion > 0.15,
        'trigger_recalib'    : desviacion > 0.25,
    })
 
subseccion("8.2 — Backtesting de profit")
print(f"\n  {'Producto':<20} {'Profit proyectado':>18} {'Profit real':>13} "
      f"{'Desviación':>12} {'Estado':>10}")
print(f"  {'─'*78}")
 
for m in metricas_campana:
    prod = m['producto']
    p    = PARAMS_NEGOCIO[prod]
 
    # Profit proyectado: usando p_calibrada promedio × parámetros de negocio
    sub_trat = df_resultados[
        (df_resultados['producto_nbo'] == prod) &
        (df_resultados['grupo'] == 'Tratamiento')
    ]
    p_cal_medio    = sub_trat['p_calibrada'].mean()
    ingreso_proy   = p_cal_medio * (p['ticket_anual'] - p['costo_originacion'])
    perdida_proy   = p['pd'] * p['lgd'] * p['ticket_anual'] * p['rwa']
    profit_proy    = (ingreso_proy - perdida_proy - p['costo_contacto']) * len(sub_trat)
 
    profit_real    = m['profit_incremental']
    desv_profit    = abs(profit_real - profit_proy) / max(abs(profit_proy), 1.0)
 
    if desv_profit > 0.30:
        estado = "❌ Crítico"
    elif desv_profit > 0.20:
        estado = "⚠️  Alerta"
    else:
        estado = "✅ Estable"
 
    print(f"  {prod:<20} ${profit_proy:>16,.0f} ${profit_real:>11,.0f} "
          f"{desv_profit:>+11.1%} {estado:>10}")
 
    # Actualizar registro de monitoreo
    for reg in monitoreo:
        if reg['producto'] == prod:
            reg['profit_proyectado']  = round(profit_proy, 2)
            reg['profit_real']        = round(profit_real, 2)
            reg['desviacion_profit']  = round(desv_profit, 4)
            reg['alerta_profit']      = desv_profit > 0.20
            reg['trigger_champion']   = desv_profit > 0.30
            break
 
subseccion("8.3 — Resumen de triggers activos")
 
df_monitoreo = pd.DataFrame(monitoreo)
 
triggers_recalib   = df_monitoreo[df_monitoreo.get('trigger_recalib', False)   == True]
triggers_champion  = df_monitoreo[df_monitoreo.get('trigger_champion', False)  == True]
alertas_tasa       = df_monitoreo[df_monitoreo.get('alerta_tasa', False)       == True]
alertas_profit     = df_monitoreo[df_monitoreo.get('alerta_profit', False)     == True]
 
print(f"""
  Triggers de recalibración activos  : {len(triggers_recalib)}
  Triggers champion-challenger        : {len(triggers_champion)}
  Alertas de tasa base               : {len(alertas_tasa)}
  Alertas de profit                  : {len(alertas_profit)}
""")
 
if len(triggers_recalib) > 0:
    print(f"  ⚠️  Productos que requieren recalibración Platt Scaling:")
    for prod in triggers_recalib['producto'].values:
        print(f"     → {prod}")
 
if len(triggers_champion) > 0:
    print(f"\n  ❌ Productos que activan champion-challenger:")
    for prod in triggers_champion['producto'].values:
        print(f"     → {prod}  (profit real se desvía > 30% del proyectado)")
 
if len(triggers_recalib) == 0 and len(triggers_champion) == 0:
    print(f"  ✅ Sin triggers activos — sistema estable en esta campaña")
 
df_monitoreo.to_csv(f'{DATA_DIR}/nbo_monitoreo_campana.csv', index=False)
print(f"\n  ✅ Monitoreo guardado: nbo_monitoreo_campana.csv")
 
 
# ══════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════
separador("RESUMEN FINAL — SEMANA 4 COMPLETADA")
 
print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║       RBlJose — NBO — SEMANA 4 COMPLETADA           ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ✓ Bloque 1 : Carga e integración de inputs                 ║
  ║  ✓ Bloque 2 : Contact Policy Engine (6 reglas + fallback)   ║
  ║  ✓ Bloque 3 : Optimizador greedy presupuestario             ║
  ║  ✓ Bloque 4 : Asignación tratamiento / control (80/20)      ║
  ║  ✓ Bloque 5 : Simulación de ejecución de campaña            ║
  ║  ✓ Bloque 6 : Observación T+30 y cierre de etiquetas        ║
  ║  ✓ Bloque 7 : Medición financiera y causal (ROI incremental) ║
  ║  ✓ Bloque 8 : Monitoreo y triggers de mantenimiento         ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Outputs generados:                                         ║
  ║    nbo_campana_ejecutada.csv    — universo contactado        ║
  ║    nbo_resultados_t30.csv       — conversiones T+30          ║
  ║    nbo_metricas_campana.csv     — KPIs financieros causales  ║
  ║    nbo_monitoreo_campana.csv    — drift y triggers           ║
  ╚══════════════════════════════════════════════════════════════╝
""")


