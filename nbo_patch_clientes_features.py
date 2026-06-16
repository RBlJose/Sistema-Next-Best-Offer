# %%
"""
RBlJose — SISTEMA NBO
Patch Semana 8 — Agregar region y features estacionales
=========================================================
Modifica SOLO nbo_clientes.csv y nbo_features.csv en-lugar,
sin regenerar nbo_ofertas.csv ni nbo_productos.csv.
 
Qué hace:
  1. Agrega columna `region` (Sierra/Costa/Amazonia) a nbo_clientes.csv
     derivada de `segmento` con distribuciones calibradas INEC Ecuador 2022.
 
  2. Agrega 8 features estacionales a nbo_features.csv cruzando con
     la región de cada cliente. Las features con lógica regional
     (es_decimo_cuarto, es_inicio_clases) se calculan correctamente
     por cliente, no por mes global.
 
Decisión de diseño — por qué patch y no regenerar:
  `region` es derivada de `segmento` que ya existe.
  Las features estacionales son funciones de (mes, region) — sin
  dependencia de ruido ni de otros estados del generador.
  No hay razón para regenerar los 20,000 clientes ni los 500k registros
  de features cuando la información adicional es determinística.
 
Nota sobre reproducibilidad:
  Se usa SEED=42 reseteado específicamente para la generación de `region`.
  Como `region` es NUEVA (no existía en el generador original), no hay
  un "estado correcto del generador" que respetar — solo necesitamos
  que el resultado sea determinístico y consistente entre ejecuciones.
 
Prerequisito:
  nbo_clientes.csv y nbo_features.csv generados por el generador original.
 
Outputs (modifica en lugar):
  nbo_clientes.csv  ← agrega columna `region`
  nbo_features.csv  ← agrega 8 columnas estacionales
"""

# %%
import numpy as np
import pandas as pd
import os
 
DATA_DIR = os.getcwd()
SEED     = 42
 
def separador(titulo):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print(f"{'='*65}")
 
def subseccion(titulo):
    print(f"\n  ── {titulo}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 1 — PATCH nbo_clientes.csv → agregar region
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 1 — PATCH nbo_clientes.csv")
 
clientes = pd.read_csv(f'{DATA_DIR}/nbo_clientes.csv')
print(f"\n  Clientes cargados : {len(clientes):,}")
print(f"  Columnas actuales : {list(clientes.columns)}")
 
if 'region' in clientes.columns:
    print(f"\n  ⚠️  Columna `region` ya existe. Se sobreescribe para garantizar consistencia.")
 
# Distribución calibrada con INEC Ecuador 2022
# Sesgo por segmento: Premium concentrado en Quito (Sierra),
# Masivo más balanceado hacia Guayaquil (Costa).
REGION_DIST = {
    'Masivo':     {'Sierra': 0.46, 'Costa': 0.49, 'Amazonia': 0.05},
    'Preferente': {'Sierra': 0.52, 'Costa': 0.44, 'Amazonia': 0.04},
    'Pyme':       {'Sierra': 0.55, 'Costa': 0.41, 'Amazonia': 0.04},
    'Premium':    {'Sierra': 0.62, 'Costa': 0.35, 'Amazonia': 0.03},
}
 
# Reset seed específico para region — variable nueva, no hay estado previo que respetar
np.random.seed(SEED)
n = len(clientes)
region = np.empty(n, dtype=object)
 
for seg, dist in REGION_DIST.items():
    mask = (clientes['segmento'] == seg).values
    region[mask] = np.random.choice(
        list(dist.keys()), size=mask.sum(), p=list(dist.values())
    )
 
clientes['region'] = region
 
# Verificación de distribución
subseccion("Distribución de region")
print(f"\n  {'Región':<15} {'N':>8} {'%':>8}")
print(f"  {'─'*34}")
for reg, cnt in clientes['region'].value_counts().items():
    print(f"  {reg:<15} {cnt:>8,} {cnt/len(clientes):>8.1%}")
 
subseccion("Distribución region × segmento (verificación de correlación)")
print()
print(pd.crosstab(
    clientes['segmento'], clientes['region'],
    normalize='index'
).round(3).to_string())
 
# Guardar
clientes.to_csv(f'{DATA_DIR}/nbo_clientes.csv', index=False)
print(f"\n  ✅ nbo_clientes.csv actualizado — columna `region` agregada")
print(f"     Total columnas: {clientes.shape[1]}")

# %%
# ══════════════════════════════════════════════════════════════════════
# BLOQUE 2 — PATCH nbo_features.csv → agregar features estacionales
# ══════════════════════════════════════════════════════════════════════
separador("BLOQUE 2 — PATCH nbo_features.csv")
 
features = pd.read_csv(f'{DATA_DIR}/nbo_features.csv')
print(f"\n  Features cargadas : {len(features):,} filas")
print(f"  Meses cubiertos   : {features['mes'].min()}–{features['mes'].max()}")
 
FEATURES_ESTACIONALES = [
    'mes_calendario', 'es_utilidades', 'es_decimo_tercero',
    'es_decimo_cuarto', 'es_inicio_clases',
    'es_navidad', 'es_impuesto_renta', 'trimestre',
]
 
ya_presentes = [c for c in FEATURES_ESTACIONALES if c in features.columns]
if ya_presentes:
    print(f"\n  ⚠️  Ya existen: {ya_presentes}. Se sobreescriben.")
 
# Traer region al DataFrame de features
region_map = clientes.set_index('id_cliente')['region']
features['_region'] = features['id_cliente'].map(region_map)
 
n_sin_region = features['_region'].isna().sum()
if n_sin_region > 0:
    raise ValueError(
        f"{n_sin_region} filas en features sin region en clientes. "
        f"Verifica que nbo_clientes.csv y nbo_features.csv provienen "
        f"del mismo generador."
    )
 
mc = ((features['mes'] - 1) % 12) + 1   # mes calendario 1-12
reg = features['_region'].values
 
# ── Features sin dependencia de región ───────────────────────────────
features['mes_calendario']    = mc.astype(np.int8)
features['es_utilidades']     = mc.isin([3, 4]).astype(np.int8)
features['es_decimo_tercero'] = (mc == 12).astype(np.int8)
features['es_navidad']        = mc.isin([11, 12]).astype(np.int8)
features['es_impuesto_renta'] = mc.isin([3, 4]).astype(np.int8)
features['trimestre']         = ((mc - 1) // 3 + 1).astype(np.int8)
 
# ── Features con lógica diferenciada por región ───────────────────────
# Décimo cuarto:
#   Costa    → marzo (mes 3)
#   Sierra   → agosto (mes 8)
#   Amazonia → agosto (mes 8, sigue calendario Sierra)
features['es_decimo_cuarto'] = np.where(
    reg == 'Costa',
    (mc == 3).astype(np.int8),
    (mc == 8).astype(np.int8)
).astype(np.int8)
 
# Inicio de clases:
#   Costa    → abril-mayo (meses 4-5)
#   Sierra   → agosto-septiembre (meses 8-9)
#   Amazonia → agosto-septiembre (sigue Sierra)
features['es_inicio_clases'] = np.where(
    reg == 'Costa',
    mc.isin([4, 5]).astype(np.int8),
    mc.isin([8, 9]).astype(np.int8)
).astype(np.int8)
 
# Eliminar columna auxiliar
features = features.drop(columns=['_region'])
 
# ── Verificación de correctitud por región ────────────────────────────
subseccion("Verificación lógica regional")
print()
 
checks = [
    ('Costa',    3,  'es_decimo_cuarto',  1, 'Costa recibe DC en marzo'),
    ('Costa',    8,  'es_decimo_cuarto',  0, 'Costa NO recibe DC en agosto'),
    ('Sierra',   8,  'es_decimo_cuarto',  1, 'Sierra recibe DC en agosto'),
    ('Sierra',   3,  'es_decimo_cuarto',  0, 'Sierra NO recibe DC en marzo'),
    ('Amazonia', 8,  'es_decimo_cuarto',  1, 'Amazonia sigue Sierra'),
    ('Costa',    4,  'es_inicio_clases',  1, 'Costa inicia clases en abril'),
    ('Costa',    8,  'es_inicio_clases',  0, 'Costa NO inicia en agosto'),
    ('Sierra',   8,  'es_inicio_clases',  1, 'Sierra inicia clases en agosto'),
    ('Sierra',   4,  'es_inicio_clases',  0, 'Sierra NO inicia en abril'),
]
 
todos_ok = True
for region_val, mes_val, col, esperado, descripcion in checks:
    region_col = features['id_cliente'].map(region_map)
    mc_col     = ((features['mes'] - 1) % 12) + 1
    sub = features[
        (region_col == region_val) & (mc_col == mes_val)
    ]
    if len(sub) == 0:
        print(f"  ⚠️  Sin datos: {descripcion}")
        continue
    real = sub[col].mean()
    ok   = abs(real - esperado) < 0.001
    flag = "✅" if ok else "❌"
    print(f"  {flag} {descripcion:<45} → {col}={real:.0f} (esp: {esperado})")
    if not ok:
        todos_ok = False
 
assert todos_ok, "❌ Falló verificación de lógica regional"
print(f"\n  ✅ Todas las verificaciones pasadas")
 
# Estadísticos básicos de las nuevas features
subseccion("Distribución de features estacionales por mes calendario")
temp = features.groupby('mes_calendario')[FEATURES_ESTACIONALES[1:]].mean().round(3)
print(f"\n  (proporción de clientes con flag = 1 por mes calendario)")
print(temp.to_string())
 
# Guardar
features.to_csv(f'{DATA_DIR}/nbo_features.csv', index=False)
print(f"\n  ✅ nbo_features.csv actualizado — {len(FEATURES_ESTACIONALES)} features estacionales agregadas")
print(f"     Total columnas: {features.shape[1]}")

# %%
# ══════════════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════════════
separador("PATCH COMPLETADO")
print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║     PATCH Semana 8 — COMPLETADO                         ║
  ╠══════════════════════════════════════════════════════════╣
  ║  ✓ nbo_clientes.csv  : columna `region` agregada        ║
  ║  ✓ nbo_features.csv  : {len(FEATURES_ESTACIONALES)} features estacionales       ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Siguiente paso:                                         ║
  ║    python nbo_regenerar_ofertas.py                      ║
  ║    → regenera nbo_ofertas.csv con señal estacional      ║
  ║      en el DAG de propensión                            ║
  ╚══════════════════════════════════════════════════════════╝
""")


