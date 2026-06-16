# 🏦 Sistema NBO — RBlJosé
### Next Best Offer · Modelado de Propensión Multiproducto · Arquitectura MLOps Bancaria

---

## 📌 Descripción del Proyecto

Sistema de recomendación bancaria (Next Best Offer) diseñado e implementado de extremo a extremo para **RBlJosé**, cubriendo 6 productos financieros, 20.000 clientes sintéticos y 25 meses de datos históricos. El proyecto simula un entorno de producción real con gobernanza de modelos, backtesting multi-período, recalibración selectiva y experimento Champion-Challenger con criterios de decisión económicamente fundamentados.

> **Nota:** Los datos utilizados son sintéticos, generados mediante un DAG causal calibrado con parámetros del mercado ecuatoriano (BCE 2024, INEC 2022). El sistema fue construido como portfolio técnico de nivel productivo.

---

## 🎯 Objetivo

Reemplazar campañas masivas no segmentadas por un sistema de scoring individual que recomienda el **producto correcto, al cliente correcto, en el momento correcto**, dentro de una ventana de predicción de 30 días.

---

## 🏗️ Arquitectura del Pipeline

```
Generador Sintético (DAG Causal)
        │
        ▼
ABT Construction (Anti-Leakage Temporal Split)
        │
        ▼
XGBoost Multiproducto (6 modelos · v1.0 y v2.0)
        │
        ▼
Platt Scaling Calibration (v1.0 → v1.1 selectiva)
        │
        ▼
Contact Policy Engine (R1–R8 · Cooling · Fatiga · Supresión)
        │
        ▼
Greedy Budget Optimizer
        │
        ▼
Backtesting Multi-Período (meses 22–25)
        │
        ▼
Champion-Challenger Experiment (v1.0 vs v2.0)
        │
        ▼
Power BI Dashboard (7 páginas · Ejecutivo + Técnico)
```

---

## 📦 Productos en Scope

| Producto | Margen Neto | PD Base | Consume RWA | Perfil Target |
|---|---|---|---|---|
| Tarjeta de Crédito | 15.5% | 5.5% | ✓ | Score >700, digital activo |
| Préstamo Personal | 13.5% | 4.5% | ✓ | Ingreso estable, sin crédito activo |
| Microcrédito | 12.0% | 9.5% | ✓ | Independiente, depósitos irregulares |
| Seguro de Vida | 17.5% | 0% | ✗ | Edad 30–55, hijos, crédito hipotecario |
| Seguro de Salud | 15.0% | 0% | ✗ | Independiente, gasto en farmacias |
| Inversión (DPF) | 3.0% | 0% | ✗ | Saldo alto, crédito recién cancelado |

---

## 📂 Estructura del Repositorio

```
├── nbo_generador_sintetico.py          # Semana 2: Generador DAG causal
├── nbo_semana3_modelos.py              # Semana 3: XGBoost v1.0 + Platt Scaling
├── NBO_Semana4_Campana.py              # Semana 4: Pipeline de campaña completo
├── NBO_Semana5_Backtesting.py       # Semana 5: Backtesting multi-período
├── NBO_Semana6_Recalibracion.py     # Semana 6: Recalibración selectiva v1.1
├── NBO_Semana7_Calibradores.py      # Semana 7: Validación operativa calibradores
├── nbo_patch_clientes_features.py      # Semana 8: Patch región + features estacionales
├── nbo_regenerar_ofertas.py            # Semana 8: Regeneración con señal estacional
├── nbo_semana8_modelo.py            # Semana 8: XGBoost v2.0 (8 features estacionales)
├── nbo_semana9_champion_challenger.py  # Semana 9: Experimento CC con bootstrap
├── NBO_Pasos1_7.docx                   # Documento de diseño del sistema
├── NBO_Informe_Final.docx              # Documento conclusivo del Poryecto 
├── Monitoreo_del_modelo.pbix           # Documento conclusivo del Poryecto 
└── models/                             # Modelos serializados (joblib + json)
```

---

## 🔬 Semanas del Proyecto

### Semana 2 — Generador Sintético
- DAG causal con 4 niveles jerárquicos (raíz → derivadas → score → propensión)
- Ruido persistente por cliente (heterogeneidad latente)
- 5 regímenes macroeconómicos (Normal → Stress → Recuperación)
- Interacciones entre productos según evidencia bancaria

### Semana 3 — Modelos de Propensión v1.0
- XGBoost binary:logistic por producto (6 modelos independientes)
- Split temporal estricto: train 1–15, val 16–18, test OOT 19–23
- Platt Scaling calibrado en validación, evaluado en test
- SHAP via `pred_contribs` (XGBoost nativo, compatible 2.x)
- AUC test: 0.71–0.81 según producto

### Semana 4 — Pipeline de Campaña
- Contact Policy Engine (R1–R8): cooling period, opt-out, producto activo, elegibilidad mínima
- Fallback a rank 2 para clientes bloqueados en rank 1
- Optimizador greedy por ratio score_nbo/costo
- Asignación aleatoria tratamiento/control (80/20) con semilla fija
- Medición causal: profit incremental vs tasa orgánica

### Semana 5 — Backtesting Multi-Período
- 4 ciclos (meses 22–25) con CPE + R7 (fatiga) + R8 (supresión)
- Ground truth corregido: `score_propension` del generador DAG (no `p_calibrada`)
- Meses 22–23: etiqueta real · Meses 24–25: simulación Bernoulli
- Output: `nbo_scores_historicos_s5.csv` (insumo para Semana 6)

### Semana 6 — Recalibración Selectiva v1.1
- Trigger: desviación >25% en tarjeta y seguro_vida (Semana 5)
- Platt Scaling ponderado con EWMA (λ=0.5, meses 22–23)
- Productos sin trigger: calibradores v1.0 preservados exactamente
- Mejora en ECE y Brier Score validada in-sample

### Semana 7 — Validación Operativa
- Evaluación comparativa v1.0 vs v1.1 sobre fases referencia y proyección
- Overlap de selección >90%: el valor principal es la corrección del sesgo en proyecciones financieras, no el ranking

### Semana 8 — Modelo v2.0 con Estacionalidad
- `region` (Sierra/Costa/Amazonia) como variable raíz derivada de INEC 2022
- 8 features estacionales: DC3, DC4 (diferenciado por región), inicio clases, navidad, utilidades, IR, mes calendario, trimestre
- Señal multiplicativa sobre `score_propension` original (flip condicional, cambio ~8–12%)
- Hiperparámetros XGBoost idénticos a v1.0 por diseño experimental

### Semana 9 — Champion-Challenger
- Split 50/50 estratificado por segmento, fijo durante 4 ciclos
- Cada brazo opera con presupuesto independiente ($25K)
- **Criterios de promoción (jerarquía estricta):**
  1. ΔProfit incremental > +2%
  2. IC 95% bootstrap no cruza cero
  3. Default rate Challenger ≤ Champion (proxy score <600)
- **Veredicto: MANTENER v1.0** — ΔProfit = -33.67%, deterioro concentrado en `prestamo`

---

## 📊 Resultados Clave

| Métrica | Valor |
|---|---|
| AUC Test (rango) | 0.71 – 0.81 |
| KS Test (rango) | 0.34 – 0.48 |
| PSI máximo (val→test) | 0.068 (préstamo) |
| Profit incremental acumulado (4 ciclos) | ~$105,000 |
| ROI incremental campaña | 5.6x |
| Cobertura CPE sobre universo scoring | ~24% elegibles finales |
| Veredicto Champion-Challenger S9 | MANTENER v1.0 |
| ΔProfit Challenger vs Champion | -33.67% |

---

## 🛡️ Principios de Gobernanza Implementados

- **Split temporal estricto** — Cross-validation prohibida en todo el pipeline
- **Ground truth íntegro** — `score_propension` del generador DAG, nunca `p_calibrada` como etiqueta
- **Champion = sistema completo** — modelo + calibración + CPE + optimizer + parámetros de negocio
- **Decisión económica, no estadística** — Profit bootstrap como criterio primario, no p-valor sobre tasa de conversión
- **Recalibración ≠ mejora de AUC** — Platt Scaling es transformación monotónica; corrige sesgo de probabilidad, no capacidad discriminante
- **Invariantes del experimento documentados** — cualquier asimetría entre brazos invalida el CC

---

## 🖥️ Dashboard Power BI

7 páginas orientadas a audiencias distintas:

| Página | Audiencia |
|---|---|
| Vista Ejecutiva | CFO / Dirección General |
| Performance del Modelo | Equipo Técnico / Riesgo de Modelos |
| Campaña y Conversiones | Marketing / Producto |
| Evolución Temporal / Backtesting | Analítica / Data Science |
| Monitoreo y Alertas | MLOps / Riesgo de Modelos |
| Champion-Challenger | Dirección de Analítica / Comité de Modelos |
| Segmentación de Clientes | CRM / Producto / Marketing |

---

## ⚙️ Stack Tecnológico

**Python:** XGBoost 2.x · scikit-learn · pandas · numpy · scipy · joblib · matplotlib

**Visualización:** Power BI Desktop (DAX · Medidas calculadas · Slicers cross-page)

**Serialización:** joblib (modelos) · pickle (calibradores) · JSON (metadata y contratos de features)

---

## 🚀 Ejecución del Pipeline

```bash
# Orden de ejecución secuencial
python nbo_generador_sintetico.py
python nbo_semana3_modelos.py
python NBO_Semana4_Campana.py
python NBO_Semana5_Backtesting_v2.py
python NBO_Semana6_Recalibracion_v2.py
python NBO_Semana7_Calibradores_v2.py
python nbo_patch_clientes_features.py
python nbo_regenerar_ofertas.py
python nbo_semana8_modelo_v2.py
python nbo_semana9_champion_challenger.py
```

---

## 📋 Requisitos

```bash
pip install xgboost scikit-learn pandas numpy scipy joblib matplotlib python-dateutil
```

---

## 🧠 Aprendizajes Clave

1. **Champion = sistema completo, no modelo aislado** — La comparación debe encapsular toda la cadena de decisión
2. **Recalibración es epistémica, no operacional inmediata** — Su valor principal es honestidad en proyecciones financieras
3. **25 meses son insuficientes para señal estacional robusta en XGBoost** — ~2 ciclos calendario bajo split temporal estricto generan varianza alta en features binarias de baja frecuencia
4. **El criterio de decisión define la cultura analítica** — Usar conversion rate como criterio de promoción es un patrón junior; profit bootstrap con guardia de riesgo es el estándar productivo
5. **Bernoulli resampling independiente introduce ruido estructural** — El flip condicional preserva correlaciones features→target

---

## 👤 Autor

**Jose** — Data Scientist · Banking Analytics  
Proyecto portfolio construido semana a semana como demostración de capacidades en MLOps bancario, modelado de propensión y arquitectura de sistemas de recomendación en entornos regulados.

---

## 📄 Licencia

Proyecto de portfolio con fines educativos y de demostración. Los datos son completamente sintéticos.
