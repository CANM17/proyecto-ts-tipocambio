# Forecasting del Tipo de Cambio USD/Guaraní (Paraguay)

Proyecto Final — Series Temporales
**Autor:** [César Núñez]

---

## 1. Descripción del problema

En Paraguay, gran parte de los créditos, precios de importación y contratos
comerciales están denominados o influenciados por el dólar estadounidense.
El tipo de cambio USD/Guaraní determina, en última instancia, cuánto termina
pagando una persona o empresa en guaraníes por obligaciones fijadas en
dólares (por ejemplo, la cuota de un préstamo en moneda extranjera).

Este proyecto aborda el problema como una tarea de **forecasting univariado**:
predecir el valor futuro del tipo de cambio a partir de su historia reciente.

## 2. Dataset utilizado

- **Fuente real:** Banco Central del Paraguay (BCP) — Cotización Referencial
  Histórica (https://www.bcp.gov.py/webapps/web/cotizacion/monedas-historica).
  Planillas oficiales descargadas manualmente, una por año (2022, 2023, 2024,
  2025), en formato de planilla (día del mes en filas, mes en columnas).
- **Procesamiento:** las 4 planillas se combinaron en una única serie diaria
  continua (`data/prepare_dataset.py`), convirtiendo el formato numérico
  es-PY (`7.037,83`) a numérico estándar y manejando "ND" (no disponible)
  como valor faltante.
- **Observaciones finales:** 996 días hábiles (2022-01-03 a 2025-12-30),
  sin duplicados. Tras reindexar a días hábiles completos, se identificaron
  46 feriados/días sin cotización, imputados por interpolación lineal.
- **Columna de fecha:** `fecha` (frecuencia de días hábiles).
- **Variable objetivo:** `tipo_cambio` (guaraníes por dólar).

## 3. Metodología aplicada

1. **Preprocesamiento** (`notebooks/01_preprocesamiento.ipynb`)
   - Reindexado a días hábiles completos e imputación de feriados sin
     cotización (interpolación lineal).
   - Análisis de rendimientos diarios (volatilidad).
   - Descomposición de tendencia (ciclo mensual aproximado, 21 días hábiles).
   - Pruebas de estacionariedad ADF y KPSS **en niveles y en la serie
     diferenciada** (ver hallazgo importante en la sección 5).
   - Gráficos de ACF/PACF sobre la serie diferenciada, como referencia
     exploratoria para el orden del modelo ARIMA.
   - Split temporal train/test (982 / 60 observaciones — el test cubre
     aproximadamente octubre-diciembre de 2025).

2. **Modelado** (`notebooks/02_modelos.ipynb`)
   - **ARIMA**, con el orden (p, d, q) elegido mediante **búsqueda por AIC**
     sobre una grilla de combinaciones (no fijado a ojo), y diagnóstico de
     residuales con el **test de Ljung-Box**.
   - **XGBoost** con features de rezago, medias móviles y calendario,
     pronosticando de forma **recursiva** a 60 días (usa sus propias
     predicciones para construir los rezagos de los pasos siguientes, nunca
     valores reales futuros — mismo horizonte y mismas reglas que ARIMA).

3. **Evaluación** (`notebooks/03_evaluacion.ipynb`)
   - Cálculo de métricas (RMSE, MAE, MAPE, sMAPE, MASE, R²).
   - Visualizaciones de predicciones vs. valores reales y comparación entre
     modelos.
   - Análisis de residuales, incluyendo cómo evoluciona el error a lo largo
     del horizonte de 60 días.

## 4. Modelos implementados

| Modelo | Categoría | Descripción |
|---|---|---|
| **ARIMA** (orden elegido por AIC) | Estadístico clásico | Autoregresión + diferenciación, univariado, pronóstico estático a 60 días |
| **XGBoost (recursivo)** | Machine Learning | Árboles de gradient boosting con rezagos, medias móviles y calendario, pronóstico recursivo a 60 días |

*Nota: no se usa un componente estacional (SARIMA), ya que el EDA no mostró
un ciclo determinístico fuerte — un tipo de cambio no tiene la
estacionalidad marcada de variables físicas/ambientales.*

## 5. Resultados y métricas

| Métrica | ARIMA | XGBoost (recursivo) |
|---|---|---|
| RMSE | **153.06** | 213.20 |
| MAE | **136.07** | 146.50 |
| MAPE (%) | **1.97** | 2.17 |
| sMAPE (%) | **1.97** | 2.12 |
| MASE | **10.66** | 11.48 |
| R² | **0.27** | -0.42 |

**ARIMA superó a XGBoost en todas las métricas.** Esto es un resultado
importante de destacar: en un pronóstico recursivo a 60 días, el error de un
modelo de árboles como XGBoost se acumula paso a paso (cada predicción se
usa para generar la siguiente), mientras que ARIMA tiene un comportamiento
de reversión a la tendencia más estable en pronósticos largos. Un diseño de
evaluación que no iguale el horizonte de pronóstico entre modelos (por
ejemplo, dejar que uno use datos reales recientes como feature y el otro no)
llevaría a conclusiones engañosas — por eso ambos modelos se evalúan aquí
bajo las mismas reglas.

### 🔎 Hallazgos relevantes de la auditoría metodológica

1. **ADF y KPSS no coinciden tras diferenciar (d=1):** el ADF indica que la
   serie diferenciada ya es estacionaria (p=0.0002), pero el KPSS sigue
   rechazando estacionariedad (p=0.01). Este patrón es típico cuando la
   serie tiene un **quiebre de tendencia/régimen** (la suba sostenida
   2022-2024 seguida de una corrección en 2025) — el KPSS es sensible a este
   tipo de cambios estructurales, distintos de una raíz unitaria pura.
   *Cuantificado (no solo visual): la pendiente lineal pasa de
   **+288.6 ₲/año (2022-2024)** a **-1.366,6 ₲/año (2025)** — cambia de
   signo y es ~5 veces más pronunciada en la corrección.*
2. **El test de Ljung-Box (aplicado correctamente) no muestra autocorrelación
   significativa en el rezago 10**, señal de un buen ajuste de corto plazo.
   Sí aparece una señal leve de autocorrelación en el rezago 20 (p=0.0065).
   *Corrección importante: una primera versión de este test usaba los
   residuales crudos del filtro de Kalman sin excluir el período de "burn-in"
   (inicialización difusa), lo que distorsionaba el resultado y hacía parecer
   que el modelo fallaba gravemente. Al usar el método correcto de
   statsmodels (`test_serial_correlation`, que excluye ese burn-in), el
   diagnóstico real es mucho más favorable para ARIMA.*
3. **El intervalo de confianza del 95% de ARIMA solo cubrió el valor real en
   el 85% de los días de test (51 de 60)**, por debajo del 95% nominal — el
   modelo subestima su propia incertidumbre. Esto se verificó empíricamente
   (no se asumió que el modelo estaba bien calibrado solo por reportar un
   intervalo), y es consistente con el quiebre de tendencia: un modelo que
   no anticipa un cambio de régimen tiende a reportar bandas de confianza
   más angostas de lo que la realidad justifica.
4. **El error de XGBoost crece mucho más rápido que el de ARIMA** a lo largo
   del horizonte de 60 días (de 55 a 387 en promedio absoluto, contra 117 a
   163 en ARIMA). Esto refleja un problema conocido del pronóstico
   recursivo con modelos de Machine Learning: los errores de cada paso se
   acumulan y realimentan en los pasos siguientes, mientras que ARIMA tiene
   un comportamiento de reversión a la tendencia más estable a largo plazo.
5. **El MASE (10.66 / 11.48) es alto y no debe leerse con la regla habitual
   de "MASE < 1 es bueno".** Ese umbral aplica cuando se compara un
   pronóstico contra un benchmark del **mismo horizonte**. Acá el
   denominador del MASE es la variación **de 1 día** en train (~12.76 ₲),
   mientras que el numerador es el error de un pronóstico **a 60 días**
   (~136-146 ₲) — son escalas distintas por construcción, no un indicio de
   que el modelo sea 10 veces peor que un ingenuo. Para una lectura de MASE
   realmente comparable, habría que construir el benchmark ingenuo también a
   60 días (por ejemplo, repetir el último valor de train durante todo el
   horizonte) en vez de usar la diferencia de 1 día.
6. **Chequeo de robustez de hiperparámetros de XGBoost:** a diferencia de
   ARIMA (orden elegido por AIC), los hiperparámetros de XGBoost se fijaron
   sin una búsqueda sistemática. Se probaron 2 configuraciones alternativas
   bajo el mismo esquema recursivo: en **ambas, ARIMA sigue ganando en
   RMSE** (200-201 vs 153.06), pero con una configuración más regularizada
   (menor profundidad, más árboles, learning rate más bajo), **la brecha en
   MAE casi desaparece** (136.25 vs 136.07 de ARIMA — prácticamente
   empatados). Esto confirma que la elección de hiperparámetros original no
   perjudicó artificialmente a XGBoost, aunque también sugiere que una
   búsqueda de hiperparámetros más sistemática (grid search o Optuna) podría
   angostar aún más la brecha en trabajos futuros.
7. **El proyecto se probó de punta a punta en un entorno virtual limpio**
   (venv nuevo, `pip install -r requirements.txt` desde cero) para confirmar
   que corre en cualquier máquina, no solo en la que se desarrolló. En la
   primera prueba (con rangos `>=` en `requirements.txt`), el venv limpio
   instaló versiones más nuevas de numpy/scipy, y **las métricas de ARIMA
   variaron levemente en la 2da-3era cifra decimal** (ej. RMSE 153.05 vs
   153.06) por la sensibilidad numérica del optimizador de máxima
   verosimilitud — las de XGBoost fueron idénticas bit a bit (tiene
   `random_state` fijo). Por eso `requirements.txt` ahora **fija versiones
   exactas** (`==`) en vez de mínimos; repitiendo la prueba con las versiones
   fijadas, **las métricas de ambos modelos salieron idénticas bit a bit**
   a las oficiales. La conclusión cualitativa (ARIMA supera a XGBoost) nunca
   estuvo en duda, pero ahora la reproducibilidad numérica exacta también
   está garantizada.

Las métricas completas quedan en [`results/metricas.csv`](results/metricas.csv)
y las predicciones detalladas en [`results/predicciones.csv`](results/predicciones.csv).

## 6. Visualizaciones

| Archivo | Contenido |
|---|---|
| `01_serie_original.png` | Serie completa del tipo de cambio (2022-2025) |
| `01_retornos.png` | Variación diaria (%) y su distribución |
| `01_descomposicion.png` | Descomposición de tendencia (ciclo mensual) |
| `01_acf_pacf.png` | ACF y PACF de la serie diferenciada |
| `02_importancia_features_xgb.png` | Importancia de features del modelo XGBoost |
| `03_pred_vs_real.png` | Predicciones vs. valores reales (ambos modelos, pronóstico a 60 días) |
| `03_comparacion_modelos.png` | Comparación de métricas (RMSE, MAE, MAPE) en barras |
| `03_analisis_residuales.png` | Análisis de residuales de ambos modelos |

## 7. Conclusiones

- El tipo de cambio USD/Guaraní mostró una **tendencia alcista** entre 2022
  y fines de 2024, con una corrección a la baja durante 2025 — un quiebre de
  régimen que resultó desafiante para ambos modelos.
- La serie **no es estacionaria en niveles**, y tras diferenciar (d=1) los
  resultados de ADF y KPSS no son unánimes — evidencia de que el quiebre de
  tendencia complica el diagnóstico clásico de estacionariedad.
- Con una comparación **metodológicamente justa** (mismo horizonte de 60
  días, ambos modelos recursivos/estáticos sin usar datos reales futuros),
  **ARIMA superó a XGBoost** en todas las métricas. El motivo principal es
  la acumulación de error en el pronóstico recursivo de XGBoost, que crece
  mucho más rápido que en ARIMA a medida que aumenta el horizonte.
- El test de Ljung-Box, aplicado correctamente (excluyendo el burn-in del
  filtro de Kalman), mostró que **ARIMA no deja autocorrelación significativa
  en el rezago 10** — buen ajuste de corto plazo — con una señal leve en el
  rezago 20 que podría ameritar seguimiento, pero no indica una falla grave.
- El intervalo de confianza del 95% de ARIMA **solo cubrió el valor real en
  el 85% de los días de test**, evidencia de que el modelo subestima su
  propia incertidumbre ante el quiebre de tendencia de 2025.
- **Trabajo futuro:** modelar la volatilidad explícitamente (GARCH/EGARCH),
  incorporar variables macroeconómicas exógenas (precio de la soja, tasa de
  política monetaria, IPC), evaluar con walk-forward validation completo
  (reentrenando en cada paso), o probar modelos de Deep Learning (LSTM).

## 8. Estructura del repositorio

```
proyecto_ts/
├── data/
│   ├── raw/
│   │   ├── cotizacion_anual2022.xls    # planillas oficiales del BCP
│   │   ├── cotizacion_anual2023.xls
│   │   ├── cotizacion_anual2024.xls
│   │   └── cotizacion_anual2025.xls
│   ├── prepare_dataset.py               # parseo de planillas a serie diaria
│   ├── datos.csv                        # dataset final (996 días hábiles)
│   ├── train.csv                        # generado por notebook 01
│   └── test.csv                         # generado por notebook 01
├── notebooks/
│   ├── 01_preprocesamiento.ipynb
│   ├── 02_modelos.ipynb
│   └── 03_evaluacion.ipynb
├── results/
│   ├── metricas.csv
│   ├── predicciones.csv
│   └── *.png
├── presentacion.pptx           # presentación de 10 diapositivas
├── README.md
└── requirements.txt
```

## 9. Cómo ejecutar el proyecto

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Opcional: regenerar data/datos.csv desde las planillas crudas del BCP
# (ya viene generado en el repo, no es necesario para correr el proyecto)
python3 data/prepare_dataset.py

# Ejecutar en orden:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_preprocesamiento.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelos.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluacion.ipynb
```

O simplemente abrir los notebooks en Jupyter/VS Code y correrlos celda a celda, en orden.


