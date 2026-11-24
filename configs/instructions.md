
# ENTREGABLE 1: Entrenamiento de modelos ML y evaluación de ejercicios en tiempo real

## 1. Rol y objetivo

Actúa como un Senior Machine Learning Engineer especializado en Computer Vision, clasificación supervisada y sistemas de inferencia en tiempo real.

El objetivo es construir el primer entregable funcional del proyecto de evaluación de ejercicios físicos.

Se dispone de dos datasets procesados:

- `data/gold/plank_features.parquet`
- `data/gold/hollow_body_features.parquet`

Debes desarrollar dos modelos de Machine Learning independientes:

1. Un modelo especializado en evaluar la ejecución de Plank.
2. Un modelo especializado en evaluar la ejecución de Hollow Body Hold.

Para cada ejercicio deberás:

- Realizar análisis exploratorio del dataset.
- Entrenar y comparar diferentes algoritmos de clasificación.
- Evaluar los modelos mediante Cross Validation.
- Seleccionar el modelo con mejor desempeño.
- Entrenar el modelo definitivo.
- Guardar los artefactos necesarios para inferencia.
- Crear un script Python que permita evaluar el ejercicio utilizando una cámara en tiempo real.

El entregable final debe permitir ejecutar dos aplicaciones independientes:

```bash
python evaluate_plank.py
python evaluate_hollow_body.py
```

Cada aplicación debe abrir la cámara, detectar la postura corporal y mostrar una evaluación en tiempo real.

---

# 2. Fase 0: Inspección del proyecto

Antes de escribir o modificar código:

1. Inspecciona la estructura actual del repositorio.
2. Examina los dos archivos Parquet.
3. Identifica todas las columnas disponibles.
4. Determina cuáles son metadata y cuáles son características predictoras.
5. Identifica las variables objetivo y sus clases.
6. Comprueba la cantidad de registros, participantes, videos y ventanas temporales.
7. Identifica valores nulos, infinitos, duplicados y posibles inconsistencias.

Presta especial atención a:

- `video_id`
- `participant_id`
- `exercise`
- `window_id`
- `label`
- `error_type`
- `camera_view`

No asumas que las columnas tienen nombres o formatos específicos sin inspeccionar los archivos.

Presenta primero un breve diagnóstico y un plan de implementación antes de realizar modificaciones.

## 2.1 Definición del problema

El objetivo inicial será una clasificación binaria:

- `correct`: ejecución correcta.
- `incorrect`: ejecución incorrecta.

La variable `label` será el target principal, siempre que su contenido real confirme esta definición.

No utilices como predictores:

- Identificadores de participantes.
- Identificadores de videos.
- Identificadores de ventanas.
- Etiquetas.
- Tipos de error.
- Metadata que revele indirectamente la clase.

`error_type` podrá reservarse para un análisis posterior de los errores específicos, pero no debe introducirse como predictor del modelo binario.

Si existen clases adicionales o inconsistencias en las etiquetas, documenta el problema antes de continuar.

---

# 3. Fase 1: Análisis exploratorio y preparación

Realiza un análisis exploratorio independiente para cada ejercicio.

Incluye:

- Número de muestras.
- Distribución de clases.
- Número de participantes.
- Número de videos.
- Distribución de muestras por participante.
- Distribución de muestras por video.
- Distribución de `error_type`, si está disponible.
- Valores faltantes por característica.
- Distribución estadística de las características.
- Correlaciones entre variables numéricas.
- Identificación de características constantes o redundantes.

Genera visualizaciones útiles para entender el dataset.

No elimines características únicamente porque estén correlacionadas. Evalúa su utilidad y documenta las decisiones.

## 3.1 Preparación de datos

Separa las columnas de metadata de las características predictoras.

Utiliza:

- `X`: características numéricas.
- `y`: variable objetivo.
- `groups`: identificador del participante.

Es fundamental conservar la trazabilidad de cada muestra.

No utilices identificadores como características de entrenamiento.

Maneja los valores faltantes mediante estrategias apropiadas, preferentemente utilizando `SimpleImputer` dentro de un Pipeline de Scikit-learn.

No realices normalización, imputación, selección de características ni transformaciones utilizando el dataset completo antes de Cross Validation.

Todas las transformaciones deben ajustarse exclusivamente con los datos de entrenamiento de cada fold.

---

# 4. Fase 2: Entrenamiento y comparación de modelos

Desarrolla un experimento independiente para cada ejercicio.

Evalúa diferentes algoritmos de clasificación.

Como mínimo, considera:

1. Logistic Regression.
2. Support Vector Machine (SVM).
3. Random Forest.
4. Extra Trees.
5. HistGradientBoosting.
6. XGBoost, si está disponible.
7. CatBoost, si está disponible.
8. LightGBM, si está disponible.

No es obligatorio utilizar todas las librerías externas si no están instaladas. Prioriza modelos que puedan ejecutarse de manera reproducible en el entorno actual.

## 4.1 Pipelines

Implementa los modelos mediante `sklearn.pipeline.Pipeline`.

Ejemplo conceptual:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])
```

Adapta el preprocesamiento a cada algoritmo.

Por ejemplo:

- Logistic Regression y SVM requieren especial atención a la escala de las características.
- Random Forest y Extra Trees no requieren escalado obligatorio.
- Los modelos basados en árboles pueden trabajar directamente con las características numéricas.

No apliques escalado innecesario a todos los algoritmos.

## 4.2 Cross Validation

La evaluación debe realizarse mediante Cross Validation de cinco folds.

**Regla crítica: utiliza validación agrupada por participante.**

Un mismo participante nunca puede aparecer simultáneamente en entrenamiento y validación dentro de un fold.

Utiliza `StratifiedGroupKFold` cuando la distribución de clases permita aplicarlo correctamente.

Si no es viable, utiliza `GroupKFold` y documenta la limitación.

No utilices `KFold` convencional ni `train_test_split` aleatorio sobre las ventanas, porque producirían Data Leakage.

Todos los modelos deben evaluarse utilizando exactamente las mismas particiones para que sus resultados sean comparables.

Si existen muy pocos participantes para cinco folds, informa del problema y propone una cantidad de folds viable. No inventes participantes ni dividas sus muestras aleatoriamente.

## 4.3 Métricas

Calcula las siguientes métricas para cada fold:

- Accuracy.
- Balanced Accuracy.
- Precision.
- Recall.
- F1-score.
- F1-score macro.
- ROC-AUC, cuando sea posible obtener probabilidades o puntuaciones de decisión.
- Average Precision / PR-AUC, cuando sea aplicable.

Presta especial atención a:

**Recall de la clase incorrecta:** permite medir qué proporción de las ejecuciones defectuosas consigue detectar el modelo.

**F1-score macro:** evita que una clase mayoritaria domine la evaluación.

**Balanced Accuracy:** permite comparar modelos cuando existe desbalance de clases.

Para cada modelo presenta:

- Media de cada métrica.
- Desviación estándar entre folds.
- Resultados individuales de cada fold.

No reportes únicamente el mejor fold.

## 4.4 Selección de hiperparámetros

Después de comparar los modelos iniciales, realiza una búsqueda de hiperparámetros para los candidatos más prometedores.

Puedes utilizar `RandomizedSearchCV` o `GridSearchCV`.

La búsqueda debe respetar la separación por grupos en todos sus niveles de validación.

Si realizas tuning y reportas el desempeño de selección utilizando los mismos folds, aclara que las métricas pueden ser optimistas. Para una estimación menos sesgada, utiliza validación anidada (Nested Cross Validation) o reserva un conjunto de participantes independiente para evaluación final.

No elijas un modelo simplemente porque tenga el mayor Accuracy.

## 4.5 Criterio de selección

Selecciona un modelo para Plank y otro para Hollow Body Hold.

Utiliza los siguientes criterios:

1. F1-score macro como métrica principal.
2. Recall de la clase incorrecta como métrica de seguridad.
3. Balanced Accuracy.
4. Estabilidad entre folds, observando la desviación estándar.

Si dos modelos presentan resultados similares, prefiere el más sencillo, rápido y estable.

No fuerces la selección de un modelo si los resultados no son suficientemente confiables.

---

# 5. Fase 3: Análisis de resultados

Genera una tabla comparativa independiente para cada ejercicio.

Ejemplo:

| Modelo | F1 Macro | Recall Incorrect | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | ... | ... | ... | ... |
| SVM | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |
| Extra Trees | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... |

Los resultados deben mostrar media y desviación estándar.

Genera también:

- Matriz de confusión.
- Classification Report.
- Curvas ROC y Precision-Recall, cuando sean aplicables.
- Comparación visual de métricas.
- Importancia de características para modelos que permitan obtenerla.
- Análisis de errores de clasificación.

Identifica especialmente los falsos negativos de la clase incorrecta: ejecuciones defectuosas clasificadas como correctas.

Documenta qué características parecen contribuir más a la clasificación.

No presentes la importancia de características como causalidad biomecánica.

---

# 6. Fase 4: Entrenamiento de los modelos definitivos

Una vez seleccionado el mejor algoritmo para cada ejercicio:

1. Recupera la configuración óptima de hiperparámetros.
2. Construye el Pipeline definitivo.
3. Entrena el Pipeline utilizando todos los datos disponibles del ejercicio.
4. Guarda el modelo completo, incluyendo imputación, escalado y clasificador.

Utiliza `joblib`.

Los modelos deben guardarse independientemente:

```text
models/
├── plank/
│   ├── model.joblib
│   └── model_metadata.json
└── hollow_body/
    ├── model.joblib
    └── model_metadata.json
```

El archivo `model_metadata.json` debe contener:

- Nombre del ejercicio.
- Algoritmo seleccionado.
- Hiperparámetros.
- Características utilizadas y su orden.
- Variable objetivo y codificación de clases.
- Métricas obtenidas.
- Estrategia de Cross Validation.
- Número de participantes y muestras.
- Versión de Python.
- Versiones de librerías relevantes.
- Fecha de entrenamiento.

Es indispensable conservar el orden exacto de las características utilizadas durante el entrenamiento.

---

# 7. Fase 5: Arquitectura de inferencia en tiempo real

Ahora desarrolla la infraestructura que permitirá utilizar los modelos con una cámara.

No implementes dos pipelines diferentes de extracción de características.

La extracción de landmarks y el feature engineering deben reutilizar las mismas funciones utilizadas durante la generación del dataset Gold.

El objetivo es evitar diferencias entre las características calculadas durante entrenamiento y las calculadas durante inferencia.

## 7.1 Captura de cámara

Utiliza OpenCV para:

- Abrir la cámara.
- Capturar los frames.
- Mostrar el video en tiempo real.
- Cerrar correctamente la cámara al presionar `Q`.
- Gestionar errores si la cámara no está disponible.

Utiliza MediaPipe Pose Landmarker para detectar los landmarks corporales.

Configura el detector de acuerdo con el procesamiento realizado previamente, utilizando los timestamps correctamente si se trabaja en modo VIDEO.

Mantén la misma definición de coordenadas, selección de landmarks, normalización y fórmulas utilizadas durante el procesamiento Gold.

No inventes nuevas fórmulas de características para la aplicación en vivo.

## 7.2 Extracción de características

Cada aplicación debe:

1. Obtener un frame de la cámara.
2. Ejecutar MediaPipe.
3. Obtener los landmarks.
4. Validar la calidad de detección.
5. Calcular las características del ejercicio correspondiente.
6. Construir una ventana temporal.
7. Aplicar las mismas transformaciones que durante el entrenamiento.
8. Obtener una predicción del modelo.

Recuerda que los modelos fueron entrenados con ventanas temporales, no necesariamente con frames individuales.

Por tanto, reproduce la misma lógica de ventanas utilizada en Gold.

Inicialmente, utiliza la configuración establecida en el procesamiento del dataset: ventanas de 2 segundos y frecuencia de muestreo de 5 FPS, siempre que coincida con la configuración real de entrenamiento.

No generes una predicción definitiva hasta disponer de una ventana válida.

Si una ventana no tiene suficientes características válidas, no fuerces una predicción.

## 7.3 Calidad de detección

Antes de realizar una predicción:

- Comprueba que se detecte una pose.
- Valida los landmarks necesarios.
- Comprueba visibility y presence cuando estén disponibles.
- Evita calcular ángulos o distancias con coordenadas inválidas.
- Detecta si la persona está fuera del encuadre.

Si la calidad de detección es insuficiente, muestra:

`No se puede evaluar: mejora la visibilidad de tu cuerpo.`

No clasifiques una detección fallida como ejecución incorrecta.

---

# 8. Fase 6: Aplicación Plank

Crea:

```text
evaluate_plank.py
```

La aplicación debe cargar exclusivamente el modelo de Plank.

Al ejecutarse:

1. Abrir la cámara.
2. Mostrar el video.
3. Detectar la pose.
4. Dibujar el esqueleto de MediaPipe sobre la imagen.
5. Calcular las características correspondientes a Plank.
6. Acumular las muestras necesarias para construir una ventana.
7. Ejecutar la predicción.
8. Mostrar el resultado.

La interfaz debe mostrar:

- Nombre del ejercicio: PLANK.
- Estado de detección de pose.
- Estado de evaluación.
- Predicción: CORRECT o INCORRECT.
- Confianza del modelo, únicamente si está disponible y correctamente interpretada.
- Estado de calidad de la detección.

Ejemplo visual conceptual:

```text
================================
             PLANK
================================

Pose: DETECTADA

Evaluación: CORRECT

Confianza: 92.4%

================================
Presiona Q para salir
```

Si el modelo predice una ejecución incorrecta, muestra:

`Postura incorrecta. Revisa tu alineación corporal.`

No inventes recomendaciones biomecánicas específicas a partir de una clasificación binaria.

---

# 9. Fase 7: Aplicación Hollow Body Hold

Crea:

```text
evaluate_hollow_body.py
```

Debe seguir la misma arquitectura general de Plank, pero utilizar exclusivamente:

- El modelo de Hollow Body Hold.
- Las características específicas de Hollow Body Hold.
- Su propio proceso de acumulación de ventanas.

La interfaz debe mostrar:

- Nombre del ejercicio: HOLLOW BODY HOLD.
- Estado de detección de pose.
- Estado de evaluación.
- Predicción: CORRECT o INCORRECT.
- Confianza, si está disponible.
- Estado de calidad de detección.

No reutilices accidentalmente el modelo ni las características de Plank.

---

# 10. Fase 8: Estabilización de predicciones

Las predicciones de cada ventana pueden variar entre sí.

Implementa un mecanismo sencillo de estabilización temporal.

Puedes utilizar:

- Suavizado mediante últimas N predicciones.
- Voto mayoritario.
- Umbral de confianza.
- Histéresis para evitar cambios constantes entre CORRECT e INCORRECT.

Documenta el mecanismo elegido.

No confundas la probabilidad estimada por el clasificador con una probabilidad real de que la postura sea biomecánicamente correcta.

Si el modelo no ofrece `predict_proba`, no inventes un porcentaje de confianza.

No permitas que una ventana antigua permanezca indefinidamente en el historial. Mantén una ventana deslizante actualizada.

---

# 11. Arquitectura de código sugerida

Organiza el proyecto de manera modular.

Una estructura inicial sugerida:

```text
project/
│
├── data/
│   └── gold/
│       ├── plank_features.parquet
│       └── hollow_body_features.parquet
│
├── models/
│   ├── plank/
│   │   ├── model.joblib
│   │   └── model_metadata.json
│   └── hollow_body/
│       ├── model.joblib
│       └── model_metadata.json
│
├── src/
│   ├── training/
│   │   ├── train_plank.py
│   │   ├── train_hollow_body.py
│   │   ├── evaluation.py
│   │   └── model_selection.py
│   │
│   ├── inference/
│   │   ├── camera.py
│   │   ├── pose_detector.py
│   │   ├── feature_extractor.py
│   │   ├── temporal_buffer.py
│   │   └── predictor.py
│   │
│   └── common/
│       ├── config.py
│       └── logger.py
│
├── reports/
│   ├── plank/
│   └── hollow_body/
│
├── evaluate_plank.py
├── evaluate_hollow_body.py
├── requirements.txt
└── README.md
```

Puedes adaptar esta arquitectura a la estructura existente del repositorio. No dupliques módulos que ya existan.

---

# 12. Fase 9: Pruebas y validación

Antes de considerar terminado el entregable, realiza las siguientes pruebas.

### Entrenamiento

- Ambos modelos se entrenan independientemente.
- Los dos datasets utilizan las características correctas.
- No existen identificadores entre las variables predictoras.
- No existe Data Leakage entre participantes.
- Todos los modelos se evalúan con las mismas particiones.
- Las métricas se guardan correctamente.
- Los modelos definitivos pueden cargarse mediante joblib.

### Inferencia

- Cada script carga el modelo correcto.
- La cámara se abre correctamente.
- MediaPipe detecta los landmarks.
- Las características calculadas tienen el mismo nombre, orden y definición que las de entrenamiento.
- Las ventanas temporales se construyen correctamente.
- El modelo produce predicciones válidas.
- La aplicación gestiona correctamente la ausencia de detección.
- La cámara se libera al cerrar la aplicación.

### Prueba de consistencia

Crea una prueba que utilice una misma muestra Gold y compare:

1. Las características originales almacenadas en Gold.
2. Las características reconstruidas utilizando el módulo de inferencia.

Deben coincidir dentro de una tolerancia numérica razonable.

Esta prueba es especialmente importante para evitar Training-Serving Skew.

---

# 13. Entregables finales

Al terminar, proporciona:

1. Modelo definitivo de Plank.
2. Modelo definitivo de Hollow Body Hold.
3. Comparación de algoritmos para cada ejercicio.
4. Métricas de Cross Validation con media y desviación estándar.
5. Matrices de confusión y reportes de clasificación.
6. Scripts de entrenamiento.
7. `evaluate_plank.py`.
8. `evaluate_hollow_body.py`.
9. Módulos reutilizables de inferencia.
10. `requirements.txt`.
11. README con instrucciones para instalar dependencias y ejecutar ambas aplicaciones.

El README debe explicar:

```bash
# Entrenar los modelos
python -m src.training.train_plank
python -m src.training.train_hollow_body

# Evaluar Plank en tiempo real
python evaluate_plank.py

# Evaluar Hollow Body Hold en tiempo real
python evaluate_hollow_body.py
```

Adapta los comandos a los módulos y puntos de entrada que realmente implementes.

---

# 14. Restricciones importantes

- No modificar ni sobrescribir los datasets Gold originales.
- No utilizar datos sintéticos en este primer experimento.
- No introducir Data Leakage.
- No realizar divisiones aleatorias de ventanas que compartan participantes.
- No utilizar metadata como características predictoras.
- No calcular características diferentes entre entrenamiento e inferencia.
- No inventar métricas ni resultados.
- No seleccionar modelos únicamente por Accuracy.
- No realizar predicciones cuando la detección de pose sea insuficiente.
- No implementar interfaces complejas: OpenCV es suficiente para este primer entregable.
- Mantener código modular, reproducible y documentado.
- Evitar sobreingeniería.

**Prioridad de implementación:** primero completar y validar los experimentos de entrenamiento y selección de modelos. Después implementar la inferencia en tiempo real. No avances a la siguiente fase si la anterior presenta errores o resultados inconsistentes.

Al finalizar, presenta un resumen técnico con los dos modelos seleccionados, sus métricas, las razones de selección y las limitaciones conocidas del sistema.