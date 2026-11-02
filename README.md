# Predicción de Nivel de Riesgo en Salud (ML_2026Q4)

Este repositorio contiene la solución modular para las **Entregas 1, 2 y 3** del curso de Machine Learning. El proyecto establece un pipeline de datos ejecutable y reproducible en Python y GitHub Codespaces, encargándose de la carga automatizada de datos, la ingeniería de características antropométricas, la construcción de modelos predictivos y la generación de reportes resumidos.

---

## 🎯 Objetivo de la Actividad 1
Implementar y automatizar un flujo de trabajo reproducible que permita:
1. Crear el repositorio y registrar avances.
2. Crear una clase que obtenga un DataFrame desde Hugging Face.
3. Crear una clase que resuma un DataFrame en un `.txt`.
4. Coordinar las dos clases y transformar los datos.

## 🎯 Objetivo de la Actividad 2
Implementar un flujo de trabajo reproducible que permita:
1. **Construcción automática del DataFrame**: Obtención de datos clínicos directamente desde Hugging Face Datasets sin descargas manuales.
2. **Preprocesamiento y Limpieza**: Separación de variables de presión arterial (`blood_pressure`), eliminación de metadatos no clínicos (`idx`, `timestamp`) y mitigación de fuga de datos (*data leakage*).
3. **Modelo de Regresión Lineal**: Predicción del Índice de Masa Corporal (`bmi`) a partir de la edad (`age`) e interpretación de métricas ($R^2$, MAE, RMSE).
4. **Modelo de Clasificación (Random Forest)**: Clasificación de categorías de riesgo clínico (`risk_category`) optimizando el *Recall* en la categoría Crítica.
5. **Generación de Reportes Automatizados**: Exportación automática de resúmenes interpretativos en archivos `.txt`.
6. **Integración CI/CD**: Ejecución automatizada de todo el pipeline mediante GitHub Actions (`ejecutando WF tareas`).

## 🎯 Objetivo de la Actividad 3
Implementar y documentar un modelo interpretable de clasificación mediante **Árboles de Decisión**:
1. **Modelado Orientado a Objetos (`ArbolDecision`)**: Encapsular la lógica de scikit-learn para preprocesamiento, entrenamiento, evaluación y visualización sin advertencias de Pandas.
2. **Visualización y Explicabilidad**: Generar el diagrama gráfico del árbol de decisión y la exportación textual de sus reglas lógicas.
3. **Reportes Clínicos Automatizados (`ReporteArbol`)**: Adaptar el generador de reportes en `.txt` con métricas específicas y lecturas interpretativas orientadas al sector salud.
4. **Documentación del Algoritmo**: Explicar y justificar frente a la pregunta de negocio el funcionamiento del modelo.

---

## 👥 Integrantes del Grupo
- Juan Fernando Cataño Higuita
- Juan David Ramírez García

## 👨‍🏫 Docente 
Andrés Felipe Callejas Jaramillo

---

# 🩺 Predicción de Nivel de Riesgo en Salud y Modelado Predictivo

## 📊 Dataset Utilizado
- **Fuente**: Hugging Face Datasets
- **Identificador**: Health Vital Signs & Risk Predictor
- **Variables principales**:
  - `age`: Edad del paciente (Predictora en Regresión).
  - `bmi`: Índice de Masa Corporal (Variable Objetivo en Regresión).
  - `blood_pressure`: Presión arterial dividida en `systolic_bp` y `diastolic_bp`.
  - `heart_rate`, `oxygen_saturation`, `body_temperature`: Signos vitales biométricos.
  - `risk_category`: Categoría de riesgo en salud (Variable Objetivo en Clasificación: *Normal*, *Elevated*, *Critical*).

---

## 🧠 Selección y Justificación del Modelo frente a la Pregunta de Negocio

- **Tipo de aprendizaje y justificación**: Usamos un modelo de **aprendizaje supervisado** llamado **Árbol de Decisión** porque el conjunto de datos ya contaba con las etiquetas que responden a nuestra pregunta de negocio. La estrategia consiste en entrenar al modelo con una porción de estos datos etiquetados para que aprenda patrones clínicos y sepa clasificar de forma automática a los nuevos pacientes que ingresen al servicio de urgencias.
- **Variables elegidas y justificación**:
  - **Variable Objetivo (`risk_category`)**: Contiene las etiquetas *normal*, *elevated* y *critical*, que representan el diagnóstico que queremos predecir para priorizar la atención médica.
  - **Variables Predictoras (`heart_rate`, `systolic_bp`, `diastolic_bp`, `oxygen_saturation`, `body_temperature`)**: Son datos fisiológicos y signos vitales fundamentales que indican objetivamente el nivel de gravedad de una persona al momento de su evaluación.

---

## 🌲 Funcionamiento del Modelo

El árbol de decisión funciona respondiendo preguntas cerradas, en este caso preguntas relacionadas con nuestra pregunta de negocio. Por ejemplo, en la primera interacción, el algoritmo evalúa todos los signos vitales del paciente y selecciona cuál de ellos separa mejor a la población (ejemplo: *¿su frecuencia cardíaca es mayor a 100 bpm?*). 

Al responder **Sí** o **No**, divide a los pacientes en dos subgrupos: los que cumplen o no la condición. Luego, en las siguientes interacciones, repite exactamente el mismo proceso dentro de cada rama analizando el resto de las variables (como la presión arterial o la temperatura) para seguir refinando la clasificación. Finalmente, este flujo de preguntas termina en las "hojas" del árbol, donde se asigna el diagnóstico de riesgo definitivo del paciente (*normal*, *elevated* o *critical*).

---

## 📈 Resultados e Interpretación de Métricas Clínicas

- **Accuracy**: El modelo acertó la categoría correcta en el **90%** de los pacientes (es decir, 18 de 20 casos). Aunque es un buen resultado, podría variar si el dataset hubiera contado con un mayor número de datos de prueba.
- **Recall**: Mide cuántos casos reales de una categoría logró capturar el modelo sin pasarlos por alto. Un **1.00 (100%)** en la clase *critical* significa que detectó a los 6 pacientes graves reales y tuvo 0 fallos por omisión, asegurando que ningún paciente crítico se quede sin atención.
- **Precisión**: Mide el porcentaje de acierto en las alarmas que enciende el modelo. Un **0.75** en *critical* significa que acertó en 3 de cada 4 alertas de paciente grave y falló en 1, clasificando por precaución a 2 pacientes de riesgo *elevado* como si fueran *críticos*.
- **F1-Score**: Mide el balance entre los casos reales capturados (Recall) y las falsas alarmas evadidas (Precisión). Un promedio de **0.89** indica que el modelo mantiene un equilibrio alto y consistente entre sus aciertos y fallos en todas las categorías.
- **Importancia de Variables**: Mide el porcentaje de influencia que tuvo cada signo vital en las decisiones del árbol:
  - `heart_rate` (Frecuencia Cardíaca): **87.03%**
  - `systolic_bp` / `diastolic_bp` (Presión Arterial): **10.31%**
  - `body_temperature` (Temperatura Corporal): **2.66%**
  - `oxygen_saturation` (Saturación de Oxígeno): **0.00%** (no requirió realizar cortes en este árbol específico).

---

## 📁 Estructura del Repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ejecutar_WF_actividad_1.yml    # Workflow de GitHub Actions
├── src/
│   ├── clases_po/                          # Módulos Python orientados a objetos
│   │   ├── __init__.py
│   │   ├── hf_loader.py                    # Carga de datos desde Hugging Face
│   │   ├── clasificacion_riesgo.py         # Modelo Random Forest
│   │   ├── arbol_decision.py               # Modelo Árbol de Decisión
│   │   ├── Reporte_clasificacion.py        # Reportes .txt para Random Forest
│   │   ├── reporte_arbol.py                # Reportes .txt para Árbol de Decisión
│   │   └── Reporte_regresion.py            # Reportes .txt para Regresión
│   └── sincronicas/
│       ├── datos/
│       │   ├── health_risk_predictor.csv   # Dataset procesado de entrada
│       │   └── healthvitalSigns.csv        # Dataset secundario
│       ├── reportes/
│       │   ├── health_risk_predictor.txt   # Resumen de ingesta
│       │   ├── reporte_clasificacion.txt  # Reporte de Random Forest
│       │   ├── reporte_arbol_decision.txt # Reporte de Árbol de Decisión
│       │   └── reporte_regresion.txt      # Reporte de Regresión
│       ├── Arbol_Decision.ipynb            # Notebook ejecutable Árbol de Decisión
│       ├── Clasificacion.ipynb             # Notebook ejecutable Random Forest
│       ├── Entrega_1.ipynb                 # Notebook de ingesta y exploración
│       └── Regresion_Lineal.ipynb          # Notebook de Regresión Lineal
├── .gitignore                              # Archivos excluidos
├── README.md                               # Documentación general
└── requirements.txt                        # Dependencias del proyecto