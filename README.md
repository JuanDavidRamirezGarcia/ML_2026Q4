# Predicción de Nivel de Riesgo en Salud (ML_2026Q4)

Este repositorio contiene la solución modular para las **Entregas 1 y 2** del curso de Machine Learning. El proyecto establece un pipeline de datos ejecutable y reproducible en Python y GitHub Codespaces, encargándose de la carga automatizada de datos, la ingeniería de características antropométricas y la generación de reportes resumidos.

## 🎯 Objetivo de la actividad 1
Implementar y automatizar un flujo de trabajo reproducible que permita:
1. Crea el repositorio y registra tus avances
2. Crea una clase que obtenga un DataFrame desde Hugging Face
3. Crea una clase que resuma un DataFrame en un .txt
4. Coordina las dos clases y transforma los datos

## 🎯 Objetivo de la actividad 2
Implementar un flujo de trabajo reproducible que permita:
1. **Construcción automática del DataFrame**: Obtención de datos clínicos directamente desde Hugging Face Datasets sin descargas manuales.
2. **Preprocesamiento y Limpieza**: Separación de variables de presión arterial (`blood_pressure`), eliminación de metadatos no clínicos (`idx`, `timestamp`) y mitigación de fuga de datos (*data leakage*).
3. **Modelo de Regresión Lineal**: Predicción del Índice de Masa Corporal (`bmi`) a partir de la edad (`age`) e interpretación de métricas ($R^2$, MAE, RMSE).
4. **Modelo de Clasificación (Random Forest)**: Clasificación de categorías de riesgo clínico (`risk_category`) optimizando el *Recall* en la categoría Crítica.
5. **Generación de Reportes Automatizados**: Exportación automática de resúmenes interpretativos en archivos `.txt`.
6. **Integración CI/CD**: Ejecución automatizada de todo el pipeline mediante GitHub Actions (`ejecutando WF tareas`).


## 👥 Integrantes del Grupo
* Juan Fernando Cataño Higuita
* Juan David Ramírez García

## Docente: 
Andrés Felipe Callejas Jaramillo

---

# Predicción de Nivel de Riesgo en Salud y Modelado Predictivo (ML_2026Q4)

Este repositorio contiene la solución modular y automatizada para el proyecto de Machine Learning. El sistema implementa un pipeline end-to-end ejecutable en Python y GitHub Actions que abarca desde la carga automatizada de datos biométricos desde Hugging Face, el preprocesamiento libre de *data leakage*, hasta el entrenamiento, evaluación y generación de reportes de modelos de **Regresión Lineal** y **Clasificación (Random Forest)**.

---

## 📊 Dataset Utilizado
- **Fuente**: Hugging Face Datasets
- **Identificador**: Health Vital Signs & Risk Predictor
- **Variables principales**:
  - `age`: Edad del paciente (Predictora en Regresión).
  - `bmi`: Índice de Masa Corporal (Variable Objetivo en Regresión).
  - `blood_pressure`: Presión arterial dividida en `systolic` y `diastolic`.
  - `heart_rate`, `oxygen_saturation`, `body_temperature`: Signos vitales biométricos.
  - `risk_category`: Categoría de riesgo en salud (Variable Objetivo en Clasificación: *Normal*, *Elevated*, *Critical*).

---

## 📁 Estructura del Repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ejecutar_WF_actividad_1.yml    # Workflow de GitHub Actions (ejecutando WF tareas)
├── src/
│   ├── clases_po/                          # Módulos Python orientados a objetos (.py)
│   │   ├── __init__.py
│   │   ├── hf_loader.py                    # Carga automática de datos desde Hugging Face
│   │   ├── clasificacion_riesgo.py         # Lógica de entrenamiento y evaluación del clasificador
│   │   ├── Reporte_clasificacion.py        # Generación de reportes .txt para Clasificación
│   │   └── Reporte_regresion.py            # Generación de reportes .txt para Regresión
│   └── sincronicas/
│       ├── datos/
│       │   ├── health_risk_predictor.csv   # Dataset procesado de entrada
│       │   └── healthvitalSigns.csv        # Dataset secundario de signos vitales
│       ├── reportes/
│       │   ├── health_risk_predictor.txt   # Resumen de ingesta de datos
│       │   ├── reporte_clasificacion.txt  # Reporte automático de Clasificación
│       │   └── reporte_regresion.txt      # Reporte automático de Regresión
│       ├── Clasificacion.ipynb             # Notebook ejecutable del modelo de Clasificación
│       ├── Entrega_1.ipynb                 # Notebook ejecutable de ingesta y exploración
│       └── Regresion_Lineal.ipynb          # Notebook ejecutable del modelo de Regresión Lineal
├── .gitignore                              # Archivos e historiales excluidos
├── README.md                               # Documentación general del proyecto
└── requirements.txt                        # Dependencias y librerías del proyecto
