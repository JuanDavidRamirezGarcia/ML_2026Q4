# Predicción de Nivel de Riesgo en Salud (ML_2026Q4)

Este repositorio contiene la solución modular para la **Entrega 1** del curso de Machine Learning. El proyecto establece un pipeline de datos ejecutable y reproducible en Python y GitHub Codespaces, encargándose de la carga automatizada de datos, la ingeniería de características antropométricas y la generación de reportes resumidos.

## 🎯 Objetivo del Trabajo
Implementar y automatizar un flujo de trabajo reproducible que permita:
1. Crea el repositorio y registra tus avances
2. Crea una clase que obtenga un DataFrame desde Hugging Face
3. Crea una clase que resuma un DataFrame en un .txt
4. Coordina las dos clases y transforma los datos


## 👥 Integrantes del Grupo
* Juan Fernando Cataño Higuita
* Juan David Ramírez García

## Docente: 
Andrés Felipe Callejas Jaramillo

---

## 📁 Estructura del Repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ejecutar-semana-1.yml          # Workflow de CI/CD para automatización
├── src/
│   ├── clases_po/                         # Módulos Python reutilizables (.py)
│   │   ├── __init__.py
│   │   ├── hf_loader.py                   # Carga de datos desde Hugging Face / CSV
│   │   └── reportes.py                    # Módulo de exportación de resúmenes .txt
│   └── sincronicas/
│       ├── datos/
│       │   └── health_risk_predictor.csv  # Dataset generado/almacenado
│       ├── reportes/
│       │   └── health_risk_predictor.txt  # Reporte generado por la ejecución
│       └── Entrega_1.ipynb                # Notebook ejecutable con análisis y LaTeX
├── .gitignore                             # Control para ignorar checkpoints y venv
├── README.md                              # Documentación del proyecto
└── requirements.txt                       # Dependencias del entorno de Python
