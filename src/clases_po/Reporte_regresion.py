"""Generación de reportes automatizados para modelos predictivos."""
from __future__ import annotations

import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ReporteRegresion:
    """Imprime y guarda un resumen legible de los resultados de regresión y clasificación."""

    def generar_reporte(self, y_test, predicciones, modelo, accuracy_clf=0.0, f1_clf=0.0) -> str:
        """Genera el contenido textual del reporte."""
        
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        r2 = r2_score(y_test, predicciones)
        w = modelo.coef_[0]
        fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lineas = [
            "==================================================",
            "REPORTE AUTOMATIZADO DE MODELOS PREDICTIVOS",
            f"Fecha de ejecución: {fecha}",
            "==================================================",
            "",
            "1. IDENTIFICACIÓN DEL DATASET Y PREPROCESAMIENTO",
            "--------------------------------------------------",
            "Dataset: Datos de salud y mediciones corporales.",
            "Preprocesamiento: Se mitigó el riesgo de fuga de datos (data leakage) descartando",
            "variables como 'weight' y 'height' para la predicción del BMI, dado que",
            "componen su fórmula matemática exacta.",
            "",
            "2. MODELO DE REGRESIÓN",
            "--------------------------------------------------",
            "- Objetivo: Estimar el Índice de Masa Corporal (BMI).",
            "- Variable Dependiente (y): bmi",
            "- Variable Predictora (X): age (seleccionada por evitar data leakage)",
            "- Algoritmo: Regresión Lineal Simple.",
            "- Configuración: Train/Test al 80/20, random_state=42.",
            "",
            "Resultados de Regresión:",
            f"- R²:   {r2:.2f}",
            f"- MAE:  {mae:.2f}",
            f"- RMSE: {rmse:.2f}",
            f"Interpretación: La edad explica el {r2*100:.0f}% de la varianza en el BMI.",
            f"El modelo presenta un error promedio absoluto de {mae:.2f} puntos al predecir casos nuevos.",
            "",
            "3. MODELO DE CLASIFICACIÓN",
            "--------------------------------------------------",
            "- Objetivo: Clasificar el nivel de riesgo de salud.",
            "- Algoritmo: Pendiente de definición.",
            "",
            "Resultados de Clasificación:",
            f"- Accuracy: {accuracy_clf:.2f}",
            f"- F1-Score: {f1_clf:.2f}",
            "",
            "4. CONCLUSIONES Y LIMITACIONES",
            "--------------------------------------------------",
            "Conclusión: Se logró modelar la tendencia del BMI de forma estadísticamente válida.",
            f"Limitaciones: Un R² de {r2:.2f} indica que faltan variables (como nutrición) para",
            "lograr estimaciones altamente precisas.",
            "=================================================="
        ]
        return "\n".join(lineas) + "\n"

    def imprimir_reporte(self, y_test, predicciones, modelo, accuracy_clf=0.0, f1_clf=0.0) -> None:
        """Imprime en la consola el reporte del modelo."""
        print(self.generar_reporte(y_test, predicciones, modelo, accuracy_clf, f1_clf), end="")

    def guardar_reporte(self, y_test, predicciones, modelo, ruta: str | Path, accuracy_clf=0.0, f1_clf=0.0) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generar_reporte(y_test, predicciones, modelo, accuracy_clf, f1_clf), encoding="utf-8")
        return path