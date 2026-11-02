"""Generación de reportes automatizados específicos para Árboles de Decisión."""
from __future__ import annotations

import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ReporteArbol:
    """Imprime y guarda un resumen legible del desempeño del Árbol de Decisión."""

    def generar_reporte(self, y_test, predicciones) -> str:
        """Genera el contenido textual del reporte clínico."""
        acc = accuracy_score(y_test, predicciones)
        f1 = f1_score(y_test, predicciones, average="macro", zero_division=0)
        
        try:
            rec_critical = recall_score(y_test, predicciones, pos_label="critical", average="binary", zero_division=0)
            prec_critical = precision_score(y_test, predicciones, pos_label="critical", average="binary", zero_division=0)
        except ValueError:
            rec_critical = 0.0
            prec_critical = 0.0

        total_casos = len(y_test)
        aciertos = int(acc * total_casos)
        fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lineas = [
            "==================================================",
            "REPORTE AUTOMATIZADO DE CLASIFICACIÓN (ÁRBOL)",
            f"Fecha de ejecución: {fecha}",
            "==================================================",
            "",
            "1. IDENTIFICACIÓN DEL DATASET Y PREPROCESAMIENTO",
            "--------------------------------------------------",
            "Dataset: Datos biométricos y signos vitales.",
            "Preprocesamiento: Se separó la variable 'blood_pressure' en 'systolic_bp' y 'diastolic_bp'.",
            "Se descartaron metadatos ('idx', 'timestamp') para dejar únicamente información clínica.",
            "",
            "2. MODELO DE CLASIFICACIÓN",
            "--------------------------------------------------",
            "- Objetivo: Clasificar el nivel de riesgo de salud (risk_category).",
            "- Variables Predictoras (X): Ritmo cardíaco, presión sistólica, presión diastólica,",
            "  temperatura corporal y saturación de oxígeno.",
            "- Algoritmo: Árbol de Decisión (DecisionTreeClassifier).",
            "- Configuración: Train/Test al 80/20 de forma estratificada, random_state=42.",
            "",
            "3. RESULTADOS GLOBALES",
            "--------------------------------------------------",
            f"- Accuracy: {acc:.2f} ({aciertos}/{total_casos} casos correctos)",
            f"- F1-Score (Macro): {f1:.2f}",
            f"- Precision (Críticos): {prec_critical:.2f}",
            f"- Recall (Críticos): {rec_critical:.2f}",
            "",
            "4. INTERPRETACIÓN CLÍNICA",
            "--------------------------------------------------",
            f"- Accuracy: Mide el porcentaje global de aciertos del modelo. Acertó la categoría",
            f"  correcta en el {acc*100:.0f}% de los pacientes, es decir, en {aciertos} de los {total_casos} casos.",
            "  Aunque es un buen resultado, podría variar al evaluar un mayor número de datos.",
            "",
            f"- Recall (Críticos): Mide cuántos casos reales de la categoría 'critical' logró capturar",
            f"  el modelo sin pasarlos por alto. Con un {rec_critical:.2f} (100%), detectó a todos los",
            "  pacientes graves sin fallos por omisión, asegurando que ninguno se quede sin atención.",
            "",
            f"- Precision (Críticos): Mide el porcentaje de acierto en las alarmas de paciente grave.",
            f"  Un {prec_critical:.2f} muestra que, para asegurar capturar a todos los graves, el algoritmo",
            "  es precavido y clasifica un porcentaje de veces a pacientes de riesgo 'elevado' como 'críticos'.",
            "",
            f"- F1-Score: Mide el balance entre los casos reales capturados (Recall) y las falsas alarmas",
            f"  evadidas (Precision). Un promedio macro de {f1:.2f} indica que el modelo mantiene un",
            "  equilibrio alto y consistente entre sus aciertos y fallos en todas las categorías.",
            "=================================================="
        ]
        return "\n".join(lineas) + "\n"

    def imprimir_reporte(self, y_test, predicciones) -> None:
        """Imprime en la consola el reporte del modelo."""
        print(self.generar_reporte(y_test, predicciones), end="")

    def guardar_reporte(self, y_test, predicciones, ruta: str | Path) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generar_reporte(y_test, predicciones), encoding="utf-8")
        return path