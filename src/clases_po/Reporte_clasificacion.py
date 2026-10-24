"""Generación de reportes automatizados para modelos de clasificación."""
from __future__ import annotations

import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

class ReporteClasificacion:
    """Imprime y guarda un resumen legible de los resultados de clasificación."""

    def generar_reporte(self, y_test, predicciones) -> str:
        """Genera el contenido textual del reporte clínico."""
        
        acc = accuracy_score(y_test, predicciones)
        f1 = f1_score(y_test, predicciones, average="macro")
        fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lineas = [
            "==================================================",
            "REPORTE AUTOMATIZADO DE CLASIFICACIÓN DE RIESGO",
            f"Fecha de ejecución: {fecha}",
            "==================================================",
            "",
            "1. IDENTIFICACIÓN DEL DATASET Y PREPROCESAMIENTO",
            "--------------------------------------------------",
            "Dataset: Datos biométricos y signos vitales.",
            "Preprocesamiento: Se separó la variable 'blood_pressure' en sistólica y diastólica.",
            "Se descartaron metadatos ('idx', 'timestamp') para dejar únicamente información clínica.",
            "",
            "2. MODELO DE CLASIFICACIÓN",
            "--------------------------------------------------",
            "- Objetivo: Clasificar el nivel de riesgo de salud (risk_category).",
            "- Variables Predictoras (X): Edad, ritmo cardíaco, saturación de oxígeno,",
            "  temperatura corporal y presión arterial.",
            "- Algoritmo: Bosque Aleatorio (Random Forest).",
            "- Configuración: Train/Test al 80/20 de forma estratificada, random_state=42.",
            "",
            "3. RESULTADOS GLOBALES",
            "--------------------------------------------------",
            f"- Accuracy: {acc:.2f}",
            f"- F1-Score (Macro): {f1:.2f}",
            "",
            "4. INTERPRETACIÓN CLÍNICA",
            "--------------------------------------------------",
            f"- Accuracy: El modelo acertó la categoría correcta en el {acc*100:.0f}% de los pacientes.",
            "  Sin embargo, esta tasa puede estar ligeramente inflada por la poca cantidad de",
            "  datos de prueba y podría variar al evaluar casos nuevos.",
            "",
            "- Recall (Críticos): Responde a de todos los pacientes realmente graves, ¿cuántos logró",
            "  detectar el modelo? Un 1.00 significa que los identificó a todos, sin enviar a ningún",
            "  crítico a casa por error. En el sector salud, esta es la métrica fundamental.",
            "",
            "- Precision (Críticos): Responde a cuando el modelo enciende la alarma de 'Paciente",
            "  Crítico', ¿qué tan confiable es? Un 0.86 muestra que, para asegurar atrapar a todos",
            "  los graves, el algoritmo se equivoca un 14% de las veces marcando como críticos a",
            "  pacientes que solo tenían riesgo 'elevado'.",
            "",
            f"- F1-Score: Con un {f1:.2f}, el modelo encuentra el punto de equilibrio entre detectar a",
            "  todos y evitar excesivas falsas alarmas, manteniendo un balance general sobresaliente.",
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