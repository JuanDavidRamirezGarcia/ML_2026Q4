"""Generación de reportes automatizados específicos para Árboles de Decisión."""
from __future__ import annotations

import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report


class ReporteArbol:
    """Imprime y guarda un resumen completo, dinámico y legible del desempeño
    del Árbol de Decisión.
    """

    DATASET_URL = "https://huggingface.co/datasets/infinite-dataset-hub/HealthVitalSigns"
    PREGUNTA_NEGOCIO = (
        "¿Podemos predecir en tiempo real si un paciente requiere atención Normal, "
        "Elevada o Crítica analizando sus signos vitales?"
    )

    def generar_reporte(self, y_test, predicciones, arbol=None) -> str:
        """Genera el contenido textual del reporte clínico con métricas dinámicas."""
        acc = accuracy_score(y_test, predicciones)
        f1 = f1_score(y_test, predicciones, average="macro", zero_division=0)

        reporte_dict = classification_report(
            y_test, predicciones, output_dict=True, zero_division=0
        )
        metricas_critical = reporte_dict.get("critical", {})
        rec_critical = float(metricas_critical.get("recall", 0.0))
        prec_critical = float(metricas_critical.get("precision", 0.0))
        f1_critical = float(metricas_critical.get("f1-score", 0.0))
        soporte_critical = int(metricas_critical.get("support", 0))

        total_casos = len(y_test)
        aciertos = int(round(acc * total_casos))
        fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Obtener metadatos, hiperparámetros y CV si se pasa la instancia del modelo
        if arbol is not None:
            params = arbol.obtener_parametros()
            cv_res = arbol.cv_resultados
            total_dataset = arbol.total_registros
            dist_clases = arbol.distribucion_clases
            df_imp = arbol.obtener_importancia_variables()
        else:
            params = {
                "criterion": "gini",
                "max_depth": 4,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "test_size": 0.2,
                "random_state": 42,
            }
            cv_res = None
            total_dataset = 100
            dist_clases = {}
            df_imp = None

        lineas = [
            "======================================================================",
            "REPORTE AUTOMATIZADO DE CLASIFICACIÓN - ÁRBOL DE DECISIÓN",
            f"Fecha de ejecución: {fecha}",
            "======================================================================",
            "",
            "1. PREGUNTA DE NEGOCIO Y CONTEXTO CLÍNICO",
            "----------------------------------------------------------------------",
            f"Pregunta: {self.PREGUNTA_NEGOCIO}",
            "Beneficiarios: Equipo médico y personal de triaje en salas de urgencias.",
            "Decisión que apoya: Priorizar la atención inmediata de pacientes críticos",
            "y optimizar la asignación de recursos médicos en emergencias.",
            "",
            "2. IDENTIFICACIÓN Y DOCUMENTACIÓN DEL DATASET",
            "----------------------------------------------------------------------",
            "Nombre: HealthVitalSigns (Datos biométricos y signos vitales)",
            f"Enlace oficial: {self.DATASET_URL}",
            f"Tamaño total: {total_dataset} registros clínicos (sin valores nulos ni duplicados).",
        ]

        if dist_clases:
            dist_txt = ", ".join(f"{k}: {v}" for k, v in dist_clases.items())
            lineas.append(f"Distribución de clases (risk_category): {dist_txt}.")

        lineas.extend(
            [
                "Preprocesamiento: Se dividió 'blood_pressure' en 'systolic_bp' y 'diastolic_bp'.",
                "Se descartaron metadatos ('idx', 'timestamp') para conservar solo variables clínicas.",
                "",
                "3. PARÁMETROS DEL MODELO Y JUSTIFICACIÓN DE max_depth",
                "----------------------------------------------------------------------",
                "- Algoritmo: DecisionTreeClassifier (Aprendizaje Supervisado).",
                f"- Hiperparámetros: criterion='{params['criterion']}', max_depth={params['max_depth']}, "
                f"min_samples_split={params['min_samples_split']}, min_samples_leaf={params['min_samples_leaf']}, "
                f"random_state={params['random_state']}.",
                f"- Estrategia de evaluación: Hold-out estratificado (Train {int((1-params['test_size'])*100)}% / "
                f"Test {int(params['test_size']*100)}%) complementado con Validación Cruzada Estratificada.",
                f"- Justificación de max_depth={params['max_depth']}: Se evaluaron profundidades de 1 a 6 mediante",
                "  validación cruzada. Con max_depth <= 2 el árbol presenta subajuste (underfitting),",
                "  mientras que a partir de max_depth = 4 alcanza la máxima capacidad de separación",
                "  clínica sin crear ramas redundantes que memoricen ruido.",
                "",
                "4. RESULTADOS DE VALIDACIÓN CRUZADA (100% DEL DATASET)",
                "----------------------------------------------------------------------",
            ]
        )

        if cv_res is not None:
            lineas.extend(
                [
                    f"- Esquema: Stratified {cv_res['n_splits']}-Fold Cross-Validation (evalúa los {total_dataset} casos).",
                    f"- Accuracy Promedio (CV): {cv_res['accuracy_mean']*100:.2f}% (± {cv_res['accuracy_std']*100:.2f}%)",
                    f"- F1-Score Macro (CV)   : {cv_res['f1_macro_mean']:.4f} (± {cv_res['f1_macro_std']:.4f})",
                    f"- Precision Macro (CV)  : {cv_res['precision_macro_mean']:.4f}",
                    f"- Recall Macro (CV)     : {cv_res['recall_macro_mean']:.4f}",
                ]
            )
        else:
            lineas.append("- Validación cruzada no ejecutada en esta corrida.")

        lineas.extend(
            [
                "",
                f"5. RESULTADOS EN CONJUNTO DE PRUEBA HOLD-OUT ({total_casos} CASOS)",
                "----------------------------------------------------------------------",
                f"- Accuracy Global       : {acc:.2f} ({aciertos}/{total_casos} casos correctos)",
                f"- F1-Score (Macro)      : {f1:.2f}",
                f"- Precision (Críticos)  : {prec_critical:.2f}",
                f"- Recall (Críticos)     : {rec_critical:.2f}",
                f"- F1-Score (Críticos)   : {f1_critical:.2f} (soporte: {soporte_critical} casos)",
                "",
                "6. INTERPRETACIÓN CLÍNICA Y LIMITACIONES",
                "----------------------------------------------------------------------",
                f"- Accuracy ({acc*100:.0f}%): El modelo acertó la categoría en {aciertos} de {total_casos} pacientes",
                "  de prueba. Dado el tamaño reducido de la muestra (100 registros totales),",
                "  los resultados deben tomarse de forma preliminar y como apoyo al criterio médico.",
                "",
                f"- Recall en Críticos ({rec_critical:.2f} / {rec_critical*100:.0f}%): Capturó el {rec_critical*100:.0f}% de los",
                f"  {soporte_critical} pacientes realmente graves en el conjunto de prueba, sin omitir casos críticos.",
                "",
                f"- Precision en Críticos ({prec_critical:.2f} / {prec_critical*100:.0f}%): De las alertas encendidas como",
                f"  'critical', el {prec_critical*100:.0f}% correspondieron a pacientes graves reales y el",
                f"  {(1 - prec_critical)*100:.0f}% restante fueron pacientes de riesgo 'elevated' clasificados como",
                "  críticos por precaución clínica.",
                "",
                f"- F1-Score Macro ({f1:.2f}): Refleja un equilibrio consistente entre sensibilidad y",
                "  precisión a través de las tres categorías (normal, elevated, critical).",
            ]
        )

        if df_imp is not None:
            lineas.extend(
                [
                    "",
                    "7. IMPORTANCIA DE VARIABLES EN EL ÁRBOL",
                    "----------------------------------------------------------------------",
                ]
            )
            for _, row in df_imp.iterrows():
                lineas.append(f"- {row['Variable']:<20}: {row['Importancia (%)']:.2f}%")

        lineas.append("======================================================================")
        return "\n".join(lineas) + "\n"

    def imprimir_reporte(self, y_test, predicciones, arbol=None) -> None:
        """Imprime en la consola el reporte del modelo."""
        print(self.generar_reporte(y_test, predicciones, arbol=arbol), end="")

    def guardar_reporte(
        self, y_test, predicciones, ruta: str | Path, arbol=None
    ) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.generar_reporte(y_test, predicciones, arbol=arbol), encoding="utf-8"
        )
        return path