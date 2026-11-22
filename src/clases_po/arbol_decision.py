"""Módulo para la clasificación de riesgo de salud usando Árbol de Decisión."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Aumentar el límite de recursión a nivel global para renderizado gráfico
sys.setrecursionlimit(10000)


class ArbolDecision:
    """Encapsula el flujo completo de preprocesamiento, entrenamiento,
    validación cruzada y evaluación usando un Árbol de Decisión.
    """

    def __init__(
        self,
        target_column: str = "label",
        max_depth: int = 4,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.target_column = target_column
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.modelo = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones = None
        self.cv_resultados: dict[str, float] | None = None
        self.df_profundidades: pd.DataFrame | None = None
        self.total_registros: int = 0
        self.distribucion_clases: dict[str, int] = {}

    def preprocesar_datos(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Limpia el dataframe y prepara las variables X e y."""
        datos = df.copy()

        if self.target_column in datos.columns and self.target_column == "label":
            datos = datos.rename(columns={"label": "risk_category"})
            target_final = "risk_category"
        elif "risk_category" in datos.columns:
            target_final = "risk_category"
        else:
            target_final = self.target_column

        # Normalizar etiquetas a minúsculas sin espacios para evitar errores de clave
        datos[target_final] = datos[target_final].astype(str).str.strip().str.lower()

        if "blood_pressure" in datos.columns:
            bp_split = (
                datos["blood_pressure"]
                .astype(str)
                .str.split("/", expand=True)
                .astype(float)
            )
            datos = datos.assign(
                systolic_bp=bp_split[0].values,
                diastolic_bp=bp_split[1].values,
            ).drop(columns=["blood_pressure"])

        columnas_basura = ["idx", "timestamp"]
        datos = datos.drop(columns=[col for col in columnas_basura if col in datos.columns])

        X = datos.drop(columns=[target_final])
        y = datos[target_final]

        self.total_registros = len(datos)
        self.distribucion_clases = y.value_counts().to_dict()
        return X, y

    def justificar_max_depth(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        profundidades: list[int] | None = None,
        cv: int = 5,
    ) -> pd.DataFrame:
        """Evalúa distintas profundidades máximas mediante validación cruzada estratificada
        para justificar empíricamente la selección de max_depth.
        """
        if profundidades is None:
            profundidades = [1, 2, 3, 4, 5, 6]

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        filas = []

        for depth in profundidades:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
            scores = cross_validate(
                clf,
                X,
                y,
                cv=skf,
                scoring=["accuracy", "f1_macro"],
                return_train_score=True,
            )
            filas.append(
                {
                    "max_depth": depth,
                    "Train Accuracy (%)": round(scores["train_accuracy"].mean() * 100, 2),
                    "CV Accuracy (%)": round(scores["test_accuracy"].mean() * 100, 2),
                    "CV Std (±%)": round(scores["test_accuracy"].std() * 100, 2),
                    "CV F1-Macro": round(scores["test_f1_macro"].mean(), 4),
                }
            )

        self.df_profundidades = pd.DataFrame(filas)
        print(f"=== JUSTIFICACIÓN DE HIPERPARÁMETRO max_depth (Stratified {cv}-Fold CV) ===")
        print(self.df_profundidades.to_string(index=False))
        return self.df_profundidades

    def validacion_cruzada(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, float]:
        """Ejecuta validación cruzada estratificada sobre todo el dataset."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_validate(
            self.modelo,
            X,
            y,
            cv=skf,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        )

        self.cv_resultados = {
            "n_splits": cv,
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"])),
            "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "f1_macro_std": float(np.std(scores["test_f1_macro"])),
            "precision_macro_mean": float(np.mean(scores["test_precision_macro"])),
            "recall_macro_mean": float(np.mean(scores["test_recall_macro"])),
        }

        print(f"\n=== VALIDACIÓN CRUZADA ESTRATIFICADA ({cv} PLIEGUES - 100% DATOS) ===")
        print(
            f"Accuracy promedio : {self.cv_resultados['accuracy_mean']*100:.2f}% "
            f"(± {self.cv_resultados['accuracy_std']*100:.2f}%)"
        )
        print(
            f"F1-Score Macro    : {self.cv_resultados['f1_macro_mean']:.4f} "
            f"(± {self.cv_resultados['f1_macro_std']:.4f})"
        )
        return self.cv_resultados

    def dividir_datos(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Divide los datos en entrenamiento y prueba de forma estratificada."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def entrenar(self) -> None:
        """Entrena el modelo de Árbol de Decisión."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Debes ejecutar 'dividir_datos' antes de entrenar.")
        self.modelo.fit(self.X_train, self.y_train)

    def predecir(self) -> pd.Series:
        """Realiza predicciones sobre el conjunto de prueba."""
        if self.X_test is None:
            raise ValueError("No hay datos de prueba para predecir.")
        self.predicciones = self.modelo.predict(self.X_test)
        return self.predicciones

    def evaluar(self) -> dict[str, float]:
        """Calcula y muestra el reporte de métricas y la matriz de confusión."""
        if self.y_test is None or self.predicciones is None:
            raise ValueError("Debes ejecutar 'predecir' antes de evaluar.")

        acc = accuracy_score(self.y_test, self.predicciones)
        f1 = f1_score(self.y_test, self.predicciones, average="macro")

        print("\n=== REPORTE DE CLASIFICACIÓN (HOLD-OUT 70/30) ===")
        print(classification_report(self.y_test, self.predicciones, zero_division=0))

        matriz = confusion_matrix(self.y_test, self.predicciones)
        df_matriz = pd.DataFrame(
            matriz,
            index=[f"Real: {c}" for c in self.modelo.classes_],
            columns=[f"Pred: {c}" for c in self.modelo.classes_],
        )
        print("=== MATRIZ DE CONFUSIÓN ===")
        print(df_matriz)

        return {"accuracy": acc, "f1_score": f1}

    def obtener_importancia_variables(self) -> pd.DataFrame:
        """Retorna un DataFrame ordenado con el peso de cada variable predictora."""
        if self.X_train is None:
            raise ValueError("El modelo debe entrenarse antes de consultar la importancia de variables.")

        return pd.DataFrame(
            {
                "Variable": self.X_train.columns,
                "Importancia (%)": self.modelo.feature_importances_ * 100,
            }
        ).sort_values(by="Importancia (%)", ascending=False)

    def graficar_matriz_confusion(
        self,
        ruta_guardado: str | Path | None = None,
    ) -> None:
        """Muestra y guarda visualmente la matriz de confusión del conjunto de prueba."""
        if self.y_test is None or self.predicciones is None:
            raise ValueError("Debes ejecutar 'predecir' antes de graficar la matriz.")

        matriz = confusion_matrix(
            self.y_test,
            self.predicciones,
            labels=self.modelo.classes_,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=self.modelo.classes_,
        ).plot(cmap="Blues", values_format="d", ax=ax)
        ax.set_title("Matriz de confusión - Conjunto de prueba (70/30)")
        ax.set_xlabel("Riesgo predicho")
        ax.set_ylabel("Riesgo real")
        plt.tight_layout()

        if ruta_guardado is not None:
            path = Path(ruta_guardado)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        plt.show()

    def obtener_parametros(self) -> dict[str, object]:
        """Retorna los hiperparámetros activos del modelo."""
        return {
            "criterion": self.modelo.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.modelo.min_samples_split,
            "min_samples_leaf": self.modelo.min_samples_leaf,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "profundidad_real_alcanzada": (
                self.modelo.get_depth() if hasattr(self.modelo, "tree_") else self.max_depth
            ),
        }

    def graficar_arbol(
        self,
        figsize: tuple[int, int] = (16, 8),
        ruta_guardado: str | Path | None = None,
    ) -> None:
        """Genera, guarda (opcional) y muestra el diagrama del Árbol de Decisión."""
        if self.X_train is None:
            raise ValueError("El modelo debe estar entrenado antes de graficarlo.")

        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            self.modelo,
            feature_names=list(self.X_train.columns),
            class_names=[str(c) for c in self.modelo.classes_],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
        )
        ax.set_title(
            f"Diagrama de Decisiones del Modelo Clínico (max_depth={self.max_depth})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if ruta_guardado is not None:
            path = Path(ruta_guardado)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        plt.show()

    def mostrar_reglas(self) -> str:
        """Retorna las reglas de decisión formateadas en texto plano."""
        if self.X_train is None:
            raise ValueError("El modelo debe entrenado antes de mostrar las reglas.")
        return export_text(self.modelo, feature_names=list(self.X_train.columns))