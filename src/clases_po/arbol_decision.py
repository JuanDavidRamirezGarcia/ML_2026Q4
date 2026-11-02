"""Módulo para la clasificación de riesgo de salud usando Árbol de Decisión."""
from __future__ import annotations

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Aumentar el límite de recursión a nivel global para renderizado gráfico en Python 3.14
sys.setrecursionlimit(10000)

class ArbolDecision:
    """Encapsula el flujo completo de preprocesamiento, entrenamiento y evaluación
    usando un Árbol de Decisión.
    """

    def __init__(
        self,
        target_column: str = "label",
        max_depth: int = 4,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.target_column = target_column
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.modelo = DecisionTreeClassifier(
            max_depth=self.max_depth, 
            random_state=self.random_state
        )
        
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones = None

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

        # Asignación limpia usando .assign() para evitar FutureWarning de Pandas
        if "blood_pressure" in datos.columns and datos["blood_pressure"].dtype == "object":
            bp_split = datos["blood_pressure"].str.split("/", expand=True).astype(float)
            datos = datos.assign(
                systolic_bp=bp_split[0].values,
                diastolic_bp=bp_split[1].values
            ).drop(columns=["blood_pressure"])
            
        columnas_basura = ["idx", "timestamp"]
        datos = datos.drop(columns=[col for col in columnas_basura if col in datos.columns])
        
        X = datos.drop(columns=[target_final])
        y = datos[target_final]
        return X, y

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
        
        print("=== REPORTE DE CLASIFICACIÓN ===")
        print(classification_report(self.y_test, self.predicciones))
        
        matriz = confusion_matrix(self.y_test, self.predicciones)
        df_matriz = pd.DataFrame(
            matriz,
            index=[f"Real: {c}" for c in self.modelo.classes_],
            columns=[f"Pred: {c}" for c in self.modelo.classes_]
        )
        print("\n=== MATRIZ DE CONFUSIÓN ===")
        print(df_matriz)
        
        return {"accuracy": acc, "f1_score": f1}

    def obtener_importancia_variables(self) -> pd.DataFrame:
        """Retorna un DataFrame ordenado con el peso de cada variable predictora."""
        if self.X_train is None:
            raise ValueError("El modelo debe entrenarse antes de consultar la importancia de variables.")
            
        return pd.DataFrame({
            'Variable': self.X_train.columns,
            'Importancia (%)': self.modelo.feature_importances_ * 100
        }).sort_values(by='Importancia (%)', ascending=False)

    def graficar_arbol(self, figsize: tuple[int, int] = (16, 8)) -> None:
        """Genera y muestra el diagrama del Árbol de Decisión."""
        if self.X_train is None:
            raise ValueError("El modelo debe estar entrenado antes de graficarlo.")

        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            self.modelo,
            feature_names=list(self.X_train.columns),
            class_names=[str(c) for c in self.modelo.classes_],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        ax.set_title("Diagrama de Decisiones del Modelo Clínico", fontsize=14, fontweight="bold")
        plt.show()

    def mostrar_reglas(self) -> str:
        """Retorna las reglas de decisión formateadas en texto plano."""
        if self.X_train is None:
            raise ValueError("El modelo debe estar entrenado antes de mostrar las reglas.")
        return export_text(self.modelo, feature_names=list(self.X_train.columns))