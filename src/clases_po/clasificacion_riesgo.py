"""Clasificador de riesgo de salud usando Random Forest."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

class ClasificadorRiesgo:
    """Encapsula el flujo de clasificación para el dataset de salud."""

    def __init__(
        self,
        target_column: str = "label",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.modelo = RandomForestClassifier(random_state=self.random_state)
        
        # Atributos para guardar el estado de los datos
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones = None

    def preprocesar_datos(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Limpia el dataframe y prepara las variables X e y."""
        datos = df.copy()
        
        # 1. Renombrar la columna objetivo si se llama 'label'
        if self.target_column in datos.columns and self.target_column == "label":
            datos = datos.rename(columns={"label": "risk_category"})
            target_final = "risk_category"
        else:
            target_final = self.target_column

        # 2. Separar la presión arterial ("120/80" -> 120.0 y 80.0)
        if "blood_pressure" in datos.columns:
            datos[["systolic_bp", "diastolic_bp"]] = datos["blood_pressure"].str.split("/", expand=True).astype(float)
            datos = datos.drop(columns=["blood_pressure"])
            
        # 3. Eliminar columnas que no sirven para predecir
        columnas_basura = ["idx", "timestamp"]
        datos = datos.drop(columns=[col for col in columnas_basura if col in datos.columns])
        
        # 4. Separar X (predictoras) e y (objetivo)
        X = datos.drop(columns=[target_final])
        y = datos[target_final]
        return X, y

    def dividir_datos(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Divide los datos en entrenamiento y prueba de forma estratificada."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def entrenar(self) -> None:
        """Entrena el modelo de Random Forest."""
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
        """Calcula y muestra las métricas de evaluación."""
        if self.y_test is None or self.predicciones is None:
            raise ValueError("Debes ejecutar 'predecir' antes de evaluar.")
            
        acc = accuracy_score(self.y_test, self.predicciones)
        f1 = f1_score(self.y_test, self.predicciones, average="macro")
        
        print(f"Accuracy: {acc:.2f}")
        print(f"F1-Score (Macro): {f1:.2f}")
        print("\nReporte Detallado:")
        print(classification_report(self.y_test, self.predicciones))
        
        return {"accuracy": acc, "f1_score": f1}