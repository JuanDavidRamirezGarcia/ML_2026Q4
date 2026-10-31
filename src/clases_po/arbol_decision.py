"""Módulo para el entrenamiento y evaluación del modelo de Árbol de Decisión."""
from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

class ArbolDecision:
    """Clase encargada de entrenar y evaluar el modelo de clasificación médica."""

    def __init__(self, max_depth: int = 4, random_state: int = 42):
        self.modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def entrenar(self, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
        """Guarda las variables de entrenamiento y ajusta el modelo."""
        self.X_train = X_train
        self.y_train = y_train
        self.modelo.fit(self.X_train, self.y_train)
        return self.modelo

    def predecir(self, X_test: pd.DataFrame = None) -> pd.Series:
        """Genera las predicciones sobre el conjunto de prueba."""
        if X_test is not None:
            self.X_test = X_test
        return self.modelo.predict(self.X_test)

    def obtener_matriz_confusion(self, y_test: pd.Series, predicciones) -> list:
        """Calcula la matriz de confusión."""
        self.y_test = y_test
        return confusion_matrix(self.y_test, predicciones)