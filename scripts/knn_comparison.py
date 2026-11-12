"""Compara distancias euclidiana y Manhattan para KNN en el dataset WDBC."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "wdbc.data"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carga las 30 características y codifica el diagnóstico como 0/1."""
    data = pd.read_csv(DATA_PATH, header=None)
    if data.shape[1] != 32:
        raise ValueError(f"Se esperaban 32 columnas en el WDBC; se encontraron {data.shape[1]}.")

    labels = data.iloc[:, 1].map({"B": 0, "M": 1})
    if labels.isna().any():
        raise ValueError("El dataset contiene diagnósticos distintos de B y M.")

    features = data.iloc[:, 2:].astype(float)
    return features, labels.astype(int)


def prepare_data(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Divide los datos de forma estratificada y escala usando solo train."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def main() -> None:
    """Carga, divide y escala los datos para el análisis KNN."""
    features, labels = load_data()
    print(f"Registros: {len(features)}")
    print(f"Características: {features.shape[1]}")
    print(f"Clases (0 = benigno, 1 = maligno):\n{labels.value_counts().sort_index()}")
    X_train, X_test, y_train, y_test = prepare_data(features, labels)
    print(f"\nEntrenamiento: {len(y_train)} | Prueba: {len(y_test)}")
    print(f"Features escaladas: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
