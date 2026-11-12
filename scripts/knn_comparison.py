"""Compara distancias euclidiana y Manhattan para KNN en el dataset WDBC."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "wdbc.data"


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


def main() -> None:
    """Carga y resume los datos para el análisis KNN."""
    features, labels = load_data()
    print(f"Registros: {len(features)}")
    print(f"Características: {features.shape[1]}")
    print(f"Clases (0 = benigno, 1 = maligno):\n{labels.value_counts().sort_index()}")


if __name__ == "__main__":
    main()
