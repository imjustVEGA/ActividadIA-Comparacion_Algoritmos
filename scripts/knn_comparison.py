"""Compara distancias euclidiana y Manhattan para KNN en el dataset WDBC."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "wdbc.data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "resultados"
RANDOM_STATE = 42
CLASS_NAMES = ["Benigno", "Maligno"]


@dataclass
class KNNResult:
    """Métricas y predicciones del mejor modelo para una distancia."""

    metric: str
    k: int
    estimator: KNeighborsClassifier
    predictions: np.ndarray
    accuracy: float


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


def search_k_values(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evalúa k=1..20 en el conjunto de prueba para ambas distancias."""
    rows = []
    for metric in ("euclidean", "manhattan"):
        for k in range(1, 21):
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rows.append(
                {
                    "distancia": metric,
                    "k": k,
                    "accuracy": accuracy_score(y_test, predictions),
                }
            )
    return pd.DataFrame(rows)


def plot_accuracy(scores: pd.DataFrame) -> Path:
    """Guarda en una sola figura las curvas de accuracy para ambas distancias."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, label in (("euclidean", "Euclidiana"), ("manhattan", "Manhattan")):
        metric_scores = scores[scores["distancia"] == metric]
        ax.plot(metric_scores["k"], metric_scores["accuracy"], marker="o", label=label)

    ax.set(
        title="Accuracy de KNN según k",
        xlabel="Número de vecinos (k)",
        ylabel="Accuracy",
        xticks=range(1, 21),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "knn_accuracy_vs_k.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def fit_best_models(
    scores: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, KNNResult]:
    """Entrena el modelo de mejor accuracy de cada métrica y reporta resultados."""
    results = {}
    for metric in ("euclidean", "manhattan"):
        best_row = (
            scores[scores["distancia"] == metric]
            .sort_values(["accuracy", "k"], ascending=[False, True])
            .iloc[0]
        )
        best_k = int(best_row["k"])
        model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[metric] = KNNResult(metric, best_k, model, predictions, accuracy)

        print(f"\n===== KNN {metric} (mejor k={best_k}) =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )
    return results


def main() -> None:
    """Carga, prepara los datos y compara los valores de k."""
    features, labels = load_data()
    print(f"Registros: {len(features)}")
    print(f"Características: {features.shape[1]}")
    print(f"Clases (0 = benigno, 1 = maligno):\n{labels.value_counts().sort_index()}")
    X_train, X_test, y_train, y_test = prepare_data(features, labels)
    print(f"\nEntrenamiento: {len(y_train)} | Prueba: {len(y_test)}")
    print(f"Features escaladas: {X_train.shape[1]}")
    scores = search_k_values(X_train, X_test, y_train, y_test)
    print("\nAccuracy por distancia y k:")
    print(scores.pivot(index="k", columns="distancia", values="accuracy").to_string())
    print(f"\nGráfica guardada en: {plot_accuracy(scores)}")
    fit_best_models(scores, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
