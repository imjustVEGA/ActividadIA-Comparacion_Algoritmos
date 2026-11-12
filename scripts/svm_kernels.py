"""Compara kernels lineal y RBF para SVM en el dataset Breast Cancer.

Script exploratorio de Owen Arturo Bermon Vazquez. La versión oficial (integrada) está en
la sección 4 de notebooks/comparacion_knn_svm_mlp.ipynb, que usa la MISMA división
entrenamiento/prueba que KNN y MLP. Ejecutar desde la raíz del repositorio:
    python scripts/svm_kernels.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_searches() -> dict[str, GridSearchCV]:
    """Construye una búsqueda de hiperparámetros para cada kernel."""
    common_steps = [
        ("scaler", StandardScaler()),
    ]
    searches = {
        "lineal": GridSearchCV(
            Pipeline(common_steps + [("svc", SVC(kernel="linear"))]),
            param_grid={"svc__C": [0.01, 0.1, 1, 10, 100]},
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "RBF": GridSearchCV(
            Pipeline(common_steps + [("svc", SVC(kernel="rbf"))]),
            param_grid={
                "svc__C": [0.01, 0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
    }
    return searches


def print_comparison(searches: dict[str, GridSearchCV]) -> tuple[str, GridSearchCV]:
    """Muestra los mejores resultados de CV y devuelve el ganador."""
    print("Resumen de validacion cruzada:")
    for kernel_name, search in searches.items():
        print(
            f"- Kernel {kernel_name}: mejores hiperparametros="
            f"{search.best_params_}, accuracy CV={search.best_score_:.4f}"
        )

    best_kernel = max(searches, key=lambda name: searches[name].best_score_)
    return best_kernel, searches[best_kernel]


def evaluate_best_model(
    best_kernel: str,
    best_search: GridSearchCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: np.ndarray,
) -> None:
    """Evalua el mejor estimador en test e imprime y guarda sus resultados."""
    best_estimator = best_search.best_estimator_
    predictions = best_estimator.predict(X_test)

    print(f"\nEvaluacion en test del kernel {best_kernel}:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Classification report:\n"
        + classification_report(y_test, predictions, target_names=target_names)
    )

    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix (NumPy):")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=target_names,
    )
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title(f"Matriz de confusion: kernel {best_kernel}")
    display.figure_.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    display.figure_.savefig("resultados/svm_confusion_matrix_best.png", dpi=150)
    plt.close(display.figure_)
    print("Figura guardada en resultados/svm_confusion_matrix_best.png")


def main() -> None:
    """Carga los datos, ajusta las busquedas y evalua el kernel ganador."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    searches = create_searches()
    for search in searches.values():
        search.fit(X_train, y_train)

    best_kernel, best_search = print_comparison(searches)
    evaluate_best_model(
        best_kernel,
        best_search,
        X_test,
        y_test,
        dataset.target_names,
    )


if __name__ == "__main__":
    main()
