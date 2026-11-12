# Comparación de algoritmos de aprendizaje supervisado: KNN, SVM y MLP

Este proyecto implementa y compara tres algoritmos de aprendizaje supervisado —K-Nearest Neighbors (KNN), Support Vector Machine (SVM) y Multilayer Perceptron (MLP)— para resolver un problema de clasificación binaria: identificar si un tumor de mama es benigno o maligno.

Se utiliza el conjunto de datos Wisconsin Diagnostic Breast Cancer (WDBC). Cada registro contiene 30 características numéricas calculadas a partir de imágenes digitalizadas de muestras de masas mamarias. La variable objetivo es diagnosis, que contiene las clases B (benigno) y M (maligno).

El objetivo es evaluar el desempeño de los tres algoritmos utilizando la misma preparación de datos y el mismo conjunto de prueba, para que sus resultados puedan compararse de manera justa.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imjustVEGA/ActividadIA-Comparacion_Algoritmos/blob/main/notebooks/comparacion_knn_svm_mlp.ipynb)

## Equipo

| Integrante | Rol |
|---|---|
| Briceño Omaña Ismael Vladimir (líder) | Carga, exploración, preprocesamiento, MLP, integración y repositorio |
| Romero Arteaga Azael Fernando | KNN |
| Bermon Vazquez Owen Arturo | SVM |
| Chan Puc Mario Santiago | Comparación de modelos y visualización |
| Dzul Chi Frida Natalia | Análisis y conclusiones |
| Herrera Gomez Alison Valeria | Análisis, conclusiones y documentación (README) |

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**: 569 registros, 30 características numéricas
(10 rasgos de los núcleos celulares × media, error estándar y peor valor) y la variable objetivo
`diagnosis` (M = maligno, B = benigno). La descripción completa está en `data/wdbc.names`.

## Estructura del repositorio

```
.
├── notebooks/
│   └── comparacion_knn_svm_mlp.ipynb   # notebook con el desarrollo completo y ejecutado
├── data/
│   ├── wdbc.data                        # dataset (sin encabezado)
│   └── wdbc.names                       # descripción de las variables
├── scripts/
│   ├── knn_comparison.py                 # búsqueda y evaluación independiente de KNN
│   └── svm_kernels.py                   # exploración de kernels SVM (versión oficial en el notebook)
├── resultados/
│   ├── knn_accuracy_vs_k.png             # accuracy de KNN según k y distancia
│   ├── knn_confusion_matrices.png         # matrices de confusión de KNN
│   └── svm_confusion_matrix_best.png    # figura generada por scripts/svm_kernels.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Metodología

1. Carga y exploración del dataset (registros, variables, clases, nulos, distribución de clases).
2. Eliminación de la columna `id`; separación de X e y (maligno = 1, benigno = 0).
3. División 80/20 estratificada con `random_state=42` (la misma para los tres modelos).
4. Estandarización con `StandardScaler` ajustado solo con el entrenamiento (evita *data leakage*).
5. Entrenamiento de KNN, SVM y MLP (MLP optimizado con `GridSearchCV` y validación cruzada de 5 pliegues).
6. Evaluación con los mismos datos de prueba: accuracy, precision, recall, F1-score y matriz de confusión.
7. Tabla comparativa, gráficas y conclusiones.

## Resultados

| Modelo | Accuracy | Precision | Recall | F1-score | Errores |
|---|---|---|---|---|---|
| KNN | | | | | |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 3 (3 FN, 0 FP) |
| MLP | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 3 (3 FN, 0 FP) |

_Completar con la tabla de la sección 6 del notebook._

## Ejecución independiente de KNN

El script `scripts/knn_comparison.py` compara las distancias euclidiana y Manhattan con
`k` de 1 a 20. Divide el dataset local con `test_size=0.2`, `stratify=y` y
`random_state=42`; ajusta `StandardScaler` únicamente con el conjunto de entrenamiento.
Selecciona para cada distancia el `k` con mayor accuracy en el conjunto de prueba (en
caso de empate, elige el menor `k`) y muestra el reporte de clasificación, las matrices
de confusión y una tabla comparativa.

Desde la raíz del repositorio, ejecútalo con:

```bash
python scripts/knn_comparison.py
```

El script guarda `resultados/knn_accuracy_vs_k.png` (ambas curvas en una figura) y
`resultados/knn_confusion_matrices.png`.

## Cómo ejecutarlo

**En Google Colab (recomendado):** abrir el notebook con el botón "Abrir en Colab" y ejecutar
**Entorno de ejecución → Ejecutar todo**. El dataset se lee directamente de este repositorio
(variable `URL_GITHUB` en la sección 1.2).

**En local:**

```bash
git clone https://github.com/imjustVEGA/ActividadIA-Comparacion_Algoritmos.git
cd ActividadIA-Comparacion_Algoritmos
pip install -r requirements.txt
jupyter notebook notebooks/comparacion_knn_svm_mlp.ipynb
```
