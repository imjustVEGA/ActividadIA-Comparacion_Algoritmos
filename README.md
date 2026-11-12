# Comparación de algoritmos de aprendizaje supervisado: KNN, SVM y MLP

Actividad práctica: implementar y comparar **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**
y una **red neuronal artificial (MLP)** para clasificar tumores de mama como **benignos** o **malignos**,
usando el mismo conjunto de datos y la misma división de entrenamiento y prueba.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imjustVEGA/ActividadIA-Comparacion_Algoritmos/blob/main/notebooks/comparacion_knn_svm_mlp.ipynb)

## Equipo

| Integrante | Rol |
|---|---|
| _Nombre (líder)_ | Carga, exploración, preprocesamiento, MLP e integración |
| _Nombre_ | KNN |
| _Nombre_ | SVM |
| _Nombre_ | Comparación de modelos |
| _Nombre_ | Conclusiones |

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
| SVM | | | | | |
| MLP | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 3 (3 FN, 0 FP) |

_Completar con la tabla de la sección 6 del notebook._

## Cómo ejecutarlo

**En Google Colab (recomendado):** abrir el notebook con el botón "Abrir en Colab" y ejecutar
**Entorno de ejecución → Ejecutar todo**. El dataset se lee directamente de este repositorio
(variable `URL_GITHUB` en la sección 1.2).

**En local:**

```bash
git clone https://github.com/imjustVEGA/ActividadIA-Comparacion_Algoritmos.git
cd REPOSITORIO
pip install -r requirements.txt
jupyter notebook notebooks/comparacion_knn_svm_mlp.ipynb
```
