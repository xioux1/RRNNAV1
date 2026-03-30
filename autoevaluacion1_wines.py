"""Autoevaluación 1: Clasificación del país de origen de vinos.

Modelos:
- Árbol de Decisión
- MLP (6, 12)
- MLP (100, 200)
- MLP (100, 200) + SMOTE

Uso:
    python autoevaluacion1_wines.py --data winemag-data-130k-v2.csv
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 17
TEST_SIZE_RATIO = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="winemag-data-130k-v2.csv",
        help="Ruta al CSV del dataset.",
    )
    return parser.parse_args()


def load_and_clean_dataset(path: str) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    datos = pd.read_csv(path)

    columnas_a_eliminar = [
        "region_2",
        "taster_twitter_handle",
        "designation",
        "Unnamed: 0",
        "description",
        "title",
        "winery",
    ]
    datos_copy = datos.drop(columns=[c for c in columnas_a_eliminar if c in datos.columns]).copy(deep=True)

    datos_copy = datos_copy.dropna(subset=["country"])

    if "price" in datos_copy.columns:
        datos_copy["price"] = datos_copy["price"].fillna(datos_copy["price"].median())

    for col in ["province", "region_1", "taster_name", "variety"]:
        if col in datos_copy.columns:
            datos_copy[col] = datos_copy[col].fillna("Unknown")

    numeric_features = ["price", "points"]
    categorical_features = ["province", "region_1", "taster_name", "variety"]

    required = numeric_features + categorical_features + ["country"]
    missing = [c for c in required if c not in datos_copy.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    X = datos_copy[numeric_features + categorical_features]
    y_raw = datos_copy["country"]

    min_samples_for_smote_train = 2
    min_class_samples_before_split = int(np.ceil(min_samples_for_smote_train / (1 - TEST_SIZE_RATIO)))

    class_counts = y_raw.value_counts()
    rare_classes = class_counts[class_counts < min_class_samples_before_split].index

    keep_idx = y_raw[~y_raw.isin(rare_classes)].index
    X = X.loc[keep_idx]
    y_raw = y_raw.loc[keep_idx]

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
    nombres_paises = list(le_target.classes_)

    return X, y, nombres_paises, numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> float:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    return acc


def main() -> None:
    args = parse_args()

    X, y, nombres_paises, numeric_features, categorical_features = load_and_clean_dataset(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline_arbol = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)),
        ]
    )
    pipeline_arbol.fit(X_train, y_train)
    acc_arbol = evaluate_model(
        "Árbol de Decisión (max_depth=8)",
        y_test,
        pipeline_arbol.predict(X_test),
        nombres_paises,
    )

    pipeline_mlp1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "estimator",
                MLPClassifier(hidden_layer_sizes=(6, 12), max_iter=100, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline_mlp1.fit(X_train, y_train)
    acc_mlp1 = evaluate_model(
        "MLP v1 (6, 12)",
        y_test,
        pipeline_mlp1.predict(X_test),
        nombres_paises,
    )

    pipeline_mlp2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "estimator",
                MLPClassifier(hidden_layer_sizes=(100, 200), max_iter=100, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline_mlp2.fit(X_train, y_train)
    acc_mlp2 = evaluate_model(
        "MLP v2 (100, 200)",
        y_test,
        pipeline_mlp2.predict(X_test),
        nombres_paises,
    )

    pipeline_mlp_smote = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
            (
                "estimator",
                MLPClassifier(hidden_layer_sizes=(100, 200), max_iter=100, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline_mlp_smote.fit(X_train, y_train)
    acc_mlp_smote = evaluate_model(
        "MLP + SMOTE (100, 200)",
        y_test,
        pipeline_mlp_smote.predict(X_test),
        nombres_paises,
    )

    print("\n=== COMPARACIÓN DE MODELOS ===")
    print(f"Árbol de Decisión (max_depth=8): {acc_arbol:.4f}")
    print(f"MLP v1 (6, 12):                 {acc_mlp1:.4f}")
    print(f"MLP v2 (100, 200):              {acc_mlp2:.4f}")
    print(f"MLP + SMOTE (100, 200):         {acc_mlp_smote:.4f}")


if __name__ == "__main__":
    main()
