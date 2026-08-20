"""Random Forest training, evaluation, and model persistence."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DEFAULT_CSV_PATH,
    DEFAULT_IMPORTANCE_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
)

THRESHOLDS = (0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10)


def _choose_threshold(
    labels: pd.Series, probabilities: np.ndarray
) -> tuple[float, list[dict[str, float | int]]]:
    """Choose the highest-F1 threshold that meets the recall target."""
    results: list[dict[str, float | int]] = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        _, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        results.append(
            {
                "threshold": threshold,
                "recall": recall,
                "precision": precision,
                "f1": f1_score,
                "false_negatives": int(fn),
            }
        )

    eligible = [row for row in results if row["recall"] >= 0.90]
    candidates = eligible or results
    best = max(candidates, key=lambda row: (row["f1"], row["precision"]))
    return float(best["threshold"]), results


def _save_feature_importance(
    classifier: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False)

    figure = Figure(figsize=(10, 9))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    sns.barplot(
        data=importance.head(30),
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="viridis",
        ax=axis,
    )
    axis.set_title("Random Forest feature importance")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    figure.clear()
    return importance


def train_model(
    csv_path: Path = DEFAULT_CSV_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    importance_path: Path = DEFAULT_IMPORTANCE_PATH,
) -> None:
    """Train and evaluate the classifier, then save the complete model package."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature table not found: {csv_path}")

    table = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna()
    missing = [name for name in [*FEATURE_NAMES, "Label"] if name not in table.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {', '.join(missing)}")
    if table["Label"].nunique() < 2:
        raise ValueError("Training requires both nevus and melanoma samples")

    features = table[FEATURE_NAMES]
    labels = table["Label"].astype(int)
    train_x, remaining_x, train_y, remaining_y = train_test_split(
        features,
        labels,
        test_size=0.4,
        random_state=42,
        stratify=labels,
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        remaining_x,
        remaining_y,
        test_size=0.5,
        random_state=42,
        stratify=remaining_y,
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    test_scaled = scaler.transform(test_x)
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(train_scaled, train_y)

    validation_probabilities = classifier.predict_proba(validation_scaled)[:, 1]
    threshold, threshold_results = _choose_threshold(
        validation_y, validation_probabilities
    )
    test_probabilities = classifier.predict_proba(test_scaled)[:, 1]
    predictions = (test_probabilities >= threshold).astype(int)
    importance = _save_feature_importance(
        classifier, list(FEATURE_NAMES), importance_path
    )

    package = {
        "model": classifier,
        "scaler": scaler,
        "threshold": threshold,
        "features": list(FEATURE_NAMES),
        "pipeline_version": "1.0.0",
        "scikit_learn_version": sklearn.__version__,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_path)

    print("\nValidation threshold evaluation")
    print(pd.DataFrame(threshold_results).to_string(index=False))
    print(f"\nSelected threshold: {threshold:.2f}")
    print("\nClassification report")
    print(
        classification_report(
            test_y,
            predictions,
            labels=[0, 1],
            target_names=["Nevus", "Melanoma"],
            zero_division=0,
        )
    )
    print("Confusion matrix")
    print(confusion_matrix(test_y, predictions, labels=[0, 1]))
    print("\nMost important features")
    print(importance.head(10).to_string(index=False))
    print(f"\nModel saved to {model_path}")
    print(f"Feature importance chart saved to {importance_path}")
