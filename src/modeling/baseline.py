from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.technical import FEATURE_COLUMNS, TARGET_COLUMN


MIN_TRAIN_ROWS = 60
MIN_TEST_ROWS = 20


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """Metrics and predictions from one chronological holdout evaluation."""

    train_rows: int
    test_rows: int
    train_up_rate: float
    test_up_rate: float
    always_up_accuracy: float
    always_up_balanced_accuracy: float
    logistic_accuracy: float
    logistic_balanced_accuracy: float
    predictions: pd.DataFrame


def validate_model_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate model inputs before splitting, training, or evaluation."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Model dataset must be a pandas DataFrame.")

    if dataset.empty:
        raise ValueError("Model dataset cannot be empty.")

    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        missing_values = ", ".join(missing_columns)
        raise ValueError(
            "Model dataset is missing required model columns: "
            f"{missing_values}."
        )

    if dataset.index.has_duplicates:
        raise ValueError("Model dataset cannot contain duplicate dates.")

    validated = dataset.copy().sort_index()
    feature_columns = list(FEATURE_COLUMNS)

    validated[feature_columns] = validated[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    target = pd.to_numeric(
        validated[TARGET_COLUMN],
        errors="coerce",
    )

    if validated[feature_columns].isna().any().any():
        raise ValueError("Model dataset contains missing or invalid features.")

    if target.isna().any():
        raise ValueError("Model dataset contains missing or invalid target values.")

    if not target.isin([0, 1]).all():
        raise ValueError("Model target values must contain only 0 or 1.")

    validated[TARGET_COLUMN] = target.astype("int64")

    return validated


def chronological_holdout_split(
    dataset: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into older training rows and newer test rows."""
    if isinstance(train_fraction, bool) or not isinstance(
        train_fraction,
        Real,
    ):
        raise TypeError("train_fraction must be a number.")

    fraction = float(train_fraction)

    if not 0.5 <= fraction < 1.0:
        raise ValueError(
            "train_fraction must be at least 0.50 and less than 1.00."
        )

    validated = validate_model_dataset(dataset)
    split_position = int(len(validated) * fraction)

    train_dataset = validated.iloc[:split_position].copy()
    test_dataset = validated.iloc[split_position:].copy()

    if len(train_dataset) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Chronological holdout requires at least {MIN_TRAIN_ROWS} "
            "training rows."
        )

    if len(test_dataset) < MIN_TEST_ROWS:
        raise ValueError(
            f"Chronological holdout requires at least {MIN_TEST_ROWS} "
            "test rows."
        )

    return train_dataset, test_dataset


def fit_logistic_regression(train_dataset: pd.DataFrame) -> Pipeline:
    """Train a scaled logistic-regression classifier on historical features."""
    validated = validate_model_dataset(train_dataset)

    features = validated.loc[:, list(FEATURE_COLUMNS)]
    target = validated[TARGET_COLUMN]

    if target.nunique() < 2:
        raise ValueError(
            "Training data must contain both target classes: 0 and 1."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(features, target)

    return model


def evaluate_logistic_holdout(
    dataset: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
) -> HoldoutEvaluation:
    """Evaluate logistic regression against an always-up baseline."""
    train_dataset, test_dataset = chronological_holdout_split(
        dataset,
        train_fraction=train_fraction,
    )

    model = fit_logistic_regression(train_dataset)

    test_features = test_dataset.loc[:, list(FEATURE_COLUMNS)]
    actual_targets = test_dataset[TARGET_COLUMN].to_numpy()

    always_up_predictions = np.ones(
        shape=len(test_dataset),
        dtype="int64",
    )

    logistic_predictions = model.predict(test_features).astype("int64")

    up_class_position = int(
        np.flatnonzero(model.classes_ == 1)[0]
    )

    probability_next_up = model.predict_proba(test_features)[
        :,
        up_class_position,
    ]

    predictions = pd.DataFrame(
        {
            "actual_next_up": actual_targets,
            "always_up_prediction": always_up_predictions,
            "logistic_prediction": logistic_predictions,
            "probability_next_up": probability_next_up,
        },
        index=test_dataset.index,
    )

    predictions.index.name = test_dataset.index.name

    return HoldoutEvaluation(
        train_rows=len(train_dataset),
        test_rows=len(test_dataset),
        train_up_rate=float(train_dataset[TARGET_COLUMN].mean()),
        test_up_rate=float(test_dataset[TARGET_COLUMN].mean()),
        always_up_accuracy=float(
            accuracy_score(
                actual_targets,
                always_up_predictions,
            )
        ),
        always_up_balanced_accuracy=float(
            balanced_accuracy_score(
                actual_targets,
                always_up_predictions,
            )
        ),
        logistic_accuracy=float(
            accuracy_score(
                actual_targets,
                logistic_predictions,
            )
        ),
        logistic_balanced_accuracy=float(
            balanced_accuracy_score(
                actual_targets,
                logistic_predictions,
            )
        ),
        predictions=predictions,
    )