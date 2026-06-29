from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.features.technical import FEATURE_COLUMNS, TARGET_COLUMN
from src.modeling.baseline import (
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    fit_logistic_regression,
    validate_model_dataset,
)


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluation:
    """Metrics and predictions from chronological walk-forward validation."""

    n_splits: int
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    always_up_accuracy: float
    always_up_balanced_accuracy: float
    logistic_accuracy: float
    logistic_balanced_accuracy: float


def _validate_n_splits(n_splits: int) -> int:
    """Validate the requested number of chronological evaluation folds."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer.")

    if not 2 <= n_splits <= 10:
        raise ValueError("n_splits must be between 2 and 10.")

    return n_splits


def evaluate_logistic_walk_forward(
    dataset: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> WalkForwardEvaluation:
    """Evaluate logistic regression across expanding chronological folds."""
    validated = validate_model_dataset(dataset)
    validated_n_splits = _validate_n_splits(n_splits)

    splitter = TimeSeriesSplit(n_splits=validated_n_splits)

    try:
        split_indices = list(splitter.split(validated))
    except ValueError as error:
        raise ValueError(
            "Not enough rows to create the requested walk-forward folds."
        ) from error

    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_positions, test_positions) in enumerate(
        split_indices,
        start=1,
    ):
        train_dataset = validated.iloc[train_positions].copy()
        test_dataset = validated.iloc[test_positions].copy()

        if len(train_dataset) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Fold {fold_number} has fewer than {MIN_TRAIN_ROWS} "
                "training rows."
            )

        if len(test_dataset) < MIN_TEST_ROWS:
            raise ValueError(
                f"Fold {fold_number} has fewer than {MIN_TEST_ROWS} "
                "test rows."
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

        fold_predictions = pd.DataFrame(
            {
                "fold": fold_number,
                "actual_next_up": actual_targets,
                "always_up_prediction": always_up_predictions,
                "logistic_prediction": logistic_predictions,
                "probability_next_up": probability_next_up,
            },
            index=test_dataset.index,
        )

        prediction_frames.append(fold_predictions)

        fold_metric_rows.append(
            {
                "fold": fold_number,
                "train_rows": len(train_dataset),
                "test_rows": len(test_dataset),
                "train_start": train_dataset.index.min(),
                "train_end": train_dataset.index.max(),
                "test_start": test_dataset.index.min(),
                "test_end": test_dataset.index.max(),
                "train_up_rate": float(
                    train_dataset[TARGET_COLUMN].mean()
                ),
                "test_up_rate": float(
                    test_dataset[TARGET_COLUMN].mean()
                ),
                "always_up_accuracy": float(
                    accuracy_score(
                        actual_targets,
                        always_up_predictions,
                    )
                ),
                "always_up_balanced_accuracy": float(
                    balanced_accuracy_score(
                        actual_targets,
                        always_up_predictions,
                    )
                ),
                "logistic_accuracy": float(
                    accuracy_score(
                        actual_targets,
                        logistic_predictions,
                    )
                ),
                "logistic_balanced_accuracy": float(
                    balanced_accuracy_score(
                        actual_targets,
                        logistic_predictions,
                    )
                ),
                "logistic_up_rate": float(
                    logistic_predictions.mean()
                ),
            }
        )

    predictions = pd.concat(prediction_frames).sort_index()
    predictions.index.name = validated.index.name

    actual_targets = predictions["actual_next_up"].to_numpy()
    always_up_predictions = predictions["always_up_prediction"].to_numpy()
    logistic_predictions = predictions["logistic_prediction"].to_numpy()

    return WalkForwardEvaluation(
        n_splits=validated_n_splits,
        fold_metrics=pd.DataFrame(fold_metric_rows),
        predictions=predictions,
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
    )