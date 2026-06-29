import numpy as np
import pandas as pd
import pytest

from src.features.technical import FEATURE_COLUMNS, TARGET_COLUMN
from src.modeling.baseline import (
    chronological_holdout_split,
    evaluate_logistic_holdout,
)


def _model_dataset(rows: int = 150) -> pd.DataFrame:
    """Create predictable synthetic feature data for model tests."""
    dates = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="B",
    )

    target = np.resize(
        np.array([0, 1], dtype="int64"),
        rows,
    )

    direction = (target * 2) - 1

    return pd.DataFrame(
        {
            "return_1d": direction * 0.01,
            "return_5d": direction * 0.02,
            "close_to_sma_5": direction * 0.01,
            "close_to_sma_20": direction * 0.015,
            "volatility_10d": 0.01 + (target * 0.005),
            "rsi_14": 45.0 + (target * 10.0),
            "volume_change_1d": direction * 0.10,
            "volume_to_sma_20": direction * 0.08,
            "intraday_range_pct": 0.02 + (target * 0.01),
            "close_location": 0.25 + (target * 0.50),
            TARGET_COLUMN: target,
        },
        index=dates,
    )


def test_chronological_split_keeps_newer_rows_for_testing():
    """Training rows must be older than test rows."""
    dataset = _model_dataset()

    train_dataset, test_dataset = chronological_holdout_split(dataset)

    assert len(train_dataset) == 120
    assert len(test_dataset) == 30
    assert train_dataset.index.max() < test_dataset.index.min()
    assert train_dataset.index.equals(dataset.index[:120])
    assert test_dataset.index.equals(dataset.index[120:])


def test_evaluation_returns_metrics_and_probabilities():
    """Evaluation should compare both models and retain test predictions."""
    evaluation = evaluate_logistic_holdout(_model_dataset())

    assert evaluation.train_rows == 120
    assert evaluation.test_rows == 30
    assert 0.0 <= evaluation.always_up_accuracy <= 1.0
    assert 0.0 <= evaluation.logistic_accuracy <= 1.0
    assert 0.0 <= evaluation.logistic_balanced_accuracy <= 1.0
    assert evaluation.logistic_accuracy >= evaluation.always_up_accuracy

    assert list(evaluation.predictions.columns) == [
        "actual_next_up",
        "always_up_prediction",
        "logistic_prediction",
        "probability_next_up",
    ]

    assert len(evaluation.predictions) == 30
    assert evaluation.predictions["probability_next_up"].between(0, 1).all()


def test_evaluation_rejects_training_data_with_one_target_class():
    """Logistic regression requires both up and non-up outcomes in training."""
    dataset = _model_dataset()
    dataset[TARGET_COLUMN] = 1

    with pytest.raises(ValueError, match="both target classes"):
        evaluate_logistic_holdout(dataset)


def test_split_rejects_invalid_train_fraction():
    """The holdout fraction must leave meaningful train and test periods."""
    with pytest.raises(ValueError, match="at least 0.50"):
        chronological_holdout_split(
            _model_dataset(),
            train_fraction=1.0,
        )


def test_evaluation_rejects_missing_model_feature():
    """Every configured feature must be present before training."""
    dataset = _model_dataset().drop(
        columns=FEATURE_COLUMNS[0],
    )

    with pytest.raises(ValueError, match="missing required model columns"):
        evaluate_logistic_holdout(dataset)