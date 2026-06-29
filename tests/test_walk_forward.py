import numpy as np
import pandas as pd
import pytest

from src.features.technical import TARGET_COLUMN
from src.modeling.walk_forward import evaluate_logistic_walk_forward


def _model_dataset(rows: int = 360) -> pd.DataFrame:
    """Create predictable synthetic market features for walk-forward tests."""
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


def test_walk_forward_keeps_training_before_testing():
    """Every fold must train on older rows and test on newer rows."""
    evaluation = evaluate_logistic_walk_forward(
        _model_dataset(),
        n_splits=4,
    )

    metrics = evaluation.fold_metrics

    assert list(metrics["fold"]) == [1, 2, 3, 4]
    assert (metrics["train_end"] < metrics["test_start"]).all()
    assert evaluation.predictions.index.is_monotonic_increasing
    assert len(evaluation.predictions) == metrics["test_rows"].sum()


def test_walk_forward_returns_probabilities_and_metrics():
    """Walk-forward evaluation should return usable probabilities and scores."""
    evaluation = evaluate_logistic_walk_forward(
        _model_dataset(),
        n_splits=4,
    )

    assert evaluation.n_splits == 4
    assert 0.0 <= evaluation.always_up_accuracy <= 1.0
    assert 0.0 <= evaluation.logistic_accuracy <= 1.0
    assert 0.0 <= evaluation.logistic_balanced_accuracy <= 1.0
    assert evaluation.logistic_accuracy >= evaluation.always_up_accuracy

    assert evaluation.predictions["probability_next_up"].between(
        0,
        1,
    ).all()


def test_walk_forward_rejects_invalid_split_count():
    """The project should reject unusable numbers of validation folds."""
    with pytest.raises(ValueError, match="between 2 and 10"):
        evaluate_logistic_walk_forward(
            _model_dataset(),
            n_splits=1,
        )


def test_walk_forward_rejects_training_data_with_one_target_class():
    """Every fold needs both target classes for logistic regression."""
    dataset = _model_dataset()
    dataset[TARGET_COLUMN] = 1

    with pytest.raises(ValueError, match="both target classes"):
        evaluate_logistic_walk_forward(
            dataset,
            n_splits=4,
        )