from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.amortized_posterior import (
    _validate_encoded_sequence,
    load_trained_model,
    predict_state_probabilities,
)
from src.models.train_bayesflow import (
    BiLSTMPosteriorEstimator,
)


def test_validate_encoded_sequence_accepts_correct_shape() -> None:
    x = np.zeros(
        (25, 20),
        dtype=np.float32,
    )

    result = _validate_encoded_sequence(x)

    assert result.shape == (25, 20)
    assert result.dtype == np.float32


def test_validate_encoded_sequence_rejects_wrong_rank() -> None:
    x = np.zeros(
        (2, 25, 20),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        _validate_encoded_sequence(x)


def test_validate_encoded_sequence_rejects_wrong_features() -> None:
    x = np.zeros(
        (25, 19),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="20 amino-acid features",
    ):
        _validate_encoded_sequence(x)


def test_prediction_shape_and_normalization() -> None:
    """
    Inference should work with an already-created model,
    without loading the final trained checkpoint.
    """

    model = BiLSTMPosteriorEstimator(
        hidden_dim=8,
        dense_dim=8,
        dropout=0.0,
    )

    encoded_sequence = np.zeros(
        (30, 20),
        dtype=np.float32,
    )

    # Create valid one-hot rows.
    encoded_sequence[:, 0] = 1.0

    probabilities = predict_state_probabilities(
        encoded_sequence=encoded_sequence,
        model=model,
        device=torch.device("cpu"),
    )

    assert probabilities.shape == (30, 2)
    assert probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    assert np.allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-6,
    )


def test_checkpoint_save_and_load(
    tmp_path: Path,
) -> None:
    """
    A model checkpoint should be loadable with the inference utility.
    """

    torch.manual_seed(1)

    original_model = BiLSTMPosteriorEstimator(
        input_dim=20,
        hidden_dim=8,
        dense_dim=8,
        output_dim=2,
        dropout=0.0,
    )

    checkpoint_path = tmp_path / "model.pt"

    checkpoint = {
        "model_state_dict": original_model.state_dict(),
        "config": {
            "input_dim": 20,
            "hidden_dim": 8,
            "dense_dim": 8,
            "output_dim": 2,
            "dropout": 0.0,
        },
        "state_order": [
            "other",
            "alpha",
        ],
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    loaded_model = load_trained_model(
        model_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    x = torch.randn(1, 12, 20)

    original_model.eval()
    loaded_model.eval()

    with torch.no_grad():
        original_output = original_model(x)
        loaded_output = loaded_model(x)

    assert torch.allclose(
        original_output,
        loaded_output,
        atol=1e-6,
    )