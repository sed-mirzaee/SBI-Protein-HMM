"""Prediction utilities for the amortized posterior estimator.

This module loads the trained PyTorch BiLSTM model and predicts posterior
state probabilities for protein sequences.

Important convention:
    output column 0 = other
    output column 1 = alpha

This file does NOT redefine HMM matrices, emission probabilities, state
order, or amino-acid order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.preprocessing.encoding import one_hot_encode_sequence
from src.models.train_bayesflow import (
    DEFAULT_MODEL_PATH,
    BiLSTMPosteriorEstimator,
)


def load_trained_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> BiLSTMPosteriorEstimator:
    """Load a trained PyTorch BiLSTM posterior model."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_path}. "
            "Train it first with: python -m src.models__.train_bayesflow"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = BiLSTMPosteriorEstimator(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        dense_dim=config["dense_dim"],
        output_dim=config["output_dim"],
        dropout=config["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def _validate_encoded_sequence(encoded_sequence: np.ndarray) -> np.ndarray:
    """Validate encoded sequence format."""

    encoded_sequence = np.asarray(encoded_sequence, dtype=np.float32)

    if encoded_sequence.ndim != 2:
        raise ValueError(
            "encoded_sequence must have shape (sequence_length, 20). "
            f"Got shape {encoded_sequence.shape}."
        )

    if encoded_sequence.shape[1] != 20:
        raise ValueError(
            "encoded_sequence must have 20 amino-acid features. "
            f"Got shape {encoded_sequence.shape}."
        )

    return encoded_sequence


def predict_state_probabilities(
    encoded_sequence: np.ndarray,
    model: BiLSTMPosteriorEstimator | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict posterior hidden-state probabilities.

    Parameters
    ----------
    encoded_sequence:
        One-hot encoded amino-acid sequence.
        Shape: (sequence_length, 20)

    model:
        Optional already-loaded model.

    model_path:
        Path to trained checkpoint. Used if model is None.

    device:
        Optional PyTorch device.

    Returns
    -------
    np.ndarray
        Shape: (sequence_length, 2)

        Column order:
            0 = other
            1 = alpha
    """

    encoded_sequence = _validate_encoded_sequence(encoded_sequence)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_trained_model(model_path=model_path, device=device)

    x = torch.tensor(
        encoded_sequence[None, :, :],
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=-1)

    probabilities_np = probabilities.squeeze(0).cpu().numpy()

    row_sums = probabilities_np.sum(axis=-1, keepdims=True)
    probabilities_np = probabilities_np / np.clip(row_sums, 1e-8, None)

    return probabilities_np.astype(np.float32)


def predict_from_sequence(
    sequence: str | list[str],
    model: BiLSTMPosteriorEstimator | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict posterior probabilities from a synthetic amino-acid sequence."""

    encoded_sequence = one_hot_encode_sequence(sequence)

    return predict_state_probabilities(
        encoded_sequence=encoded_sequence,
        model=model,
        model_path=model_path,
        device=device,
    )


if __name__ == "__main__":
    example_sequence = "ARNDCEQGHILKMFPSTWYV" * 5

    probabilities = predict_from_sequence(example_sequence)

    print("Predicted probabilities shape:", probabilities.shape)
    print("First five rows:")
    print(probabilities[:5])
    print("Column order: 0 = other, 1 = alpha")