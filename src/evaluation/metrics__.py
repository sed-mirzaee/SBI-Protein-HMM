"""Small evaluation___ metrics used by the final project evaluation___.

State convention:
    0 = other
    1 = alpha
"""

from __future__ import annotations

import numpy as np


def posterior_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return MSE between posterior-probability arrays of equal shape."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.ndim < 2 or y_true.shape[-1] != 2:
        raise ValueError(
            "Posterior arrays must end in two columns: [other, alpha]."
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Posterior arrays contain NaN or infinite values.")

    return float(np.mean((y_true - y_pred) ** 2))


def probabilities_to_states(probabilities: np.ndarray) -> np.ndarray:
    """Convert [other, alpha] probabilities into hard states using argmax."""
    probabilities = np.asarray(probabilities, dtype=np.float32)

    if probabilities.ndim < 2 or probabilities.shape[-1] != 2:
        raise ValueError(
            "Probability array must end in two columns: [other, alpha]."
        )

    return np.argmax(probabilities, axis=-1).astype(np.int64)


def state_accuracy(y_true_states: np.ndarray, y_pred_states: np.ndarray) -> float:
    """Return per-residue classification accuracy for two state arrays."""
    y_true_states = np.asarray(y_true_states, dtype=np.int64)
    y_pred_states = np.asarray(y_pred_states, dtype=np.int64)

    if y_true_states.shape != y_pred_states.shape:
        raise ValueError(
            f"Shape mismatch: true={y_true_states.shape}, "
            f"predicted={y_pred_states.shape}"
        )
    if y_true_states.size == 0:
        raise ValueError("Cannot compute accuracy on empty arrays.")

    return float(np.mean(y_true_states == y_pred_states))