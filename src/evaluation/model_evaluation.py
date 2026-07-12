"""
Evaluation functions for model predictions.

We compare:
1. true posterior probabilities from Forward-Backward
2. predicted posterior probabilities from the neural network / BayesFlow
"""

import numpy as np


def check_shapes(y_true, y_pred):
    """
    Check that true and predicted probabilities have the same shape.
    Expected shape:
    (sequence_length, 2)
    or
    (n_sequences, sequence_length, 2)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}"
        )

    if y_true.shape[-1] != 2:
        raise ValueError("Last dimension must be 2: [alpha-helix, other].")

    return y_true, y_pred


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error between true and predicted probabilities.
    """

    y_true, y_pred = check_shapes(y_true, y_pred)

    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error between true and predicted probabilities.
    """

    y_true, y_pred = check_shapes(y_true, y_pred)

    return np.mean(np.abs(y_true - y_pred))


def cross_entropy(y_true, y_pred):
    """
    Cross-entropy between true and predicted probabilities.
    """

    y_true, y_pred = check_shapes(y_true, y_pred)

    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def state_accuracy(y_true, y_pred):
    """
    Convert probabilities to states using argmax and compute accuracy.

    State 0 = alpha-helix
    State 1 = other
    """

    y_true, y_pred = check_shapes(y_true, y_pred)

    true_states = np.argmax(y_true, axis=-1)
    pred_states = np.argmax(y_pred, axis=-1)

    return np.mean(true_states == pred_states)


def evaluate_model_prediction(y_true, y_pred):
    """
    Compute all evaluation metrics.
    """

    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "cross_entropy": cross_entropy(y_true, y_pred),
        "state_accuracy": state_accuracy(y_true, y_pred),
    }