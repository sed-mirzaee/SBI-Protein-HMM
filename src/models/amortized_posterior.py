"""Amortized posterior estimator.

The neural network architecture and training procedure will be
implemented by the team member responsible for BayesFlow training.
"""

import numpy as np


def predict_state_probabilities(
    encoded_sequence: np.ndarray,
) -> np.ndarray:
    """
    Predict posterior hidden-state probabilities for a protein sequence.

    Parameters
    ----------
    encoded_sequence:
        One-hot encoded amino-acid sequence.

        Expected shape:
        (sequence_length, 20)

    Returns
    -------
    np.ndarray
        Predicted state probabilities.

        Expected shape:
        (sequence_length, 2)

        Column order:
        0 = other
        1 = alpha
    """

    raise NotImplementedError(
        "The amortized posterior model has not been trained yet."
    )