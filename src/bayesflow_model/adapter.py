"""
Official BayesFlow Adapter for offline protein training pairs.
"""

from __future__ import annotations

import os

# Must be set before importing BayesFlow or Keras.
os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf


def build_protein_adapter() -> bf.adapters.Adapter:
    """
    Map project-specific arrays to BayesFlow canonical keys.

    Raw project keys
    ----------------
    protein_sequence:
        Padded one-hot amino-acid sequences.

    state_probabilities:
        Forward-Backward posterior probabilities.

    encoder_mask:
        Mask used by the BiLSTM summary network.

    target_mask:
        Mask forwarded to the inference network.

    loss_weight:
        Temporal sample weights used to exclude padding
        from the scoring-rule loss.

    Adapted BayesFlow keys
    ----------------------
    summary_variables
    inference_variables
    summary_mask
    inference_mask
    sample_weight
    """

    return bf.approximators.Approximator.build_adapter(
        inference_variables="state_probabilities",
        summary_variables="protein_sequence",

        summary_mask="encoder_mask",
        inference_mask="target_mask",

        # Important: padding must not contribute to loss.
        sample_weight="loss_weight",
    )