"""Official BayesFlow adapter for offline protein training pairs."""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf


def build_protein_adapter() -> bf.adapters.Adapter:
    """
    Map the project's raw offline data to BayesFlow's canonical keys.

    Raw project keys:
        protein_sequence
        state_probabilities
        encoder_mask
        target_mask

    BayesFlow keys:
        summary_variables
        inference_variables
        summary_mask
        inference_mask
    """

    adapter = bf.approximators.Approximator.build_adapter(
        # y: Forward-Backward posterior targets
        inference_variables="state_probabilities",

        # x: one-hot protein sequences
        summary_variables="protein_sequence",

        # Padding mask passed as mask=... to the BiLSTM summary network
        summary_mask="encoder_mask",

        # Padding mask passed to the later inference component
        inference_mask="target_mask",
    )

    return adapter