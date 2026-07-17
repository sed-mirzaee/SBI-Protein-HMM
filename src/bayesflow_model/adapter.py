"""Official BayesFlow Adapter for offline protein training pairs."""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf


def build_protein_adapter() -> bf.adapters.Adapter:
    """
    Map project-specific keys to BayesFlow canonical keys.

    Raw keys:
        protein_sequence
        state_probabilities
        encoder_mask
        target_mask

    Adapted keys:
        summary_variables
        inference_variables
        summary_mask
        inference_mask
    """

    return bf.approximators.Approximator.build_adapter(
        # y: Forward–Backward probabilities used as targets.
        inference_variables="state_probabilities",

        # x: padded one-hot protein sequences.
        summary_variables="protein_sequence",

        # Forwarded as mask=... to the summary network.
        summary_mask="encoder_mask",

        # Forwarded as mask=... to the later
        # inference network.
        inference_mask="target_mask",
    )