"""Test the sequence-preserving BayesFlow BiLSTM network."""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import numpy as np

from src.bayesflow_model.data import prepare_bayesflow_data
from src.bayesflow_model.summary_network import (
    ProteinBiLSTMSequenceNetwork,
)


def main() -> None:
    data = prepare_bayesflow_data(
        split_name="train",
        limit=8,
    )

    x = data["protein_sequence"]
    mask = data["sequence_mask"]

    print("Protein sequence shape:", x.shape)
    print("Sequence mask shape:", mask.shape)

    assert x.shape == (8, 250, 20)
    assert mask.shape == (8, 250)

    # Validate the stored data contract.
    row_sums = x.sum(axis=-1)

    assert np.allclose(
        row_sums[mask],
        1.0,
        atol=1e-6,
    )

    assert np.allclose(
        row_sums[~mask],
        0.0,
        atol=1e-6,
    )

    network = ProteinBiLSTMSequenceNetwork(
        hidden_dim=64,
        dropout=0.10,
    )

    sequence_features = network(
        x,
        mask=mask,
        training=False,
    )

    sequence_features = np.asarray(
        sequence_features
    )

    print(
        "Sequence feature shape:",
        sequence_features.shape,
    )

    print(
        "Sequence feature dtype:",
        sequence_features.dtype,
    )

    assert sequence_features.shape == (
        8,
        250,
        128,
    )

    assert np.isfinite(
        sequence_features
    ).all()

    print()
    print("✓ Offline data loaded")
    print("✓ Stored mask used directly")
    print("✓ Valid positions are one-hot encoded")
    print("✓ Padding positions are all zero")
    print("✓ BiLSTM preserved all 250 positions")
    print("✓ Each position received 128 contextual features")
    print("✓ Sequence encoder is ready")


if __name__ == "__main__":
    main()