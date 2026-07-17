"""Test the sequence-preserving BayesFlow BiLSTM network."""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import numpy as np

from src.bayesflow_model.data import (
    prepare_bayesflow_data,
)
from src.bayesflow_model.summary_network import (
    ProteinBiLSTMSequenceNetwork,
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load a small subset
    # ---------------------------------------------------------
    data = prepare_bayesflow_data(
        split_name="train",
        limit=8,
    )

    protein_sequence = data[
        "protein_sequence"
    ]

    encoder_mask = data[
        "encoder_mask"
    ]

    state_probabilities = data[
        "state_probabilities"
    ]

    print(
        "Protein sequence shape:",
        protein_sequence.shape,
    )

    print(
        "Encoder mask shape:",
        encoder_mask.shape,
    )

    print(
        "Target shape:",
        state_probabilities.shape,
    )

    # ---------------------------------------------------------
    # 2. Validate input shapes
    # ---------------------------------------------------------
    assert protein_sequence.shape == (
        8,
        250,
        20,
    )

    assert encoder_mask.shape == (
        8,
        250,
    )

    assert state_probabilities.shape == (
        8,
        250,
        2,
    )

    assert encoder_mask.dtype == np.bool_

    # ---------------------------------------------------------
    # 3. Validate real and padded positions
    # ---------------------------------------------------------
    valid_x = protein_sequence[
        encoder_mask
    ]

    padded_x = protein_sequence[
        ~encoder_mask
    ]

    assert np.isfinite(
        protein_sequence
    ).all()

    # Every real position is one-hot encoded.
    assert np.allclose(
        valid_x.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # Every padded position contains twenty zeros.
    assert np.allclose(
        padded_x,
        0.0,
        atol=1e-7,
    )

    # ---------------------------------------------------------
    # 4. Run the sequence-preserving BiLSTM
    # ---------------------------------------------------------
    network = ProteinBiLSTMSequenceNetwork(
        hidden_dim=64,
        dropout=0.10,
    )

    sequence_features = network(
        protein_sequence,
        mask=encoder_mask,
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

    # 64 forward + 64 backward features.
    assert sequence_features.shape == (
        8,
        250,
        128,
    )

    assert sequence_features.dtype == np.float32

    assert np.isfinite(
        sequence_features
    ).all()

    # ---------------------------------------------------------
    # 5. Result
    # ---------------------------------------------------------
    print()
    print("✓ Offline data loaded")
    print("✓ Stored encoder mask used directly")
    print("✓ Valid positions are one-hot encoded")
    print("✓ Padding positions are all zero")
    print("✓ BiLSTM preserved all 250 positions")
    print("✓ Each position received 128 contextual features")
    print("✓ Sequence encoder is ready")


if __name__ == "__main__":
    main()