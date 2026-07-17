"""Check the official BayesFlow Adapter for protein training pairs."""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import numpy as np

from src.bayesflow_model.adapter import (
    build_protein_adapter,
)
from src.bayesflow_model.data import (
    prepare_bayesflow_data,
)
from src.bayesflow_model.summary_network import (
    ProteinBiLSTMSequenceNetwork,
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load raw offline training pairs
    # ---------------------------------------------------------
    raw_data = prepare_bayesflow_data(
        split_name="train",
        limit=8,
    )

    print("Raw data keys:")
    print(sorted(raw_data.keys()))

    print("\nRaw shapes:")
    print(
        "protein_sequence:",
        raw_data["protein_sequence"].shape,
    )
    print(
        "state_probabilities:",
        raw_data["state_probabilities"].shape,
    )
    print(
        "encoder_mask:",
        raw_data["encoder_mask"].shape,
    )
    print(
        "target_mask:",
        raw_data["target_mask"].shape,
    )

    # ---------------------------------------------------------
    # 2. Check raw contract
    # ---------------------------------------------------------
    assert raw_data["protein_sequence"].shape == (
        8,
        250,
        20,
    )

    assert raw_data["state_probabilities"].shape == (
        8,
        250,
        2,
    )

    assert raw_data["encoder_mask"].shape == (
        8,
        250,
    )

    assert raw_data["target_mask"].shape == (
        8,
        250,
    )

    assert raw_data["encoder_mask"].dtype == np.bool_
    assert raw_data["target_mask"].dtype == np.bool_

    assert np.array_equal(
        raw_data["encoder_mask"],
        raw_data["target_mask"],
    )

    # ---------------------------------------------------------
    # 3. Run the official BayesFlow Adapter
    # ---------------------------------------------------------
    adapter = build_protein_adapter()

    adapted_data = adapter(
        raw_data
    )

    print("\nAdapted data keys:")
    print(sorted(adapted_data.keys()))

    expected_keys = {
        "summary_variables",
        "inference_variables",
        "summary_mask",
        "inference_mask",
    }

    assert set(adapted_data.keys()) == expected_keys

    summary_variables = adapted_data[
        "summary_variables"
    ]

    inference_variables = adapted_data[
        "inference_variables"
    ]

    summary_mask = adapted_data[
        "summary_mask"
    ]

    inference_mask = adapted_data[
        "inference_mask"
    ]

    print("\nAdapted shapes:")
    print(
        "summary_variables:",
        summary_variables.shape,
    )
    print(
        "inference_variables:",
        inference_variables.shape,
    )
    print(
        "summary_mask:",
        summary_mask.shape,
    )
    print(
        "inference_mask:",
        inference_mask.shape,
    )

    print("\nAdapted dtypes:")
    print(
        "summary_variables:",
        summary_variables.dtype,
    )
    print(
        "inference_variables:",
        inference_variables.dtype,
    )
    print(
        "summary_mask:",
        summary_mask.dtype,
    )
    print(
        "inference_mask:",
        inference_mask.dtype,
    )

    # ---------------------------------------------------------
    # 4. Validate adapted shapes and dtypes
    # ---------------------------------------------------------
    assert summary_variables.shape == (
        8,
        250,
        20,
    )

    assert inference_variables.shape == (
        8,
        250,
        2,
    )

    assert summary_mask.shape == (
        8,
        250,
    )

    assert inference_mask.shape == (
        8,
        250,
    )

    assert summary_variables.dtype == np.float32
    assert inference_variables.dtype == np.float32

    # The official Adapter converts masks to float32.
    assert summary_mask.dtype == np.float32
    assert inference_mask.dtype == np.float32

    # Mask values must remain binary.
    assert np.all(
        np.isin(
            summary_mask,
            [0.0, 1.0],
        )
    )

    assert np.all(
        np.isin(
            inference_mask,
            [0.0, 1.0],
        )
    )

    # Convert masks to Boolean for NumPy indexing.
    summary_mask_bool = (
        summary_mask > 0.5
    )

    inference_mask_bool = (
        inference_mask > 0.5
    )

    # ---------------------------------------------------------
    # 5. Confirm masks were preserved
    # ---------------------------------------------------------
    assert np.array_equal(
        summary_mask_bool,
        inference_mask_bool,
    )

    assert np.array_equal(
        summary_mask_bool,
        raw_data["encoder_mask"],
    )

    assert np.array_equal(
        inference_mask_bool,
        raw_data["target_mask"],
    )

    # ---------------------------------------------------------
    # 6. Confirm x and y were preserved
    # ---------------------------------------------------------
    assert np.array_equal(
        summary_variables,
        raw_data["protein_sequence"],
    )

    assert np.array_equal(
        inference_variables,
        raw_data["state_probabilities"],
    )

    # ---------------------------------------------------------
    # 7. Validate real and padded positions
    # ---------------------------------------------------------
    valid_x = summary_variables[
        summary_mask_bool
    ]

    padded_x = summary_variables[
        ~summary_mask_bool
    ]

    valid_y = inference_variables[
        inference_mask_bool
    ]

    padded_y = inference_variables[
        ~inference_mask_bool
    ]

    assert np.isfinite(valid_x).all()
    assert np.isfinite(valid_y).all()

    # Real input rows are one-hot encoded.
    assert np.allclose(
        valid_x.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # Padded input rows contain only zeros.
    assert np.allclose(
        padded_x,
        0.0,
        atol=1e-7,
    )

    # Real target rows are posterior probabilities.
    assert np.allclose(
        valid_y.sum(axis=-1),
        1.0,
        atol=1e-5,
    )

    assert np.all(
        valid_y >= 0.0
    )

    assert np.all(
        valid_y <= 1.0
    )

    # Padded target rows contain only zeros.
    assert np.allclose(
        padded_y,
        0.0,
        atol=1e-7,
    )

    # ---------------------------------------------------------
    # 8. Test adapted input with the BiLSTM
    # ---------------------------------------------------------
    encoder = ProteinBiLSTMSequenceNetwork(
        hidden_dim=64,
        dropout=0.10,
    )

    sequence_features = encoder(
        summary_variables,
        mask=summary_mask,
        training=False,
    )

    sequence_features = np.asarray(
        sequence_features
    )

    print(
        "\nSequence features:",
        sequence_features.shape,
    )

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
    # 9. Result
    # ---------------------------------------------------------
    print()
    print("✓ Raw offline pairs (x, y) loaded")
    print("✓ Official BayesFlow Adapter executed")
    print("✓ x mapped to summary_variables")
    print("✓ y mapped to inference_variables")
    print("✓ encoder_mask mapped to summary_mask")
    print("✓ target_mask mapped to inference_mask")
    print("✓ Float masks converted safely for validation")
    print("✓ Adapter preserved every array value")
    print("✓ BiLSTM accepted adapted data and mask")
    print("✓ Adapter stage is ready")


if __name__ == "__main__":
    main()