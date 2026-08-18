"""Check the official BayesFlow inference component."""

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
from src.bayesflow_model.inference_network import (
    build_inference_network,
)
from src.bayesflow_model.summary_network import (
    ProteinBiLSTMSequenceNetwork,
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load and adapt a small offline batch
    # ---------------------------------------------------------
    raw_data = prepare_bayesflow_data(
        split_name="train",
        limit=8,
    )

    adapter = build_protein_adapter()

    adapted_data = adapter(raw_data)

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

    # ---------------------------------------------------------
    # 2. Produce position-wise contextual features
    # ---------------------------------------------------------
    summary_network = ProteinBiLSTMSequenceNetwork(
        hidden_dim=64,
        dropout=0.10,
    )

    sequence_features = summary_network(
        summary_variables,
        mask=summary_mask,
        training=False,
    )

    print(
        "sequence_features:",
        tuple(sequence_features.shape),
    )

    assert tuple(sequence_features.shape) == (
        8,
        250,
        128,
    )

    # ---------------------------------------------------------
    # 3. Build and run BayesFlow inference network
    # ---------------------------------------------------------
    inference_network = build_inference_network(
        position_dim=64,
        condition_dim=128,
        dropout=0.10,
    )

    estimates = inference_network(
        xz=inference_variables,
        conditions=sequence_features,
        mask=inference_mask,
        training=False,
    )

    print("\nEstimate keys:")
    print(estimates.keys())

    print(
        "posterior_mean keys:",
        estimates["posterior_mean"].keys(),
    )

    predicted_probabilities = estimates[
        "posterior_mean"
    ]["value"]

    predicted_probabilities = np.asarray(
        predicted_probabilities
    )

    print(
        "\nPredicted probabilities:",
        predicted_probabilities.shape,
    )

    print(
        "Predicted dtype:",
        predicted_probabilities.dtype,
    )

    # ---------------------------------------------------------
    # 4. Validate prediction contract
    # ---------------------------------------------------------
    assert predicted_probabilities.shape == (
        8,
        250,
        2,
    )

    assert predicted_probabilities.dtype == np.float32

    assert np.isfinite(
        predicted_probabilities
    ).all()

    inference_mask_bool = (
        inference_mask > 0.5
    )

    valid_predictions = predicted_probabilities[
        inference_mask_bool
    ]

    assert np.all(
        valid_predictions >= 0.0
    )

    assert np.all(
        valid_predictions <= 1.0
    )

    assert np.allclose(
        valid_predictions.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # ---------------------------------------------------------
    # 5. Check that BayesFlow can compute its official loss
    # ---------------------------------------------------------
    loss_weights = inference_mask[..., None]

    assert loss_weights.shape == (
        8,
        250,
        1,
    )

    metrics = inference_network.compute_metrics(
        x=inference_variables,
        conditions=sequence_features,
        sample_weight=loss_weights,
        stage="validation",
    )

    print("\nMetric keys:")
    print(metrics.keys())

    loss = float(
        np.asarray(metrics["loss"])
    )

    print(
        "Initial untrained loss:",
        loss,
    )

    assert np.isfinite(loss)

    # ---------------------------------------------------------
    # 6. Display a few untrained outputs
    # ---------------------------------------------------------
    print("\nFirst five valid predictions:")
    print(valid_predictions[:5])

    print()
    print("✓ Adapter output loaded")
    print("✓ BiLSTM contextual features produced")
    print("✓ Official ScoringRuleNetwork executed")
    print("✓ Output shape matches y")
    print("✓ Softmax produced valid probabilities")
    print("✓ Valid probability rows sum to one")
    print("✓ MeanScore loss was computed")
    print("✓ Inference component is ready")


if __name__ == "__main__":
    main()