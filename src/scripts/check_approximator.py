"""
Smoke test for the complete native BayesFlow approximator.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import numpy as np

from src.bayesflow_model.approximator import (
    build_protein_approximator,
)
from src.bayesflow_model.data import (
    prepare_bayesflow_data,
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load a small raw batch
    # ---------------------------------------------------------

    raw_data = prepare_bayesflow_data(
        split_name="train",
        limit=8,
    )

    print("Raw keys:")
    print(sorted(raw_data.keys()))

    print("\nRaw shapes:")
    for key, value in raw_data.items():
        print(f"{key}: {value.shape}")

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

    assert raw_data["loss_weight"].shape == (
        8,
        250,
        1,
    )

    # ---------------------------------------------------------
    # 2. Build complete BayesFlow approximator
    # ---------------------------------------------------------

    approximator = build_protein_approximator(
        hidden_dim=64,
        position_dim=64,
        condition_dim=128,
        dropout=0.10,
        learning_rate=1e-3,
    )

    # ---------------------------------------------------------
    # 3. Adapt data
    # ---------------------------------------------------------

    adapted_data = approximator.adapter(
        raw_data,
    )

    print("\nAdapted keys:")
    print(sorted(adapted_data.keys()))

    expected_keys = {
        "summary_variables",
        "inference_variables",
        "summary_mask",
        "inference_mask",
        "sample_weight",
    }

    assert set(adapted_data.keys()) == expected_keys

    print("\nAdapted shapes:")
    for key, value in adapted_data.items():
        print(f"{key}: {value.shape}")

    assert adapted_data["summary_variables"].shape == (
        8,
        250,
        20,
    )

    assert adapted_data["inference_variables"].shape == (
        8,
        250,
        2,
    )

    assert adapted_data["summary_mask"].shape == (
        8,
        250,
    )

    assert adapted_data["inference_mask"].shape == (
        8,
        250,
    )

    assert adapted_data["sample_weight"].shape == (
        8,
        250,
        1,
    )

    # ---------------------------------------------------------
    # 4. Build model from one adapted batch
    # ---------------------------------------------------------

    approximator.build_from_data(
        adapted_data
    )

    # ---------------------------------------------------------
    # 5. Compute initial metrics
    # ---------------------------------------------------------

    metrics = approximator.compute_metrics(
        **adapted_data,
        stage="validation",
    )

    print("\nMetric keys:")
    print(sorted(metrics.keys()))

    for key, value in metrics.items():
        scalar_value = float(np.asarray(value))
        print(f"{key}: {scalar_value:.6f}")

        assert np.isfinite(scalar_value)

    # ---------------------------------------------------------
    # 6. Display model summary
    # ---------------------------------------------------------

    print("\nModel summary:")
    approximator.summary(
        expand_nested=True,
    )

    print()
    print("✓ Complete BayesFlow approximator built")
    print("✓ Adapter connected")
    print("✓ Summary network connected")
    print("✓ Inference network connected")
    print("✓ Padding mask mapped to sample_weight")
    print("✓ BayesFlow loss computed")
    print("✓ Model is ready for smoke training")


if __name__ == "__main__":
    main()