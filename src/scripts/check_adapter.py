"""Check the official BayesFlow Adapter for protein training pairs."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


# ============================================================
# Settings for the artificial test batch
# ============================================================

BATCH_SIZE = 8
MAX_LENGTH = 250
N_AMINO_ACIDS = 20
N_STATES = 2
RANDOM_SEED = 1


# ============================================================
# Helper functions
# ============================================================

def to_numpy(value: Any) -> np.ndarray:
    """
    Convert NumPy arrays or Torch tensors to NumPy arrays.
    """

    if isinstance(value, np.ndarray):
        return value

    # Torch tensor
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)


def get_project_adapter():
    """
    Load and build the adapter defined in
    src.bayesflow_model.adapter.
    """

    module = importlib.import_module(
        "src.bayesflow_model.adapter"
    )

    if not hasattr(module, "build_protein_adapter"):
        raise AttributeError(
            "Function 'build_protein_adapter' was not found in "
            "'src.bayesflow_model.adapter'."
        )

    print("Using adapter builder: build_protein_adapter")

    return module.build_protein_adapter()

def apply_adapter(adapter: Any, raw_data: dict[str, np.ndarray]) -> dict:
    """
    Apply the BayesFlow adapter to one raw batch.

    BayesFlow adapter versions may support either:
        adapter(raw_data)
    or:
        adapter.adapt(raw_data)
    """

    if callable(adapter):
        return adapter(raw_data)

    if hasattr(adapter, "adapt"):
        return adapter.adapt(raw_data)

    raise TypeError(
        "The loaded adapter is neither callable nor does it provide "
        "an 'adapt' method."
    )


# ============================================================
# Create a small artificial batch
# ============================================================

def create_raw_batch() -> dict[str, np.ndarray]:
    """
    Create one artificial padded batch with the same structure
    as the project's offline datasets.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    # Different valid lengths for the sequences
    sequence_lengths = np.array(
        [250, 225, 200, 175, 150, 125, 100, 75],
        dtype=np.int64,
    )

    protein_sequence = np.zeros(
        (BATCH_SIZE, MAX_LENGTH, N_AMINO_ACIDS),
        dtype=np.float32,
    )

    state_probabilities = np.zeros(
        (BATCH_SIZE, MAX_LENGTH, N_STATES),
        dtype=np.float32,
    )

    encoder_mask = np.zeros(
        (BATCH_SIZE, MAX_LENGTH),
        dtype=np.float32,
    )

    target_mask = np.zeros(
        (BATCH_SIZE, MAX_LENGTH),
        dtype=np.float32,
    )

    loss_weight = np.zeros(
        (BATCH_SIZE, MAX_LENGTH),
        dtype=np.float32,
    )

    for sample_index, length in enumerate(sequence_lengths):
        # ----------------------------------------------------
        # One-hot encoded amino-acid sequence
        # ----------------------------------------------------
        amino_acid_indices = rng.integers(
            low=0,
            high=N_AMINO_ACIDS,
            size=length,
        )

        protein_sequence[
            sample_index,
            np.arange(length),
            amino_acid_indices,
        ] = 1.0

        # ----------------------------------------------------
        # Artificial posterior probabilities
        # [P(other), P(alpha)]
        # ----------------------------------------------------
        alpha_probability = rng.uniform(
            low=0.0,
            high=1.0,
            size=length,
        ).astype(np.float32)

        state_probabilities[
            sample_index,
            :length,
            0,
        ] = 1.0 - alpha_probability

        state_probabilities[
            sample_index,
            :length,
            1,
        ] = alpha_probability

        # ----------------------------------------------------
        # Valid-position masks
        # ----------------------------------------------------
        encoder_mask[sample_index, :length] = 1.0
        target_mask[sample_index, :length] = 1.0
        loss_weight[sample_index, :length] = 1.0

    return {
        "protein_sequence": protein_sequence,
        "state_probabilities": state_probabilities,
        "encoder_mask": encoder_mask,
        "target_mask": target_mask,
        "loss_weight": loss_weight,
        "sequence_lengths": sequence_lengths,
    }


# ============================================================
# Checks
# ============================================================

def check_raw_batch(raw_data: dict[str, np.ndarray]) -> None:
    """
    Validate the batch before passing it to the adapter.
    """

    expected_raw_keys = {
        "protein_sequence",
        "state_probabilities",
        "encoder_mask",
        "target_mask",
        "loss_weight",
        "sequence_lengths",
    }

    assert set(raw_data.keys()) == expected_raw_keys

    assert raw_data["protein_sequence"].shape == (
        BATCH_SIZE,
        MAX_LENGTH,
        N_AMINO_ACIDS,
    )

    assert raw_data["state_probabilities"].shape == (
        BATCH_SIZE,
        MAX_LENGTH,
        N_STATES,
    )

    assert raw_data["encoder_mask"].shape == (
        BATCH_SIZE,
        MAX_LENGTH,
    )

    assert raw_data["target_mask"].shape == (
        BATCH_SIZE,
        MAX_LENGTH,
    )

    assert raw_data["loss_weight"].shape == (
        BATCH_SIZE,
        MAX_LENGTH,
    )

    assert raw_data["sequence_lengths"].shape == (BATCH_SIZE,)

    # Valid one-hot positions must sum to one.
    valid_positions = raw_data["encoder_mask"].astype(bool)

    one_hot_sums = raw_data["protein_sequence"].sum(axis=-1)

    assert np.allclose(
        one_hot_sums[valid_positions],
        1.0,
    )

    # Padded sequence positions must remain zero.
    assert np.allclose(
        raw_data["protein_sequence"][~valid_positions],
        0.0,
    )

    # Posterior probabilities at valid positions must sum to one.
    posterior_sums = raw_data["state_probabilities"].sum(axis=-1)

    assert np.allclose(
        posterior_sums[valid_positions],
        1.0,
        atol=1e-6,
    )

    print("* Raw batch is valid")


def check_adapted_batch(
    raw_data: dict[str, np.ndarray],
    adapted_data: dict,
) -> None:
    """
    Validate the adapter output.
    """

    expected_adapted_keys = {
        "summary_variables",
        "inference_variables",
        "summary_mask",
        "inference_mask",
        "sample_weight",
    }

    actual_keys = set(adapted_data.keys())

    assert actual_keys == expected_adapted_keys, (
        "\nUnexpected adapted keys.\n"
        f"Expected: {sorted(expected_adapted_keys)}\n"
        f"Actual:   {sorted(actual_keys)}"
    )

    print("* Adapted keys are correct")

    summary_variables = to_numpy(
        adapted_data["summary_variables"]
    )

    inference_variables = to_numpy(
        adapted_data["inference_variables"]
    )

    summary_mask = to_numpy(
        adapted_data["summary_mask"]
    )

    inference_mask = to_numpy(
        adapted_data["inference_mask"]
    )

    sample_weight = to_numpy(
        adapted_data["sample_weight"]
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert summary_variables.shape == (
        BATCH_SIZE,
        MAX_LENGTH,
        N_AMINO_ACIDS,
    )

    assert inference_variables.shape == (
        BATCH_SIZE,
        MAX_LENGTH,
        N_STATES,
    )

    assert summary_mask.shape == (
        BATCH_SIZE,
        MAX_LENGTH,
    )

    assert inference_mask.shape == (
        BATCH_SIZE,
        MAX_LENGTH,
    )

    assert sample_weight.shape == (
        BATCH_SIZE,
        MAX_LENGTH,
    )

    print("* Adapted shapes are correct")

    # --------------------------------------------------------
    # Mapping checks
    # --------------------------------------------------------

    assert np.allclose(
        summary_variables,
        raw_data["protein_sequence"],
    )

    assert np.allclose(
        inference_variables,
        raw_data["state_probabilities"],
    )

    assert np.allclose(
        summary_mask,
        raw_data["encoder_mask"],
    )

    assert np.allclose(
        inference_mask,
        raw_data["target_mask"],
    )

    assert np.allclose(
        sample_weight,
        raw_data["loss_weight"],
    )

    print("* Raw-to-adapted variable mapping is correct")

    # --------------------------------------------------------
    # Mask and weight checks
    # --------------------------------------------------------

    assert np.all(
        np.isin(summary_mask, [0.0, 1.0])
    )

    assert np.all(
        np.isin(inference_mask, [0.0, 1.0])
    )

    assert np.all(sample_weight >= 0.0)

    assert np.allclose(
        summary_mask,
        inference_mask,
    )

    assert np.allclose(
        inference_mask,
        sample_weight,
    )

    print("* Masks and sample weights are valid")

    # --------------------------------------------------------
    # Probability checks
    # --------------------------------------------------------

    valid_positions = inference_mask.astype(bool)

    valid_probabilities = inference_variables[valid_positions]

    assert np.all(valid_probabilities >= 0.0)
    assert np.all(valid_probabilities <= 1.0)

    assert np.allclose(
        valid_probabilities.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    print("* Adapted posterior probabilities are valid")

    # sequence_lengths is intentionally not part of the final
    # BayesFlow dictionary because the masks already contain
    # the valid sequence-length information.
    assert "sequence_lengths" not in adapted_data

    print("* sequence_lengths is correctly represented by masks")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 60)
    print("Checking native BayesFlow adapter")
    print("=" * 60)

    raw_data = create_raw_batch()

    print("\nRaw data keys:")
    print(sorted(raw_data.keys()))

    print("\nRaw shapes:")
    for key, value in raw_data.items():
        print(f"{key}: {value.shape}")

    check_raw_batch(raw_data)

    adapter = get_project_adapter()
    adapted_data = apply_adapter(adapter, raw_data)

    print("\nAdapted data keys:")
    print(sorted(adapted_data.keys()))

    print("\nAdapted shapes:")
    for key, value in adapted_data.items():
        print(f"{key}: {to_numpy(value).shape}")

    check_adapted_batch(
        raw_data=raw_data,
        adapted_data=adapted_data,
    )

    print("\n" + "=" * 60)
    print("Adapter check passed successfully.")
    print("No files were created or overwritten.")
    print("=" * 60)


if __name__ == "__main__":
    main()