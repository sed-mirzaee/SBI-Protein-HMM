"""Load offline protein datasets for native BayesFlow training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.configs.config import N_TRAIN_SAMPLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"


def get_split_path(split_name: str) -> Path:
    """
    Return the dataset path for one experiment split.

    The filename suffix identifies the training experiment.
    For example, validation_15000.npz belongs to the experiment
    whose training set contains 15,000 sequences.
    """

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    if split_name not in valid_splits:
        raise ValueError(
            f"split_name must be one of {sorted(valid_splits)}, "
            f"but received {split_name!r}."
        )

    return DATA_DIR / f"{split_name}_{N_TRAIN_SAMPLES}.npz"


def load_split(
    split_name: str,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Load and validate one saved synthetic-data split."""

    path = get_split_path(split_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with np.load(path) as stored_data:
        required_keys = {
            "x",
            "y",
            "mask",
            "lengths",
            "hidden_states",
        }

        missing_keys = required_keys.difference(
            stored_data.files
        )

        if missing_keys:
            raise KeyError(
                f"{path.name} is missing keys: "
                f"{sorted(missing_keys)}"
            )

        data = {
            "x": stored_data["x"].astype(
                np.float32
            ),
            "y": stored_data["y"].astype(
                np.float32
            ),
            "mask": stored_data["mask"].astype(
                bool
            ),
            "lengths": stored_data["lengths"].astype(
                np.int32
            ),
            "hidden_states": stored_data[
                "hidden_states"
            ].astype(np.int32),
        }

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit must be a positive integer."
            )

        data = {
            key: values[:limit]
            for key, values in data.items()
        }

    _validate_split(
        data=data,
        split_name=split_name,
    )

    return data


def _validate_split(
    data: dict[str, np.ndarray],
    split_name: str,
) -> None:
    """Validate shapes, masks and probability values."""

    x = data["x"]
    y = data["y"]
    mask = data["mask"]
    lengths = data["lengths"]
    hidden_states = data["hidden_states"]

    if x.ndim != 3 or x.shape[-1] != 20:
        raise ValueError(
            f"{split_name}: x must have shape "
            f"(samples, sequence_length, 20), "
            f"but received {x.shape}."
        )

    expected_y_shape = (
        x.shape[0],
        x.shape[1],
        2,
    )

    if y.shape != expected_y_shape:
        raise ValueError(
            f"{split_name}: y must have shape "
            f"{expected_y_shape}, but received {y.shape}."
        )

    expected_mask_shape = x.shape[:2]

    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"{split_name}: mask must have shape "
            f"{expected_mask_shape}, "
            f"but received {mask.shape}."
        )

    if hidden_states.shape != expected_mask_shape:
        raise ValueError(
            f"{split_name}: hidden_states must have shape "
            f"{expected_mask_shape}, "
            f"but received {hidden_states.shape}."
        )

    if lengths.shape != (x.shape[0],):
        raise ValueError(
            f"{split_name}: lengths must have shape "
            f"({x.shape[0]},), but received "
            f"{lengths.shape}."
        )

    if not np.array_equal(
        mask.sum(axis=1),
        lengths,
    ):
        raise ValueError(
            f"{split_name}: mask and lengths "
            f"are inconsistent."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"{split_name}: x contains NaN or Inf."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{split_name}: y contains NaN or Inf."
        )

    valid_x = x[mask]
    valid_y = y[mask]
    padded_x = x[~mask]
    padded_y = y[~mask]

    # Every real amino acid must be one-hot encoded.
    if not np.allclose(
        valid_x.sum(axis=-1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{split_name}: valid amino-acid rows "
            f"are not one-hot encoded."
        )

    # Padding rows must contain only zeros.
    if not np.allclose(
        padded_x,
        0.0,
        atol=1e-7,
    ):
        raise ValueError(
            f"{split_name}: padded x positions "
            f"are not all zero."
        )

    # Forward–Backward posterior probabilities.
    if not np.allclose(
        valid_y.sum(axis=-1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{split_name}: valid posterior rows "
            f"do not sum to one."
        )

    if np.any(valid_y < 0.0) or np.any(
        valid_y > 1.0
    ):
        raise ValueError(
            f"{split_name}: posterior probabilities "
            f"must lie between zero and one."
        )

    if not np.allclose(
        padded_y,
        0.0,
        atol=1e-7,
    ):
        raise ValueError(
            f"{split_name}: padded y positions "
            f"are not all zero."
        )


def prepare_bayesflow_data(
    split_name: str,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Prepare stored pairs for the BayesFlow Adapter.

    protein_sequence:
        Padded one-hot amino-acid sequences.

    state_probabilities:
        Forward-Backward posterior probabilities.

    encoder_mask:
        Padding mask used by the BiLSTM summary network.

    target_mask:
        Padding mask forwarded to the inference network.

    loss_weight:
        Temporal sample weights used to exclude padding
        from the BayesFlow scoring-rule loss.
    """

    data = load_split(
        split_name=split_name,
        limit=limit,
    )

    sequence_mask = data["mask"].astype(bool)

    return {
        "protein_sequence": data["x"],
        "state_probabilities": data["y"],

        "encoder_mask": sequence_mask.copy(),
        "target_mask": sequence_mask.copy(),

        # Shape: (samples, sequence_length, 1)
        # This broadcasts over the final state dimension.
        "loss_weight": sequence_mask.astype(
            np.float32
        )[..., None],

        "sequence_lengths": data["lengths"],
    }