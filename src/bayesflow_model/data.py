"""Load the offline protein datasets for native BayesFlow training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.configs.config import N_TRAIN_SAMPLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"


def get_split_path(split_name: str) -> Path:
    """
    Return the dataset path for the selected experiment.

    The suffix identifies the training run. For example,
    validation_15000.npz belongs to the experiment trained
    with 15,000 training sequences.
    """

    valid_splits = {"train", "validation", "test"}

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
    """Load and validate one saved synthetic split."""

    path = get_split_path(split_name)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path) as data:
        required_keys = {
            "x",
            "y",
            "mask",
            "lengths",
            "hidden_states",
        }

        missing_keys = required_keys.difference(data.files)

        if missing_keys:
            raise KeyError(
                f"{path.name} is missing keys: "
                f"{sorted(missing_keys)}"
            )

        result = {
            "x": data["x"].astype(np.float32),
            "y": data["y"].astype(np.float32),
            "mask": data["mask"].astype(bool),
            "lengths": data["lengths"].astype(np.int32),
            "hidden_states": data["hidden_states"].astype(np.int32),
        }

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive.")

        result = {
            key: values[:limit]
            for key, values in result.items()
        }

    _validate_split(result, split_name)

    return result


def _validate_split(
    data: dict[str, np.ndarray],
    split_name: str,
) -> None:
    """Check the stored data contract."""

    x = data["x"]
    y = data["y"]
    mask = data["mask"]
    lengths = data["lengths"]

    if x.ndim != 3 or x.shape[-1] != 20:
        raise ValueError(
            f"{split_name}: x must have shape "
            f"(samples, length, 20), got {x.shape}."
        )

    if y.shape != (*x.shape[:2], 2):
        raise ValueError(
            f"{split_name}: y must have shape "
            f"{(*x.shape[:2], 2)}, got {y.shape}."
        )

    if mask.shape != x.shape[:2]:
        raise ValueError(
            f"{split_name}: mask must have shape "
            f"{x.shape[:2]}, got {mask.shape}."
        )

    if lengths.shape != (x.shape[0],):
        raise ValueError(
            f"{split_name}: lengths has wrong shape "
            f"{lengths.shape}."
        )

    if not np.array_equal(mask.sum(axis=1), lengths):
        raise ValueError(
            f"{split_name}: mask and lengths are inconsistent."
        )

    valid_y = y[mask]

    if not np.isfinite(x).all():
        raise ValueError(f"{split_name}: x contains NaN or Inf.")

    if not np.isfinite(valid_y).all():
        raise ValueError(f"{split_name}: y contains NaN or Inf.")

    if not np.allclose(
        valid_y.sum(axis=-1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{split_name}: posterior rows do not sum to one."
        )

def prepare_bayesflow_data(
    split_name: str,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Prepare one offline dataset split for BayesFlow.

    The saved Forward-Backward probabilities are the training targets.
    No new posterior is calculated here.
    """

    data = load_split(
        split_name=split_name,
        limit=limit,
    )

    protein_sequence = data["x"].astype(np.float32)
    state_probabilities = data["y"].astype(np.float32)
    sequence_mask = data["mask"].astype(bool)

    return {
        # Input x
        "protein_sequence": protein_sequence,

        # Target y, already calculated by Forward-Backward
        "state_probabilities": state_probabilities,

        # Same Boolean values, but two distinct roles
        "encoder_mask": sequence_mask.copy(),
        "target_mask": sequence_mask.copy(),

        # Kept in raw data for checks, but not required by the network
        "sequence_lengths": data["lengths"].astype(np.int32),
    }