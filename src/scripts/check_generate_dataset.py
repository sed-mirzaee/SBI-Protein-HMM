from pathlib import Path

import numpy as np
import torch
from src.configs.config import (N_TRAIN_SAMPLES)

DATA_PATH = Path(f"../../data/synthetic/train_{N_TRAIN_SAMPLES}.npz")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = np.load(DATA_PATH)

    print("Keys:", data.files)

    x = data["x"]
    y = data["y"]
    mask = data["mask"]
    lengths = data["lengths"]
    hidden_states = data["hidden_states"]

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("mask shape:", mask.shape)
    print("lengths shape:", lengths.shape)

    print("x dtype:", x.dtype)
    print("y dtype:", y.dtype)
    print("mask dtype:", mask.dtype)

    print("Minimum length:", lengths.min())
    print("Maximum length:", lengths.max())
    print("Mean length:", lengths.mean())

    assert x.ndim == 3
    assert y.ndim == 3
    assert mask.ndim == 2

    assert x.shape[0] == y.shape[0] == mask.shape[0]
    assert x.shape[1] == y.shape[1] == mask.shape[1]

    assert x.shape[2] == 20
    assert y.shape[2] == 2

    assert len(lengths) == x.shape[0]

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    assert mask.dtype == np.bool_
    print("Part2, No Problem.")

    sample_id = 0
    length = int(lengths[sample_id])

    print("Sample length:", length)

    assert mask[sample_id, :length].all()
    assert not mask[sample_id, length:].any()
    assert np.allclose(
        x[sample_id, length:],
        0.0,
    )
    assert np.allclose(
        y[sample_id, length:],
        0.0,
    )
    print("Part3, No Problem.")

    # ---------------------------------------------------------
    # Part 4: Final dataset consistency checks
    # ---------------------------------------------------------

    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == mask.shape[0]
    assert x.shape[0] == lengths.shape[0]
    assert x.shape[0] == hidden_states.shape[0]

    assert x.shape[1] == y.shape[1]
    assert x.shape[1] == mask.shape[1]
    assert x.shape[1] == hidden_states.shape[1]

    assert x.shape[2] == 20
    assert y.shape[2] == 2

    # Valid positions in x must be one-hot encoded
    valid_x = x[mask]
    assert np.allclose(valid_x.sum(axis=1), 1.0)

    # Valid posterior probabilities must sum to one
    valid_y = y[mask]
    assert np.allclose(valid_y.sum(axis=1), 1.0, atol=1e-6)

    # Padded positions must be zero
    assert np.allclose(x[~mask], 0.0)
    assert np.allclose(y[~mask], 0.0)

    print("Part4, No Problem.")
    print("\nDataset generation check passed successfully.")

if __name__ == "__main__":
    main()