from pathlib import Path

import numpy as np
import torch
from src.configs.config import (N_TRAIN_SAMPLES)
from src.models.train_bayesflow import BiLSTMPosteriorEstimator

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


    x_batch = torch.tensor(x[:16], dtype=torch.float32)
    y_batch = torch.tensor(y[:16], dtype=torch.float32)
    mask_batch = torch.tensor(mask[:16], dtype=torch.bool)

    model = BiLSTMPosteriorEstimator()

    model.eval()

    with torch.no_grad():
        logits = model(x_batch)

    print("Logits shape:", logits.shape)

    assert logits.shape == y_batch.shape
    assert torch.isfinite(logits).all()

    print("Forward pass successful!")

if __name__ == "__main__":
    main()