"""
Train the native BayesFlow protein posterior estimator.
"""

from __future__ import annotations

from src.configs.config import N_TRAIN_SAMPLES
import os
from pathlib import Path

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf
import keras
import numpy as np
import time

from src.bayesflow_model.approximator import (
    build_protein_approximator,
)
from src.bayesflow_model.data import (
    prepare_bayesflow_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / f"training_{N_TRAIN_SAMPLES}"
)

MODEL_PATH = (
    OUTPUT_DIR / f"protein_bayesflow_{N_TRAIN_SAMPLES}.keras"
)

HISTORY_PATH = (
    OUTPUT_DIR / f"training_history_{N_TRAIN_SAMPLES}.npz"
)


def build_offline_dataset(
    split_name: str,
    adapter: bf.adapters.Adapter,
    batch_size: int,
    limit: int | None = None,
    shuffle: bool = True,
) -> bf.datasets.OfflineDataset:
    """
    Load one split and wrap it in a BayesFlow OfflineDataset.
    """

    raw_data = prepare_bayesflow_data(
        split_name=split_name,
        limit=limit,
    )

    return bf.datasets.OfflineDataset(
        data=raw_data,
        batch_size=batch_size,
        adapter=adapter,
        shuffle=shuffle,
    )


def main() -> None:
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    batch_size = 32
    epochs = 20
    learning_rate = 5e-4

    train_limit = None
    validation_limit = None

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    approximator = build_protein_approximator(
        hidden_dim=64,
        position_dim=64,
        condition_dim=128,
        dropout=0.10,
        learning_rate=learning_rate,
    )

    # ---------------------------------------------------------
    # Build offline datasets
    # ---------------------------------------------------------

    train_dataset = build_offline_dataset(
        split_name="train",
        adapter=approximator.adapter,
        batch_size=batch_size,
        limit=train_limit,
        shuffle=True,
    )

    validation_dataset = build_offline_dataset(
        split_name="validation",
        adapter=approximator.adapter,
        batch_size=batch_size,
        limit=validation_limit,
        shuffle=False,
    )

    print("Training batches:", len(train_dataset))
    print(
        "Validation batches:",
        len(validation_dataset),
    )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            min_delta=1e-6,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    training_start_time = time.perf_counter()

    history = approximator.fit(
        dataset=train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    training_end_time = time.perf_counter()

    training_duration_seconds = (
            training_end_time
            - training_start_time
    )

    training_duration_minutes = (
            training_duration_seconds / 60.0
    )

    print()
    print("Training time")
    print("=" * 40)
    print(
        f"Seconds: "
        f"{training_duration_seconds:.2f}"
    )
    print(
        f"Minutes: "
        f"{training_duration_minutes:.2f}"
    )

    # ---------------------------------------------------------
    # Save final restored model
    # ---------------------------------------------------------

    approximator.save(
        MODEL_PATH
    )

    # ---------------------------------------------------------
    # Save history
    # ---------------------------------------------------------

    history_arrays = {
        key: np.asarray(values)
        for key, values in history.history.items()
    }

    history_arrays["training_duration_seconds"] = (
        np.asarray(
            [training_duration_seconds],
            dtype=np.float64,
        )
    )

    history_arrays["training_duration_minutes"] = (
        np.asarray(
            [training_duration_minutes],
            dtype=np.float64,
        )
    )

    history_arrays["number_of_train_sequences"] = (
        np.asarray(
            [len(train_dataset) * batch_size],
            dtype=np.int64,
        )
    )

    history_arrays["batch_size"] = np.asarray(
        [batch_size],
        dtype=np.int64,
    )

    history_arrays["requested_epochs"] = np.asarray(
        [epochs],
        dtype=np.int64,
    )

    np.savez(
        HISTORY_PATH,
        **history_arrays,
    )

    print()
    print("Training completed.")
    print("Model saved to:")
    print(MODEL_PATH)
    print("History saved to:")
    print(HISTORY_PATH)


if __name__ == "__main__":
    main()