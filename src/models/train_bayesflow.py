"""Training entry point for the amortized posterior estimator.

Fast baseline implementation for the ML part of the project.

This module trains a PyTorch BiLSTM model to learn:

    encoded amino-acid sequence -> posterior state probabilities

Important convention:
    output column 0 = other
    output column 1 = alpha

This file does NOT redefine HMM matrices, emission probabilities, state
order, or amino-acid order. It uses the existing simulator, encoding,
and Forward-Backward implementation from the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.simulator.protein_hmm import simulate_hmm
from src.preprocessing.encoding import one_hot_encode_sequence
from src.inference.forward_backward import forward_backward
from src.configs.config import (N_AMINO_ACIDS,
                                N_STATES,
                                MIN_SEQUENCE_LENGTH,
                                MAX_SEQUENCE_LENGTH,
                                N_TRAIN_SAMPLES,
                                N_VALIDATION_SAMPLES,
                                N_TEST_SAMPLES)

DEFAULT_MODEL_PATH = Path("outputs/models/amortized_posterior.pt")
DEFAULT_HISTORY_PATH = Path("outputs/models/training_history.json")
SEQUENCE_LENGTH = 100

class BiLSTMPosteriorEstimator(nn.Module):
    """BiLSTM model for posterior state probability estimation.

    Input shape:
        (batch_size, sequence_length, 20)

    Output shape:
        (batch_size, sequence_length, 2)

    Output column order:
        0 = other
        1 = alpha
    """

    def __init__(
        self,
        input_dim: int = N_AMINO_ACIDS,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        output_dim: int = N_STATES,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (batch, sequence_length, 2)."""

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.head(lstm_out)

        return logits

#
# def simulate_training_sample(
#     sequence_length: int = SEQUENCE_LENGTH,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Generate one training sample from the existing repo pipeline.
#
#     Returns
#     -------
#     encoded_sequence:
#         shape (sequence_length, 20)
#
#     state_probabilities:
#         shape (sequence_length, 2)
#
#     Column order:
#         0 = other
#         1 = alpha
#     """
#
#     protein = simulate_hmm(sequence_length)
#     sequence = protein["AminoAcid"].tolist()
#
#     encoded_sequence = one_hot_encode_sequence(sequence).astype(np.float32)
#     state_probabilities = forward_backward(sequence).astype(np.float32)
#
#     return encoded_sequence, state_probabilities
#
#
# def make_dataset(
#     num_samples: int,
#     sequence_length: int = SEQUENCE_LENGTH,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Create an offline simulated dataset."""
#
#     xs = []
#     ys = []
#
#     for _ in range(num_samples):
#         x, y = simulate_training_sample(sequence_length=sequence_length)
#         xs.append(x)
#         ys.append(y)
#
#     return np.stack(xs, axis=0), np.stack(ys, axis=0)

#SED: Change by Mask
def soft_cross_entropy(
    logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for soft Forward-Backward targets.

    target_probabilities has shape:
        (batch, sequence_length, 2)

    logits has shape:
        (batch, sequence_length, 2)
    """

    log_probs = torch.log_softmax(logits, dim=-1)
    position_loss = -(target_probabilities * log_probs).sum(dim=-1)
    valid_loss = position_loss[mask]

    return valid_loss.mean()

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate loss, MAE, and MSE against Forward-Backward targets.

    Only real sequence positions are included.
    Padded positions are ignored using the mask.
    """

    model.eval()

    total_loss_sum = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_valid_positions = 0

    with torch.no_grad():

        for x_batch, y_batch, mask_batch in data_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            logits = model(x_batch)
            probabilities = torch.softmax(logits, dim=-1)

            # Loss for each sequence position:
            # shape = (batch_size, max_sequence_length)
            log_probabilities = torch.log_softmax(
                logits,
                dim=-1,
            )

            position_loss = -(
                y_batch * log_probabilities
            ).sum(dim=-1)

            # Error for each output probability:
            # shape = (batch_size, max_sequence_length, 2)
            absolute_error = torch.abs(
                probabilities - y_batch
            )

            squared_error = (
                probabilities - y_batch
            ) ** 2

            # Expand mask from:
            # (batch, length)
            # to:
            # (batch, length, 1)
            probability_mask = mask_batch.unsqueeze(-1)

            total_loss_sum += float(
                position_loss[mask_batch].sum().item()
            )

            total_absolute_error += float(
                absolute_error[
                    probability_mask.expand_as(absolute_error)
                ].sum().item()
            )

            total_squared_error += float(
                squared_error[
                    probability_mask.expand_as(squared_error)
                ].sum().item()
            )

            valid_positions = int(mask_batch.sum().item())

            total_valid_positions += valid_positions

    # Loss: one loss value per valid sequence position
    mean_loss = (
        total_loss_sum
        / max(total_valid_positions, 1)
    )

    # MAE and MSE: two probability values per position
    total_valid_probabilities = (
        total_valid_positions * N_STATES
    )

    mean_mae = (
        total_absolute_error
        / max(total_valid_probabilities, 1)
    )

    mean_mse = (
        total_squared_error
        / max(total_valid_probabilities, 1)
    )

    return {
        "loss": mean_loss,
        "mae": mean_mae,
        "mse": mean_mse,
    }

def train_model(
    num_train: int = 1000,
    num_val: int = 200,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    dense_dim: int = 64,
    dropout: float = 0.10,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    seed: int = 123,
) -> tuple[BiLSTMPosteriorEstimator, dict[str, list[float]]]:
    """Train the PyTorch BiLSTM amortized posterior baseline."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    model_path = Path(model_path)
    history_path = Path(history_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # print("Generating training data...")
    # x_train, y_train = make_dataset(num_train, sequence_length=sequence_length)
    #
    # print("Generating validation data...")
    # x_val, y_val = make_dataset(num_val, sequence_length=sequence_length)

    # SED: Change Make Data --> Load
    print("Loading training data...")

    train_data = np.load(
        f"../../data/synthetic/train_{N_TRAIN_SAMPLES}.npz"
    )

    x_train = train_data["x"]
    y_train = train_data["y"]
    train_mask = train_data["mask"]

    print("Loading validation data...")

    val_data = np.load(
        f"../../data/synthetic/validation_{N_TRAIN_SAMPLES}.npz"
    )

    x_val = val_data["x"]
    y_val = val_data["y"]
    val_mask = val_data["mask"]


    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("train_mask shape:", train_mask.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("val_mask shape:", val_mask.shape)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(train_mask, dtype=torch.bool),
    )

    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(val_mask, dtype=torch.bool),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = BiLSTMPosteriorEstimator(
        input_dim=N_AMINO_ACIDS,
        hidden_dim=hidden_dim,
        dense_dim=dense_dim,
        output_dim=N_STATES,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_mse": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        # SED: Change for mask
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = soft_cross_entropy(logits, y_batch, mask_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate_model(model, val_loader, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_mse"].append(val_metrics["mse"])

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_mse={val_metrics['mse']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": N_AMINO_ACIDS,
                    "hidden_dim": hidden_dim,
                    "dense_dim": dense_dim,
                    "output_dim": N_STATES,
                    "dropout": dropout,
                    "sequence_length": sequence_length,
                },
                "state_order": ["other", "alpha"],
                "description": (
                    "BiLSTM amortized posterior baseline. "
                    "Output column 0 = other, column 1 = alpha."
                ),
            }

            torch.save(checkpoint, model_path)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Saved best model to: {model_path}")
    print(f"Saved training history to: {history_path}")

    return model, history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyTorch BiLSTM posterior estimator."
    )

    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dense-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--history-path", type=str, default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--seed", type=int, default=123)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    train_model(
        num_train=args.num_train,
        num_val=args.num_val,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dense_dim=args.dense_dim,
        dropout=args.dropout,
        model_path=args.model_path,
        history_path=args.history_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()