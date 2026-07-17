"""
Create the seven final figures for the SBI Protein HMM project.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.amortized_posterior import load_trained_model


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "final_training_history.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "evaluation" / "final_results.json"
)
TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "synthetic" / "test_2000.npz"
)
MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "amortized_posterior.pt"
)
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


# ============================================================
# General helpers
# ============================================================

def save_figure(filename: str) -> None:
    """Save the current matplotlib figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def load_json(path: Path) -> dict:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_test_data() -> dict[str, np.ndarray]:
    """Load the final synthetic test dataset."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {TEST_DATA_PATH}")

    with np.load(TEST_DATA_PATH) as data:
        return {
            "x": data["x"].astype(np.float32),
            "y": data["y"].astype(np.float32),
            "mask": data["mask"].astype(bool),
            "lengths": data["lengths"].astype(int),
            "hidden_states": data["hidden_states"].astype(int),
        }


def predict_test_posteriors(
    x: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Predict posterior probabilities for all padded test sequences.

    Output shape:
        (num_sequences, max_length, 2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )
    model.eval()

    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))

            x_batch = torch.tensor(
                x[start:end],
                dtype=torch.float32,
                device=device,
            )

            logits = model(x_batch)
            probabilities = torch.softmax(logits, dim=-1)

            predictions.append(
                probabilities.cpu().numpy().astype(np.float32)
            )

    return np.concatenate(predictions, axis=0)


def confusion_matrix_binary(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
) -> np.ndarray:
    """
    Compute a 2x2 confusion matrix.

    State order:
        0 = other
        1 = alpha
    """
    true_states = np.asarray(true_states).ravel()
    predicted_states = np.asarray(predicted_states).ravel()

    matrix = np.zeros((2, 2), dtype=int)

    for true_value, predicted_value in zip(
        true_states,
        predicted_states,
    ):
        matrix[int(true_value), int(predicted_value)] += 1

    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Plot one confusion matrix with counts and row percentages."""
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(row_percentages, vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("True state")

    labels = ["Other", "Alpha"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)

    for row in range(2):
        for column in range(2):
            count = matrix[row, column]
            percentage = row_percentages[row, column] * 100

            ax.text(
                column,
                row,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Row proportion")
    save_figure(filename)


# ============================================================
# Figure 1: Training loss
# ============================================================

def figure_1_training_loss(history: dict) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Soft cross-entropy loss")
    plt.title("BiLSTM Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure_01_training_loss.png")


# ============================================================
# Figure 2: Validation loss
# ============================================================

def figure_2_validation_loss(history: dict) -> None:
    epochs = np.arange(1, len(history["val_loss"]) + 1)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_value = float(np.min(history["val_loss"]))

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch = {best_epoch}",
    )

    plt.scatter(
        [best_epoch],
        [best_value],
        zorder=3,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Soft cross-entropy loss")
    plt.title("BiLSTM Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("figure_02_validation_loss.png")


# ============================================================
# Figure 3: Posterior MSE
# ============================================================

def figure_3_posterior_mse(results: dict) -> None:
    mse = float(results["synthetic_test"]["posterior_mse"])

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        ["NN vs.\nForward–Backward"],
        [mse],
    )

    plt.ylabel("Posterior MSE")
    plt.title("Synthetic Test: Posterior Approximation Error")
    plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    plt.grid(axis="y", alpha=0.3)

    plt.bar_label(
        bars,
        labels=[f"{mse:.6g}"],
        padding=5,
    )

    save_figure("figure_03_posterior_mse.png")


# ============================================================
# Figure 4: Real-data accuracy comparison
# ============================================================

def figure_4_real_accuracy(results: dict) -> None:
    real_results = results["real_protein_test"]

    labels = ["Forward–Backward", "BiLSTM"]
    values = [
        float(real_results["hmm_state_accuracy"]),
        float(real_results["nn_state_accuracy"]),
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)

    lower_limit = max(0.0, min(values) - 0.03)
    upper_limit = min(1.0, max(values) + 0.03)

    plt.ylim(lower_limit, upper_limit)
    plt.ylabel("State accuracy")
    plt.title("Real Protein Data: State Accuracy")
    plt.grid(axis="y", alpha=0.3)

    plt.bar_label(
        bars,
        labels=[f"{value * 100:.2f}%" for value in values],
        padding=5,
    )

    save_figure("figure_04_real_accuracy.png")


# ============================================================
# Figures 5 and 6: Confusion matrices on synthetic test data
# ============================================================

def figures_5_and_6_confusion_matrices(
    test_data: dict[str, np.ndarray],
    nn_posteriors: np.ndarray,
) -> None:
    mask = test_data["mask"]

    true_states = test_data["hidden_states"][mask]
    hmm_states = np.argmax(test_data["y"], axis=-1)[mask]
    nn_states = np.argmax(nn_posteriors, axis=-1)[mask]

    hmm_matrix = confusion_matrix_binary(
        true_states=true_states,
        predicted_states=hmm_states,
    )

    nn_matrix = confusion_matrix_binary(
        true_states=true_states,
        predicted_states=nn_states,
    )

    plot_confusion_matrix(
        matrix=hmm_matrix,
        title="Synthetic Test: Forward–Backward vs. True States",
        filename="figure_05_hmm_confusion_matrix.png",
    )

    plot_confusion_matrix(
        matrix=nn_matrix,
        title="Synthetic Test: BiLSTM vs. True States",
        filename="figure_06_nn_confusion_matrix.png",
    )


# ============================================================
# Figure 7: Posterior probabilities for one sequence
# ============================================================

def figure_7_sequence_posterior(
    test_data: dict[str, np.ndarray],
    nn_posteriors: np.ndarray,
    sequence_index: int | None = None,
) -> None:
    """
    Plot true hidden states, Forward–Backward posterior, and NN posterior.

    If sequence_index is None, select a sequence with several state changes,
    because it is more informative for the presentation.
    """
    lengths = test_data["lengths"]
    hidden_states = test_data["hidden_states"]

    if sequence_index is None:
        best_index = 0
        most_transitions = -1

        for index, length in enumerate(lengths):
            states = hidden_states[index, :length]
            transitions = int(np.sum(states[1:] != states[:-1]))

            if transitions > most_transitions:
                best_index = index
                most_transitions = transitions

        sequence_index = best_index

    length = int(lengths[sequence_index])
    positions = np.arange(1, length + 1)

    true_states = hidden_states[sequence_index, :length]
    hmm_alpha = test_data["y"][sequence_index, :length, 1]
    nn_alpha = nn_posteriors[sequence_index, :length, 1]

    plt.figure(figsize=(12, 5))

    plt.step(
        positions,
        true_states,
        where="mid",
        linewidth=1.5,
        label="True state (0=Other, 1=Alpha)",
    )

    plt.plot(
        positions,
        hmm_alpha,
        linewidth=2,
        label="Forward–Backward P(Alpha)",
    )

    plt.plot(
        positions,
        nn_alpha,
        linestyle="--",
        linewidth=2,
        label="BiLSTM P(Alpha)",
    )

    plt.axhline(
        0.5,
        linestyle=":",
        linewidth=1,
        label="Decision threshold = 0.5",
    )

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sequence position")
    plt.ylabel("Alpha state / probability")
    plt.title(
        "Posterior Probabilities Along One Synthetic Protein "
        f"(sequence {sequence_index}, length {length})"
    )
    plt.grid(alpha=0.3)
    plt.legend(loc="best")

    save_figure("figure_07_sequence_posterior.png")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading training history...")
    history = load_json(HISTORY_PATH)

    print("Loading final evaluation results...")
    results = load_json(RESULTS_PATH)

    print("Creating Figures 1-4...")
    figure_1_training_loss(history)
    figure_2_validation_loss(history)
    figure_3_posterior_mse(results)
    figure_4_real_accuracy(results)

    print("Loading synthetic test data...")
    test_data = load_test_data()

    print("Predicting synthetic test posteriors...")
    nn_posteriors = predict_test_posteriors(
        x=test_data["x"],
        batch_size=64,
    )

    print("Creating Figures 5-7...")
    figures_5_and_6_confusion_matrices(
        test_data=test_data,
        nn_posteriors=nn_posteriors,
    )

    figure_7_sequence_posterior(
        test_data=test_data,
        nn_posteriors=nn_posteriors,
    )

    print("\nAll figures were created successfully.")
    print(f"Output directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
