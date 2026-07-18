"""
Create final presentation/report figures___ for the native BayesFlow model.

The script reads already-saved outputs from:

    outputs/native_bayesflow/training_<N>/
    outputs/native_bayesflow/evaluation/

It does not load the model and does not run inference again.

Run from the project root:

    python -m src.evaluation.make_final_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.configs.config import N_TRAIN_SAMPLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / f"training_{N_TRAIN_SAMPLES}"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / "evaluation"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / "figures___"
)

HISTORY_PATH = (
    TRAINING_DIR
    / f"training_history_{N_TRAIN_SAMPLES}.npz"
)

SYNTHETIC_PREDICTIONS_PATH = (
    EVALUATION_DIR
    / f"test_predictions_{N_TRAIN_SAMPLES}.npz"
)

REAL_PREDICTIONS_PATH = (
    EVALUATION_DIR
    / f"real_protein_predictions_{N_TRAIN_SAMPLES}.npz"
)

RESULTS_PATH = (
    EVALUATION_DIR
    / f"evaluation_results_{N_TRAIN_SAMPLES}.json"
)


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save one figure and close it."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FIGURE_DIR / filename

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {output_path}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load all arrays from one npz file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def load_json(path: Path) -> dict:
    """Load one JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_history_array(
    history: dict[str, np.ndarray],
    possible_names: tuple[str, ...],
) -> np.ndarray:
    """
    Return the first available training-history series.

    Keras usually saves loss and val_loss, but this helper
    keeps the plotting script robust to slightly different names.
    """

    for name in possible_names:
        if name in history:
            values = np.asarray(
                history[name],
                dtype=float,
            ).reshape(-1)

            if values.size > 0:
                return values

    raise KeyError(
        "None of these history keys were found: "
        f"{possible_names}. "
        f"Available keys: {sorted(history)}"
    )


def confusion_matrix_binary(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
) -> np.ndarray:
    """
    Compute a 2 x 2 confusion matrix.

    Row: true state
    Column: predicted state

    State 0 = other
    State 1 = alpha
    """

    true_states = np.asarray(
        true_states,
        dtype=int,
    ).reshape(-1)

    predicted_states = np.asarray(
        predicted_states,
        dtype=int,
    ).reshape(-1)

    if true_states.shape != predicted_states.shape:
        raise ValueError(
            "True and predicted states have "
            "different shapes."
        )

    matrix = np.zeros(
        (2, 2),
        dtype=int,
    )

    for true_value, predicted_value in zip(
        true_states,
        predicted_states,
    ):
        matrix[
            int(true_value),
            int(predicted_value),
        ] += 1

    return matrix


def add_bar_labels(
    axis: plt.Axes,
    bars,
    values: list[float],
    percentage: bool = False,
) -> None:
    """Add readable labels above bars."""

    if percentage:
        labels = [
            f"{value * 100:.2f}%"
            for value in values
        ]
    else:
        labels = [
            f"{value:.6f}"
            for value in values
        ]

    axis.bar_label(
        bars,
        labels=labels,
        padding=4,
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Plot counts and row percentages."""

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(
        figsize=(6, 5),
    )

    image = axis.imshow(
        row_percentages,
        vmin=0.0,
        vmax=1.0,
    )

    labels = ["Other", "Alpha"]

    axis.set_xticks(
        [0, 1],
        labels=labels,
    )

    axis.set_yticks(
        [0, 1],
        labels=labels,
    )

    axis.set_xlabel("Predicted state")
    axis.set_ylabel("True state")
    axis.set_title(title)

    for row in range(2):
        for column in range(2):
            count = matrix[
                row,
                column,
            ]

            percentage = (
                row_percentages[
                    row,
                    column,
                ]
                * 100.0
            )

            axis.text(
                column,
                row,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Row proportion",
    )

    save_figure(
        figure,
        filename,
    )


def figure_01_training_history(
    history: dict[str, np.ndarray],
) -> None:
    """Plot training and validation loss together."""

    train_loss = get_history_array(
        history,
        (
            "loss",
            "train_loss",
        ),
    )

    validation_loss = get_history_array(
        history,
        (
            "val_loss",
            "validation_loss",
        ),
    )

    number_of_epochs = min(
        len(train_loss),
        len(validation_loss),
    )

    train_loss = train_loss[
        :number_of_epochs
    ]

    validation_loss = validation_loss[
        :number_of_epochs
    ]

    epochs = np.arange(
        1,
        number_of_epochs + 1,
    )

    best_index = int(
        np.argmin(validation_loss)
    )

    best_epoch = best_index + 1
    best_validation_loss = float(
        validation_loss[best_index]
    )

    figure, axis = plt.subplots(
        figsize=(9, 5),
    )

    axis.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training loss",
    )

    axis.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation loss",
    )

    axis.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch = {best_epoch}",
    )

    axis.scatter(
        [best_epoch],
        [best_validation_loss],
        zorder=3,
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(
        "Native BayesFlow Training History"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    save_figure(
        figure,
        "figure_01_training_history.png",
    )


def choose_synthetic_sequence(
    hidden_states: np.ndarray,
    lengths: np.ndarray,
) -> int:
    """
    Choose a visually informative synthetic sequence.

    The selected sequence has the greatest number of
    hidden-state transitions.
    """

    best_index = 0
    most_transitions = -1

    for index, raw_length in enumerate(
        lengths
    ):
        length = int(raw_length)

        states = hidden_states[
            index,
            :length,
        ]

        if length <= 1:
            transitions = 0
        else:
            transitions = int(
                np.sum(
                    states[1:]
                    != states[:-1]
                )
            )

        if transitions > most_transitions:
            best_index = index
            most_transitions = transitions

    return best_index


def figure_02_synthetic_posterior_example(
    synthetic: dict[str, np.ndarray],
) -> None:
    """
    Plot hidden states and posterior probabilities
    for one synthetic protein.
    """

    predictions = synthetic["predictions"]
    targets = synthetic["targets"]
    hidden_states = synthetic[
        "hidden_states"
    ]
    lengths = synthetic["lengths"]

    sequence_index = choose_synthetic_sequence(
        hidden_states=hidden_states,
        lengths=lengths,
    )

    length = int(
        lengths[sequence_index]
    )

    positions = np.arange(
        1,
        length + 1,
    )

    true_states = hidden_states[
        sequence_index,
        :length,
    ]

    hmm_alpha = targets[
        sequence_index,
        :length,
        1,
    ]

    neural_alpha = predictions[
        sequence_index,
        :length,
        1,
    ]

    figure, axis = plt.subplots(
        figsize=(12, 5),
    )

    axis.step(
        positions,
        true_states,
        where="mid",
        linewidth=1.4,
        label=(
            "True state "
            "(0=Other, 1=Alpha)"
        ),
    )

    axis.plot(
        positions,
        hmm_alpha,
        linewidth=2,
        label=(
            "Forward–Backward "
            "P(Alpha)"
        ),
    )

    axis.plot(
        positions,
        neural_alpha,
        linestyle="--",
        linewidth=2,
        label=(
            "Native BayesFlow "
            "P(Alpha)"
        ),
    )

    axis.axhline(
        0.5,
        linestyle=":",
        linewidth=1,
        label="Decision threshold = 0.5",
    )

    axis.set_ylim(
        -0.05,
        1.05,
    )

    axis.set_xlabel(
        "Sequence position"
    )

    axis.set_ylabel(
        "Alpha state / probability"
    )

    axis.set_title(
        "Synthetic Protein: "
        "Posterior Probability by Position"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    save_figure(
        figure,
        "figure_02_synthetic_posterior_example.png",
    )


def figure_03_posterior_scatter(
    synthetic: dict[str, np.ndarray],
    maximum_points: int = 30000,
) -> None:
    """
    Scatter plot of BayesFlow and Forward-Backward
    alpha probabilities.
    """

    predictions = synthetic["predictions"]
    targets = synthetic["targets"]
    mask = synthetic["mask"].astype(bool)

    target_alpha = targets[
        :,
        :,
        1,
    ][mask]

    predicted_alpha = predictions[
        :,
        :,
        1,
    ][mask]

    number_of_points = len(
        target_alpha
    )

    if number_of_points > maximum_points:
        random_generator = (
            np.random.default_rng(123)
        )

        selected_indices = (
            random_generator.choice(
                number_of_points,
                size=maximum_points,
                replace=False,
            )
        )

        target_alpha = target_alpha[
            selected_indices
        ]

        predicted_alpha = predicted_alpha[
            selected_indices
        ]

    figure, axis = plt.subplots(
        figsize=(6, 6),
    )

    axis.scatter(
        target_alpha,
        predicted_alpha,
        alpha=0.20,
        s=8,
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect agreement",
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.set_xlabel(
        "Forward–Backward P(Alpha)"
    )

    axis.set_ylabel(
        "Native BayesFlow P(Alpha)"
    )

    axis.set_title(
        "Synthetic Test: "
        "Posterior Probability Agreement"
    )

    axis.grid(alpha=0.3)
    axis.legend()

    save_figure(
        figure,
        "figure_03_posterior_scatter.png",
    )


def figure_04_posterior_error_distribution(
    synthetic: dict[str, np.ndarray],
) -> None:
    """Plot errors in alpha posterior probabilities."""

    predictions = synthetic["predictions"]
    targets = synthetic["targets"]
    mask = synthetic["mask"].astype(bool)

    errors = (
        predictions[:, :, 1]
        - targets[:, :, 1]
    )[mask]

    mean_error = float(
        np.mean(errors)
    )

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    axis.hist(
        errors,
        bins=60,
    )

    axis.axvline(
        0.0,
        linestyle="--",
        label="Zero error",
    )

    axis.axvline(
        mean_error,
        linestyle=":",
        label=(
            f"Mean error = "
            f"{mean_error:.4f}"
        ),
    )

    axis.set_xlabel(
        "BayesFlow P(Alpha) "
        "− Forward–Backward P(Alpha)"
    )

    axis.set_ylabel(
        "Number of positions"
    )

    axis.set_title(
        "Synthetic Test: "
        "Posterior Error Distribution"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    save_figure(
        figure,
        "figure_04_posterior_error_distribution.png",
    )


def figure_05_synthetic_accuracy(
    results: dict,
) -> None:
    """Compare main synthetic state metrics."""

    metrics = results[
        "synthetic_test"
    ]

    labels = [
        "Forward–Backward\naccuracy",
        "Native BayesFlow\naccuracy",
        "Model/HMM\nagreement",
    ]

    values = [
        float(
            metrics[
                "forward_backward_state_accuracy"
            ]
        ),
        float(
            metrics[
                "neural_state_accuracy"
            ]
        ),
        float(
            metrics[
                "neural_forward_backward_agreement"
            ]
        ),
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    bars = axis.bar(
        labels,
        values,
    )

    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Proportion")
    axis.set_title(
        "Synthetic Test: "
        "State-Level Performance"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    add_bar_labels(
        axis=axis,
        bars=bars,
        values=values,
        percentage=True,
    )

    save_figure(
        figure,
        "figure_05_synthetic_accuracy_comparison.png",
    )


def figures_06_and_07_synthetic_confusion(
    synthetic: dict[str, np.ndarray],
) -> None:
    """Create two synthetic confusion matrices."""

    mask = synthetic[
        "mask"
    ].astype(bool)

    true_states = synthetic[
        "hidden_states"
    ][mask]

    hmm_states = np.argmax(
        synthetic["targets"],
        axis=-1,
    )[mask]

    neural_states = np.argmax(
        synthetic["predictions"],
        axis=-1,
    )[mask]

    hmm_matrix = confusion_matrix_binary(
        true_states=true_states,
        predicted_states=hmm_states,
    )

    neural_matrix = confusion_matrix_binary(
        true_states=true_states,
        predicted_states=neural_states,
    )

    plot_confusion_matrix(
        matrix=hmm_matrix,
        title=(
            "Synthetic Test: "
            "Forward–Backward vs True States"
        ),
        filename=(
            "figure_06_synthetic_hmm_confusion.png"
        ),
    )

    plot_confusion_matrix(
        matrix=neural_matrix,
        title=(
            "Synthetic Test: "
            "Native BayesFlow vs True States"
        ),
        filename=(
            "figure_07_synthetic_bayesflow_confusion.png"
        ),
    )


def choose_real_sequence(
    sequence_ids: np.ndarray,
    ground_truth: np.ndarray,
) -> int:
    """
    Choose an informative real sequence.

    The selected sequence has the most annotation transitions.
    """

    unique_ids = np.unique(
        sequence_ids
    )

    best_id = int(
        unique_ids[0]
    )

    most_transitions = -1

    for sequence_id in unique_ids:
        selected = (
            sequence_ids
            == sequence_id
        )

        states = ground_truth[
            selected
        ]

        transitions = int(
            np.sum(
                states[1:]
                != states[:-1]
            )
        )

        if transitions > most_transitions:
            best_id = int(sequence_id)
            most_transitions = transitions

    return best_id


def figure_08_real_posterior_example(
    real: dict[str, np.ndarray],
) -> None:
    """
    Plot annotation, Forward-Backward posterior,
    and Native BayesFlow posterior for one real protein.
    """

    sequence_ids = real["sequence_id"]
    positions = real["position"]
    ground_truth = real["ground_truth"]

    selected_id = choose_real_sequence(
        sequence_ids=sequence_ids,
        ground_truth=ground_truth,
    )

    selected = (
        sequence_ids
        == selected_id
    )

    selected_positions = positions[
        selected
    ]

    true_states = ground_truth[
        selected
    ]

    hmm_alpha = real[
        "hmm_probabilities"
    ][selected, 1]

    neural_alpha = real[
        "neural_probabilities"
    ][selected, 1]

    figure, axis = plt.subplots(
        figsize=(12, 5),
    )

    axis.step(
        selected_positions,
        true_states,
        where="mid",
        linewidth=1.4,
        label=(
            "Real annotation "
            "(0=Other, 1=Alpha)"
        ),
    )

    axis.plot(
        selected_positions,
        hmm_alpha,
        linewidth=2,
        label=(
            "Forward–Backward "
            "P(Alpha)"
        ),
    )

    axis.plot(
        selected_positions,
        neural_alpha,
        linestyle="--",
        linewidth=2,
        label=(
            "Native BayesFlow "
            "P(Alpha)"
        ),
    )

    axis.axhline(
        0.5,
        linestyle=":",
        linewidth=1,
        label="Decision threshold = 0.5",
    )

    axis.set_ylim(
        -0.05,
        1.05,
    )

    axis.set_xlabel(
        "Protein position"
    )

    axis.set_ylabel(
        "Alpha state / probability"
    )

    axis.set_title(
        "Real Protein: "
        "Posterior Probability by Position "
        "(maximum 250 residues)"
    )

    axis.grid(alpha=0.3)
    axis.legend(loc="best")

    save_figure(
        figure,
        "figure_08_real_protein_posterior.png",
    )


def figure_09_real_accuracy(
    results: dict,
) -> None:
    """
    Compare real-data baseline, Forward-Backward,
    and Native BayesFlow accuracy.
    """

    real_metrics = results.get(
        "real_protein_test"
    )

    if real_metrics is None:
        print(
            "Skipping real accuracy figure: "
            "real_protein_test is null."
        )
        return

    labels = [
        "Always Other\nbaseline",
        "Forward–Backward",
        "Native BayesFlow",
    ]

    values = [
        float(
            real_metrics[
                "always_other_baseline"
            ]
        ),
        float(
            real_metrics[
                "hmm_state_accuracy"
            ]
        ),
        float(
            real_metrics[
                "neural_state_accuracy"
            ]
        ),
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    bars = axis.bar(
        labels,
        values,
    )

    axis.set_ylim(0, 1.05)
    axis.set_ylabel("State accuracy")
    axis.set_title(
        "Real Protein Data: "
        "Accuracy Comparison"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    add_bar_labels(
        axis=axis,
        bars=bars,
        values=values,
        percentage=True,
    )

    save_figure(
        figure,
        "figure_09_real_accuracy_comparison.png",
    )


def figures_10_and_11_real_confusion(
    real: dict[str, np.ndarray],
) -> None:
    """Create real-data confusion matrices."""

    true_states = real[
        "ground_truth"
    ]

    hmm_states = real[
        "hmm_states"
    ]

    neural_states = real[
        "neural_states"
    ]

    hmm_matrix = confusion_matrix_binary(
        true_states=true_states,
        predicted_states=hmm_states,
    )

    neural_matrix = confusion_matrix_binary(
        true_states=true_states,
        predicted_states=neural_states,
    )

    plot_confusion_matrix(
        matrix=hmm_matrix,
        title=(
            "Real Proteins: "
            "Forward–Backward vs Annotation"
        ),
        filename=(
            "figure_10_real_hmm_confusion.png"
        ),
    )

    plot_confusion_matrix(
        matrix=neural_matrix,
        title=(
            "Real Proteins: "
            "Native BayesFlow vs Annotation"
        ),
        filename=(
            "figure_11_real_bayesflow_confusion.png"
        ),
    )


def main() -> None:
    """Create all available figures___."""

    print("Loading training history...")
    history = load_npz(
        HISTORY_PATH
    )

    print(
        "Loading synthetic evaluation outputs..."
    )
    synthetic = load_npz(
        SYNTHETIC_PREDICTIONS_PATH
    )

    print(
        "Loading evaluation metrics..."
    )
    results = load_json(
        RESULTS_PATH
    )

    print()
    print("Creating training figure...")
    figure_01_training_history(
        history
    )

    print(
        "Creating synthetic figures___..."
    )
    figure_02_synthetic_posterior_example(
        synthetic
    )
    figure_03_posterior_scatter(
        synthetic
    )
    figure_04_posterior_error_distribution(
        synthetic
    )
    figure_05_synthetic_accuracy(
        results
    )
    figures_06_and_07_synthetic_confusion(
        synthetic
    )

    if REAL_PREDICTIONS_PATH.exists():
        print(
            "Loading real-protein outputs..."
        )

        real = load_npz(
            REAL_PREDICTIONS_PATH
        )

        print(
            "Creating real-protein figures___..."
        )

        figure_08_real_posterior_example(
            real
        )
        figure_09_real_accuracy(
            results
        )
        figures_10_and_11_real_confusion(
            real
        )
    else:
        print()
        print(
            "Real-protein prediction file "
            "was not found."
        )
        print(
            "Synthetic and training figures___ "
            "were still created."
        )
        print(
            "Missing file:"
        )
        print(
            REAL_PREDICTIONS_PATH
        )

    print()
    print(
        "All available figures___ "
        "were created successfully."
    )
    print(
        f"Output directory:\n{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
