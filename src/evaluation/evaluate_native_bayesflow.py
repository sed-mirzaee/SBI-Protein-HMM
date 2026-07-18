"""
Evaluate the saved native BayesFlow model on:

1. the complete synthetic test split, and
2. real proteins with sst3 annotations (first 250 positions).

For real proteins, the project uses:
    H -> alpha-helix -> state 1
    C/E -> other      -> state 0

The script compares:
- Native BayesFlow posterior vs Forward-Backward posterior on synthetic data
- Forward-Backward hard states vs real annotations
- Native BayesFlow hard states vs real annotations

Run from the project root:

    python -m src.bayesflow_model.evaluate_native_bayesflow

Optional arguments:

    python -m src.bayesflow_model.evaluate_native_bayesflow \
        --real-data-path data/processed/real_proteins.csv \
        --max-real-proteins 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd

# Register custom classes before loading the saved Keras model.
from src.bayesflow_model.approximator import TorchScoringRuleApproximator
from src.bayesflow_model.inference_network import (
    ProbabilityMeanScore,
    SequenceConditionSubnet,
)
from src.bayesflow_model.summary_network import ProteinBiLSTMSequenceNetwork
from src.bayesflow_model.data import load_split
from src.configs.config import N_TRAIN_SAMPLES
from src.inference.forward_backward import forward_backward
from src.preprocessing.encoding import one_hot_encode_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / f"training_{N_TRAIN_SAMPLES}"
)

MODEL_PATH = (
    TRAINING_OUTPUT_DIR
    / f"protein_bayesflow_{N_TRAIN_SAMPLES}.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / "evaluation___"
)

SYNTHETIC_METRICS_PATH = (
    OUTPUT_DIR
    / f"test_metrics_{N_TRAIN_SAMPLES}.csv"
)

SYNTHETIC_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / f"test_predictions_{N_TRAIN_SAMPLES}.npz"
)

REAL_METRICS_PATH = (
    OUTPUT_DIR
    / f"real_protein_metrics_{N_TRAIN_SAMPLES}.csv"
)

REAL_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / f"real_protein_predictions_{N_TRAIN_SAMPLES}.npz"
)

FINAL_RESULTS_PATH = (
    OUTPUT_DIR
    / f"evaluation_results_{N_TRAIN_SAMPLES}.json"
)

DEFAULT_REAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_proteins.csv"
)

# The trained inference subnet was built with sequences padded to 250.
# Its final Dense layer therefore expects a fixed flattened input size.
MODEL_SEQUENCE_LENGTH = 250


def predict_batch(
    model: Any,
    protein_sequences: np.ndarray,
    sequence_mask: np.ndarray,
) -> np.ndarray:
    """
    Predict posterior probabilities without using true targets.

    Parameters
    ----------
    model:
        Loaded native BayesFlow model.
    protein_sequences:
        One-hot encoded sequences with shape
        (batch_size, sequence_length, 20).
    sequence_mask:
        Boolean mask with shape
        (batch_size, sequence_length).

    Returns
    -------
    np.ndarray
        Posterior probabilities with shape
        (batch_size, sequence_length, 2).
    """

    sequence_features = model.summary_network(
        protein_sequences,
        mask=sequence_mask,
        training=False,
    )

    batch_size = protein_sequences.shape[0]
    sequence_length = protein_sequences.shape[1]

    dummy_inference_variables = np.zeros(
        (
            batch_size,
            sequence_length,
            2,
        ),
        dtype=np.float32,
    )

    estimates = model.inference_network(
        xz=dummy_inference_variables,
        conditions=sequence_features,
        mask=sequence_mask,
        training=False,
    )

    predictions = estimates["posterior_mean"]["value"]

    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()

    return np.asarray(
        predictions,
        dtype=np.float32,
    )


def validate_probabilities(
    probabilities: np.ndarray,
    mask: np.ndarray,
    name: str,
) -> None:
    """Validate probabilities only at non-padding positions."""

    valid_probabilities = probabilities[mask.astype(bool)]

    if valid_probabilities.size == 0:
        raise ValueError(f"{name} contains no valid positions.")

    if not np.isfinite(valid_probabilities).all():
        raise ValueError(f"{name} contains NaN or Inf.")

    if np.any(valid_probabilities < 0.0):
        raise ValueError(f"{name} contains probabilities below zero.")

    if np.any(valid_probabilities > 1.0):
        raise ValueError(f"{name} contains probabilities above one.")

    if not np.allclose(
        valid_probabilities.sum(axis=-1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Rows of {name} do not sum to one."
        )


def probabilities_to_states(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Convert [P(other), P(alpha)] into states 0 and 1.
    """

    return np.argmax(
        probabilities,
        axis=-1,
    ).astype(np.int32)


def annotation_to_binary(
    annotation: str,
) -> np.ndarray:
    """
    Map sst3 annotation to the two project states.

    H -> 1 = alpha
    C -> 0 = other
    E -> 0 = other
    """

    annotation = str(annotation).strip().upper()

    mapping = {
        "H": 1,
        "C": 0,
        "E": 0,
    }

    unknown_symbols = sorted(
        set(annotation) - set(mapping)
    )

    if unknown_symbols:
        raise ValueError(
            "Unsupported sst3 symbols: "
            f"{unknown_symbols}"
        )

    return np.asarray(
        [mapping[symbol] for symbol in annotation],
        dtype=np.int32,
    )


def compute_synthetic_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    hidden_states: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """
    Compute synthetic-test metrics on non-padding positions.

    The Forward-Backward posterior is the soft target.
    Hidden states are available because the data are simulated.
    """

    valid_mask = mask.astype(bool)

    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    valid_hidden_states = hidden_states[valid_mask]

    errors = valid_predictions - valid_targets

    mae = float(
        np.mean(np.abs(errors))
    )

    mse = float(
        np.mean(errors ** 2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    alpha_probability_mae = float(
        np.mean(
            np.abs(
                valid_predictions[:, 1]
                - valid_targets[:, 1]
            )
        )
    )

    epsilon = 1e-7

    clipped_predictions = np.clip(
        valid_predictions,
        epsilon,
        1.0 - epsilon,
    )

    soft_cross_entropy = float(
        -np.mean(
            np.sum(
                valid_targets
                * np.log(clipped_predictions),
                axis=-1,
            )
        )
    )

    predicted_states = probabilities_to_states(
        valid_predictions
    )

    forward_backward_states = probabilities_to_states(
        valid_targets
    )

    neural_state_accuracy = float(
        np.mean(
            predicted_states
            == valid_hidden_states
        )
    )

    forward_backward_state_accuracy = float(
        np.mean(
            forward_backward_states
            == valid_hidden_states
        )
    )

    neural_forward_backward_agreement = float(
        np.mean(
            predicted_states
            == forward_backward_states
        )
    )

    return {
        "number_of_sequences": int(
            predictions.shape[0]
        ),
        "number_of_valid_positions": int(
            valid_mask.sum()
        ),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "soft_cross_entropy": soft_cross_entropy,
        "alpha_probability_mae": (
            alpha_probability_mae
        ),
        "neural_state_accuracy": (
            neural_state_accuracy
        ),
        "forward_backward_state_accuracy": (
            forward_backward_state_accuracy
        ),
        "neural_forward_backward_agreement": (
            neural_forward_backward_agreement
        ),
    }


def save_metrics_csv(
    metrics: dict[str, float | int],
    output_path: Path,
) -> None:
    """Save one metric dictionary as a two-column CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            ["metric", "value"]
        )

        for metric_name, value in metrics.items():
            writer.writerow(
                [metric_name, value]
            )


def evaluate_synthetic_data(
    model: Any,
    batch_size: int = 32,
) -> tuple[
    dict[str, float | int],
    dict[str, np.ndarray],
]:
    """
    Run inference on the complete synthetic test split.
    """

    print("Loading synthetic test data...")

    test_data = load_split(
        split_name="test",
    )

    x_test = test_data["x"]
    y_test = test_data["y"]
    mask_test = test_data["mask"]
    hidden_states_test = test_data[
        "hidden_states"
    ]
    lengths_test = test_data["lengths"]

    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)
    print("mask_test:", mask_test.shape)

    all_predictions: list[np.ndarray] = []

    number_of_samples = x_test.shape[0]

    for start_index in range(
        0,
        number_of_samples,
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            number_of_samples,
        )

        batch_predictions = predict_batch(
            model=model,
            protein_sequences=x_test[
                start_index:end_index
            ],
            sequence_mask=mask_test[
                start_index:end_index
            ],
        )

        all_predictions.append(
            batch_predictions
        )

        print(
            f"Predicted synthetic sequences: "
            f"{end_index}/{number_of_samples}"
        )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    if predictions.shape != y_test.shape:
        raise ValueError(
            "Synthetic prediction shape mismatch: "
            f"{predictions.shape} vs {y_test.shape}"
        )

    validate_probabilities(
        probabilities=predictions,
        mask=mask_test,
        name="synthetic predictions",
    )

    metrics = compute_synthetic_metrics(
        predictions=predictions,
        targets=y_test,
        hidden_states=hidden_states_test,
        mask=mask_test,
    )

    outputs = {
        "predictions": predictions,
        "targets": y_test,
        "mask": mask_test,
        "lengths": lengths_test,
        "hidden_states": hidden_states_test,
    }

    return metrics, outputs


def load_real_dataset(
    csv_path: str | Path,
    sequence_column: str,
    annotation_column: str,
) -> pd.DataFrame:
    """Load and validate the prepared real-protein CSV."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Real-protein CSV not found:\n"
            f"{csv_path.resolve()}\n\n"
            "The CSV must contain the protein sequence "
            "and sst3 annotation columns."
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        sequence_column,
        annotation_column,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing real-data columns: "
            f"{sorted(missing_columns)}\n"
            "Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    return dataframe


def prepare_real_batch(
    sequences: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    One-hot encode real proteins and pad them to the exact length
    expected by the trained model.

    The current inference subnet contains a Dense layer after flattening.
    Therefore, inference inputs must have the same sequence length used
    during training: 250 positions.

    Sequences longer than 250 must already have been truncated before
    this function is called.
    """

    if not sequences:
        raise ValueError(
            "Cannot prepare an empty real-protein batch."
        )

    lengths = np.asarray(
        [len(sequence) for sequence in sequences],
        dtype=np.int32,
    )

    if np.any(lengths > MODEL_SEQUENCE_LENGTH):
        raise ValueError(
            "A real sequence is longer than "
            f"{MODEL_SEQUENCE_LENGTH}. "
            "Truncate it before prepare_real_batch()."
        )

    encoded_batch = np.zeros(
        (
            len(sequences),
            MODEL_SEQUENCE_LENGTH,
            20,
        ),
        dtype=np.float32,
    )

    mask = np.zeros(
        (
            len(sequences),
            MODEL_SEQUENCE_LENGTH,
        ),
        dtype=bool,
    )

    for index, sequence in enumerate(sequences):
        encoded_sequence = one_hot_encode_sequence(
            sequence
        )

        sequence_length = encoded_sequence.shape[0]

        encoded_batch[
            index,
            :sequence_length,
            :,
        ] = encoded_sequence

        mask[
            index,
            :sequence_length,
        ] = True

    return encoded_batch, mask, lengths

def compute_real_metrics(
    true_states: np.ndarray,
    hmm_states: np.ndarray,
    neural_states: np.ndarray,
    hmm_probabilities: np.ndarray,
    neural_probabilities: np.ndarray,
    evaluated_proteins: int,
    skipped_proteins: int,
) -> dict[str, float | int]:
    """
    Compute micro-averaged real-protein metrics.
    """

    total_positions = int(
        true_states.size
    )

    if total_positions == 0:
        raise RuntimeError(
            "No valid real-protein positions were evaluated."
        )

    alpha_count = int(
        np.sum(true_states == 1)
    )

    other_count = int(
        np.sum(true_states == 0)
    )

    hmm_accuracy = float(
        np.mean(hmm_states == true_states)
    )

    neural_accuracy = float(
        np.mean(neural_states == true_states)
    )

    hmm_neural_agreement = float(
        np.mean(hmm_states == neural_states)
    )

    posterior_mae = float(
        np.mean(
            np.abs(
                neural_probabilities
                - hmm_probabilities
            )
        )
    )

    posterior_mse = float(
        np.mean(
            (
                neural_probabilities
                - hmm_probabilities
            ) ** 2
        )
    )

    alpha_probability_mae = float(
        np.mean(
            np.abs(
                neural_probabilities[:, 1]
                - hmm_probabilities[:, 1]
            )
        )
    )

    alpha_fraction = float(
        alpha_count / total_positions
    )

    other_fraction = float(
        other_count / total_positions
    )

    return {
        "evaluated_proteins": int(
            evaluated_proteins
        ),
        "skipped_proteins": int(
            skipped_proteins
        ),
        "evaluated_positions": (
            total_positions
        ),
        "hmm_state_accuracy": (
            hmm_accuracy
        ),
        "neural_state_accuracy": (
            neural_accuracy
        ),
        "hmm_neural_state_agreement": (
            hmm_neural_agreement
        ),
        "neural_hmm_posterior_mae": (
            posterior_mae
        ),
        "neural_hmm_posterior_mse": (
            posterior_mse
        ),
        "alpha_probability_mae": (
            alpha_probability_mae
        ),
        "alpha_count": alpha_count,
        "other_count": other_count,
        "alpha_fraction": alpha_fraction,
        "other_fraction": other_fraction,
        "always_other_baseline": (
            other_fraction
        ),
        "hmm_improvement_over_baseline": float(
            hmm_accuracy - other_fraction
        ),
        "neural_improvement_over_baseline": float(
            neural_accuracy - other_fraction
        ),
    }


def evaluate_real_proteins(
    model: Any,
    dataframe: pd.DataFrame,
    sequence_column: str,
    annotation_column: str,
    max_proteins: int,
    batch_size: int,
) -> tuple[
    dict[str, float | int],
    dict[str, np.ndarray],
    dict[str, str | np.ndarray] | None,
]:
    """
    Evaluate Forward-Backward and native BayesFlow
    against real sst3 annotations.

    Accuracy is micro-averaged over all evaluated positions.
    """

    if max_proteins <= 0:
        raise ValueError(
            "max_proteins must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    sample_size = min(
        max_proteins,
        len(dataframe),
    )

    selected = dataframe.sample(
        n=sample_size,
        random_state=123,
    )

    valid_sequences: list[str] = []
    valid_annotations: list[np.ndarray] = []
    skipped_proteins = 0

    for _, row in selected.iterrows():
        sequence = str(
            row[sequence_column]
        ).strip().upper()

        annotation = str(
            row[annotation_column]
        ).strip().upper()

        if (
            not sequence
            or not annotation
            or sequence == "NAN"
            or annotation == "NAN"
            or len(sequence) != len(annotation)
        ):
            skipped_proteins += 1
            continue

        # The trained model expects exactly 250 padded positions.
        # Keep sequence and annotation aligned when truncating.
        sequence = sequence[:MODEL_SEQUENCE_LENGTH]
        annotation = annotation[:MODEL_SEQUENCE_LENGTH]

        try:
            true_states = annotation_to_binary(
                annotation
            )

            # This also checks that all amino acids are supported.
            one_hot_encode_sequence(sequence)

        except (TypeError, ValueError):
            skipped_proteins += 1
            continue

        valid_sequences.append(sequence)
        valid_annotations.append(true_states)

    if not valid_sequences:
        raise RuntimeError(
            "No valid real proteins were found. "
            "Check seq/sst3 columns and amino-acid symbols."
        )

    all_true_states: list[np.ndarray] = []
    all_hmm_states: list[np.ndarray] = []
    all_neural_states: list[np.ndarray] = []
    all_hmm_probabilities: list[np.ndarray] = []
    all_neural_probabilities: list[np.ndarray] = []
    all_sequence_ids: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []

    example: (
        dict[str, str | np.ndarray]
        | None
    ) = None

    evaluated_proteins = 0

    for start_index in range(
        0,
        len(valid_sequences),
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            len(valid_sequences),
        )

        batch_sequences = valid_sequences[
            start_index:end_index
        ]

        batch_annotations = valid_annotations[
            start_index:end_index
        ]

        encoded_batch, batch_mask, lengths = (
            prepare_real_batch(
                batch_sequences
            )
        )

        batch_neural_probabilities = predict_batch(
            model=model,
            protein_sequences=encoded_batch,
            sequence_mask=batch_mask,
        )

        validate_probabilities(
            probabilities=(
                batch_neural_probabilities
            ),
            mask=batch_mask,
            name="real-protein neural predictions",
        )

        for local_index, sequence in enumerate(
            batch_sequences
        ):
            sequence_length = int(
                lengths[local_index]
            )

            true_states = batch_annotations[
                local_index
            ]

            neural_probabilities = (
                batch_neural_probabilities[
                    local_index,
                    :sequence_length,
                ]
            )

            hmm_probabilities = forward_backward(
                sequence
            ).astype(np.float32)

            if (
                neural_probabilities.shape
                != hmm_probabilities.shape
            ):
                raise ValueError(
                    "HMM/neural posterior shape mismatch "
                    f"for protein {start_index + local_index}: "
                    f"{hmm_probabilities.shape} vs "
                    f"{neural_probabilities.shape}"
                )

            if (
                hmm_probabilities.shape[0]
                != true_states.shape[0]
            ):
                raise ValueError(
                    "Prediction/annotation length mismatch "
                    f"for protein {start_index + local_index}."
                )

            hmm_states = probabilities_to_states(
                hmm_probabilities
            )

            neural_states = probabilities_to_states(
                neural_probabilities
            )

            protein_index = (
                start_index
                + local_index
            )

            positions = np.arange(
                1,
                sequence_length + 1,
                dtype=np.int32,
            )

            sequence_ids = np.full(
                sequence_length,
                protein_index,
                dtype=np.int32,
            )

            all_true_states.append(
                true_states
            )

            all_hmm_states.append(
                hmm_states
            )

            all_neural_states.append(
                neural_states
            )

            all_hmm_probabilities.append(
                hmm_probabilities
            )

            all_neural_probabilities.append(
                neural_probabilities
            )

            all_sequence_ids.append(
                sequence_ids
            )

            all_positions.append(
                positions
            )

            evaluated_proteins += 1

            if example is None:
                example = {
                    "sequence": sequence,
                    "ground_truth": true_states,
                    "hmm_states": hmm_states,
                    "neural_states": neural_states,
                    "hmm_alpha_probability": (
                        hmm_probabilities[:, 1]
                    ),
                    "neural_alpha_probability": (
                        neural_probabilities[:, 1]
                    ),
                }

        print(
            "Predicted real proteins: "
            f"{end_index}/{len(valid_sequences)}"
        )

    true_states_array = np.concatenate(
        all_true_states
    )

    hmm_states_array = np.concatenate(
        all_hmm_states
    )

    neural_states_array = np.concatenate(
        all_neural_states
    )

    hmm_probabilities_array = np.concatenate(
        all_hmm_probabilities,
        axis=0,
    )

    neural_probabilities_array = np.concatenate(
        all_neural_probabilities,
        axis=0,
    )

    sequence_ids_array = np.concatenate(
        all_sequence_ids
    )

    positions_array = np.concatenate(
        all_positions
    )

    metrics = compute_real_metrics(
        true_states=true_states_array,
        hmm_states=hmm_states_array,
        neural_states=neural_states_array,
        hmm_probabilities=hmm_probabilities_array,
        neural_probabilities=(
            neural_probabilities_array
        ),
        evaluated_proteins=evaluated_proteins,
        skipped_proteins=skipped_proteins,
    )

    outputs = {
        "sequence_id": sequence_ids_array,
        "position": positions_array,
        "ground_truth": true_states_array,
        "hmm_states": hmm_states_array,
        "neural_states": neural_states_array,
        "hmm_probabilities": (
            hmm_probabilities_array
        ),
        "neural_probabilities": (
            neural_probabilities_array
        ),
    }

    return metrics, outputs, example


def print_metrics(
    title: str,
    metrics: dict[str, float | int],
) -> None:
    """Print metrics with readable formatting."""

    print()
    print(title)
    print("=" * len(title))

    for metric_name, value in metrics.items():
        if isinstance(value, int):
            print(
                f"{metric_name}: {value}"
            )
        else:
            print(
                f"{metric_name}: {value:.6f}"
            )


def print_real_example(
    example: (
        dict[str, str | np.ndarray]
        | None
    ),
    number_of_positions: int = 80,
) -> None:
    """
    Print one compact qualitative real-protein comparison.
    """

    if example is None:
        print(
            "\nNo valid example protein was available."
        )
        return

    sequence = str(
        example["sequence"]
    )

    ground_truth = np.asarray(
        example["ground_truth"]
    )

    hmm_states = np.asarray(
        example["hmm_states"]
    )

    neural_states = np.asarray(
        example["neural_states"]
    )

    hmm_alpha_probability = np.asarray(
        example["hmm_alpha_probability"]
    )

    neural_alpha_probability = np.asarray(
        example["neural_alpha_probability"]
    )

    number_to_show = min(
        number_of_positions,
        len(sequence),
    )

    print()
    print(
        "Example real protein "
        "(0=other, 1=alpha)"
    )
    print("=" * 45)

    print(
        "Amino acids: ",
        sequence[:number_to_show],
    )

    print(
        "Ground truth:",
        "".join(
            map(
                str,
                ground_truth[:number_to_show],
            )
        ),
    )

    print(
        "Forward-Backward:",
        "".join(
            map(
                str,
                hmm_states[:number_to_show],
            )
        ),
    )

    print(
        "Native BayesFlow:",
        "".join(
            map(
                str,
                neural_states[:number_to_show],
            )
        ),
    )

    print()
    print(
        "First alpha probabilities:"
    )

    for position in range(
        min(10, number_to_show)
    ):
        print(
            f"Position {position + 1:3d}: "
            f"HMM={hmm_alpha_probability[position]:.4f}, "
            f"BayesFlow={neural_alpha_probability[position]:.4f}, "
            f"truth={ground_truth[position]}"
        )


def save_final_json(
    synthetic_metrics: dict[str, float | int],
    real_metrics: dict[str, float | int] | None,
) -> None:
    """Save synthetic and real metrics in one JSON file."""

    payload = {
        "training_samples": N_TRAIN_SAMPLES,
        "synthetic_test": synthetic_metrics,
        "real_protein_test": real_metrics,
    }

    FINAL_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_RESULTS_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the native BayesFlow model "
            "on synthetic and real proteins."
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--real-data-path",
        type=Path,
        default=DEFAULT_REAL_DATA_PATH,
    )

    parser.add_argument(
        "--sequence-column",
        type=str,
        default="seq",
    )

    parser.add_argument(
        "--annotation-column",
        type=str,
        default="sst3",
    )

    parser.add_argument(
        "--max-real-proteins",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--skip-real",
        action="store_true",
        help=(
            "Run only the synthetic evaluation___."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete evaluation___ pipeline."""

    args = parse_arguments()

    if not args.model_path.exists():
        raise FileNotFoundError(
            "Model not found:\n"
            f"{args.model_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading native BayesFlow model...")

    model = keras.models.load_model(
        args.model_path,
        compile=False,
    )

    # -----------------------------------------------------
    # 1. Synthetic test evaluation___
    # -----------------------------------------------------

    synthetic_metrics, synthetic_outputs = (
        evaluate_synthetic_data(
            model=model,
            batch_size=args.batch_size,
        )
    )

    print_metrics(
        title="Synthetic test metrics",
        metrics=synthetic_metrics,
    )

    save_metrics_csv(
        metrics=synthetic_metrics,
        output_path=SYNTHETIC_METRICS_PATH,
    )

    np.savez_compressed(
        SYNTHETIC_PREDICTIONS_PATH,
        **synthetic_outputs,
    )

    real_metrics: (
        dict[str, float | int]
        | None
    ) = None

    # -----------------------------------------------------
    # 2. Real protein evaluation___
    # -----------------------------------------------------

    if not args.skip_real:
        print()
        print("Loading real protein data...")

        real_dataframe = load_real_dataset(
            csv_path=args.real_data_path,
            sequence_column=(
                args.sequence_column
            ),
            annotation_column=(
                args.annotation_column
            ),
        )

        (
            real_metrics,
            real_outputs,
            example,
        ) = evaluate_real_proteins(
            model=model,
            dataframe=real_dataframe,
            sequence_column=(
                args.sequence_column
            ),
            annotation_column=(
                args.annotation_column
            ),
            max_proteins=(
                args.max_real_proteins
            ),
            batch_size=args.batch_size,
        )

        print_metrics(
            title="Real protein metrics",
            metrics=real_metrics,
        )

        print_real_example(example)

        save_metrics_csv(
            metrics=real_metrics,
            output_path=REAL_METRICS_PATH,
        )

        np.savez_compressed(
            REAL_PREDICTIONS_PATH,
            **real_outputs,
        )

    save_final_json(
        synthetic_metrics=synthetic_metrics,
        real_metrics=real_metrics,
    )

    print()
    print("Saved outputs")
    print("=" * 45)
    print(SYNTHETIC_METRICS_PATH)
    print(SYNTHETIC_PREDICTIONS_PATH)

    if real_metrics is not None:
        print(REAL_METRICS_PATH)
        print(REAL_PREDICTIONS_PATH)

    print(FINAL_RESULTS_PATH)

    print()
    print("✓ Synthetic test inference completed")
    print("✓ Padding excluded from synthetic metrics")

    if real_metrics is not None:
        print("✓ Real-protein inference completed")
        print("✓ Forward-Backward compared with annotations")
        print("✓ Native BayesFlow compared with annotations")
        print("✓ Real posterior probabilities saved")


if __name__ == "__main__":
    main()
