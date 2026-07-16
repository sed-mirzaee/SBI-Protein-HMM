"""Final lightweight evaluation pipeline.

This module is intentionally limited to the two agreed metrics:

1. Synthetic test data:
   MSE between Forward-Backward posterior probabilities and neural-network
   posterior probabilities.
2. Real protein data:
   Per-residue state accuracy for both Forward-Backward and the neural network
   against sst3 ground-truth annotations.

Run from the project root after a checkpoint is available:

    python -m src.evaluate_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    posterior_mse,
    probabilities_to_states,
    state_accuracy,
)
from src.inference.forward_backward import forward_backward
from src.models.amortized_posterior import (
    DEFAULT_MODEL_PATH,
    load_trained_model,
    predict_from_sequence,
    predict_state_probabilities,
)


DEFAULT_REAL_DATA_PATH = Path("data/processed/real_proteins.csv")
DEFAULT_RESULTS_PATH = Path("outputs/evaluation/final_results.json")


def annotation_to_binary(annotation: str) -> np.ndarray:
    """Map sst3 annotation to project states: H=1 and C/E=0."""
    annotation = str(annotation).strip().upper()
    mapping = {"H": 1, "C": 0, "E": 0}

    unknown = sorted(set(annotation) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported sst3 symbols: {unknown}")

    return np.asarray([mapping[symbol] for symbol in annotation], dtype=np.int64)




def load_synthetic_test(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fixed synthetic test data."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic test file not found: {path.resolve()}"
        )

    data = np.load(path, allow_pickle=False)

    required_keys = {"x", "y", "mask"}
    missing = required_keys - set(data.files)

    if missing:
        raise ValueError(
            f"Missing keys in synthetic test file: {sorted(missing)}"
        )

    x_test = data["x"].astype(np.float32)
    y_test = data["y"].astype(np.float32)
    mask = data["mask"].astype(bool)

    if x_test.shape[:2] != mask.shape:
        raise ValueError("x and mask shapes are incompatible.")

    if y_test.shape[:2] != mask.shape:
        raise ValueError("y and mask shapes are incompatible.")

    return x_test, y_test, mask

def evaluate_synthetic_test(
    model: Any,
    test_path: str | Path,
) -> dict[str, float | int]:
    """Evaluate NN against Forward–Backward posterior using masked MSE."""

    x_test, y_test, mask = load_synthetic_test(test_path)

    squared_error_sum = 0.0
    valid_probability_count = 0

    for i in range(len(x_test)):
        valid_positions = mask[i]

        if not np.any(valid_positions):
            continue

        encoded_sequence = x_test[i][valid_positions]
        true_posterior = y_test[i][valid_positions] 

        predicted_posterior = predict_state_probabilities(
            encoded_sequence=encoded_sequence,
            model=model,
        )

        if predicted_posterior.shape != true_posterior.shape:
            raise ValueError(
                f"Prediction shape mismatch for sample {i}: "
                f"{predicted_posterior.shape} vs {true_posterior.shape}"
            )

        difference = true_posterior - predicted_posterior

        squared_error_sum += float(np.sum(difference ** 2))
        valid_probability_count += int(difference.size)
        if valid_probability_count == 0:
            raise RuntimeError(
                "Synthetic test set contains no valid, unmasked positions."
        )

    final_mse = squared_error_sum / valid_probability_count

    return {
        "num_sequences": int(len(x_test)),
        "posterior_mse": float(final_mse),
    }


def load_real_dataset(
    csv_path: str | Path = DEFAULT_REAL_DATA_PATH,
    sequence_column: str = "seq",
    annotation_column: str = "sst3",
) -> pd.DataFrame:
    """Load and validate the prepared real-protein CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Real-protein CSV not found at {csv_path.resolve()}. "
            "Save df_clean[['seq', 'sst3']] from Data.ipynb first."
        )

    dataframe = pd.read_csv(csv_path)
    required = {sequence_column, annotation_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing)}. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    return dataframe

def evaluate_hmm_on_real_proteins(
    dataframe: pd.DataFrame,
    sequence_column: str = "seq",
    annotation_column: str = "sst3",
    max_proteins: int = 500,
) -> tuple[dict[str, float | int], dict[str, np.ndarray | str] | None]:
    """Evaluate Forward-Backward states against real sst3 annotations."""

    if max_proteins <= 0:
        raise ValueError("max_proteins must be positive.")

    hmm_correct = 0
    total_positions = 0
    evaluated = 0
    skipped = 0
    example: dict[str, np.ndarray | str] | None = None

    sample_size = min(max_proteins, len(dataframe))
    selected = dataframe.sample(
        n=sample_size,
        random_state=123,
    )

    for _, row in selected.iterrows():
        sequence = str(row[sequence_column]).strip().upper()
        annotation = str(row[annotation_column]).strip().upper()

        if not sequence or len(sequence) != len(annotation):
            skipped += 1
            continue

        try:
            # Real experimental labels:
            # H = alpha = 1
            # C/E = other = 0
            true_states = annotation_to_binary(annotation)

            # HMM posterior probabilities for the real sequence
            hmm_posterior = forward_backward(
                list(sequence)
            ).astype(np.float32)

            if hmm_posterior.shape[0] != len(true_states):
                raise ValueError(
                    "HMM prediction length does not match annotation length."
                )

            # Convert [P(other), P(alpha)] into state 0 or 1
            hmm_states = probabilities_to_states(hmm_posterior)

        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        hmm_correct += int(
            np.sum(hmm_states == true_states)
        )
        total_positions += int(true_states.size)
        evaluated += 1

        if example is None:
            example = {
                "sequence": sequence,
                "ground_truth": true_states,
                "hmm_states": hmm_states,
            }

    if total_positions == 0:
        raise RuntimeError(
            "No valid real proteins were evaluated. "
            "Check seq/sst3 columns and amino-acid symbols."
        )

    metrics: dict[str, float | int] = {
        "requested_proteins": sample_size,
        "evaluated_proteins": evaluated,
        "skipped_proteins": skipped,
        "evaluated_positions": total_positions,
        "hmm_state_accuracy": float(
            hmm_correct / total_positions
        ),
    }

    return metrics, example
def print_hmm_example(
    example: dict[str, np.ndarray | str] | None,
    num_positions: int = 80,
) -> None:
    """Print ground truth and HMM prediction for one protein."""

    if example is None:
        print("No valid example protein was available.")
        return

    ground_truth = np.asarray(example["ground_truth"])
    hmm_states = np.asarray(example["hmm_states"])

    n = min(num_positions, len(ground_truth))

    print("\nExample real protein (0=other, 1=alpha)")
    print("-----------------------------------------")
    print(
        "Ground truth:",
        "".join(map(str, ground_truth[:n])),
    )
    print(
        "HMM:         ",
        "".join(map(str, hmm_states[:n])),
    )

def evaluate_real_proteins(
    dataframe: pd.DataFrame,
    model: Any,
    sequence_column: str = "seq",
    annotation_column: str = "sst3",
    max_proteins: int = 500,
) -> tuple[dict[str, float | int], dict[str, np.ndarray | str] | None]:
    """Evaluate HMM and NN hard-state predictions against real annotations.

    Accuracy is micro-averaged over all valid amino-acid positions.
    """
    if max_proteins <= 0:
        raise ValueError("max_proteins must be positive.")

    hmm_correct = 0
    nn_correct = 0
    total_positions = 0
    evaluated = 0
    skipped = 0
    example: dict[str, np.ndarray | str] | None = None

    # Fixed random sample avoids evaluating only the first rows of the file.
    sample_size = min(max_proteins, len(dataframe))
    selected = dataframe.sample(n=sample_size, random_state=123)

    for _, row in selected.iterrows():
        sequence = str(row[sequence_column]).strip().upper()
        annotation = str(row[annotation_column]).strip().upper()

        if not sequence or len(sequence) != len(annotation):
            skipped += 1
            continue

        try:
            true_states = annotation_to_binary(annotation)
            hmm_posterior = forward_backward(list(sequence)).astype(np.float32)
            nn_posterior = predict_from_sequence(sequence=sequence, model=model)

            if hmm_posterior.shape != nn_posterior.shape:
                raise ValueError(
                    f"HMM/NN shape mismatch: {hmm_posterior.shape} vs "
                    f"{nn_posterior.shape}"
                )
            if hmm_posterior.shape[0] != len(true_states):
                raise ValueError("Prediction length does not match annotation length.")

            hmm_states = probabilities_to_states(hmm_posterior)
            nn_states = probabilities_to_states(nn_posterior)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        hmm_correct += int(np.sum(hmm_states == true_states))
        nn_correct += int(np.sum(nn_states == true_states))
        total_positions += int(true_states.size)
        evaluated += 1
        

        if example is None:
            example = {
                "sequence": sequence,
                "ground_truth": true_states,
                "hmm_states": hmm_states,
                "nn_states": nn_states,
            }

    if total_positions == 0:
        raise RuntimeError(
            "No valid real proteins were evaluated. Check sequence/annotation "
            "columns and supported amino-acid symbols."
        )

    metrics: dict[str, float | int] = {
        "requested_proteins": sample_size,
        "evaluated_proteins": evaluated,
        "skipped_proteins": skipped,
        "evaluated_positions": total_positions,
        "hmm_state_accuracy": float(hmm_correct / total_positions),
        "nn_state_accuracy": float(nn_correct / total_positions),
    }

    return metrics, example


def print_example(
    example: dict[str, np.ndarray | str] | None,
    num_positions: int = 80,
) -> None:
    """Print a compact qualitative comparison for one real protein."""
    if example is None:
        print("No valid example protein was available.")
        return

    ground_truth = np.asarray(example["ground_truth"])
    hmm_states = np.asarray(example["hmm_states"])
    nn_states = np.asarray(example["nn_states"])
    n = min(num_positions, len(ground_truth))

    print("\nExample real protein (0=other, 1=alpha)")
    print("-----------------------------------------")
    print("Ground truth:", "".join(map(str, ground_truth[:n])))
    print("HMM:         ", "".join(map(str, hmm_states[:n])))
    print("Neural net:  ", "".join(map(str, nn_states[:n])))


def save_results(
    synthetic_metrics: dict[str, float | int],
    real_metrics: dict[str, float | int],
    output_path: str | Path = DEFAULT_RESULTS_PATH,
) -> None:
    """Save metrics and a compact CSV table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "synthetic_test": synthetic_metrics,
        "real_protein_test": real_metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "comparison": "Synthetic: NN vs Forward-Backward",
                "metric": "MSE",
                "value": synthetic_metrics["posterior_mse"],
            },
            {
                "comparison": "Real: Forward-Backward vs Ground Truth",
                "metric": "State Accuracy",
                "value": real_metrics["hmm_state_accuracy"],
            },
            {
                "comparison": "Real: Neural Network vs Ground Truth",
                "metric": "State Accuracy",
                "value": real_metrics["nn_state_accuracy"],
            },
        ]
    ).to_csv(csv_path, index=False)

    print(f"\nSaved JSON results to: {output_path}")
    print(f"Saved CSV results to:  {csv_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final project evaluation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--real-data-path", type=Path, default=DEFAULT_REAL_DATA_PATH)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--max-real-proteins", type=int, default=500)
    parser.add_argument("--sequence-column", type=str, default="seq")
    parser.add_argument("--annotation-column", type=str, default="sst3")
    parser.add_argument("--synthetic-test-path", type=Path, default=Path("data/synthetic/test_2000.npz"),)
    parser.add_argument(
    "--hmm-only",
    action="store_true",
    help=(
        "Evaluate only Forward-Backward against real annotations "
        "without loading a neural-network checkpoint."
    ),
)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # HMM-only evaluation does not require Sara's trained model.
    if args.hmm_only:
        real_dataframe = load_real_dataset(
            csv_path=args.real_data_path,
            sequence_column=args.sequence_column,
            annotation_column=args.annotation_column,
        )

        hmm_metrics, example = evaluate_hmm_on_real_proteins(
            dataframe=real_dataframe,
            sequence_column=args.sequence_column,
            annotation_column=args.annotation_column,
            max_proteins=args.max_real_proteins,
        )

        print("\nHMM real-protein evaluation")
        print("---------------------------")
        print(
            f"Requested proteins: "
            f"{hmm_metrics['requested_proteins']}"
        )
        print(
            f"Evaluated proteins: "
            f"{hmm_metrics['evaluated_proteins']}"
        )
        print(
            f"Skipped proteins:   "
            f"{hmm_metrics['skipped_proteins']}"
        )
        print(
            f"Evaluated positions: "
            f"{hmm_metrics['evaluated_positions']}"
        )
        print(
            f"HMM state accuracy: "
            f"{hmm_metrics['hmm_state_accuracy']:.4f}"
        )

        print_hmm_example(example)

        output_path = Path(
            "outputs/evaluation/hmm_real_results.json"
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(hmm_metrics, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved HMM results to: {output_path}")
        return

    # Full evaluation starts here 
    print(f"Loading checkpoint: {args.model_path}")
    model = load_trained_model(
        model_path=args.model_path
    )

    # Keep the remainder of your existing main() unchanged.

    synthetic_metrics = evaluate_synthetic_test(
    model=model,
    test_path=args.synthetic_test_path,
)
    print("\nSynthetic test")
    print("--------------")
    print(f"Sequences: {synthetic_metrics['num_sequences']}")
    print(f"Posterior MSE: {synthetic_metrics['posterior_mse']:.8f}")

    real_dataframe = load_real_dataset(
        csv_path=args.real_data_path,
        sequence_column=args.sequence_column,
        annotation_column=args.annotation_column,
    )
    real_metrics, example = evaluate_real_proteins(
        dataframe=real_dataframe,
        model=model,
        sequence_column=args.sequence_column,
        annotation_column=args.annotation_column,
        max_proteins=args.max_real_proteins,
    )

    print("\nReal protein test")
    print("-----------------")
    print(f"Evaluated proteins: {real_metrics['evaluated_proteins']}")
    print(f"Skipped proteins:   {real_metrics['skipped_proteins']}")
    print(f"HMM state accuracy: {real_metrics['hmm_state_accuracy']:.4f}")
    print(f"NN state accuracy:  {real_metrics['nn_state_accuracy']:.4f}")

    print_example(example)
    save_results(synthetic_metrics, real_metrics, args.results_path)


if __name__ == "__main__":
    main()