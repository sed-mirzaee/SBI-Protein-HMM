"""
Evaluate the saved native BayesFlow model
on the complete synthetic test split.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

# Register custom classes before loading.
from src.bayesflow_model.approximator import (
    TorchScoringRuleApproximator,
)
from src.bayesflow_model.inference_network import (
    ProbabilityMeanScore,
    SequenceConditionSubnet,
)
from src.bayesflow_model.summary_network import (
    ProteinBiLSTMSequenceNetwork,
)
from src.bayesflow_model.data import load_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / "protein_bayesflow.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / "evaluation"
)

METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "test_predictions.npz"
)


def predict_batch(
    model,
    protein_sequences: np.ndarray,
    sequence_mask: np.ndarray,
) -> np.ndarray:
    """
    Predict posterior probabilities without using true targets.
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

    return np.asarray(
        predictions,
        dtype=np.float32,
    )


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    hidden_states: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Compute metrics only on non-padding positions.
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
        np.mean(errors**2)
    )

    rmse = float(
        np.sqrt(mse)
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

    predicted_states = np.argmax(
        valid_predictions,
        axis=-1,
    )

    forward_backward_states = np.argmax(
        valid_targets,
        axis=-1,
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

    neural_fb_agreement = float(
        np.mean(
            predicted_states
            == forward_backward_states
        )
    )

    alpha_probability_mae = float(
        np.mean(
            np.abs(
                valid_predictions[:, 1]
                - valid_targets[:, 1]
            )
        )
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "soft_cross_entropy": soft_cross_entropy,
        "alpha_probability_mae": alpha_probability_mae,
        "neural_state_accuracy": neural_state_accuracy,
        "forward_backward_state_accuracy": (
            forward_backward_state_accuracy
        ),
        "neural_forward_backward_agreement": (
            neural_fb_agreement
        ),
        "number_of_sequences": float(
            predictions.shape[0]
        ),
        "number_of_valid_positions": float(
            valid_mask.sum()
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """
    Save evaluation metrics as a two-column CSV.
    """

    with open(
        METRICS_PATH,
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


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load model and test data
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading model...")

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

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

    # ---------------------------------------------------------
    # 2. Batched inference
    # ---------------------------------------------------------

    batch_size = 32

    all_predictions = []

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
            f"Predicted "
            f"{end_index}/{number_of_samples}"
        )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    assert predictions.shape == y_test.shape

    # ---------------------------------------------------------
    # 3. Probability validation
    # ---------------------------------------------------------

    valid_predictions = predictions[
        mask_test
    ]

    assert np.isfinite(
        valid_predictions
    ).all()

    assert np.all(valid_predictions >= 0.0)
    assert np.all(valid_predictions <= 1.0)

    assert np.allclose(
        valid_predictions.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # ---------------------------------------------------------
    # 4. Metrics
    # ---------------------------------------------------------

    metrics = compute_metrics(
        predictions=predictions,
        targets=y_test,
        hidden_states=hidden_states_test,
        mask=mask_test,
    )

    print("\nSynthetic test metrics")
    print("=" * 45)

    for metric_name, value in metrics.items():
        if metric_name.startswith(
            "number_of"
        ):
            print(
                f"{metric_name}: {int(value)}"
            )
        else:
            print(
                f"{metric_name}: {value:.6f}"
            )

    # ---------------------------------------------------------
    # 5. Save outputs
    # ---------------------------------------------------------

    save_metrics(metrics)

    np.savez_compressed(
        PREDICTIONS_PATH,
        predictions=predictions,
        targets=y_test,
        mask=mask_test,
        lengths=lengths_test,
        hidden_states=hidden_states_test,
    )

    print("\nSaved metrics:")
    print(METRICS_PATH)

    print("\nSaved predictions:")
    print(PREDICTIONS_PATH)

    print()
    print("✓ Full test inference completed")
    print("✓ Padding excluded from metrics")
    print("✓ Forward-Backward comparison completed")
    print("✓ Hidden-state accuracy calculated")
    print("✓ Evaluation outputs saved")


if __name__ == "__main__":
    main()
    