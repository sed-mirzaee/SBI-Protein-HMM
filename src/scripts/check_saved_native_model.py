"""
Check loading and inference of the saved native BayesFlow model.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

# These imports register the custom serializable classes
# before keras loads the saved model.
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
from src.bayesflow_model.data import (
    prepare_bayesflow_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "native_bayesflow"
    / "protein_bayesflow.keras"
)


def extract_predictions(
    model,
    protein_sequences: np.ndarray,
    sequence_mask: np.ndarray,
    inference_template: np.ndarray,
) -> np.ndarray:
    """
    Run one forward pass through the loaded model.

    inference_template determines only the required output shape.
    It must not provide information used for prediction.
    """

    sequence_features = model.summary_network(
        protein_sequences,
        mask=sequence_mask,
        training=False,
    )

    estimates = model.inference_network(
        xz=inference_template,
        conditions=sequence_features,
        mask=sequence_mask,
        training=False,
    )

    predictions = estimates["posterior_mean"]["value"]

    return np.asarray(
        predictions,
        dtype=np.float32,
    )


def main() -> None:
    # ---------------------------------------------------------
    # 1. Check saved model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}"
        )

    print("Loading model:")
    print(MODEL_PATH)

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("\nLoaded model type:")
    print(type(model).__name__)

    print("\nSummary network type:")
    print(type(model.summary_network).__name__)

    print("\nInference network type:")
    print(type(model.inference_network).__name__)

    # ---------------------------------------------------------
    # 2. Load a small test batch
    # ---------------------------------------------------------

    raw_data = prepare_bayesflow_data(
        split_name="test",
        limit=8,
    )

    protein_sequences = raw_data["protein_sequence"]
    targets = raw_data["state_probabilities"]
    sequence_mask = raw_data["encoder_mask"]

    print("\nInput shapes:")
    print("protein_sequences:", protein_sequences.shape)
    print("targets:", targets.shape)
    print("sequence_mask:", sequence_mask.shape)

    # ---------------------------------------------------------
    # 3. Prediction using the real target as template
    # ---------------------------------------------------------

    predictions_with_targets = extract_predictions(
        model=model,
        protein_sequences=protein_sequences,
        sequence_mask=sequence_mask,
        inference_template=targets,
    )

    # ---------------------------------------------------------
    # 4. Prediction without target information
    # ---------------------------------------------------------

    dummy_template = np.zeros_like(
        targets,
        dtype=np.float32,
    )

    predictions_without_targets = extract_predictions(
        model=model,
        protein_sequences=protein_sequences,
        sequence_mask=sequence_mask,
        inference_template=dummy_template,
    )

    print("\nPrediction shape:")
    print(predictions_without_targets.shape)

    # ---------------------------------------------------------
    # 5. Validate prediction contract
    # ---------------------------------------------------------

    expected_shape = (
        protein_sequences.shape[0],
        protein_sequences.shape[1],
        2,
    )

    assert predictions_without_targets.shape == expected_shape

    assert np.isfinite(
        predictions_without_targets
    ).all()

    valid_mask = sequence_mask.astype(bool)

    valid_predictions = predictions_without_targets[
        valid_mask
    ]

    assert np.all(valid_predictions >= 0.0)
    assert np.all(valid_predictions <= 1.0)

    assert np.allclose(
        valid_predictions.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # ---------------------------------------------------------
    # 6. Critical target-leakage check
    # ---------------------------------------------------------

    maximum_difference = float(
        np.max(
            np.abs(
                predictions_with_targets
                - predictions_without_targets
            )
        )
    )

    print("\nMaximum prediction difference")
    print("real target vs dummy target:")
    print(maximum_difference)

    assert np.allclose(
        predictions_with_targets,
        predictions_without_targets,
        atol=1e-6,
    ), (
        "Predictions depend on inference_variables. "
        "This indicates target leakage."
    )

    # ---------------------------------------------------------
    # 7. Basic test metrics
    # ---------------------------------------------------------

    valid_targets = targets[valid_mask]

    mae = float(
        np.mean(
            np.abs(
                valid_predictions
                - valid_targets
            )
        )
    )

    mse = float(
        np.mean(
            (
                valid_predictions
                - valid_targets
            ) ** 2
        )
    )

    print("\nSmall-batch metrics:")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")

    print("\nFirst five valid predictions:")
    print(valid_predictions[:5])

    print("\nFirst five Forward-Backward targets:")
    print(valid_targets[:5])

    print()
    print("✓ Saved model loaded successfully")
    print("✓ Custom BayesFlow classes restored")
    print("✓ Inference works without true targets")
    print("✓ No target leakage detected")
    print("✓ Output probabilities are valid")


if __name__ == "__main__":
    main()