"""
Check native BayesFlow inference on one real protein sequence.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

# Import custom classes before loading the model.
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
from src.inference.forward_backward import (
    forward_backward,
)
from src.preprocessing.encoding import (
    check_sequence,
    encode_for_neural_network,
)


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
    / "real_protein"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "real_protein_predictions.csv"
)

MAX_LENGTH = 250


# Replace this with the real protein sequence.
REAL_SEQUENCE = (
    "GIVEQCCTSICSLYQLENYCN"
)


def clean_input_sequence(
    sequence: str,
) -> str:
    """
    Remove whitespace and convert the sequence to uppercase.
    """

    cleaned = "".join(
        sequence.split()
    ).upper()

    if not cleaned:
        raise ValueError(
            "The protein sequence is empty."
        )

    check_sequence(cleaned)

    if len(cleaned) > MAX_LENGTH:
        raise ValueError(
            f"Sequence length is {len(cleaned)}, "
            f"but the model supports at most "
            f"{MAX_LENGTH} positions."
        )

    return cleaned


def prepare_real_protein(
    sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode and pad one real protein sequence.

    Returns
    -------
    x:
        Shape (1, 250, 20).

    mask:
        Shape (1, 250).
    """

    sequence_length = len(sequence)

    encoded = encode_for_neural_network(
        sequence=sequence,
        max_length=MAX_LENGTH,
        one_hot=True,
    ).astype(np.float32)

    mask = np.zeros(
        MAX_LENGTH,
        dtype=np.float32,
    )

    mask[:sequence_length] = 1.0

    return (
        encoded[None, ...],
        mask[None, ...],
    )


def predict_posterior(
    model,
    protein_sequence: np.ndarray,
    sequence_mask: np.ndarray,
) -> np.ndarray:
    """
    Predict position-wise state probabilities.
    """

    summary_features = model.summary_network(
        protein_sequence,
        mask=sequence_mask,
        training=False,
    )

    batch_size = protein_sequence.shape[0]
    sequence_length = protein_sequence.shape[1]

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
        conditions=summary_features,
        mask=sequence_mask,
        training=False,
    )

    predictions = estimates[
        "posterior_mean"
    ]["value"]

    return np.asarray(
        predictions,
        dtype=np.float32,
    )


def save_results(
    sequence: str,
    neural_posterior: np.ndarray,
    fb_posterior: np.ndarray,
) -> None:
    """
    Save position-wise probabilities as CSV.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "position",
                "amino_acid",
                "neural_other_probability",
                "neural_alpha_probability",
                "forward_backward_other_probability",
                "forward_backward_alpha_probability",
                "neural_predicted_state",
                "forward_backward_predicted_state",
            ]
        )

        for position, amino_acid in enumerate(
            sequence,
            start=1,
        ):
            index = position - 1

            neural_state = (
                "alpha"
                if neural_posterior[index, 1]
                >= neural_posterior[index, 0]
                else "other"
            )

            fb_state = (
                "alpha"
                if fb_posterior[index, 1]
                >= fb_posterior[index, 0]
                else "other"
            )

            writer.writerow(
                [
                    position,
                    amino_acid,
                    float(
                        neural_posterior[
                            index,
                            0,
                        ]
                    ),
                    float(
                        neural_posterior[
                            index,
                            1,
                        ]
                    ),
                    float(
                        fb_posterior[
                            index,
                            0,
                        ]
                    ),
                    float(
                        fb_posterior[
                            index,
                            1,
                        ]
                    ),
                    neural_state,
                    fb_state,
                ]
            )


def main() -> None:
    # ---------------------------------------------------------
    # 1. Validate real sequence
    # ---------------------------------------------------------

    sequence = clean_input_sequence(
        REAL_SEQUENCE
    )

    sequence_length = len(sequence)

    print("Real protein length:")
    print(sequence_length)

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}"
        )

    print("\nLoading model:")
    print(MODEL_PATH)

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    # ---------------------------------------------------------
    # 3. Encode sequence
    # ---------------------------------------------------------

    x_real, mask_real = prepare_real_protein(
        sequence
    )

    print("\nPrepared shapes:")
    print("x_real:", x_real.shape)
    print("mask_real:", mask_real.shape)

    assert x_real.shape == (
        1,
        MAX_LENGTH,
        20,
    )

    assert mask_real.shape == (
        1,
        MAX_LENGTH,
    )

    assert int(mask_real.sum()) == sequence_length

    # Confirm valid one-hot positions.
    assert np.allclose(
        x_real[
            0,
            :sequence_length,
        ].sum(axis=-1),
        1.0,
    )

    # Confirm padded positions are zero.
    assert np.allclose(
        x_real[
            0,
            sequence_length:,
        ],
        0.0,
    )

    # ---------------------------------------------------------
    # 4. Neural inference
    # ---------------------------------------------------------

    full_neural_posterior = predict_posterior(
        model=model,
        protein_sequence=x_real,
        sequence_mask=mask_real,
    )

    neural_posterior = full_neural_posterior[
        0,
        :sequence_length,
    ]

    # ---------------------------------------------------------
    # 5. Forward-Backward reference
    # ---------------------------------------------------------

    fb_posterior = forward_backward(
        sequence
    ).astype(np.float32)

    assert neural_posterior.shape == (
        sequence_length,
        2,
    )

    assert fb_posterior.shape == (
        sequence_length,
        2,
    )

    # ---------------------------------------------------------
    # 6. Validate probabilities
    # ---------------------------------------------------------

    assert np.isfinite(
        neural_posterior
    ).all()

    assert np.all(
        neural_posterior >= 0.0
    )

    assert np.all(
        neural_posterior <= 1.0
    )

    assert np.allclose(
        neural_posterior.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # ---------------------------------------------------------
    # 7. Compare with Forward-Backward
    # ---------------------------------------------------------

    mae = float(
        np.mean(
            np.abs(
                neural_posterior
                - fb_posterior
            )
        )
    )

    alpha_mae = float(
        np.mean(
            np.abs(
                neural_posterior[:, 1]
                - fb_posterior[:, 1]
            )
        )
    )

    neural_states = np.argmax(
        neural_posterior,
        axis=-1,
    )

    fb_states = np.argmax(
        fb_posterior,
        axis=-1,
    )

    state_agreement = float(
        np.mean(
            neural_states == fb_states
        )
    )

    # ---------------------------------------------------------
    # 8. Save output
    # ---------------------------------------------------------

    save_results(
        sequence=sequence,
        neural_posterior=neural_posterior,
        fb_posterior=fb_posterior,
    )

    # ---------------------------------------------------------
    # 9. Display results
    # ---------------------------------------------------------

    print("\nReal-protein comparison")
    print("=" * 45)
    print(f"MAE: {mae:.6f}")
    print(
        f"Alpha probability MAE: "
        f"{alpha_mae:.6f}"
    )
    print(
        f"State agreement: "
        f"{state_agreement:.6f}"
    )

    print("\nFirst five neural probabilities:")
    print(neural_posterior[:5])

    print(
        "\nFirst five Forward-Backward "
        "probabilities:"
    )
    print(fb_posterior[:5])

    print("\nSaved results:")
    print(OUTPUT_PATH)

    print()
    print("✓ Real sequence validated")
    print("✓ One-hot encoding completed")
    print("✓ Padding and mask validated")
    print("✓ Neural inference completed")
    print("✓ Forward-Backward comparison completed")
    print("✓ Position-wise results saved")


if __name__ == "__main__":
    main()