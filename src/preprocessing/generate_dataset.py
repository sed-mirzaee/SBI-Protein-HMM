from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
from src.configs.hmm_parameters import (
    STATE_TO_INDEX,
)

def generate_split(
    num_samples: int,
    seed: int,
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, np.ndarray]:
    """
    Generate one offline synthetic dataset split.

    All sequences are padded to max_length.
    The mask identifies real positions and padded positions.
    """

    np.random.seed(seed)

    x = np.zeros(
        (num_samples, max_length, N_AMINO_ACIDS),
        dtype=np.float32,
    )

    y = np.zeros(
        (num_samples, max_length, N_STATES),
        dtype=np.float32,
    )

    mask = np.zeros(
        (num_samples, max_length),
        dtype=bool,
    )

    lengths = np.zeros(
        num_samples,
        dtype=np.int16,
    )

    hidden_states = np.full(
        (num_samples, max_length),
        fill_value=-1,
        dtype=np.int8,
    )

    for i in range(num_samples):

        sequence_length = int(
            np.random.randint(  # this function give a value in interval [a, b)
                min_length,
                max_length + 1,
            )
        )

        protein = simulate_hmm(sequence_length)

        sequence = protein["AminoAcid"].tolist()

        encoded_sequence = one_hot_encode_sequence(
            sequence
        ).astype(np.float32)

        state_probabilities = forward_backward(
            sequence
        ).astype(np.float32)

        state_values = (
            protein["State"]
            .map(STATE_TO_INDEX)
            .to_numpy(dtype=np.int8)
        )

        x[i, :sequence_length] = encoded_sequence
        y[i, :sequence_length] = state_probabilities
        hidden_states[i, :sequence_length] = state_values

        mask[i, :sequence_length] = True
        lengths[i] = sequence_length

        if (i + 1) % 100 == 0:
            print(
                f"Generated {i + 1}/{num_samples}"
            )

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "lengths": lengths,
        "hidden_states": hidden_states,
    }

def save_split(
    dataset: dict[str, np.ndarray],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        **dataset,
    )

    print(f"Saved dataset to: {output_path}")

def main() -> None:

    output_dir = Path("../../data/synthetic")

    split_configs = {
        "train": {
            "num_samples": N_TRAIN_SAMPLES,
            "seed": 202605,
        },
        "validation": {
            "num_samples": N_VALIDATION_SAMPLES,
            "seed": 202606,
        },
        "test": {
            "num_samples": N_TEST_SAMPLES,
            "seed": 202607,
        },
    }

    for split_name, config in split_configs.items():

        print(f"\nGenerating {split_name}...")

        dataset = generate_split(
            num_samples=config["num_samples"],
            seed=config["seed"],
            min_length=MIN_SEQUENCE_LENGTH,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        save_split(
            dataset,
            output_dir / f"{split_name}_{N_TRAIN_SAMPLES}.npz",
        )

    metadata = {
        "length_distribution": "discrete uniform",
        "minimum_length": MIN_SEQUENCE_LENGTH,
        "maximum_length": MAX_SEQUENCE_LENGTH,
        "state_order": [
            "other",
            "alpha",
        ],
        "splits": split_configs,
    }

    with (
        output_dir / f"metadata_{N_TRAIN_SAMPLES}.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()