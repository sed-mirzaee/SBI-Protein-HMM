import numpy as np

from src.simulator.protein_hmm import simulate_hmm
from src.preprocessing.encoding import one_hot_encode_sequence
from src.inference.forward_backward import forward_backward


SEQUENCE_LENGTH = 100


def simulate_training_pair() -> dict[str, np.ndarray]:
    """
    Simulate one protein sequence and prepare one training pair.

    Returns
    -------
    encoded_sequence:
        One-hot encoded amino-acid sequence.
        Shape: (sequence_length, 20)

    state_probabilities:
        Forward-Backward posterior state probabilities.
        Shape: (sequence_length, 2)
    """

    # 1. Simulate one protein sequence
    protein = simulate_hmm(SEQUENCE_LENGTH)

    # protein["AminoAcid"] is a pandas Series
    sequence = protein["AminoAcid"].to_list()

    # 2. Neural-network input
    encoded_sequence = one_hot_encode_sequence(sequence)

    # 3. Training target
    state_probabilities = forward_backward(sequence)

    return {
        "encoded_sequence": encoded_sequence.astype(np.float32),
        "state_probabilities": state_probabilities.astype(np.float32),
    }

sample = simulate_training_pair()

print(sample.keys())
print("Input shape:", sample["encoded_sequence"].shape)
print("Target shape:", sample["state_probabilities"].shape)

print(
    "Input row sums:",
    sample["encoded_sequence"].sum(axis=1)[:5],
)

print(
    "Target row sums:",
    sample["state_probabilities"].sum(axis=1)[:5],
)