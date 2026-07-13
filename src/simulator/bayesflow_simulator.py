"""
        BayesFlow-compatible wrapper for the protein HMM simulator.

"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import bayesflow as bf

from src.simulator.protein_hmm import simulate_hmm
from src.preprocessing.encoding import one_hot_encode_sequence
from src.inference.forward_backward import forward_backward


# For the first implementation, all sequences have a fixed length.
SEQUENCE_LENGTH = 100


def simulate_sample() -> dict[str, np.ndarray]:
    """
    Simulate one protein sequence and prepare one training sample.

    Returns
    -------
    dict
        encoded_sequence:
            One-hot encoded amino-acid sequence.
            Shape: (SEQUENCE_LENGTH, 20)

        state_probabilities:
            Posterior hidden-state probabilities computed with the
            Forward-Backward algorithm.
            Shape: (SEQUENCE_LENGTH, 2)
    """

    # 1. Simulate one protein sequence.
    protein = simulate_hmm(SEQUENCE_LENGTH)

    # Convert the amino-acid column into a list such as:
    # ["A", "L", "E", ...]
    sequence = protein["AminoAcid"].tolist()

    # 2. Prepare the neural-network input.
    encoded_sequence = one_hot_encode_sequence(sequence)

    # 3. Prepare the training target.
    state_probabilities = forward_backward(sequence)

    return {
        "encoded_sequence": encoded_sequence.astype(np.float32),
        "state_probabilities": state_probabilities.astype(np.float32),
    }


# BayesFlow repeatedly calls simulate_sample() when sample(batch_size)
# is requested.
bayesflow_simulator = bf.make_simulator(simulate_sample)