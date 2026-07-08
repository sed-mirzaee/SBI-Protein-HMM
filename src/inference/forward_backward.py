"""
Forward-Backward algorithm for the two-state protein HMM.

States:
0 = alpha-helix
1 = other
"""

import numpy as np


TRANSITION_MATRIX = np.array([
    [0.90, 0.10],  # from alpha-helix to [alpha-helix, other]
    [0.05, 0.95],  # from other to [alpha-helix, other]
])


EMISSION_ALPHA = {
    "A": 0.12, "R": 0.06, "N": 0.03, "D": 0.05, "C": 0.01,
    "E": 0.09, "Q": 0.05, "G": 0.04, "H": 0.02, "I": 0.07,
    "L": 0.12, "K": 0.06, "M": 0.03, "F": 0.04, "P": 0.02,
    "S": 0.05, "T": 0.04, "W": 0.01, "Y": 0.03, "V": 0.06,
}


EMISSION_OTHER = {
    "A": 0.06, "R": 0.05, "N": 0.05, "D": 0.06, "C": 0.02,
    "E": 0.05, "Q": 0.03, "G": 0.09, "H": 0.03, "I": 0.05,
    "L": 0.08, "K": 0.06, "M": 0.02, "F": 0.04, "P": 0.06,
    "S": 0.07, "T": 0.06, "W": 0.01, "Y": 0.04, "V": 0.07,
}


def get_emission_probabilities(amino_acid):
    """
    Return emission probabilities for one amino acid.

    Output order:
    [P(amino_acid | alpha-helix), P(amino_acid | other)]
    """

    amino_acid = amino_acid.upper()

    if amino_acid not in EMISSION_ALPHA:
        raise ValueError(f"Invalid amino acid: {amino_acid}")

    return np.array([
        EMISSION_ALPHA[amino_acid],
        EMISSION_OTHER[amino_acid],
    ])


def forward_backward(sequence):
    """
    Compute posterior state probabilities for a protein sequence.

    Input:
    sequence: string or list of amino acids

    Output:
    posterior: array with shape (sequence_length, 2)

    posterior[:, 0] = P(alpha-helix | sequence)
    posterior[:, 1] = P(other | sequence)
    """

    if isinstance(sequence, str):
        sequence = list(sequence)

    sequence_length = len(sequence)

    forward = np.zeros((sequence_length, 2))
    backward = np.zeros((sequence_length, 2))

    # The project says every sequence starts in "other"
    initial_probabilities = np.array([0.0, 1.0])

    # ---------- Forward pass ----------
    forward[0] = initial_probabilities * get_emission_probabilities(sequence[0])
    forward[0] = forward[0] / forward[0].sum()

    for t in range(1, sequence_length):
        emission = get_emission_probabilities(sequence[t])

        forward[t] = emission * (forward[t - 1] @ TRANSITION_MATRIX)
        forward[t] = forward[t] / forward[t].sum()

    # ---------- Backward pass ----------
    backward[-1] = np.array([1.0, 1.0])

    for t in range(sequence_length - 2, -1, -1):
        next_emission = get_emission_probabilities(sequence[t + 1])

        backward[t] = TRANSITION_MATRIX @ (next_emission * backward[t + 1])
        backward[t] = backward[t] / backward[t].sum()

    # ---------- Posterior ----------
    posterior = forward * backward
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    return posterior