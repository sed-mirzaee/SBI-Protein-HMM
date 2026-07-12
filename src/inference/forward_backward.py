"""
Forward-Backward algorithm for the two-state protein HMM.

States:
0 = other
1 = alpha-helix
"""

import numpy as np

from src.configs.hmm_parameters import (
    AMINO_ACIDS,
    TRANSITION_MATRIX,
    EMISSION_ALPHA,
    EMISSION_OTHER,
)


emission_alpha_dict = dict(zip(AMINO_ACIDS, EMISSION_ALPHA))
emission_other_dict = dict(zip(AMINO_ACIDS, EMISSION_OTHER))


def get_emission_probabilities(amino_acid):
    """
    Return emission probabilities for one amino acid.

    Output order:
    [P(amino_acid | other), P(amino_acid | alpha-helix)]
    """

    amino_acid = amino_acid.upper()

    if amino_acid not in emission_alpha_dict:
        raise ValueError(f"Invalid amino acid: {amino_acid}")

    return np.array([
        emission_other_dict[amino_acid],
        emission_alpha_dict[amino_acid],
    ])


def forward_backward(sequence):
    """
    Compute posterior state probabilities for a protein sequence.

    Input:
    sequence: string or list of amino acids

    Output:
    posterior: array with shape (sequence_length, 2)

    posterior[:, 0] = P(other | sequence)
    posterior[:, 1] = P(alpha-helix | sequence)
    """

    if isinstance(sequence, str):
        sequence = list(sequence)

    sequence_length = len(sequence)

    forward = np.zeros((sequence_length, 2))
    backward = np.zeros((sequence_length, 2))

    # The project says every sequence starts in "other"
    initial_probabilities = np.array([1.0, 0.0])

    # ---------- Forward pass ----------
    forward[0] = (initial_probabilities * get_emission_probabilities(sequence[0]))
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