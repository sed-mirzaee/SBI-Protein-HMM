"""
                 Simulation Based Inference - Project  work
              Inference of protein secondary structure motifs
                                  05. July 2026
"""

import numpy as np
import pandas as pd
from src.configs.hmm_parameters import (
    STATES,
    AMINO_ACIDS,
    TRANSITION_MATRIX,
    EMISSION_ALPHA,
    EMISSION_OTHER,
)

# Generate the hidden-state sequence.
#     Every sequence starts in the "other" state.

#---------------------------------------------------
#     Generate the hidden-state sequence.
#     Every sequence starts in the "other" state.
#---------------------------------------------------
def simulate_states(n):

    hidden = np.empty(n, dtype=object)

    hidden[0] = "other"

    for i in range(1, n):

        current_state = hidden[i - 1]

        if current_state == "other":
            current_row = 0
        else:
            current_row = 1

        hidden[i] = np.random.choice(
            STATES,
            p=TRANSITION_MATRIX[current_row]
        )

    return hidden

#-----------------------------------------------------
#     Generate one amino acid for each hidden state.
#-----------------------------------------------------
def simulate_sequence(hidden):
    sequence = np.empty(len(hidden), dtype=object)

    for i, state in enumerate(hidden):

        if state == "alpha":
            probabilities = EMISSION_ALPHA
        else:
            probabilities = EMISSION_OTHER

        sequence[i] = np.random.choice(
            AMINO_ACIDS,
            p=probabilities
        )

    return sequence

#-----------------------------------------------------
#     Generate one complete protein sequence.
#-----------------------------------------------------
def simulate_hmm(n):
    hidden = simulate_states(n)

    sequence = simulate_sequence(hidden)

    protein = pd.DataFrame({
        "Position": np.arange(1, n + 1),
        "State": hidden,
        "AminoAcid": sequence
    })

    return protein


if __name__ == "__main__":

    np.random.seed(1)

    protein = simulate_hmm(30)

    print(protein)