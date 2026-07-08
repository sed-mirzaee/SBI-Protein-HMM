"""
Protein HMM simulator for the Simulation-Based Inference project.

This module is a Python/NumPy translation and refactor of the original R
simulator. It models a two-state Hidden Markov Model (HMM) for protein
secondary structure:

    hidden states: "other", "alpha"
    observations: 20 amino acids

The simulator always starts in the "other" state, matching the project
specification in the original R code.
"""

# TODO: output structure for next steps
# {
#     "states": [0, 1, 1, ...]

#     "sequence": [10, 1, 4, ...],
#     "posterior_alpha": [0.03, 0.05, 0.12, ...],
#     "posterior_other": [0.97, 0.95, 0.88, ...]
# }


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProteinHMMSimulator:
    """Two-state HMM simulator for protein amino acid sequences."""

    amino_acids: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                "A", "R", "N", "D", "C",
                "E", "Q", "G", "H", "I",
                "L", "K", "M", "F", "P",
                "S", "T", "W", "Y", "V",
            ]
        )
    )

    states: np.ndarray = field(
        default_factory=lambda: np.array(["other", "alpha"])
    )

    transition: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.95, 0.05],  # other -> other / alpha
                [0.10, 0.90],  # alpha -> other / alpha
            ],
            dtype=float,
        )
    )

    emit_alpha: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.12, 0.06, 0.03, 0.05, 0.01,
                0.09, 0.05, 0.04, 0.02, 0.07,
                0.12, 0.06, 0.03, 0.04, 0.02,
                0.05, 0.04, 0.01, 0.03, 0.06,
            ],
            dtype=float,
        )
    )

    emit_other: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.06, 0.05, 0.05, 0.06, 0.02,
                0.05, 0.03, 0.09, 0.03, 0.05,
                0.08, 0.06, 0.02, 0.04, 0.06,
                0.07, 0.06, 0.01, 0.04, 0.07,
            ],
            dtype=float,
        )
    )

    random_seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        self._validate_probabilities()

    def _validate_probabilities(self) -> None:
        """Check that all probability vectors/matrices are valid."""
        if self.transition.shape != (2, 2):
            raise ValueError("transition must be a 2x2 matrix.")

        if not np.allclose(self.transition.sum(axis=1), 1.0):
            raise ValueError("Each row of transition must sum to 1.")

        if not np.isclose(self.emit_alpha.sum(), 1.0):
            raise ValueError("emit_alpha probabilities must sum to 1.")

        if not np.isclose(self.emit_other.sum(), 1.0):
            raise ValueError("emit_other probabilities must sum to 1.")

        if len(self.amino_acids) != len(self.emit_alpha):
            raise ValueError("amino_acids and emit_alpha must have the same length.")

        if len(self.amino_acids) != len(self.emit_other):
            raise ValueError("amino_acids and emit_other must have the same length.")

    def simulate_states(self, n: int) -> np.ndarray:
        """
        Simulate the hidden state sequence.

        Parameters
        ----------
        n:
            Length of the sequence.

        Returns
        -------
        np.ndarray
            Array of hidden states, each either "other" or "alpha".
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        hidden = np.empty(n, dtype=object)
        hidden[0] = "other"

        for i in range(1, n):
            current_state = hidden[i - 1]
            current_index = np.where(self.states == current_state)[0][0]
            hidden[i] = self.rng.choice(
                self.states,
                p=self.transition[current_index],
            )

        return hidden

    def simulate_sequence(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Generate an amino acid sequence from hidden states.

        Parameters
        ----------
        hidden_states:
            Array of hidden states.

        Returns
        -------
        np.ndarray
            Simulated amino acid sequence.
        """
        sequence = np.empty(len(hidden_states), dtype=object)

        for i, state in enumerate(hidden_states):
            if state == "alpha":
                probs = self.emit_alpha
            elif state == "other":
                probs = self.emit_other
            else:
                raise ValueError(f"Unknown hidden state: {state}")

            sequence[i] = self.rng.choice(self.amino_acids, p=probs)

        return sequence

    def simulate(self, n: int, as_dataframe: bool = True) -> pd.DataFrame | dict[str, Any]:
        """
        Simulate a complete HMM protein sequence.

        Parameters
        ----------
        n:
            Length of the protein sequence.
        as_dataframe:
            If True, return a pandas DataFrame.
            If False, return a dictionary with metadata and arrays.

        Returns
        -------
        pandas.DataFrame or dict
            Simulated protein sequence and hidden states.
        """
        hidden_states = self.simulate_states(n)
        sequence = self.simulate_sequence(hidden_states)

        dataframe = pd.DataFrame(
            {
                "Position": np.arange(1, n + 1),
                "State": hidden_states,
                "AminoAcid": sequence,
            }
        )

        if as_dataframe:
            return dataframe

        return {
            "parameters": self.get_parameters(),
            "hidden_states": hidden_states,
            "sequence": sequence,
            "dataframe": dataframe,
        }

    def get_parameters(self) -> dict[str, Any]:
        """Return model parameters in a reusable dictionary format."""
        return {
            "states": self.states.copy(),
            "amino_acids": self.amino_acids.copy(),
            "transition": self.transition.copy(),
            "emission": {
                "alpha": self.emit_alpha.copy(),
                "other": self.emit_other.copy(),
            },
        }
    


if __name__ == "__main__":
    simulator = ProteinHMMSimulator(random_seed=1)

    protein = simulator.simulate(30)
    print(protein)

    # Simple sanity checks matching the original R script
    hidden = simulator.simulate_states(100_000)

    from_other = hidden[:-1] == "other"
    to_alpha = hidden[1:] == "alpha"
    print("Estimated P(other -> alpha):", np.mean(to_alpha[from_other]))

    from_alpha = hidden[:-1] == "alpha"
    to_alpha = hidden[1:] == "alpha"
    print("Estimated P(alpha -> alpha):", np.mean(to_alpha[from_alpha]))
