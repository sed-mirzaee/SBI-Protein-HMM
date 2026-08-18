import numpy as np
import pandas as pd

from src.simulator.protein_hmm import (
    simulate_states,
    simulate_sequence,
    simulate_hmm,
)


# ---------------------------------------------------------
# Test 1: Initial state
# ---------------------------------------------------------

np.random.seed(1)

protein = simulate_hmm(100)

print("First state:")
print(protein.loc[0, "State"])


# ---------------------------------------------------------
# Test 2: Transition probabilities
# ---------------------------------------------------------

np.random.seed(1)

hidden = simulate_states(100_000)

print("\nState counts:")
print(pd.Series(hidden).value_counts())


previous_states = hidden[:-1]
next_states = hidden[1:]


from_other = previous_states == "other"
to_alpha = next_states == "alpha"

p_alpha_given_other = np.mean(to_alpha[from_other])

print("\nEstimated P(alpha | other):")
print(p_alpha_given_other)


from_alpha = previous_states == "alpha"
to_alpha = next_states == "alpha"

p_alpha_given_alpha = np.mean(to_alpha[from_alpha])

print("\nEstimated P(alpha | alpha):")
print(p_alpha_given_alpha)


# ---------------------------------------------------------
# Test 3: Emission probabilities
# ---------------------------------------------------------

np.random.seed(1)

alpha_states = np.repeat("alpha", 100_000)
seq_alpha = simulate_sequence(alpha_states)

print("\nEstimated emission probabilities for alpha:")
print(
    pd.Series(seq_alpha)
    .value_counts(normalize=True)
    .sort_index()
)


other_states = np.repeat("other", 100_000)
seq_other = simulate_sequence(other_states)

print("\nEstimated emission probabilities for other:")
print(
    pd.Series(seq_other)
    .value_counts(normalize=True)
    .sort_index()
)

# ---------------------------------------------------------
# Test 3: Initial State
# ---------------------------------------------------------

n = 100
protein = simulate_hmm(n)

assert protein.loc[0, "State"] == "other"
print("* Initial state is correct")

assert len(protein) == n
print("* Sequence length is correct")

print("\nUnique amino acids:")
print(sorted(protein["AminoAcid"].unique()))