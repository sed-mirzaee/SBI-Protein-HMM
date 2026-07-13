import numpy as np

from src.simulator.protein_hmm import simulate_hmm
from src.inference.forward_backward import forward_backward


# ---------------------------------------------------------
# Test 1: Generate one protein
# ---------------------------------------------------------

np.random.seed(1)

protein = simulate_hmm(30)

print(protein.head())

assert len(protein) == 30

print("* Simulator works.\n")


# ---------------------------------------------------------
# Test 2: Extract amino acid sequence
# ---------------------------------------------------------

sequence = protein["AminoAcid"].tolist()

print(sequence)

assert len(sequence) == 30

print("* Sequence extracted.\n")


# ---------------------------------------------------------
# Test 3: Run Forward-Backward
# ---------------------------------------------------------

posterior = forward_backward(sequence)

print(posterior)

assert posterior.shape == (30, 2)

print("* Forward-Backward works.\n")


# ---------------------------------------------------------
# Test 4: Posterior normalization
# ---------------------------------------------------------

row_sums = posterior.sum(axis=1)

print(row_sums)

assert np.allclose(row_sums, 1.0)

print("* Posterior is normalized.\n")


# ---------------------------------------------------------
# Test 5: Initial state
# ---------------------------------------------------------

print(posterior[0])

assert np.allclose(
    posterior[0],
    np.array([1.0, 0.0])
)

print("* Initial state is correct.\n")


print("* Simulator + Forward-Backward pipeline works correctly.")