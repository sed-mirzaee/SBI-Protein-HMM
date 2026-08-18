import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from src.inference.forward_backward import (
    get_emission_probabilities,
    forward_backward,
)


# ---------------------------------------------------------
# Test 1: Emission probabilities
# ---------------------------------------------------------

emission_A = get_emission_probabilities("A")

print("Emission for amino acid A:")
print(emission_A)

assert np.allclose(
    emission_A,
    np.array([0.06, 0.12])
)

print("* Test 1 passed.\n")

# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

sequence = "ALEKGV"

posterior = forward_backward(sequence)

print(posterior)

# ---------------------------------------------------------
# Test 2: Posterior shape
# ---------------------------------------------------------

print("Posterior shape:")
print(posterior.shape)

assert posterior.shape == (len(sequence), 2)

print("* Test 2 passed.\n")


# ---------------------------------------------------------
# Test 3: Posterior sums
# ---------------------------------------------------------

row_sums = posterior.sum(axis=1)

print("Row sums:")
print(row_sums)

assert np.allclose(row_sums, 1.0)

print("* Test 3 passed.\n")


# ---------------------------------------------------------
# Test 4: Initial state
# ---------------------------------------------------------

print("Posterior at first position:")
print(posterior[0])

assert np.allclose(
    posterior[0],
    np.array([1.0, 0.0])
)

print("* Test 4 passed.\n")



