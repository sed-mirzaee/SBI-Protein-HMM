import numpy as np


# Hidden states
# The order is important:
# index 0 = other
# index 1 = alpha
STATES = np.array([
    "other",
    "alpha"
])

STATE_TO_INDEX = {
    state: index
    for index, state in enumerate(STATES)
}

OTHER_STATE_INDEX = STATE_TO_INDEX["other"]
ALPHA_STATE_INDEX = STATE_TO_INDEX["alpha"]

# Amino acids
AMINO_ACIDS = np.array([
    "A", "R", "N", "D", "C",
    "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
])

# Transition probabilities
#
# Rows: current state
# Columns: next state
#
#              next:
#              other  alpha
# current other 0.95   0.05
# current alpha 0.10   0.90

TRANSITION_MATRIX = np.array([
    [0.95, 0.05],  # other -> other, alpha
    [0.10, 0.90]   # alpha -> other, alpha
])

# Emission probabilities for alpha-helix
EMISSION_ALPHA = np.array([
    0.12, 0.06, 0.03, 0.05, 0.01,
    0.09, 0.05, 0.04, 0.02, 0.07,
    0.12, 0.06, 0.03, 0.04, 0.02,
    0.05, 0.04, 0.01, 0.03, 0.06
])

# Emission probabilities for other
EMISSION_OTHER = np.array([
    0.06, 0.05, 0.05, 0.06, 0.02,
    0.05, 0.03, 0.09, 0.03, 0.05,
    0.08, 0.06, 0.02, 0.04, 0.06,
    0.07, 0.06, 0.01, 0.04, 0.07
])