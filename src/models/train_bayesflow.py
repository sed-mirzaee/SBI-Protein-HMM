"""Training entry point for the BayesFlow posterior estimator.

This module is intentionally left for the team member responsible
for model architecture and training.
"""


def train_model():
    """
    Train the amortized posterior estimator.

    Expected simulator output
    -------------------------
    encoded_sequence:
        Shape (batch_size, sequence_length, 20)

    state_probabilities:
        Shape (batch_size, sequence_length, 2)
    """

    raise NotImplementedError(
        "BayesFlow training will be implemented by the training team member."
    )


if __name__ == "__main__":
    train_model()