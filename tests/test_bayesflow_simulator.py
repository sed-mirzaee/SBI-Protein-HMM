import numpy as np

from src.simulator.bayesflow_simulator import (
    SEQUENCE_LENGTH,
    simulate_sample,
    bayesflow_simulator,
)


def test_simulate_sample_keys():
    """The wrapper must return the expected variables."""

    sample = simulate_sample()

    assert set(sample.keys()) == {
        "encoded_sequence",
        "state_probabilities",
    }


def test_simulate_sample_shapes():
    """One simulated sample must have the expected shapes."""

    sample = simulate_sample()

    assert sample["encoded_sequence"].shape == (
        SEQUENCE_LENGTH,
        20,
    )

    assert sample["state_probabilities"].shape == (
        SEQUENCE_LENGTH,
        2,
    )


def test_encoded_sequence_is_one_hot():
    """Each sequence position must contain one active amino acid."""

    sample = simulate_sample()

    encoded_sequence = sample["encoded_sequence"]

    assert np.all(
        np.logical_or(
            encoded_sequence == 0.0,
            encoded_sequence == 1.0,
        )
    )

    assert np.allclose(
        encoded_sequence.sum(axis=1),
        1.0,
    )


def test_state_probabilities_are_valid():
    """Forward-Backward probabilities must be valid."""

    sample = simulate_sample()

    state_probabilities = sample["state_probabilities"]

    assert np.all(state_probabilities >= 0.0)
    assert np.all(state_probabilities <= 1.0)

    assert np.allclose(
        state_probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )


def test_initial_state_probability():
    """
    Every simulated sequence starts in the 'other' state.

    State order:
        column 0 = other
        column 1 = alpha
    """

    sample = simulate_sample()

    first_position = sample["state_probabilities"][0]

    assert np.allclose(
        first_position,
        np.array([1.0, 0.0]),
        atol=1e-6,
    )


def test_bayesflow_batch():
    """BayesFlow must be able to generate a complete batch."""

    batch_size = 4

    batch = bayesflow_simulator.sample(batch_size)

    assert batch["encoded_sequence"].shape == (
        batch_size,
        SEQUENCE_LENGTH,
        20,
    )

    assert batch["state_probabilities"].shape == (
        batch_size,
        SEQUENCE_LENGTH,
        2,
    )

    assert batch["encoded_sequence"].dtype == np.float32
    assert batch["state_probabilities"].dtype == np.float32

    assert np.allclose(
        batch["encoded_sequence"].sum(axis=2),
        1.0,
    )

    assert np.allclose(
        batch["state_probabilities"].sum(axis=2),
        1.0,
        atol=1e-6,
    )