"""
Encoding utilities for protein sequences.

This module converts amino acid sequences into numerical formats
that can be used by a neural network.
"""

import numpy as np


# Use the same amino acid order as the simulator
AMINO_ACIDS = np.array([
    "A", "R", "N", "D", "C",
    "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
])

AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def clean_sequence(sequence):
    """
    Convert input sequence to a clean list of uppercase amino acid letters.
    """

    if isinstance(sequence, str):
        sequence = list(sequence)

    return [aa.upper() for aa in sequence]


def check_sequence(sequence):
    """
    Check whether all amino acids are standard amino acids.
    """

    sequence = clean_sequence(sequence)

    invalid = [aa for aa in sequence if aa not in AA_TO_INDEX]

    if invalid:
        raise ValueError(f"Invalid amino acids found: {invalid}")

    return True


def integer_encode_sequence(sequence):
    """
    Convert amino acid letters to integer IDs.

    Example:
    ["A", "L", "E"] -> [0, 10, 5]
    """

    sequence = clean_sequence(sequence)
    check_sequence(sequence)

    encoded = [AA_TO_INDEX[aa] for aa in sequence]

    return np.array(encoded, dtype=np.int64)


def one_hot_encode_sequence(sequence):
    """
    Convert amino acid sequence to one-hot format.

    Output shape:
    sequence_length x 20
    """

    encoded = integer_encode_sequence(sequence)

    one_hot = np.zeros((len(encoded), len(AMINO_ACIDS)), dtype=np.float32)

    one_hot[np.arange(len(encoded)), encoded] = 1.0

    return one_hot


def pad_sequence(encoded_sequence, max_length, padding_value=0):
    """
    Pad or cut one encoded sequence to a fixed length.
    """

    encoded_sequence = np.asarray(encoded_sequence)

    if len(encoded_sequence) > max_length:
        return encoded_sequence[:max_length]

    padded = np.full(max_length, padding_value, dtype=encoded_sequence.dtype)
    padded[:len(encoded_sequence)] = encoded_sequence

    return padded


def encode_for_neural_network(sequence, max_length=None, one_hot=False):
    """
    Main function for neural network input.

    If one_hot=False:
        returns integer encoding.

    If one_hot=True:
        returns one-hot encoding.

    If max_length is given:
        output is padded/cut to max_length.
    """

    if one_hot:
        encoded = one_hot_encode_sequence(sequence)

        if max_length is not None:
            if encoded.shape[0] > max_length:
                encoded = encoded[:max_length]

            elif encoded.shape[0] < max_length:
                padded = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)
                padded[:encoded.shape[0], :] = encoded
                encoded = padded

        return encoded

    encoded = integer_encode_sequence(sequence)

    if max_length is not None:
        encoded = pad_sequence(encoded, max_length)

    return encoded


def encode_batch_for_neural_network(sequences, max_length=None, one_hot=False):
    """
    Encode a list of protein sequences for neural network input.
    """

    encoded_sequences = [
        encode_for_neural_network(seq, max_length=max_length, one_hot=one_hot)
        for seq in sequences
    ]

    return np.array(encoded_sequences)