"""Sequence-preserving BiLSTM network for protein sequences."""

from __future__ import annotations

import os

# Must be set before importing BayesFlow or Keras.
os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf
import keras


@bf.utils.serialization.serializable(
    "sbi_protein_hmm"
)
class ProteinBiLSTMSequenceNetwork(
    bf.networks.SummaryNetwork
):
    """
    Produce one contextual feature vector per sequence position.

    Input:
        protein_sequence:
            shape
            (batch_size, sequence_length, 20)

        mask:
            shape
            (batch_size, sequence_length)

    Output:
        shape
        (
            batch_size,
            sequence_length,
            2 * hidden_dim,
        )

    With hidden_dim=64, each sequence position receives
    128 contextual features:
        64 from the forward LSTM
        64 from the backward LSTM
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.supports_masking = True

        self.bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=hidden_dim,
                return_sequences=True,
                dropout=dropout,
            )
        )

    def call(
        self,
        protein_sequence,
        mask=None,
        training: bool = False,
        **kwargs,
    ):
        """
        Encode every amino-acid position using both directions.
        """

        # BayesFlow's default Adapter converts masks
        # to float32. Keras LSTM expects a Boolean mask.
        if mask is not None:
            mask = keras.ops.cast(
                mask,
                dtype="bool",
            )

        return self.bilstm(
            protein_sequence,
            mask=mask,
            training=training,
        )

    def compute_mask(
            self,
            inputs,
            mask=None,
    ):
        """
        Do not propagate the Keras mask downstream.

        The mask has already been used by the BiLSTM.
        Target padding is handled separately through sample weights.
        """
        return None

    def get_config(self) -> dict:
        """Return configuration for Keras serialization."""

        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout_rate,
            }
        )

        return config