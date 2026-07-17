"""Sequence encoder for the protein HMM posterior estimator."""
from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf
import keras


@bf.utils.serialization.serializable("sbi_protein_hmm")
class ProteinBiLSTMSequenceNetwork(
    bf.networks.SummaryNetwork
):
    """
    Produce a contextual representation for every sequence position.

    Input:
        protein_sequence:
            (batch_size, max_length, 20)

        mask:
            (batch_size, max_length)

    Output:
        (batch_size, max_length, 2 * hidden_dim)
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
        if mask is not None:
            mask = keras.ops.cast(
                mask,
                "bool",
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
        # Output is still sequence-shaped, so preserve the mask.
        return mask

    def get_config(self) -> dict:
        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout_rate,
            }
        )

        return config

@bf.utils.serialization.serializable("sbi_protein_hmm")
class ProteinBiLSTMSummaryNetwork(
    bf.networks.SummaryNetwork
):
    def __init__(
        self,
        hidden_dim: int = 64,
        summary_dim: int = 64,
        dropout: float = 0.10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.hidden_dim = hidden_dim
        self.summary_dim = summary_dim
        self.dropout_rate = dropout

        self.bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=hidden_dim,
                return_sequences=False,
                dropout=dropout,
            )
        )

        self.summary_layer = keras.layers.Dense(
            units=summary_dim,
            activation="relu",
        )

    def call(
        self,
        inputs: dict,
        training: bool = False,
        **kwargs,
    ):
        protein_sequence = inputs["protein_sequence"]
        sequence_mask = inputs["sequence_mask"]

        summary = self.bilstm(
            protein_sequence,
            mask=sequence_mask,
            training=training,
        )

        return self.summary_layer(summary)

    def compute_mask(
        self,
        inputs,
        mask=None,
    ):
        # The result is one fixed-size vector per protein.
        return None

    def get_config(self) -> dict:
        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "summary_dim": self.summary_dim,
                "dropout": self.dropout_rate,
            }
        )

        return config