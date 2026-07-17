"""BayesFlow inference component for position-wise state probabilities."""

from __future__ import annotations

import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch",
)

import bayesflow as bf
import keras


@bf.utils.serialization.serializable(
    "sbi_protein_hmm"
)
class SequenceConditionSubnet(keras.layers.Layer):
    """
    Prepare sequence-shaped BiLSTM features for ScoringRuleNetwork.

    Input:
        sequence_features:
            (batch_size, sequence_length, feature_dim)

        mask:
            (batch_size, sequence_length)

    Output:
        (batch_size, output_dim)

    The sequence positions are preserved until the final flattening.
    Padded positions are zeroed before flattening.
    """

    def __init__(
        self,
        position_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True
        self.position_dim = position_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.position_layer = keras.layers.Dense(
            units=position_dim,
            activation="relu",
        )

        self.dropout_layer = keras.layers.Dropout(
            rate=dropout,
        )

        self.flatten_layer = keras.layers.Flatten()

        self.output_layer = keras.layers.Dense(
            units=output_dim,
            activation="relu",
        )

    def call(
        self,
        sequence_features,
        mask=None,
        training: bool = False,
        **kwargs,
    ):
        """
        Convert contextual sequence features to one condition vector.
        """

        hidden = self.position_layer(
            sequence_features,
        )

        hidden = self.dropout_layer(
            hidden,
            training=training,
        )

        if mask is not None:
            mask = keras.ops.cast(
                mask,
                hidden.dtype,
            )

            hidden = hidden * mask[..., None]

        hidden = self.flatten_layer(hidden)

        return self.output_layer(hidden)

    def compute_output_shape(
        self,
        input_shape,
    ):
        return (
            input_shape[0],
            self.output_dim,
        )

    def get_config(self) -> dict:
        config = super().get_config()

        config.update(
            {
                "position_dim": self.position_dim,
                "output_dim": self.output_dim,
                "dropout": self.dropout_rate,
            }
        )

        return config


@bf.utils.serialization.serializable(
    "sbi_protein_hmm"
)
class ProbabilityMeanScore(
    bf.scoring_rules.MeanScore
):
    """
    Squared-error score with a softmax probability link.

    The final dimension has size two:

        0 = other
        1 = alpha

    Softmax guarantees that both values lie between zero and one
    and sum to one at every sequence position.
    """

    def get_link(
        self,
        key: str,
    ) -> keras.layers.Layer:
        if key == "value":
            return keras.layers.Softmax(
                axis=-1,
            )

        return super().get_link(key)


def build_inference_network(
    position_dim: int = 64,
    condition_dim: int = 128,
    dropout: float = 0.10,
) -> bf.networks.ScoringRuleNetwork:
    """
    Build the official BayesFlow inference component.

    The network estimates the Forward-Backward posterior
    probabilities from BiLSTM contextual features.
    """

    subnet = SequenceConditionSubnet(
        position_dim=position_dim,
        output_dim=condition_dim,
        dropout=dropout,
    )

    inference_network = bf.networks.ScoringRuleNetwork(
        scoring_rules={
            "posterior_mean": ProbabilityMeanScore(),
        },
        subnet=subnet,
    )

    return inference_network