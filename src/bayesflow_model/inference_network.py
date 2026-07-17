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

    def build(self, input_shape) -> None:
        """
        Explicitly build the internal layers.

        Parameters
        ----------
        input_shape:
            Shape of the sequence features:

            (
                batch_size,
                sequence_length,
                feature_dimension,
            )

            For the current project this is normally:

            (
                None,
                250,
                128,
            )
        """

        # ---------------------------------------------------------
        # 1. Build the position-wise Dense layer
        # ---------------------------------------------------------

        self.position_layer.build(
            input_shape
        )

        # Output of position_layer:
        #
        # (
        #     batch_size,
        #     sequence_length,
        #     position_dim,
        # )

        position_output_shape = (
            input_shape[0],
            input_shape[1],
            self.position_dim,
        )

        # ---------------------------------------------------------
        # 2. Build Dropout and Flatten
        # ---------------------------------------------------------

        self.dropout_layer.build(
            position_output_shape
        )

        self.flatten_layer.build(
            position_output_shape
        )

        # ---------------------------------------------------------
        # 3. Compute flattened feature dimension
        # ---------------------------------------------------------

        sequence_length = input_shape[1]

        if sequence_length is None:
            raise ValueError(
                "SequenceConditionSubnet requires a fixed "
                "sequence length. The current model expects "
                "sequences padded to length 250."
            )

        flattened_dimension = (
                sequence_length
                * self.position_dim
        )

        # ---------------------------------------------------------
        # 4. Build the final Dense layer
        # ---------------------------------------------------------

        self.output_layer.build(
            (
                input_shape[0],
                flattened_dimension,
            )
        )

        # ---------------------------------------------------------
        # 5. Mark the custom layer as built
        # ---------------------------------------------------------

        super().build(input_shape)

    def call(
            self,
            inputs,
            mask=None,
            training: bool = False,
    ):
        """
        Convert sequence-level BiLSTM features into
        a fixed-size condition vector.

        Parameters
        ----------
        inputs:
            Sequence features with shape:

            (
                batch_size,
                sequence_length,
                feature_dimension,
            )

        mask:
            Valid-position mask with shape:

            (
                batch_size,
                sequence_length,
            )

        training:
            Whether the layer is being used during training.

        Returns
        -------
        Tensor with shape:

            (
                batch_size,
                output_dim,
            )
        """

        # ---------------------------------------------------------
        # 1. Position-wise feature transformation
        # ---------------------------------------------------------

        hidden = self.position_layer(
            inputs
        )

        hidden = self.dropout_layer(
            hidden,
            training=training,
        )

        # ---------------------------------------------------------
        # 2. Explicitly zero padded positions
        # ---------------------------------------------------------

        if mask is not None:
            numeric_mask = keras.ops.cast(
                mask,
                hidden.dtype,
            )

            hidden = (
                    hidden
                    * numeric_mask[..., None]
            )

        # ---------------------------------------------------------
        # 3. Flatten all sequence positions
        # ---------------------------------------------------------

        hidden = self.flatten_layer(
            hidden
        )

        # ---------------------------------------------------------
        # 4. Create final condition vector
        # ---------------------------------------------------------

        conditions = self.output_layer(
            hidden
        )

        return conditions

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