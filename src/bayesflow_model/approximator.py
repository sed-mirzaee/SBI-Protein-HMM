"""
Build the native BayesFlow approximator for protein-state inference.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# This must be set before importing BayesFlow or Keras.
os.environ.setdefault("KERAS_BACKEND", "torch")

import bayesflow as bf
import keras

from src.bayesflow_model.adapter import build_protein_adapter
from src.bayesflow_model.inference_network import build_inference_network
from src.bayesflow_model.summary_network import (
    ProteinBiLSTMSequenceNetwork,
)


@keras.saving.register_keras_serializable(
    package="protein_bayesflow"
)
class TorchScoringRuleApproximator(
    bf.approximators.ScoringRuleApproximator
):
    """
    ScoringRuleApproximator compatible with dictionary batches
    produced by OfflineDataset under the Keras Torch backend.

    With the Torch backend, Keras may call the model as:

        model(batch_dictionary)

    instead of:

        model(**batch_dictionary)

    BayesFlow's compute_metrics method expects the second form.
    This wrapper converts the first form into the second form.
    """

    def call(
            self,
            inputs=None,
            training=False,
            **kwargs,
    ):
        """
        Unpack dictionary batches produced by OfflineDataset.
        """

        stage = kwargs.pop(
            "stage",
            "training" if training else "validation",
        )

        if isinstance(inputs, Mapping):
            batch = dict(inputs)
            batch.update(kwargs)

            return self.compute_metrics(
                **batch,
                stage=stage,
            )

        if inputs is not None:
            return self.compute_metrics(
                inputs,
                stage=stage,
                **kwargs,
            )

        return self.compute_metrics(
            stage=stage,
            **kwargs,
        )

def build_protein_approximator(
    hidden_dim: int = 64,
    position_dim: int = 64,
    condition_dim: int = 128,
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
) -> TorchScoringRuleApproximator:
    """
    Build and compile the native BayesFlow model.

    Pipeline
    --------
    protein sequence
        -> BayesFlow Adapter
        -> sequence-preserving BiLSTM
        -> ScoringRuleNetwork
        -> position-wise posterior probabilities
    """

    adapter = build_protein_adapter()

    summary_network = ProteinBiLSTMSequenceNetwork(
        hidden_dim=hidden_dim,
        dropout=dropout,
        name="protein_bilstm_summary",
    )

    inference_network = build_inference_network(
        position_dim=position_dim,
        condition_dim=condition_dim,
        dropout=dropout,
    )

    approximator = TorchScoringRuleApproximator(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        standardize=None,
        name="protein_scoring_rule_approximator",
    )

    approximator.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
    )

    return approximator