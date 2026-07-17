import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.train_bayesflow import (
    BiLSTMPosteriorEstimator,
    evaluate_model,
    soft_cross_entropy,
)


def test_bilstm_output_shape() -> None:
    """Model output must match posterior target dimensions."""

    model = BiLSTMPosteriorEstimator(
        input_dim=20,
        hidden_dim=8,
        dense_dim=8,
        output_dim=2,
        dropout=0.0,
    )

    x = torch.randn(4, 25, 20)

    logits = model(x)

    assert logits.shape == (4, 25, 2)
    assert torch.isfinite(logits).all()


def test_soft_cross_entropy_is_finite() -> None:
    """Masked soft cross-entropy must return one finite scalar."""

    logits = torch.randn(2, 5, 2)

    targets = torch.tensor(
        [
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.1, 0.9],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.9, 0.1],
                [0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )

    loss = soft_cross_entropy(
        logits=logits,
        target_probabilities=targets,
        mask=mask,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_soft_cross_entropy_ignores_padding() -> None:
    """
    Changing targets at padded positions must not change the loss.
    """

    torch.manual_seed(1)

    logits = torch.randn(1, 4, 2)

    targets_1 = torch.tensor(
        [
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    targets_2 = targets_1.clone()

    # Change only padded positions.
    targets_2[0, 2] = torch.tensor([1.0, 0.0])
    targets_2[0, 3] = torch.tensor([0.0, 1.0])

    mask = torch.tensor(
        [[True, True, False, False]]
    )

    loss_1 = soft_cross_entropy(
        logits,
        targets_1,
        mask,
    )

    loss_2 = soft_cross_entropy(
        logits,
        targets_2,
        mask,
    )

    assert torch.allclose(
        loss_1,
        loss_2,
        atol=1e-7,
    )


class ZeroLogitModel(nn.Module):
    """
    Return equal logits for both states.

    Softmax output will therefore be [0.5, 0.5].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        return torch.zeros(
            batch_size,
            sequence_length,
            2,
            device=x.device,
        )


def test_evaluate_model_uses_only_valid_positions() -> None:
    """Evaluation must exclude padded positions."""

    x = torch.zeros(1, 4, 20)

    y = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [[True, True, False, False]]
    )

    dataset = TensorDataset(x, y, mask)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    metrics = evaluate_model(
        model=ZeroLogitModel(),
        data_loader=loader,
        device=torch.device("cpu"),
    )

    # Equal probabilities give cross-entropy log(2).
    assert math.isclose(
        metrics["loss"],
        math.log(2.0),
        rel_tol=1e-6,
    )

    assert math.isclose(
        metrics["mae"],
        0.5,
        rel_tol=1e-6,
    )

    assert math.isclose(
        metrics["mse"],
        0.25,
        rel_tol=1e-6,
    )


def test_one_optimizer_step_runs() -> None:
    """
    One forward/backward/update step must run without NaN or shape errors.
    """

    torch.manual_seed(1)

    model = BiLSTMPosteriorEstimator(
        hidden_dim=8,
        dense_dim=8,
        dropout=0.0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    x = torch.randn(2, 10, 20)

    raw_targets = torch.rand(2, 10, 2)
    targets = raw_targets / raw_targets.sum(
        dim=-1,
        keepdim=True,
    )

    mask = torch.ones(
        2,
        10,
        dtype=torch.bool,
    )

    optimizer.zero_grad()

    logits = model(x)

    loss = soft_cross_entropy(
        logits,
        targets,
        mask,
    )

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert gradients_exist
    assert torch.isfinite(loss)

    optimizer.step()