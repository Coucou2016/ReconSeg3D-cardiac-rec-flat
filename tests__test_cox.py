"""Cox partial likelihood NaN-safety."""

import math

import numpy as np
import torch

from reconseg3d.models.losses import cox_partial_likelihood
from reconseg3d.training.metrics import concordance_index


def test_cox_mixed_events_finite():
    risk = torch.tensor([0.2, -0.1, 0.5, 0.0], requires_grad=True)
    time = torch.tensor([1.0, 3.0, 2.0, 4.0])
    event = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()


def test_cox_all_censored_zero_finite():
    risk = torch.randn(4, requires_grad=True)
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.zeros(4)
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0
    loss.backward()
    assert risk.grad is not None


def test_cox_single_event():
    risk = torch.tensor([0.3, -0.2], requires_grad=True)
    time = torch.tensor([1.5, 4.0])
    event = torch.tensor([1.0, 0.0])
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()


def test_cox_tied_times_finite():
    risk = torch.tensor([0.4, -0.1, 0.2], requires_grad=True)
    time = torch.tensor([2.0, 2.0, 5.0])
    event = torch.tensor([1.0, 1.0, 0.0])
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()


def test_cox_shared_finite_mask_drops_nan_aligned():
    """NaN in one tensor must drop the same patient index in all three."""
    risk = torch.tensor([0.5, float("nan"), -0.2, 0.1], requires_grad=True)
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()
    assert risk.grad is not None
    assert torch.isfinite(risk.grad[0])
    assert risk.grad[1].item() == 0.0 or not torch.isfinite(risk.grad[1])


def test_cox_large_logits_finite():
    risk = torch.tensor([50.0, -50.0, 20.0], requires_grad=True)
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([1.0, 1.0, 0.0])
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(risk.grad).all()


def test_cox_batch_size_one_event():
    risk = torch.tensor([0.7], requires_grad=True)
    time = torch.tensor([3.0])
    event = torch.tensor([1.0])
    loss = cox_partial_likelihood(risk, time, event)
    assert torch.isfinite(loss)
    loss.backward()


def test_c_index_concordant():
    risk = np.array([2.0, 1.0, 0.0])
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0])
    c = concordance_index(risk, time, event)
    assert c == 1.0
    assert math.isnan(concordance_index(risk, time, np.zeros(3)))
