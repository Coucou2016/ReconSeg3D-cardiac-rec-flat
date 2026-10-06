"""Slice sampling zeros non-selected slices."""

import numpy as np
import torch

from reconseg3d.data.slice_sampling import apply_slice_sampling_4d, sample_sparse_sa_stack


def test_slice_sampling_zeros_unselected():
    rng = np.random.default_rng(0)
    vol = np.ones((1, 16, 32, 32), dtype=np.float32)
    sparse, mask = sample_sparse_sa_stack(vol, s_min=4, s_max=6, noise_std=0.0, rng=rng)
    assert mask.shape == (16,)
    assert 4 <= int(mask.sum()) <= 6
    assert np.allclose(sparse[0, ~mask], 0.0)
    assert float(np.abs(sparse[0, mask]).sum()) > 0.0


def test_slice_sampling_4d_same_mask_over_t():
    vol = torch.ones(1, 4, 8, 16, 16)
    sparse, mask, target = apply_slice_sampling_4d(vol, s_min=2, s_max=3, noise_std=0.0, rng=np.random.default_rng(1))
    assert target.shape == vol.shape
    assert sparse.shape == vol.shape
    d_mask = mask > 0.5
    assert int(d_mask.sum()) <= 3
    assert torch.allclose(sparse[:, :, ~d_mask], torch.zeros_like(sparse[:, :, ~d_mask]))
