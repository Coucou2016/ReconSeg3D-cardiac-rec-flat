"""Per-frame reconstruction vs broadcast ablation."""

import torch

from reconseg3d.models.reconseg3d import ReconSeg3D


def test_per_frame_recon_shape_and_varies_over_t():
    model = ReconSeg3D(in_channels=1, num_seg_classes=5, clinical_dim=0, per_frame_recon=True, predict_motion=True)
    x = torch.randn(2, 1, 4, 8, 16, 16)
    out = model(x)
    assert out.reconstruction.shape == x.shape
    assert out.reconstruction.shape == (2, 1, 4, 8, 16, 16)
    # Frames must not be copies of each other for non-constant input.
    assert not torch.allclose(out.reconstruction[:, :, 0], out.reconstruction[:, :, 1], atol=1e-6)
    assert out.flow is not None and out.flow.shape == (2, 3, 4, 8, 16, 16)
    assert out.seg_sequence is not None and out.seg_sequence.shape[2] == 4


def test_broadcast_ablation_equal_across_t():
    model = ReconSeg3D(in_channels=1, num_seg_classes=5, clinical_dim=0, per_frame_recon=False)
    x = torch.randn(1, 1, 3, 8, 16, 16)
    out = model(x)
    assert out.reconstruction.shape == x.shape
    assert torch.allclose(out.reconstruction[:, :, 0], out.reconstruction[:, :, 1])
    assert torch.allclose(out.reconstruction[:, :, 1], out.reconstruction[:, :, 2])
    assert out.flow is None
