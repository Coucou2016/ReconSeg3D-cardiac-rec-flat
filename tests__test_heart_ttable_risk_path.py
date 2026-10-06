"""HeartTTable fusion should use its risk head (not a second unused MLP)."""

import torch

from reconseg3d.models.reconseg3d import ReconSeg3D


def test_heart_ttable_fusion_uses_ttable_risk_head():
    model = ReconSeg3D(
        in_channels=1,
        num_seg_classes=5,
        clinical_dim=4,
        base_channels=8,
        fusion="heart_ttable",
        ttable_embed_dim=16,
        ttable_patch_size=4,
        per_frame_recon=True,
        predict_motion=False,
        per_frame_seg=False,
    )
    x = torch.randn(1, 1, 4, 8, 16, 16)
    clinical = torch.randn(1, 4)
    out = model(x, clinical)
    assert out.mace_logits.shape == (1,)
    assert torch.isfinite(out.mace_logits).all()
    # Gradients should flow into HeartTTable risk_head
    out.mace_logits.sum().backward()
    assert model.heart_ttable.risk_head[0].weight.grad is not None
