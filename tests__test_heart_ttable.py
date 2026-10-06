"""HeartTTable-lite forward on tiny 8×16×16×16 sequences."""

import torch

from reconseg3d.models.heart_ttable import HeartTTable
from reconseg3d.models.reconseg3d import ReconSeg3D


def test_heart_ttable_tiny_forward():
    model = HeartTTable(in_channels=1, clinical_dim=4, embed_dim=32, patch_size=4, num_heads=4, num_layers=2)
    volume = torch.randn(2, 1, 8, 16, 16, 16)
    clinical = torch.randn(2, 4)
    out = model(volume, clinical)
    assert out.features.shape == (2, 32)
    assert out.risk_logits.shape == (2,)
    out.risk_logits.sum().backward()


def test_reconseg3d_heart_ttable_fusion():
    model = ReconSeg3D(
        in_channels=1,
        num_seg_classes=4,
        base_channels=8,
        clinical_dim=4,
        fusion="heart_ttable",
        num_phenotype_classes=5,
        ttable_embed_dim=32,
        ttable_patch_size=4,
        per_frame_recon=True,
    )
    x = torch.randn(1, 1, 4, 8, 16, 16)
    clinical = torch.randn(1, 4)
    out = model(x, clinical)
    assert out.mace_logits.shape == (1,)
    assert out.phenotype_logits is not None
    assert out.phenotype_logits.shape == (1, 5)
