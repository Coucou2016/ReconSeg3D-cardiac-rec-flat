"""Standalone compact 3D recon / seg modules."""

import torch

from reconseg3d.models.volume_recon import CompactVolumeRecon
from reconseg3d.models.volume_seg import VolumeUNet3D


def test_volume_recon_3d_and_4d():
    net = CompactVolumeRecon(in_channels=1, base_channels=8, use_transformer=True, num_layers=1)
    x3 = torch.randn(2, 1, 16, 32, 32)
    y3 = net(x3)
    assert y3.shape == x3.shape
    x4 = torch.randn(1, 1, 2, 16, 32, 32)
    y4 = net(x4)
    assert y4.shape == x4.shape
    y4.mean().backward()


def test_volume_seg_classes():
    net = VolumeUNet3D(in_channels=1, num_classes=4, base_channels=8)
    x = torch.randn(1, 1, 8, 16, 16)
    logits = net(x)
    assert logits.shape == (1, 4, 8, 16, 16)
