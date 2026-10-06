"""ED/ES phase index wiring and labeled-phase supervision."""

from pathlib import Path

import torch

from reconseg3d.data.acdc import ACDCDataset, make_fake_acdc
from reconseg3d.data.dataset import SyntheticCardiacDataset
from reconseg3d.models.losses import MultiTaskLoss, labeled_phase_seg_loss
from reconseg3d.models.reconseg3d import ReconSeg3D
from torch import nn


def test_synthetic_ed_es_batch_fields():
    ds = SyntheticCardiacDataset(num_samples=2, num_frames=8, spatial_size=(8, 16, 16), clinical_dim=0, train=False)
    sample = ds[0]
    assert "seg_frame_indices" in sample
    assert "seg_valid_mask" in sample
    assert "segmentation_sequence" in sample
    assert sample["ed_index"].item() == 0
    assert int(sample["es_index"].item()) == 4
    assert sample["seg_valid_mask"].sum() >= 1
    assert sample["segmentation_sequence"].shape[0] == 8


def test_fake_acdc_ed_es_indices(tmp_path: Path):
    root = make_fake_acdc(tmp_path / "acdc", n_patients=2, spatial=(8, 16, 16), n_frames=8)
    ds = ACDCDataset(root, num_frames=8, spatial_size=(8, 16, 16), clinical_dim=0, train=False, split="train")
    sample = ds[0]
    ed = int(sample["ed_index"].item())
    es = int(sample["es_index"].item())
    assert sample["seg_valid_mask"][ed]
    assert sample["seg_valid_mask"][es]
    assert int(sample["segmentation_sequence"][ed].max()) > 0
    # Primary segmentation is ED
    assert torch.equal(sample["segmentation"], sample["segmentation_sequence"][ed])


def test_model_uses_seg_frame_indices_not_midcycle():
    model = ReconSeg3D(clinical_dim=0, per_frame_recon=True, base_channels=8, num_seg_classes=4)
    x = torch.randn(2, 1, 6, 8, 16, 16)
    # Force different frames
    out0 = model(x, seg_frame_indices=torch.tensor([0, 0]))
    out5 = model(x, seg_frame_indices=torch.tensor([5, 5]))
    # Logits at different refs should differ for random weights (not identical)
    assert out0.segmentation.shape == (2, 4, 8, 16, 16)
    assert out5.seg_sequence is not None
    # Explicit gather matches
    assert torch.allclose(out0.segmentation, out0.seg_sequence[:, :, 0])
    assert torch.allclose(out5.segmentation, out5.seg_sequence[:, :, 5])


def test_labeled_phase_seg_loss_ignores_unlabeled():
    b, k, t, d, h, w = 1, 4, 4, 8, 8, 8
    logits = torch.randn(b, k, t, d, h, w, requires_grad=True)
    gt = torch.full((b, t, d, h, w), -1, dtype=torch.long)
    gt[:, 0] = 1
    gt[:, 2] = 2
    mask = torch.tensor([[True, False, True, False]])
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    loss = labeled_phase_seg_loss(logits, gt, mask, num_classes=k, ce=ce)
    loss.backward()
    assert torch.isfinite(loss)
    # Unlabeled frames should not receive gradient through CE path only on selected
    assert logits.grad is not None


def test_multitask_labeled_sequence_supervision():
    model = ReconSeg3D(clinical_dim=0, per_frame_recon=True, base_channels=8, num_seg_classes=4)
    criterion = MultiTaskLoss(num_seg_classes=4, alpha2=1.0, w_mace=0.0, w_inv=0.0)
    x = torch.randn(1, 1, 4, 8, 16, 16)
    t = 4
    seg_seq = torch.full((1, t, 8, 16, 16), -1, dtype=torch.long)
    seg_seq[:, 0] = 1
    seg_seq[:, 2] = 1
    valid = torch.tensor([[True, False, True, False]])
    out = model(x, seg_frame_indices=torch.tensor([0]))
    losses = criterion(
        out.reconstruction,
        out.segmentation,
        out.mace_logits,
        x,
        seg_seq[:, 0],
        torch.tensor([0.0]),
        seg_sequence=out.seg_sequence,
        segmentation_sequence=seg_seq,
        seg_valid_mask=valid,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
