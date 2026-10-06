"""Model forward/backward and shape contract tests."""

import torch

from reconseg3d.data.dataset import SyntheticCardiacDataset, build_dataloader
from reconseg3d.models.losses import MultiTaskLoss
from reconseg3d.models.reconseg3d import ReconSeg3D
from reconseg3d.utils.shapes import assert_volume_shape, permute_dhw_t_to_ctdhw, validate_batch


def test_forward_shapes():
    model = ReconSeg3D(in_channels=1, num_seg_classes=5, clinical_dim=4)
    x = torch.randn(2, 1, 8, 16, 32, 32)
    clinical = torch.randn(2, 4)
    out = model(x, clinical)
    assert out.reconstruction.shape == x.shape
    assert out.segmentation.shape == (2, 5, 16, 32, 32)
    assert out.mace_logits.shape == (2,)


def test_backward_step():
    model = ReconSeg3D(clinical_dim=4)
    criterion = MultiTaskLoss(num_seg_classes=5)
    x = torch.randn(1, 1, 4, 8, 16, 16)
    clinical = torch.randn(1, 4)
    seg = torch.randint(0, 5, (1, 8, 16, 16))
    mace = torch.tensor([1.0])
    out = model(x, clinical)
    losses = criterion(out.reconstruction, out.segmentation, out.mace_logits, x, seg, mace)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_synthetic_dataset_and_batch():
    ds = SyntheticCardiacDataset(num_samples=4, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=4, seed=0)
    sample = ds[0]
    assert sample["volume"].shape == (1, 4, 8, 16, 16)
    assert sample["segmentation"].shape == (8, 16, 16)
    loader = build_dataloader(
        {"data": {"source": "synthetic", "train_samples": 4, "batch_size": 2, "num_frames": 4, "spatial_size": [8, 16, 16], "clinical_dim": 4}},
        "train",
    )
    batch = next(iter(loader))
    validate_batch(batch, in_channels=1, clinical_dim=4)


def test_permute_layout():
    x = torch.randn(1, 1, 8, 16, 16, 4)
    y = permute_dhw_t_to_ctdhw(x)
    assert y.shape == (1, 1, 4, 8, 16, 16)
    assert_volume_shape(y, in_channels=1)


def test_wrong_ndim_raises():
    try:
        assert_volume_shape(torch.randn(2, 1, 8, 16))
        raised = False
    except ValueError:
        raised = True
    assert raised
