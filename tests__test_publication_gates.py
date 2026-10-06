"""Publication fake-data hard gate and ACDC multi-phase augmentation."""

from pathlib import Path

import pytest
import torch
import yaml

from reconseg3d.data.acdc import ACDCDataset, make_fake_acdc
from reconseg3d.data.dataset import build_dataloader
from reconseg3d.data.transforms import apply_train_transforms, random_flip_3d


def test_synthetic_forbidden_when_allow_fake_false():
    cfg = {
        "data": {
            "source": "synthetic",
            "allow_fake_data": False,
            "auto_fake": False,
            "num_frames": 4,
            "spatial_size": [8, 16, 16],
            "batch_size": 1,
            "train_samples": 2,
            "clinical_dim": 0,
        },
        "model": {"num_seg_classes": 4, "num_phenotype_classes": 5},
    }
    with pytest.raises(RuntimeError, match="forbids fake data"):
        build_dataloader(cfg, "train")


def test_acdc_empty_root_hard_error(tmp_path: Path):
    cfg = {
        "data": {
            "source": "acdc",
            "root": str(tmp_path / "empty_acdc"),
            "allow_fake_data": False,
            "auto_fake": False,
            "num_frames": 4,
            "spatial_size": [8, 16, 16],
            "batch_size": 1,
            "clinical_dim": 0,
        },
        "model": {"num_seg_classes": 4},
    }
    with pytest.raises(RuntimeError, match="forbids fake data"):
        build_dataloader(cfg, "train")


def test_publication_motion_yaml_forbids_fake(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "publication_motion.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["allow_fake_data"] is False
    # Point at empty root so gate fires without needing real ACDC
    cfg["data"]["root"] = str(tmp_path / "no_patients")
    with pytest.raises(RuntimeError, match="forbids fake data"):
        build_dataloader(cfg, "train")


def test_publication_recon_yaml_not_silent_synthetic():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "publication_recon.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["allow_fake_data"] is False
    assert cfg["data"]["source"] != "synthetic"


def test_smoke_motion_allows_synthetic():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "smoke_motion.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["allow_fake_data"] is True
    assert cfg["data"]["source"] == "synthetic"
    loader = build_dataloader(cfg, "train")
    batch = next(iter(loader))
    assert "volume" in batch


def test_random_flip_co_transforms_extra_masks():
    vol = torch.arange(2 * 4 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4, 4)
    m0 = torch.arange(4 * 4 * 4).view(4, 4, 4)
    m1 = m0 + 100
    # Force flips by p=1
    out_vol, out_m0, extras = random_flip_3d(vol.clone(), m0.clone(), p=1.0, extra_masks=[m1.clone()])
    assert out_vol.shape == vol.shape
    assert extras[0].shape == m1.shape
    # Same spatial permutation: relative offset between masks preserved under flip
    assert torch.equal(extras[0] - out_m0, m1 - m0)


def test_acdc_train_aug_cotransforms_ed_and_es(tmp_path: Path, monkeypatch):
    """Forced flips must update both ED and ES slots (not ED alone)."""
    root = make_fake_acdc(tmp_path / "acdc", n_patients=2, spatial=(8, 16, 16), n_frames=4)
    # Always flip every axis (rand < 0.5).
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.0]))

    ds_val = ACDCDataset(root, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=0, train=False, split="train")
    ds_tr = ACDCDataset(root, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=0, train=True, split="train")
    base = ds_val[0]
    aug = ds_tr[0]
    ed = int(base["ed_index"].item())
    es = int(base["es_index"].item())
    assert ed != es

    # Expected: same three-axis flip on both masks (p=1 path via monkeypatch).
    exp_ed = torch.flip(base["segmentation_sequence"][ed], dims=[0, 1, 2])
    exp_es = torch.flip(base["segmentation_sequence"][es], dims=[0, 1, 2])
    assert torch.equal(aug["segmentation_sequence"][ed], exp_ed)
    assert torch.equal(aug["segmentation_sequence"][es], exp_es)
    assert torch.equal(aug["segmentation"], exp_ed)


def test_apply_train_transforms_extra_masks_roundtrip_shapes():
    vol = torch.randn(1, 4, 8, 16, 16)
    ed = torch.randint(0, 4, (8, 16, 16))
    es = torch.randint(0, 4, (8, 16, 16))
    vol_o, ed_o, clin, extras = apply_train_transforms(vol, ed, None, extra_masks=[es])
    assert clin is None
    assert vol_o.shape == vol.shape
    assert ed_o.shape == ed.shape
    assert extras[0].shape == es.shape
