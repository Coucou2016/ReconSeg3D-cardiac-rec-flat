"""Fake ACDC / MM-WHS / EMIDEC loaders."""

from pathlib import Path

import numpy as np
import torch

from reconseg3d.data.acdc import ACDCDataset, make_fake_acdc
from reconseg3d.data.emidec import EMIDECDataset, make_fake_emidec
from reconseg3d.data.mmwhs import MMWHSDataset, make_fake_mmwhs, map_mmwhs_labels


def test_fake_acdc_loader(tmp_path: Path):
    root = make_fake_acdc(tmp_path / "acdc", n_patients=4, spatial=(8, 16, 16), n_frames=4)
    ds = ACDCDataset(root, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=3, train=False, split="train")
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["volume"].shape == (1, 4, 8, 16, 16)
    assert sample["segmentation"].shape == (8, 16, 16)
    assert sample["clinical"].shape == (3,)
    assert int(sample["phenotype"].item()) in range(5)
    assert "seg_frame_indices" in sample
    assert sample["segmentation_sequence"].shape[0] == 4
    assert sample["seg_valid_mask"].dtype == torch.bool
    assert "spacing" in sample
    assert sample["spacing"].shape == (3,)


def test_acdc_clinical_no_phenotype_leakage(tmp_path: Path):
    """P0-1: phenotype/Group must not appear in clinical inputs."""
    root = make_fake_acdc(tmp_path / "acdc", n_patients=4, spatial=(8, 16, 16), n_frames=4)
    for dim in (0, 1, 2, 3, 5):
        names = ACDCDataset.clinical_feature_names(dim)
        assert "phenotype" not in names
        assert "group" not in {n.lower() for n in names}

    ds = ACDCDataset(root, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=3, train=False, split="train")
    sample = ds[0]
    clin = sample["clinical"].detach().cpu().numpy().astype(np.float64)
    phenotype = int(sample["phenotype"].item())
    assert clin.shape == (3,)
    # Height/weight/nb from Info.cfg — not phenotype id.
    assert abs(clin[0] - 161.0) < 1e-3 or clin[0] >= 160.0  # fake heights 161+
    assert clin[1] >= 60.0
    assert clin[2] == 4.0  # NbFrame
    assert not np.isclose(clin[0], float(phenotype))
    assert not np.isclose(clin[1], float(phenotype))
    # One-hot / binary encodings of the supervised target must not match clinical.
    one_hot = np.zeros(5, dtype=np.float64)
    one_hot[phenotype] = 1.0
    assert not np.allclose(clin, one_hot[:3])
    binary = np.array([1.0 if phenotype == k else 0.0 for k in range(3)], dtype=np.float64)
    assert not np.allclose(clin, binary)


def test_temporal_subsample_preserves_ed_es(tmp_path: Path):
    """P0-2: original ED/ES frames remain in the selected set after subsample."""
    from reconseg3d.data.io import select_time_indices_keep_anchors

    t, num = 16, 8
    ed, es = 0, 15
    idx = select_time_indices_keep_anchors(t, num, anchors=[ed, es])
    assert ed in idx and es in idx
    assert len(idx) == num
    assert len(idx) == len(set(idx))

    root = make_fake_acdc(tmp_path / "acdc", n_patients=2, spatial=(8, 16, 16), n_frames=16)
    # Force Info.cfg ED=1 ES=16
    for case in sorted((tmp_path / "acdc").glob("patient*")):
        info = (case / "Info.cfg").read_text(encoding="utf-8")
        lines = []
        for line in info.splitlines():
            if line.startswith("ED:"):
                lines.append("ED: 1")
            elif line.startswith("ES:"):
                lines.append("ES: 16")
            else:
                lines.append(line)
        (case / "Info.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ds = ACDCDataset(root, num_frames=8, spatial_size=(8, 16, 16), clinical_dim=0, train=False, split="train")
    sample = ds[0]
    ed_i = int(sample["ed_index"].item())
    es_i = int(sample["es_index"].item())
    assert sample["seg_valid_mask"][ed_i]
    assert sample["seg_valid_mask"][es_i]
    # GT at selected indices must be non-empty (exact original frames kept).
    assert int(sample["segmentation_sequence"][ed_i].max()) > 0
    assert int(sample["segmentation_sequence"][es_i].max()) > 0
    assert torch.equal(sample["segmentation"], sample["segmentation_sequence"][ed_i])


def test_fake_mmwhs_loader(tmp_path: Path):
    root = make_fake_mmwhs(tmp_path / "mmwhs", n_cases=2, spatial=(8, 16, 16))
    mapped = map_mmwhs_labels(np.array([0, 500, 600, 205]))
    assert list(mapped) == [0, 1, 2, 3]
    ds = MMWHSDataset(root, num_frames=1, spatial_size=(8, 16, 16), clinical_dim=0, train=False, split="train")
    sample = ds[0]
    assert sample["volume"].shape == (1, 1, 8, 16, 16)
    labels = set(sample["segmentation"].unique().tolist())
    assert labels.issubset({0, 1, 2, 3})


def test_fake_emidec_loader(tmp_path: Path):
    root = make_fake_emidec(tmp_path / "emidec", n_cases=2, spatial=(8, 16, 16))
    ds = EMIDECDataset(root, num_frames=1, spatial_size=(8, 16, 16), stack_scar_channel=True, train=False)
    sample = ds[0]
    assert sample["volume"].shape[0] == 2
    assert sample["volume"].shape[-3:] == (8, 16, 16)
    assert sample["segmentation"].max() >= 3
