"""5-fold ACDC splits, M&Ms stub, temporal backends, SVF, baselines."""

from pathlib import Path

import torch
import yaml

from reconseg3d.data.acdc import ACDCDataset, make_fake_acdc
from reconseg3d.data.mms import MMsDataset, discover_mms_patients, make_fake_mms
from reconseg3d.data.splits import load_fold_file, stratified_kfold_patients, write_acdc_folds
from reconseg3d.models.baselines import CompactVoxelMorph, FlowRegAdapter, build_baseline
from reconseg3d.models.motion import MotionNet, scaling_and_squaring
from reconseg3d.models.reconseg3d import build_model
from reconseg3d.utils.config import load_config


def test_write_and_load_acdc_folds(tmp_path: Path):
    # ≥5 patients/diagnosis so every fold bucket is non-empty under stratification.
    root = make_fake_acdc(tmp_path / "acdc", n_patients=25, spatial=(8, 16, 16), n_frames=4)
    out = tmp_path / "splits"
    paths = write_acdc_folds(root, out_dir=out, n_folds=5, seed=0, fake_if_empty=False)
    assert len(paths) == 5
    fold0 = load_fold_file(paths[0])
    assert set(fold0.keys()) >= {"train", "val", "test", "fold"}
    assert len(fold0["train"]) > 0 and len(fold0["val"]) > 0 and len(fold0["test"]) > 0
    # No overlap between train and test
    assert set(fold0["train"]).isdisjoint(set(fold0["test"]))
    ds = ACDCDataset(
        root,
        num_frames=4,
        spatial_size=(8, 16, 16),
        clinical_dim=0,
        train=False,
        split="test",
        fold_file=paths[0],
    )
    assert len(ds) == len(fold0["test"])
    sample = ds[0]
    assert "patient_id" in sample
    assert "spacing" in sample


def test_stratified_folds_cover_diagnoses(tmp_path: Path):
    root = make_fake_acdc(tmp_path / "acdc", n_patients=20, spatial=(8, 16, 16), n_frames=4)
    patients = sorted((tmp_path / "acdc").glob("patient*"))
    folds = stratified_kfold_patients(patients, n_folds=5, seed=1)
    all_test = []
    for f in folds:
        all_test.extend(f["test"])
    assert len(all_test) == len(set(all_test)) == 20


def test_repo_split_files_exist():
    root = Path(__file__).resolve().parents[1]
    for k in range(5):
        p = root / "splits" / f"acdc_fold{k}.json"
        assert p.is_file(), p
        data = load_fold_file(p)
        assert data["fold"] == k
        assert data.get("synthetic_placeholder") is True


def test_placeholder_manifest_flag():
    root = Path(__file__).resolve().parents[1]
    import json

    man = json.loads((root / "splits" / "acdc_folds_manifest.json").read_text(encoding="utf-8"))
    assert man["synthetic_placeholder"] is True
    assert man["n_patients"] < 20  # smoke-scale fake
    assert (root / "splits" / "README.md").is_file()


def test_write_acdc_folds_refuses_degenerate(tmp_path: Path):
    root = make_fake_acdc(tmp_path / "acdc", n_patients=8, spatial=(8, 16, 16), n_frames=4)
    out = tmp_path / "splits"
    try:
        write_acdc_folds(root, out_dir=out, n_folds=5, seed=0, fake_if_empty=False, allow_degenerate=False)
        raised = False
    except ValueError as e:
        raised = True
        assert "empty" in str(e).lower() or "minimum" in str(e).lower() or "degenerate" in str(e).lower() or "Refusing" in str(e)
    assert raised


def test_write_acdc_folds_allow_degenerate_smoke(tmp_path: Path):
    root = make_fake_acdc(tmp_path / "acdc", n_patients=8, spatial=(8, 16, 16), n_frames=4)
    out = tmp_path / "splits"
    paths = write_acdc_folds(
        root, out_dir=out, n_folds=5, seed=0, fake_if_empty=False, allow_degenerate=True
    )
    assert len(paths) == 5
    import json

    man = json.loads((out / "acdc_folds_manifest.json").read_text(encoding="utf-8"))
    assert man["synthetic_placeholder"] is True


def test_empty_train_hard_fail_publication_mode(tmp_path: Path):
    from reconseg3d.data.dataset import build_dataloader

    root = make_fake_acdc(tmp_path / "acdc", n_patients=8, spatial=(8, 16, 16), n_frames=4)
    fold_path = tmp_path / "empty_train_fold.json"
    fold_path.write_text(
        '{"fold":0,"train":[],"val":["patient001"],"test":["patient002"]}',
        encoding="utf-8",
    )
    cfg = {
        "data": {
            "source": "acdc",
            "root": str(root),
            "allow_fake_data": False,
            "auto_fake": False,
            "fold_file": str(fold_path),
            "spatial_size": [8, 16, 16],
            "num_frames": 4,
            "batch_size": 1,
            "clinical_dim": 0,
        },
        "model": {"num_seg_classes": 4},
    }
    try:
        build_dataloader(cfg, "train")
        raised = False
    except RuntimeError as e:
        raised = True
        assert "empty train" in str(e).lower()
    assert raised


def test_publication_configs_fold_null():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "configs/publication/publication_recon.yaml",
        "configs/publication/publication_seg.yaml",
        "configs/publication_recon.yaml",
    ):
        cfg = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert cfg["data"].get("fold") is None, rel


def test_mms_fake_and_hard_fail(tmp_path: Path):
    empty = tmp_path / "empty_mms"
    empty.mkdir()
    assert discover_mms_patients(empty) == []
    root = make_fake_mms(tmp_path / "mms", n_patients=4, spatial=(8, 16, 16), n_frames=4)
    ds = MMsDataset(root, num_frames=4, spatial_size=(8, 16, 16), clinical_dim=0, train=False, split="train")
    assert len(ds) >= 1
    s = ds[0]
    assert s["volume"].ndim == 5
    assert "ed_index" in s and "es_index" in s


def test_publication_mms_hard_fail_without_data(tmp_path: Path):
    from reconseg3d.data.dataset import build_dataloader

    cfg = {
        "data": {
            "source": "mms",
            "root": str(tmp_path / "missing_mms"),
            "allow_fake_data": False,
            "auto_fake": False,
            "spatial_size": [8, 16, 16],
            "num_frames": 4,
            "batch_size": 1,
            "clinical_dim": 0,
        },
        "model": {"num_seg_classes": 4},
    }
    try:
        build_dataloader(cfg, "train")
        raised = False
    except RuntimeError as e:
        raised = True
        assert "M&Ms" in str(e) or "mms" in str(e).lower() or "fake" in str(e).lower()
    assert raised


def test_temporal_backends_forward():
    for mode in ("temporal_conv", "conv_lstm", "temporal_attention"):
        cfg = {
            "model": {
                "arch": "reconseg3d",
                "in_channels": 1,
                "num_seg_classes": 4,
                "base_channels": 8,
                "temporal_mode": mode,
                "clinical_dim": 0,
                "per_frame_recon": True,
                "predict_motion": True,
                "per_frame_seg": True,
                "fusion": "concat",
                "task": "motion",
            }
        }
        model = build_model(cfg)
        x = torch.randn(1, 1, 4, 8, 16, 16)
        out = model(x)
        assert out.reconstruction.shape == x.shape
        assert out.flow is not None


def test_svf_scaling_and_squaring_path():
    v = torch.randn(2, 3, 8, 8, 8) * 0.1
    flow = scaling_and_squaring(v, steps=3)
    assert flow.shape == v.shape
    net = MotionNet(in_channels=1, base_channels=4, use_svf=True, svf_steps=3)
    src = torch.randn(2, 1, 8, 8, 8)
    tgt = torch.randn(2, 1, 8, 8, 8)
    f = net.forward_pair(src, tgt)
    assert f.shape == (2, 3, 8, 8, 8)


def test_compact_voxelmorph_and_flowreg_adapter():
    vm = CompactVoxelMorph(base_channels=4)
    src = torch.randn(1, 1, 8, 8, 8)
    tgt = torch.randn(1, 1, 8, 8, 8)
    out = vm(src, tgt)
    assert out["flow"].shape[1] == 3
    assert out["warped"].shape == src.shape
    adapter = FlowRegAdapter()
    assert adapter.available() is False or isinstance(adapter.available(), bool)
    m = build_baseline("voxelmorph")
    assert isinstance(m, CompactVoxelMorph)


def test_closed_cycle_docstring_mentions_loop():
    from reconseg3d.models.losses import MultiTaskLoss

    doc = MultiTaskLoss.__doc__ or ""
    assert "closed-cycle" in doc.lower() or "L_periodic" in doc or "closing" in doc.lower()
    assert "adjacent path" not in doc.lower() or "closed-cycle" in doc.lower()


def test_physical_slice_sampling_mm():
    import numpy as np

    from reconseg3d.data.slice_sampling import sample_sparse_sa_stack

    vol = np.ones((1, 8, 16, 16), dtype=np.float32)
    sparse, mask = sample_sparse_sa_stack(
        vol,
        s_min=2,
        s_max=3,
        trans_mm=(1.0, 3.0),
        spacing_dhw=(10.0, 1.5, 1.5),
        noise_std=0.0,
        rng=np.random.default_rng(0),
    )
    assert mask.sum() >= 2
    assert float(np.abs(sparse[0, mask]).sum()) > 0


def test_sampling_ratio_overrides_s_range():
    import numpy as np

    from reconseg3d.data.slice_sampling import sample_sparse_sa_stack

    vol = np.ones((16, 8, 8), dtype=np.float32)
    _, mask = sample_sparse_sa_stack(
        vol, sampling_ratio=0.25, noise_std=0.0, rng=np.random.default_rng(1)
    )
    assert int(mask.sum()) == 4  # 0.25 * 16
