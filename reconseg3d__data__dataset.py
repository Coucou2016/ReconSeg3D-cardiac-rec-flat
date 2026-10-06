"""Cardiac 4D datasets: synthetic demo, ACDC, MM-WHS, EMIDEC, NIfTI AMI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from reconseg3d.data.io import load_nifti
from reconseg3d.data.slice_sampling import apply_slice_sampling_4d, slice_sampling_config
from reconseg3d.data.transforms import apply_train_transforms, normalize_intensity

logger = logging.getLogger(__name__)

# Segmentation class indices (documented in README)
SEG_BACKGROUND = 0
SEG_LV = 1
SEG_RV = 2
SEG_MYOCARDIUM = 3
SEG_SCAR = 4
NUM_SEG_CLASSES = 5


class SyntheticCardiacDataset(Dataset):
    """Runnable dummy cine-like volumes with MACE / Cox / phenotype labels."""

    def __init__(
        self,
        num_samples: int = 32,
        in_channels: int = 1,
        num_frames: int = 8,
        spatial_size: tuple[int, int, int] = (16, 32, 32),
        num_seg_classes: int = NUM_SEG_CLASSES,
        clinical_dim: int = 4,
        seed: int = 42,
        train: bool = True,
        slice_sampling: bool = False,
        slice_kwargs: dict[str, Any] | None = None,
        num_phenotype_classes: int = 5,
    ) -> None:
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.num_seg_classes = num_seg_classes
        self.clinical_dim = clinical_dim
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.slice_sampling = slice_sampling
        self.slice_kwargs = slice_kwargs or {}
        self.num_phenotype_classes = num_phenotype_classes

    def __len__(self) -> int:
        return self.num_samples

    def _make_ellipsoid_mask(self, center: np.ndarray, radii: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        d, h, w = shape
        zz, yy, xx = np.ogrid[:d, :h, :w]
        dist = ((zz - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2 + ((xx - center[2]) / radii[2]) ** 2
        return (dist <= 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d, h, w = self.spatial_size
        t = self.num_frames
        center = self.rng.uniform([d * 0.3, h * 0.3, w * 0.3], [d * 0.7, h * 0.7, w * 0.7])
        myo = self._make_ellipsoid_mask(center, np.array([d * 0.35, h * 0.25, w * 0.25]), (d, h, w))
        lv = self._make_ellipsoid_mask(center, np.array([d * 0.2, h * 0.15, w * 0.15]), (d, h, w))
        rv_center = center + np.array([0.0, h * 0.15, 0.0])
        rv = self._make_ellipsoid_mask(rv_center, np.array([d * 0.12, h * 0.1, w * 0.1]), (d, h, w))
        scar = (myo > 0) & (self.rng.random((d, h, w)) > 0.85)
        scar = scar.astype(np.float32) * myo

        seg = np.zeros((d, h, w), dtype=np.int64)
        seg[myo > 0] = SEG_MYOCARDIUM
        seg[lv > 0] = SEG_LV
        seg[rv > 0] = SEG_RV
        if self.num_seg_classes >= 5:
            seg[scar > 0] = SEG_SCAR

        phase = np.linspace(0, 2 * np.pi, t, dtype=np.float32)
        volume = np.zeros((self.in_channels, t, d, h, w), dtype=np.float32)
        for ti, ph in enumerate(phase):
            motion = 1.0 + 0.08 * np.sin(ph)
            frame = myo * motion + lv * 1.2 + rv * 0.9 + scar * 0.5
            frame += self.rng.normal(0, 0.05, frame.shape).astype(np.float32)
            volume[0, ti] = frame
        if self.in_channels >= 2:
            volume[1] = scar[np.newaxis, ...]

        scar_burden = float(scar.sum()) / max(float(myo.sum()), 1.0)
        troponin = self.rng.uniform(0.1, 3.0)
        ef = self.rng.uniform(35.0, 60.0)
        age = self.rng.uniform(45.0, 85.0)
        mace_prob = 0.35 * (scar_burden > 0.08) + 0.25 * (troponin > 1.5) + 0.2 * (ef < 45.0) + 0.1 * (age > 70.0)
        mace = float(self.rng.random() < min(0.95, mace_prob + 0.15))
        clinical_base = np.array([troponin, ef, age, scar_burden], dtype=np.float32)
        if self.clinical_dim <= 0:
            clinical = np.zeros(0, dtype=np.float32)
        elif self.clinical_dim <= 4:
            clinical = clinical_base[: self.clinical_dim]
        else:
            extra = self.rng.normal(size=(self.clinical_dim - 4,)).astype(np.float32)
            clinical = np.concatenate([clinical_base, extra])

        n_pheno = max(self.num_phenotype_classes, 2)
        if scar_burden > 0.08:
            phenotype = 1  # MINF-like
        else:
            phenotype = int(self.rng.integers(0, n_pheno))
            if phenotype == 1:
                phenotype = 0
        time = float(self.rng.uniform(0.5, 5.0) * (0.55 if mace else 1.0))
        event = mace

        # Synthetic ED/ES: first and mid frame (not silent t//2 as sole GT).
        ed_idx = 0
        es_idx = max(0, t // 2)
        seg_seq = torch.full((t, d, h, w), -1, dtype=torch.long)
        seg_t = torch.from_numpy(seg)
        seg_seq[ed_idx] = seg_t
        seg_seq[es_idx] = seg_t  # same mask for smoke; real ACDC differs
        valid = torch.zeros(t, dtype=torch.bool)
        valid[ed_idx] = True
        valid[es_idx] = True

        sample = {
            "volume": torch.from_numpy(volume),
            "segmentation": seg_t,
            "segmentation_sequence": seg_seq,
            "seg_valid_mask": valid,
            "seg_frame_indices": torch.tensor([ed_idx, es_idx], dtype=torch.long),
            "ed_index": torch.tensor(ed_idx, dtype=torch.long),
            "es_index": torch.tensor(es_idx, dtype=torch.long),
            "spacing": torch.tensor([10.0, 1.5, 1.5], dtype=torch.float32),  # synthetic mm
            "mace": torch.tensor(mace, dtype=torch.float32),
            "case_id": torch.tensor(idx),
            "patient_id": f"synth_{idx:04d}",
            "phenotype": torch.tensor(phenotype, dtype=torch.long),
            "minf": torch.tensor(1.0 if phenotype == 1 else 0.0, dtype=torch.float32),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
        }
        if self.clinical_dim > 0:
            sample["clinical"] = torch.from_numpy(clinical)
        if self.train:
            clin = sample.get("clinical")
            vol, seg_t_out, clin = apply_train_transforms(sample["volume"], sample["segmentation"], clin)
            sample["volume"] = vol
            sample["segmentation"] = seg_t_out
            sample["segmentation_sequence"] = sample["segmentation_sequence"].clone()
            sample["segmentation_sequence"][ed_idx] = seg_t_out
            sample["segmentation_sequence"][es_idx] = seg_t_out
            if clin is not None:
                sample["clinical"] = clin
        else:
            sample["volume"] = normalize_intensity(sample["volume"])
        if self.slice_sampling:
            sparse, mask, target = apply_slice_sampling_4d(sample["volume"], **self.slice_kwargs)
            sample["volume"] = sparse
            sample["slice_mask"] = mask
            sample["volume_target"] = target
        return sample


class CardiacAMI4DDataset(Dataset):
    """
    Real AMI cases from a manifest JSON + NIfTI layout.

    Expected per case directory:
        cine.nii.gz          — 4D (D,H,W,T) or 5D with channel
        seg.nii.gz           — 3D int labels (optional)
        clinical.json        — optional tabular features
        label.json           — {"mace": 0|1}
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path = "manifest.json",
        num_frames: int | None = None,
        spatial_size: tuple[int, int, int] | None = None,
        clinical_dim: int = 0,
        train: bool = True,
        slice_sampling: bool = False,
        slice_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        with open(self.root / manifest, encoding="utf-8") as f:
            self.entries = json.load(f)
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.clinical_dim = clinical_dim
        self.train = train
        self.slice_sampling = slice_sampling
        self.slice_kwargs = slice_kwargs or {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.entries[idx]
        case_dir = self.root / entry["case_id"]
        img = load_nifti(case_dir / entry.get("volume", "cine.nii.gz")).astype(np.float32)

        if img.ndim == 4:
            d, h, w, t = img.shape
            volume = np.transpose(img, (3, 0, 1, 2))  # T,D,H,W
            volume = volume[np.newaxis, ...]  # C,T,D,H,W
        elif img.ndim == 5:
            volume = np.transpose(img, (0, 4, 1, 2, 3))
        else:
            raise ValueError(f"Unexpected volume shape {img.shape}")

        if self.num_frames is not None:
            volume = volume[:, : self.num_frames]

        seg_path = case_dir / entry.get("segmentation", "seg.nii.gz")
        if seg_path.exists():
            seg = load_nifti(seg_path).astype(np.int64)
        else:
            seg = np.zeros(volume.shape[-3:], dtype=np.int64)

        label_path = case_dir / entry.get("label", "label.json")
        with open(label_path, encoding="utf-8") as f:
            label_obj = json.load(f)
            mace = float(label_obj.get("mace", 0))
            time = float(label_obj.get("time", 5.0))
            event = float(label_obj.get("event", mace))
            phenotype = int(label_obj.get("phenotype", 0))

        clinical = torch.zeros(self.clinical_dim)
        clin_path = case_dir / entry.get("clinical", "clinical.json")
        if clin_path.exists() and self.clinical_dim > 0:
            with open(clin_path, encoding="utf-8") as f:
                feats = json.load(f)
                if isinstance(feats, dict):
                    feats = [feats[k] for k in sorted(feats)]
                clinical = torch.tensor(feats[: self.clinical_dim], dtype=torch.float32)

        volume_t = torch.from_numpy(volume.copy())
        seg_t = torch.from_numpy(seg.copy())
        if self.train:
            volume_t, seg_t, clinical = apply_train_transforms(volume_t, seg_t, clinical)
        else:
            volume_t = normalize_intensity(volume_t)

        sample = {
            "volume": volume_t,
            "segmentation": seg_t,
            "mace": torch.tensor(mace, dtype=torch.float32),
            "clinical": clinical,
            "case_id": torch.tensor(idx),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "phenotype": torch.tensor(phenotype, dtype=torch.long),
        }
        if self.slice_sampling:
            sparse, mask, target = apply_slice_sampling_4d(sample["volume"], **self.slice_kwargs)
            sample["volume"] = sparse
            sample["slice_mask"] = mask
            sample["volume_target"] = target
        return sample


def _spatial_size(data_cfg: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(data_cfg.get("spatial_size", [16, 32, 32]))  # type: ignore[return-value]


def _collate_with_patient_id(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate that keeps ``patient_id`` as a list of strings."""
    from torch.utils.data._utils.collate import default_collate

    ids = [b.get("patient_id") for b in batch]
    cleaned = []
    for b in batch:
        bb = {k: v for k, v in b.items() if k != "patient_id"}
        cleaned.append(bb)
    out = default_collate(cleaned)
    if any(x is not None for x in ids):
        out["patient_id"] = ids
    return out


def build_dataloader(cfg: dict[str, Any], split: str = "train") -> DataLoader:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    batch_size = data_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 0)
    train = split == "train"
    source = data_cfg.get("source", "synthetic")
    slice_on = bool(data_cfg.get("slice_sampling", False))
    slice_kwargs = slice_sampling_config(data_cfg) if slice_on else {}
    clinical_dim = data_cfg.get("clinical_dim", 4)
    num_frames = data_cfg.get("num_frames", 8)
    spatial = _spatial_size(data_cfg)
    num_seg = model_cfg.get("num_seg_classes", data_cfg.get("num_seg_classes", NUM_SEG_CLASSES))

    fold_path = None
    try:
        from reconseg3d.data.splits import resolve_fold_path

        fold_path = resolve_fold_path(cfg)
    except Exception:
        fold_path = None

    if source == "synthetic":
        allow_fake = bool(data_cfg.get("allow_fake_data", data_cfg.get("auto_fake", True)))
        if not allow_fake:
            raise RuntimeError(
                "Publication config forbids fake data (allow_fake_data=false) but "
                "data.source is 'synthetic'. Use a real public root (acdc/mmwhs/emidec/mms) "
                "or a smoke config with allow_fake_data: true."
            )
        n = data_cfg.get("train_samples" if train else "val_samples", 32 if train else 8)
        ds: Dataset = SyntheticCardiacDataset(
            num_samples=n,
            in_channels=data_cfg.get("in_channels", 1),
            num_frames=num_frames,
            spatial_size=spatial,
            num_seg_classes=num_seg,
            clinical_dim=clinical_dim,
            seed=data_cfg.get("seed", 42) + (0 if train else 1),
            train=train,
            slice_sampling=slice_on,
            slice_kwargs=slice_kwargs,
            num_phenotype_classes=model_cfg.get("num_phenotype_classes", 5),
        )
    elif source == "acdc":
        from reconseg3d.data.acdc import ACDCDataset, discover_acdc_patients, make_fake_acdc

        root = Path(data_cfg.get("root", "data/acdc"))
        allow_fake = bool(data_cfg.get("allow_fake_data", data_cfg.get("auto_fake", True)))
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if not discover_acdc_patients(root):
            if not allow_fake:
                raise RuntimeError(
                    f"Publication config forbids fake data (allow_fake_data=false) but "
                    f"no ACDC patients found under {root}."
                )
            if auto_fake:
                logger.warning("No ACDC patients in %s; creating fake layout (auto_fake=true)", root)
                make_fake_acdc(root, n_patients=int(data_cfg.get("fake_n_patients", 8)), spatial=spatial, n_frames=num_frames)
        if fold_path is not None:
            from reconseg3d.data.splits import assert_fold_train_nonempty, load_fold_file

            fold_data = load_fold_file(fold_path)
            assert_fold_train_nonempty(
                fold_data, path=fold_path, allow_fake_data=allow_fake
            )
        ds = ACDCDataset(
            root=root,
            num_frames=num_frames,
            spatial_size=spatial,
            clinical_dim=clinical_dim,
            train=train,
            split=split,
            slice_sampling=slice_on,
            slice_kwargs=slice_kwargs,
            num_seg_classes=num_seg,
            fold_file=fold_path,
        )
    elif source in ("mms", "mnms", "m&ms"):
        from reconseg3d.data.mms import MMsDataset, MMS_DOCS, discover_mms_patients, make_fake_mms

        root = Path(data_cfg.get("root", "data/mms"))
        allow_fake = bool(data_cfg.get("allow_fake_data", data_cfg.get("auto_fake", True)))
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if not discover_mms_patients(root):
            if not allow_fake:
                raise RuntimeError(
                    f"Publication config forbids fake data (allow_fake_data=false) but "
                    f"no M&Ms patients found under {root}.\n{MMS_DOCS}"
                )
            if auto_fake:
                logger.warning("No M&Ms patients in %s; creating fake layout (auto_fake=true)", root)
                make_fake_mms(root, n_patients=int(data_cfg.get("fake_n_patients", 8)), spatial=spatial, n_frames=num_frames)
        ds = MMsDataset(
            root=root,
            num_frames=num_frames,
            spatial_size=spatial,
            clinical_dim=clinical_dim,
            train=train,
            split=split,
            slice_sampling=slice_on,
            slice_kwargs=slice_kwargs,
            num_seg_classes=num_seg,
        )
    elif source == "mmwhs":
        from reconseg3d.data.mmwhs import MMWHSDataset, discover_mmwhs_cases, make_fake_mmwhs

        root = Path(data_cfg.get("root", "data/mmwhs"))
        allow_fake = bool(data_cfg.get("allow_fake_data", data_cfg.get("auto_fake", True)))
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if not discover_mmwhs_cases(root):
            if not allow_fake:
                raise RuntimeError(
                    f"Publication config forbids fake data (allow_fake_data=false) but "
                    f"no MM-WHS cases found under {root}."
                )
            if auto_fake:
                logger.warning("No MM-WHS cases in %s; creating fake layout (auto_fake=true)", root)
                make_fake_mmwhs(root, n_cases=int(data_cfg.get("fake_n_cases", 4)), spatial=spatial)
        ds = MMWHSDataset(
            root=root,
            num_frames=num_frames,
            spatial_size=spatial,
            clinical_dim=clinical_dim,
            train=train,
            split=split,
            slice_sampling=slice_on,
            slice_kwargs=slice_kwargs,
        )
    elif source == "emidec":
        from reconseg3d.data.emidec import EMIDECDataset, discover_emidec_cases, make_fake_emidec

        root = Path(data_cfg.get("root", "data/emidec"))
        allow_fake = bool(data_cfg.get("allow_fake_data", data_cfg.get("auto_fake", True)))
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if not discover_emidec_cases(root):
            if not allow_fake:
                raise RuntimeError(
                    f"Publication config forbids fake data (allow_fake_data=false) but "
                    f"no EMIDEC cases found under {root}."
                )
            if auto_fake:
                logger.warning("No EMIDEC cases in %s; creating fake layout (auto_fake=true)", root)
                make_fake_emidec(root, n_cases=int(data_cfg.get("fake_n_cases", 4)), spatial=spatial)
        ds = EMIDECDataset(
            root=root,
            num_frames=num_frames,
            spatial_size=spatial,
            clinical_dim=clinical_dim,
            train=train,
            split=split,
            stack_scar_channel=bool(data_cfg.get("stack_scar_channel", False)),
            slice_sampling=slice_on,
            slice_kwargs=slice_kwargs,
        )
    else:
        root = data_cfg.get("root", "data/ami")
        manifest = data_cfg.get(f"{split}_manifest", f"{split}_manifest.json")
        ds = CardiacAMI4DDataset(
            root=root,
            manifest=manifest,
            num_frames=num_frames,
            clinical_dim=clinical_dim,
            train=train,
            slice_sampling=slice_on,
            slice_kwargs=slice_kwargs,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_with_patient_id,
    )
