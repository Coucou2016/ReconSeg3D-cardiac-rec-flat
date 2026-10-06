SOURCE PACK CHUNK 4 — continue reading next chunks before proposing patches.

## FILE: reconseg3d/data/dataset.py

```
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

        sample = {
            "volume": torch.from_numpy(volume),
            "segmentation": torch.from_numpy(seg),
            "mace": torch.tensor(mace, dtype=torch.float32),
            "case_id": torch.tensor(idx),
            "phenotype": torch.tensor(phenotype, dtype=torch.long),
            "minf": torch.tensor(1.0 if phenotype == 1 else 0.0, dtype=torch.float32),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
        }
        if self.clinical_dim > 0:
            sample["clinical"] = torch.from_numpy(clinical)
        if self.train:
            clin = sample.get("clinical")
            vol, seg_t, clin = apply_train_transforms(sample["volume"], sample["segmentation"], clin)
            sample["volume"] = vol
            sample["segmentation"] = seg_t
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

    if source == "synthetic":
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
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if auto_fake and not discover_acdc_patients(root):
            logger.warning("No ACDC patients in %s; creating fake layout (auto_fake=true)", root)
            make_fake_acdc(root, n_patients=int(data_cfg.get("fake_n_patients", 8)), spatial=spatial, n_frames=num_frames)
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
        )
    elif source == "mmwhs":
        from reconseg3d.data.mmwhs import MMWHSDataset, discover_mmwhs_cases, make_fake_mmwhs

        root = Path(data_cfg.get("root", "data/mmwhs"))
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if auto_fake and not discover_mmwhs_cases(root):
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
        auto_fake = bool(data_cfg.get("auto_fake", True))
        if auto_fake and not discover_emidec_cases(root):
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
    )

```

---
## FILE: reconseg3d/data/slice_sampling.py

```
"""
Simulate sparse short-axis (SA) stacks from dense 3D volumes.

Paper setting (original ReconSeg3D): target grid ~256×256×128 (H×W×D),
sample S ∈ [8, 16] slices along depth, in-plane rotation 1–5°, translation
1–5 px, additive noise; unselected slices are zeroed.

This module supports smaller tensors for tests and CI (e.g. 32×32×16).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Paper-scale spatial grid as (D, H, W) matching this repo's tensor layout.
PAPER_SPATIAL_SIZE = (128, 256, 256)
DEFAULT_S_RANGE = (8, 16)
DEFAULT_ROT_DEG = (1.0, 5.0)
DEFAULT_TRANS_PX = (1, 5)


def _as_ctdhw(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalize to (C, T, D, H, W) and remember the original layout tag."""
    if volume.ndim == 3:
        return volume[np.newaxis, np.newaxis, ...], "dhw"
    if volume.ndim == 4:
        return volume[:, np.newaxis, ...], "cdhw"
    if volume.ndim == 5:
        return volume, "ctdhw"
    raise ValueError(f"volume must be 3D/4D/5D, got shape {volume.shape}")


def _from_ctdhw(volume: np.ndarray, layout: str) -> np.ndarray:
    if layout == "dhw":
        return volume[0, 0]
    if layout == "cdhw":
        return volume[:, 0]
    return volume


def _bilinear_sample(img: np.ndarray, src_y: np.ndarray, src_x: np.ndarray) -> np.ndarray:
    h, w = img.shape
    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = src_y - y0
    wx = src_x - x0
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    ia = img[y0c, x0c]
    ib = img[y0c, x1c]
    ic = img[y1c, x0c]
    id_ = img[y1c, x1c]
    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx
    out = wa * ia + wb * ib + wc * ic + wd * id_
    outside = (src_y < 0) | (src_y > h - 1) | (src_x < 0) | (src_x > w - 1)
    out = np.where(outside, 0.0, out)
    return out.astype(np.float32, copy=False)


def rotate_translate_slice(
    img: np.ndarray,
    angle_deg: float,
    dy: float,
    dx: float,
) -> np.ndarray:
    """In-plane rotation (degrees) then translation (pixels) with bilinear sampling."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    yt = yy - cy - dy
    xt = xx - cx - dx
    src_y = c * yt + s * xt + cy
    src_x = -s * yt + c * xt + cx
    return _bilinear_sample(img, src_y, src_x)


def sample_sparse_sa_stack(
    volume: np.ndarray | torch.Tensor,
    *,
    s_min: int = 8,
    s_max: int = 16,
    rot_deg: tuple[float, float] = DEFAULT_ROT_DEG,
    trans_px: tuple[int, int] = DEFAULT_TRANS_PX,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
    """
    Sample S depth slices, perturb in-plane, zero the rest.

    Args:
        volume: (D,H,W), (C,D,H,W), or (C,T,D,H,W). Same slice indices for all C/T.
        s_min, s_max: S is drawn uniformly in ``[s_min, s_max]`` then clipped to D.
            If D is smaller than the paper range, S ∈ [1, D] and at least one
            slice is left empty when D > 1 and S < D.

    Returns:
        sparse: same type/shape as ``volume``
        slice_mask: (D,) bool, True on selected slices
    """
    is_torch = isinstance(volume, torch.Tensor)
    vol_np = volume.detach().cpu().numpy() if is_torch else np.asarray(volume)
    rng = rng if rng is not None else np.random.default_rng()

    ctdhw, layout = _as_ctdhw(vol_np.astype(np.float32, copy=False))
    _c, _t, d, h, w = ctdhw.shape
    s_hi = max(1, min(int(s_max), d))
    s_lo = max(1, min(int(s_min), s_hi))
    s_count = int(rng.integers(s_lo, s_hi + 1))
    indices = np.sort(rng.choice(d, size=s_count, replace=False))
    mask = np.zeros(d, dtype=bool)
    mask[indices] = True

    sparse = np.zeros_like(ctdhw)
    rot_lo, rot_hi = rot_deg
    tr_lo, tr_hi = trans_px
    for zi in indices:
        angle = float(rng.uniform(rot_lo, rot_hi))
        if rng.random() < 0.5:
            angle = -angle
        dy = float(rng.integers(tr_lo, tr_hi + 1)) * (1.0 if rng.random() < 0.5 else -1.0)
        dx = float(rng.integers(tr_lo, tr_hi + 1)) * (1.0 if rng.random() < 0.5 else -1.0)
        for ci in range(ctdhw.shape[0]):
            for ti in range(ctdhw.shape[1]):
                sl = rotate_translate_slice(ctdhw[ci, ti, zi], angle, dy, dx)
                if noise_std > 0:
                    sl = sl + rng.normal(0.0, noise_std, sl.shape).astype(np.float32)
                sparse[ci, ti, zi] = sl

    out_np = _from_ctdhw(sparse, layout)
    if is_torch:
        return torch.from_numpy(out_np.copy()).to(device=volume.device, dtype=volume.dtype), mask
    return out_np, mask


def apply_slice_sampling_4d(
    volume: torch.Tensor,
    *,
    s_min: int = 8,
    s_max: int = 16,
    rot_deg: tuple[float, float] = DEFAULT_ROT_DEG,
    trans_px: tuple[int, int] = DEFAULT_TRANS_PX,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply sparse SA sampling to (C, T, D, H, W).

    Returns:
        sparse_input, slice_mask (D,) float32, dense_target (clone of input)
    """
    if volume.ndim != 5:
        raise ValueError(f"Expected (C,T,D,H,W), got {tuple(volume.shape)}")
    target = volume.clone()
    sparse, mask = sample_sparse_sa_stack(
        volume,
        s_min=s_min,
        s_max=s_max,
        rot_deg=rot_deg,
        trans_px=trans_px,
        noise_std=noise_std,
        rng=rng,
    )
    assert isinstance(sparse, torch.Tensor)
    mask_t = torch.from_numpy(mask.astype(np.float32))
    return sparse, mask_t, target


def slice_sampling_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Read slice-sampling hyperparameters from a data config mapping."""
    s_range = data_cfg.get("slice_s_range", list(DEFAULT_S_RANGE))
    rot = data_cfg.get("slice_rot_deg", list(DEFAULT_ROT_DEG))
    trans = data_cfg.get("slice_trans_px", list(DEFAULT_TRANS_PX))
    return {
        "s_min": int(s_range[0]),
        "s_max": int(s_range[1]),
        "rot_deg": (float(rot[0]), float(rot[1])),
        "trans_px": (int(trans[0]), int(trans[1])),
        "noise_std": float(data_cfg.get("slice_noise_std", 0.02)),
    }

```


