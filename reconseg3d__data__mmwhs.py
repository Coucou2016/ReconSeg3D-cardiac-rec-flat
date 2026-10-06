"""
MM-WHS MRI loader (whole-heart labels).

Official label IDs (MRI)::

    500 LV cavity, 600 RV cavity, 205 LV myocardium (LVM),
    420 LA, 550 RA, 820 aorta, 850 pulmonary artery.

This loader maps LV/RV/LVM -> {1, 2, 3} (paper 4-class mode with background).
If data is absent, call ``make_fake_mmwhs(root)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from reconseg3d.data.io import load_nifti, resize_ctdhw, resize_dhw, save_nifti, xyz_to_dhw
from reconseg3d.data.transforms import apply_train_transforms, normalize_intensity

MMWHS_LV = 500
MMWHS_RV = 600
MMWHS_LVM = 205

MMWHS_TO_REPO = {
    MMWHS_LV: 1,
    MMWHS_RV: 2,
    MMWHS_LVM: 3,
}


def map_mmwhs_labels(seg: np.ndarray) -> np.ndarray:
    out = np.zeros(seg.shape, dtype=np.int64)
    for src, dst in MMWHS_TO_REPO.items():
        out[seg == src] = dst
    return out


def discover_mmwhs_cases(root: str | Path) -> list[tuple[Path, Path]]:
    """Return (image, label) path pairs."""
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []
    patterns = [
        "*_image.nii.gz",
        "*_image.nii",
        "mr_train_*_image.nii.gz",
    ]
    images: list[Path] = []
    for pat in patterns:
        images.extend(root.glob(pat))
        images.extend(root.glob(f"**/{pat}"))
    seen: set[Path] = set()
    for img in sorted(set(images)):
        if img in seen:
            continue
        stem = img.name.replace("_image.nii.gz", "").replace("_image.nii", "")
        label = img.with_name(stem + "_label.nii.gz")
        if not label.exists():
            label = img.with_name(stem + "_label.nii")
        if label.exists():
            pairs.append((img, label))
            seen.add(img)
    return pairs


def make_fake_mmwhs(
    root: str | Path,
    n_cases: int = 2,
    spatial: tuple[int, int, int] = (16, 32, 32),
    seed: int = 0,
) -> Path:
    """Write MM-WHS-like ``mr_train_*_image/label.nii.gz`` files (X,Y,Z)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d, h, w = spatial
    zz, yy, xx = np.ogrid[:d, :h, :w]
    for i in range(n_cases):
        cid = 1001 + i
        cy, cx = h * 0.5, w * 0.5
        lv = ((zz - d * 0.5) / (d * 0.22)) ** 2 + ((yy - cy) / (h * 0.18)) ** 2 + ((xx - cx) / (w * 0.18)) ** 2 <= 1.0
        lvm = ((zz - d * 0.5) / (d * 0.3)) ** 2 + ((yy - cy) / (h * 0.26)) ** 2 + ((xx - cx) / (w * 0.26)) ** 2 <= 1.0
        rv = ((zz - d * 0.5) / (d * 0.2)) ** 2 + ((yy - (cy + h * 0.16)) / (h * 0.12)) ** 2 + (
            (xx - cx) / (w * 0.12)
        ) ** 2 <= 1.0
        label_dhw = np.zeros((d, h, w), dtype=np.int16)
        label_dhw[lvm] = MMWHS_LVM
        label_dhw[lv] = MMWHS_LV
        label_dhw[rv] = MMWHS_RV
        img_dhw = (label_dhw > 0).astype(np.float32)
        img_dhw = img_dhw + 0.3 * (label_dhw == MMWHS_LV) + rng.normal(0, 0.02, img_dhw.shape).astype(np.float32)
        img_xyz = np.transpose(img_dhw, (1, 2, 0))
        lab_xyz = np.transpose(label_dhw, (1, 2, 0))
        save_nifti(root / f"mr_train_{cid}_image.nii.gz", img_xyz)
        save_nifti(root / f"mr_train_{cid}_label.nii.gz", lab_xyz)
    return root


class MMWHSDataset(Dataset):
    """3D whole-heart MRI; optionally tiled along T to satisfy 4D models."""

    def __init__(
        self,
        root: str | Path,
        num_frames: int = 1,
        spatial_size: tuple[int, int, int] | None = (16, 32, 32),
        clinical_dim: int = 0,
        train: bool = True,
        split: str = "train",
        train_ratio: float = 0.75,
        slice_sampling: bool = False,
        slice_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        pairs = discover_mmwhs_cases(self.root)
        if not pairs:
            raise FileNotFoundError(
                f"No MM-WHS *_image.nii.gz in {self.root}. "
                "Download MM-WHS MRI or call make_fake_mmwhs(root)."
            )
        n = len(pairs)
        cut = max(1, min(n - 1, int(round(train_ratio * n)))) if n > 1 else 1
        self.pairs = pairs[:cut] if split == "train" else pairs[cut:] or pairs[-1:]
        self.num_frames = max(1, int(num_frames))
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None
        self.clinical_dim = clinical_dim
        self.train = train
        self.slice_sampling = slice_sampling
        self.slice_kwargs = slice_kwargs or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, lab_path = self.pairs[idx]
        img = load_nifti(img_path).astype(np.float32)
        lab = load_nifti(lab_path)
        if img.ndim != 3:
            raise ValueError(f"Expected 3D MM-WHS image, got {img.shape}")
        dhw = xyz_to_dhw(img)
        seg = map_mmwhs_labels(xyz_to_dhw(lab.astype(np.int64)))
        volume = torch.from_numpy(np.ascontiguousarray(dhw[np.newaxis, np.newaxis, ...]))
        if self.num_frames > 1:
            volume = volume.repeat(1, self.num_frames, 1, 1, 1)
        seg_t = torch.from_numpy(np.ascontiguousarray(seg))
        if self.spatial_size is not None:
            volume = resize_ctdhw(volume, self.spatial_size)
            seg_t = resize_dhw(seg_t, self.spatial_size, is_label=True)

        sample: dict[str, torch.Tensor] = {
            "volume": volume,
            "segmentation": seg_t.long(),
            "mace": torch.tensor(0.0, dtype=torch.float32),
            "phenotype": torch.tensor(0, dtype=torch.long),
            "time": torch.tensor(5.0, dtype=torch.float32),
            "event": torch.tensor(0.0, dtype=torch.float32),
            "case_id": torch.tensor(idx),
        }
        if self.clinical_dim > 0:
            sample["clinical"] = torch.zeros(self.clinical_dim, dtype=torch.float32)

        if self.train:
            clin = sample.get("clinical")
            vol, seg_out, clin = apply_train_transforms(sample["volume"], sample["segmentation"], clin)
            sample["volume"] = vol
            sample["segmentation"] = seg_out
            if clin is not None:
                sample["clinical"] = clin
        else:
            sample["volume"] = normalize_intensity(sample["volume"])

        if self.slice_sampling:
            from reconseg3d.data.slice_sampling import apply_slice_sampling_4d

            sparse, mask, target = apply_slice_sampling_4d(sample["volume"], **self.slice_kwargs)
            sample["volume"] = sparse
            sample["slice_mask"] = mask
            sample["volume_target"] = target
        return sample
