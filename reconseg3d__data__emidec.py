"""
EMIDEC LGE / scar stub.

Expected layout (challenge-like)::

    root/Case_N/Images/Case_N.nii.gz
    root/Case_N/Contours/Case_N.nii.gz

Contour labels commonly: 0 bg, 1 myocardium, 2 LV cavity, 3 infarct, 4 MVO.
Mapped into this repo as LV=1, MYO=3, scar=4 (RV usually absent).

If data is absent, call ``make_fake_emidec(root)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from reconseg3d.data.io import load_nifti, resize_ctdhw, resize_dhw, save_nifti, xyz_to_dhw
from reconseg3d.data.transforms import apply_train_transforms, normalize_intensity

EMIDEC_MYO = 1
EMIDEC_CAVITY = 2
EMIDEC_INFARCT = 3
EMIDEC_MVO = 4


def map_emidec_labels(seg: np.ndarray) -> np.ndarray:
    """Map EMIDEC contours to repo classes: 0 bg, 1 LV, 3 MYO, 4 scar."""
    out = np.zeros(seg.shape, dtype=np.int64)
    out[seg == EMIDEC_CAVITY] = 1
    out[seg == EMIDEC_MYO] = 3
    out[(seg == EMIDEC_INFARCT) | (seg == EMIDEC_MVO)] = 4
    return out


def discover_emidec_cases(root: str | Path) -> list[Path]:
    root = Path(root)
    cases = sorted(p for p in root.glob("Case_*") if p.is_dir())
    if not cases:
        cases = sorted(p for p in root.glob("case_*") if p.is_dir())
    return cases


def make_fake_emidec(
    root: str | Path,
    n_cases: int = 2,
    spatial: tuple[int, int, int] = (16, 32, 32),
    seed: int = 0,
) -> Path:
    """Write EMIDEC-like Case_N/Images + Contours NIfTIs (X,Y,Z)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d, h, w = spatial
    zz, yy, xx = np.ogrid[:d, :h, :w]
    cy, cx = h * 0.5, w * 0.5
    for i in range(1, n_cases + 1):
        name = f"Case_{i}"
        img_dir = root / name / "Images"
        con_dir = root / name / "Contours"
        img_dir.mkdir(parents=True, exist_ok=True)
        con_dir.mkdir(parents=True, exist_ok=True)
        lv = ((zz - d * 0.5) / (d * 0.2)) ** 2 + ((yy - cy) / (h * 0.16)) ** 2 + ((xx - cx) / (w * 0.16)) ** 2 <= 1.0
        myo = ((zz - d * 0.5) / (d * 0.28)) ** 2 + ((yy - cy) / (h * 0.24)) ** 2 + ((xx - cx) / (w * 0.24)) ** 2 <= 1.0
        scar = myo & (~lv) & (xx > cx)
        mvo = scar & ((zz - d * 0.5) ** 2 + (yy - cy) ** 2 + (xx - (cx + w * 0.08)) ** 2 < (min(h, w) * 0.04) ** 2)
        lab = np.zeros((d, h, w), dtype=np.int16)
        lab[myo] = EMIDEC_MYO
        lab[lv] = EMIDEC_CAVITY
        lab[scar] = EMIDEC_INFARCT
        lab[mvo] = EMIDEC_MVO
        img = 0.4 * myo.astype(np.float32) + 1.0 * lv.astype(np.float32) + 1.4 * scar.astype(np.float32)
        img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
        save_nifti(img_dir / f"{name}.nii.gz", np.transpose(img, (1, 2, 0)))
        save_nifti(con_dir / f"{name}.nii.gz", np.transpose(lab, (1, 2, 0)))
    return root


class EMIDECDataset(Dataset):
    """LGE volume + scar labels; optional second channel = scar mask."""

    def __init__(
        self,
        root: str | Path,
        num_frames: int = 1,
        spatial_size: tuple[int, int, int] | None = (16, 32, 32),
        clinical_dim: int = 0,
        train: bool = True,
        split: str = "train",
        train_ratio: float = 0.75,
        stack_scar_channel: bool = False,
        slice_sampling: bool = False,
        slice_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        cases = discover_emidec_cases(self.root)
        if not cases:
            raise FileNotFoundError(
                f"No EMIDEC Case_* folders in {self.root}. "
                "Download EMIDEC or call make_fake_emidec(root)."
            )
        n = len(cases)
        cut = max(1, min(n - 1, int(round(train_ratio * n)))) if n > 1 else 1
        self.cases = cases[:cut] if split == "train" else cases[cut:] or cases[-1:]
        self.num_frames = max(1, int(num_frames))
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None
        self.clinical_dim = clinical_dim
        self.train = train
        self.stack_scar_channel = stack_scar_channel
        self.slice_sampling = slice_sampling
        self.slice_kwargs = slice_kwargs or {}

    def __len__(self) -> int:
        return len(self.cases)

    def _case_niftis(self, case: Path) -> tuple[Path, Path | None]:
        name = case.name
        img = case / "Images" / f"{name}.nii.gz"
        con = case / "Contours" / f"{name}.nii.gz"
        if not img.exists():
            matches = list((case / "Images").glob("*.nii*")) if (case / "Images").exists() else list(case.glob("*.nii*"))
            if not matches:
                raise FileNotFoundError(f"No EMIDEC image in {case}")
            img = matches[0]
        if not con.exists():
            matches = list((case / "Contours").glob("*.nii*")) if (case / "Contours").exists() else []
            con_path = matches[0] if matches else None
        else:
            con_path = con
        return img, con_path

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        case = self.cases[idx]
        img_path, con_path = self._case_niftis(case)
        img = load_nifti(img_path).astype(np.float32)
        dhw = xyz_to_dhw(img) if img.ndim == 3 else img
        if con_path is not None:
            lab = load_nifti(con_path)
            seg = map_emidec_labels(xyz_to_dhw(lab.astype(np.int64)) if lab.ndim == 3 else lab.astype(np.int64))
        else:
            seg = np.zeros(dhw.shape, dtype=np.int64)

        cine = torch.from_numpy(np.ascontiguousarray(dhw[np.newaxis, np.newaxis, ...]))
        if self.num_frames > 1:
            cine = cine.repeat(1, self.num_frames, 1, 1, 1)
        seg_t = torch.from_numpy(np.ascontiguousarray(seg))
        if self.spatial_size is not None:
            cine = resize_ctdhw(cine, self.spatial_size)
            seg_t = resize_dhw(seg_t, self.spatial_size, is_label=True)

        if self.stack_scar_channel:
            scar = (seg_t == 4).float().unsqueeze(0).unsqueeze(0).expand_as(cine)
            volume = torch.cat([cine, scar], dim=0)
        else:
            volume = cine

        sample: dict[str, torch.Tensor] = {
            "volume": volume,
            "segmentation": seg_t.long(),
            "mace": torch.tensor(float((seg_t == 4).any()), dtype=torch.float32),
            "phenotype": torch.tensor(1 if (seg_t == 4).any() else 0, dtype=torch.long),
            "time": torch.tensor(3.0, dtype=torch.float32),
            "event": torch.tensor(float((seg_t == 4).any()), dtype=torch.float32),
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
