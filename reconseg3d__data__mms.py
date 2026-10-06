"""
M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease) cardiac MRI loader stub.

Official challenge: https://www.ub.edu/mnms/

Expected layout (after download / unzip)::

    root/
      patient101/
        patient101_4d.nii.gz   # or vendor-specific cine naming
        Info.cfg               # optional ED/ES/Group
        ..._gt.nii.gz

This module provides discovery + hard-fail publication gates. Real multi-site
tables remain **待补充** until licensed data are mounted — never invent numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from reconseg3d.data.acdc import (
    ACDC_GROUP_NAMES,
    parse_info_cfg,
    phenotype_from_group,
)
from reconseg3d.data.io import (
    acdc_xyzt_to_ctdhw,
    load_nifti,
    resize_ctdhw,
    resize_dhw,
    save_nifti,
    scale_spacing_dhw,
    subsample_time_keep_anchors,
    xyz_to_dhw,
)
from reconseg3d.data.transforms import apply_train_transforms, normalize_intensity

logger = logging.getLogger(__name__)

MMS_DOWNLOAD_URL = "https://www.ub.edu/mnms/"
MMS_DOCS = (
    "1. Request access / download M&Ms from the challenge site "
    f"({MMS_DOWNLOAD_URL}).\n"
    "2. Unzip so each case is a folder under data.root (patient*/Case*).\n"
    "3. Set data.source: mms (or mnms) and allow_fake_data: false for publication.\n"
    "4. Tables stay 待补充 until real data are evaluated — no invented Dice/HD95."
)


def discover_mms_patients(root: str | Path) -> list[Path]:
    root = Path(root)
    patients = sorted(p for p in root.glob("patient*") if p.is_dir())
    if not patients:
        patients = sorted(p for p in root.glob("Case*") if p.is_dir())
    if not patients:
        for sub in ("training", "Training", "train"):
            d = root / sub
            if d.is_dir():
                patients = sorted(p for p in d.glob("patient*") if p.is_dir())
                if not patients:
                    patients = sorted(p for p in d.glob("Case*") if p.is_dir())
                if patients:
                    break
    return patients


def make_fake_mms(
    root: str | Path,
    n_patients: int = 4,
    spatial: tuple[int, int, int] = (16, 32, 32),
    n_frames: int = 8,
    seed: int = 0,
) -> Path:
    """Tiny M&Ms-like tree for CI only (not clinical data)."""
    from reconseg3d.data.acdc import make_fake_acdc

    # Reuse ACDC fake writer; M&Ms loader accepts the same patient*/Info.cfg layout.
    return make_fake_acdc(root, n_patients=n_patients, spatial=spatial, n_frames=n_frames, seed=seed)


class MMsDataset(Dataset):
    """External-generalization stub loader (ACDC-compatible fields)."""

    def __init__(
        self,
        root: str | Path,
        num_frames: int | None = 8,
        spatial_size: tuple[int, int, int] | None = (16, 32, 32),
        clinical_dim: int = 0,
        train: bool = True,
        split: str = "train",
        train_ratio: float = 0.75,
        slice_sampling: bool = False,
        slice_kwargs: dict[str, Any] | None = None,
        num_seg_classes: int = 4,
    ) -> None:
        self.root = Path(root)
        patients = discover_mms_patients(self.root)
        if not patients:
            raise FileNotFoundError(
                f"No M&Ms patients under {self.root}.\n{MMS_DOCS}"
            )
        n = len(patients)
        cut = max(1, min(n - 1, int(round(train_ratio * n)))) if n > 1 else 1
        if split == "train":
            self.patients = patients[:cut]
        elif split == "test":
            held = patients[cut:] or patients[-1:]
            self.patients = held[len(held) // 2 :] or held[-1:]
        else:
            held = patients[cut:] or patients[-1:]
            self.patients = held[: max(1, len(held) // 2)]
        self.num_frames = num_frames
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None
        self.clinical_dim = clinical_dim
        self.train = train
        self.slice_sampling = slice_sampling
        self.slice_kwargs = slice_kwargs or {}
        self.num_seg_classes = num_seg_classes

    def __len__(self) -> int:
        return len(self.patients)

    def _load_gt(self, case: Path, pid: str, frame_1based: int, shape: tuple[int, int, int]) -> np.ndarray:
        from reconseg3d.data.acdc import _map_acdc_labels

        candidates = [
            case / f"{pid}_frame{frame_1based:02d}_gt.nii.gz",
            case / f"{pid}_gt.nii.gz",
            case / "gt.nii.gz",
        ]
        for gt_path in candidates:
            if gt_path.exists():
                gt = load_nifti(gt_path)
                if gt.ndim == 3:
                    return _map_acdc_labels(xyz_to_dhw(gt.astype(np.int64)))
        return np.zeros(shape, dtype=np.int64)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case = self.patients[idx]
        pid = case.name
        info = parse_info_cfg(case / "Info.cfg") if (case / "Info.cfg").exists() else {}
        ed_1 = int(float(info.get("ED", 1)))
        es_1 = int(float(info.get("ES", ed_1)))
        group = info.get("Group", "NOR")
        phenotype = phenotype_from_group(group)

        four_d = case / f"{pid}_4d.nii.gz"
        spacing_dhw = (1.0, 1.0, 1.0)
        affine = np.eye(4, dtype=np.float64)
        if four_d.exists():
            meta = load_nifti(four_d, with_meta=True)
            volume = acdc_xyzt_to_ctdhw(meta.data.astype(np.float32))
            spacing_dhw = meta.spacing_dhw
            affine = meta.affine
        else:
            # Single-frame fallback
            frame_path = next(case.glob("*frame*.nii.gz"), None)
            if frame_path is None:
                raise FileNotFoundError(f"No cine found in {case}")
            meta = load_nifti(frame_path, with_meta=True)
            sl = meta.data.astype(np.float32)
            spacing_dhw = meta.spacing_dhw
            affine = meta.affine
            dhw = xyz_to_dhw(sl) if sl.ndim == 3 else sl
            volume = dhw[np.newaxis, np.newaxis, ...]

        orig_t = int(volume.shape[1])
        ed_orig = max(0, min(ed_1 - 1, orig_t - 1))
        es_orig = max(0, min(es_1 - 1, orig_t - 1))
        seg_ed = self._load_gt(case, pid, ed_1, volume.shape[-3:])
        seg_es = self._load_gt(case, pid, es_1, volume.shape[-3:])

        volume_t = torch.from_numpy(np.ascontiguousarray(volume))
        seg_ed_t = torch.from_numpy(np.ascontiguousarray(seg_ed))
        seg_es_t = torch.from_numpy(np.ascontiguousarray(seg_es))

        selected = list(range(orig_t))
        if self.num_frames is not None:
            volume_t, selected = subsample_time_keep_anchors(
                volume_t, self.num_frames, anchors=[ed_orig, es_orig]
            )
        new_t = int(volume_t.shape[1])
        ed_idx = selected.index(ed_orig)
        es_idx = selected.index(es_orig)

        src_spatial = tuple(int(x) for x in volume_t.shape[-3:])
        if self.spatial_size is not None:
            volume_t = resize_ctdhw(volume_t, self.spatial_size)
            seg_ed_t = resize_dhw(seg_ed_t, self.spatial_size, is_label=True)
            seg_es_t = resize_dhw(seg_es_t, self.spatial_size, is_label=True)
            spacing_dhw = scale_spacing_dhw(spacing_dhw, src_spatial, self.spatial_size)

        seg_seq = torch.full((new_t, *seg_ed_t.shape), -1, dtype=torch.long)
        seg_seq[ed_idx] = seg_ed_t.long()
        seg_seq[es_idx] = seg_es_t.long()
        valid = torch.zeros(new_t, dtype=torch.bool)
        valid[ed_idx] = True
        valid[es_idx] = True

        sample: dict[str, Any] = {
            "volume": volume_t,
            "segmentation": seg_ed_t.long(),
            "segmentation_sequence": seg_seq,
            "seg_valid_mask": valid,
            "seg_frame_indices": torch.tensor([ed_idx, es_idx], dtype=torch.long),
            "ed_index": torch.tensor(ed_idx, dtype=torch.long),
            "es_index": torch.tensor(es_idx, dtype=torch.long),
            "spacing": torch.tensor(spacing_dhw, dtype=torch.float32),
            "affine": torch.from_numpy(np.asarray(affine, dtype=np.float32)),
            "mace": torch.tensor(0.0, dtype=torch.float32),
            "phenotype": torch.tensor(phenotype, dtype=torch.long),
            "time": torch.tensor(5.0, dtype=torch.float32),
            "event": torch.tensor(0.0, dtype=torch.float32),
            "case_id": torch.tensor(idx),
            "patient_id": pid,
        }
        if self.clinical_dim > 0:
            sample["clinical"] = torch.zeros(self.clinical_dim, dtype=torch.float32)

        if self.train:
            clin = sample.get("clinical")
            vol, seg_out, clin = apply_train_transforms(sample["volume"], sample["segmentation"], clin)
            sample["volume"] = vol
            sample["segmentation"] = seg_out
            sample["segmentation_sequence"] = sample["segmentation_sequence"].clone()
            sample["segmentation_sequence"][ed_idx] = seg_out
            if clin is not None:
                sample["clinical"] = clin
        else:
            sample["volume"] = normalize_intensity(sample["volume"])

        if self.slice_sampling:
            from reconseg3d.data.slice_sampling import apply_slice_sampling_4d

            skw = dict(self.slice_kwargs)
            if skw.get("trans_mm") is not None and skw.get("spacing_dhw") is None:
                skw["spacing_dhw"] = spacing_dhw
            sparse, mask, target = apply_slice_sampling_4d(sample["volume"], **skw)
            sample["volume"] = sparse
            sample["slice_mask"] = mask
            sample["volume_target"] = target
        return sample
