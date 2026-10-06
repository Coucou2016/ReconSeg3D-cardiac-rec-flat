"""
ACDC cine-MRI loader.

Expected layout (official challenge)::

    root/patientXXX/
        Info.cfg
        patientXXX_4d.nii.gz
        patientXXX_frameXX.nii.gz
        patientXXX_frameXX_gt.nii.gz

If real data is absent, call ``make_fake_acdc(root)`` to write a matching tree
so unit tests can run without downloading the challenge archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

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

# Clinical covariates allowed in publication inputs (never phenotype / Group).
ACDC_CLINICAL_FEATURES = ("height", "weight", "nb_frame")
from reconseg3d.data.transforms import apply_train_transforms, normalize_intensity

# ACDC diagnosis groups used as phenotype labels (5-class).
ACDC_GROUPS = {
    "NOR": 0,
    "MINF": 1,
    "DCM": 2,
    "HCM": 3,
    "RV": 4,
    "ARV": 4,
}
ACDC_GROUP_NAMES = ("NOR", "MINF", "DCM", "HCM", "RV")


def parse_info_cfg(path: str | Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        info[key.strip()] = val.strip()
    return info


def phenotype_from_group(group: str) -> int:
    return ACDC_GROUPS.get(group.upper(), 0)


def discover_acdc_patients(root: str | Path) -> list[Path]:
    root = Path(root)
    patients = sorted(p for p in root.glob("patient*") if p.is_dir())
    if not patients:
        training = root / "training"
        if training.is_dir():
            patients = sorted(p for p in training.glob("patient*") if p.is_dir())
    return patients


def make_fake_acdc(
    root: str | Path,
    n_patients: int = 4,
    spatial: tuple[int, int, int] = (16, 32, 32),
    n_frames: int = 8,
    seed: int = 0,
) -> Path:
    """
    Write an ACDC-like directory tree with tiny NIfTIs.

    ``spatial`` is (D, H, W) in this repo's layout; files are stored as
    (X, Y, Z, T) = (H, W, D, T) to mimic official ACDC.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d, h, w = spatial
    for i in range(1, n_patients + 1):
        pid = f"patient{i:03d}"
        case = root / pid
        case.mkdir(parents=True, exist_ok=True)
        group = ACDC_GROUP_NAMES[(i - 1) % len(ACDC_GROUP_NAMES)]
        ed, es = 1, max(2, n_frames)
        info = (
            f"ED: {ed}\n"
            f"ES: {es}\n"
            f"Height: {160 + i}\n"
            f"Weight: {60 + i}\n"
            f"NbFrame: {n_frames}\n"
            f"Group: {group}\n"
        )
        (case / "Info.cfg").write_text(info, encoding="utf-8")

        cine = np.zeros((h, w, d, n_frames), dtype=np.float32)
        zz, yy, xx = np.ogrid[:d, :h, :w]
        cy, cx = h * 0.5, w * 0.5
        for t in range(n_frames):
            scale = 1.0 + 0.1 * np.sin(2 * np.pi * t / max(n_frames, 1))
            lv = ((zz - d * 0.5) / (d * 0.2)) ** 2 + ((yy - cy) / (h * 0.18 * scale)) ** 2 + (
                (xx - cx) / (w * 0.18 * scale)
            ) ** 2 <= 1.0
            myo = ((zz - d * 0.5) / (d * 0.28)) ** 2 + ((yy - cy) / (h * 0.26 * scale)) ** 2 + (
                (xx - cx) / (w * 0.26 * scale)
            ) ** 2 <= 1.0
            rv = ((zz - d * 0.5) / (d * 0.18)) ** 2 + ((yy - (cy + h * 0.18)) / (h * 0.12)) ** 2 + (
                (xx - cx) / (w * 0.12)
            ) ** 2 <= 1.0
            frame = myo.astype(np.float32) * 0.8 + lv.astype(np.float32) * 1.1 + rv.astype(np.float32) * 0.9
            frame += rng.normal(0, 0.02, frame.shape).astype(np.float32)
            cine[:, :, :, t] = np.transpose(frame, (1, 2, 0))  # D,H,W -> H,W,D stored as X,Y,Z

        save_nifti(case / f"{pid}_4d.nii.gz", cine)

        def _label_xyz(scale: float) -> np.ndarray:
            lv = ((zz - d * 0.5) / (d * 0.2)) ** 2 + ((yy - cy) / (h * 0.18 * scale)) ** 2 + (
                (xx - cx) / (w * 0.18 * scale)
            ) ** 2 <= 1.0
            myo = ((zz - d * 0.5) / (d * 0.28)) ** 2 + ((yy - cy) / (h * 0.26 * scale)) ** 2 + (
                (xx - cx) / (w * 0.26 * scale)
            ) ** 2 <= 1.0
            rv = ((zz - d * 0.5) / (d * 0.18)) ** 2 + ((yy - (cy + h * 0.18)) / (h * 0.12)) ** 2 + (
                (xx - cx) / (w * 0.12)
            ) ** 2 <= 1.0
            lab = np.zeros((d, h, w), dtype=np.int16)
            lab[myo] = 2  # ACDC: 1=RV, 2=MYO, 3=LV
            lab[lv] = 3
            lab[rv] = 1
            return np.transpose(lab, (1, 2, 0))  # (H, W, D)

        for frame_idx, scale in ((ed, 1.1), (es, 0.9)):
            sl = cine[:, :, :, frame_idx - 1]
            save_nifti(case / f"{pid}_frame{frame_idx:02d}.nii.gz", sl)
            save_nifti(case / f"{pid}_frame{frame_idx:02d}_gt.nii.gz", _label_xyz(scale))
    return root


def _map_acdc_labels(seg: np.ndarray) -> np.ndarray:
    """Official ACDC GT: 0 bg, 1 RV, 2 myocardium, 3 LV -> repo 0/1=LV/2=RV/3=MYO."""
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 3] = 1
    out[seg == 1] = 2
    out[seg == 2] = 3
    return out


class ACDCDataset(Dataset):
    """Load ACDC patients as (C, T, D, H, W) with ED/ES labeled phases.

    Batch fields (in addition to volume / clinical / phenotype):
        - ``segmentation``: ED GT (B,D,H,W) for backward-compatible single-frame heads
        - ``segmentation_sequence``: (T,D,H,W) with GT at ED/ES and -1 elsewhere
        - ``seg_valid_mask``: (T,) bool for labeled frames
        - ``seg_frame_indices``: (2,) [ed_idx, es_idx] in the (possibly subsampled) timeline
        - ``ed_index`` / ``es_index``: scalar 0-based indices
        - ``patient_id``: string id (collate may keep as list)
        - ``spacing``: (3,) float32 (sz, sy, sx) mm
    """

    def __init__(
        self,
        root: str | Path,
        num_frames: int | None = 8,
        spatial_size: tuple[int, int, int] | None = (16, 32, 32),
        clinical_dim: int = 3,
        train: bool = True,
        split: str = "train",
        train_ratio: float = 0.75,
        slice_sampling: bool = False,
        slice_kwargs: dict[str, Any] | None = None,
        num_seg_classes: int = 4,
        patient_ids: list[str] | None = None,
        fold_file: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        patients = discover_acdc_patients(self.root)
        if not patients:
            raise FileNotFoundError(
                f"No ACDC patient* folders in {self.root}. "
                "Download the ACDC challenge data or call make_fake_acdc(root)."
            )
        by_name = {p.name: p for p in patients}
        if fold_file is not None:
            from reconseg3d.data.splits import load_fold_file

            fold = load_fold_file(fold_file)
            key = "train" if split == "train" else ("test" if split == "test" else "val")
            ids = list(fold.get(key, []))
            if not ids and split == "test":
                # Fall back to val list if an older fold file lacks test.
                ids = list(fold.get("val", []))
            selected = [by_name[i] for i in ids if i in by_name]
            if not selected:
                raise FileNotFoundError(
                    f"Fold file {fold_file} split={split} matched 0 patients under {self.root}"
                )
            self.patients = selected
        elif patient_ids is not None:
            selected = [by_name[i] for i in patient_ids if i in by_name]
            if not selected:
                raise FileNotFoundError(f"patient_ids matched 0 patients under {self.root}")
            self.patients = selected
        else:
            n = len(patients)
            cut = max(1, min(n - 1, int(round(train_ratio * n)))) if n > 1 else 1
            if split == "train":
                self.patients = patients[:cut]
            elif split == "test":
                # Held-out tail beyond val when no fold file: last 20% (min 1).
                n_test = max(1, n - cut)
                self.patients = patients[cut:] or patients[-1:]
                # Use second half of held-out as test when enough patients.
                if n_test >= 2:
                    mid = cut + n_test // 2
                    self.patients = patients[mid:] or patients[-1:]
            else:
                # val
                held = patients[cut:] or patients[-1:]
                if len(held) >= 2:
                    self.patients = held[: max(1, len(held) // 2)]
                else:
                    self.patients = held
        self.num_frames = num_frames
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None
        self.clinical_dim = clinical_dim
        self.train = train
        self.slice_sampling = slice_sampling
        self.slice_kwargs = slice_kwargs or {}
        self.num_seg_classes = num_seg_classes

    def __len__(self) -> int:
        return len(self.patients)

    def _load_gt_frame(self, case: Path, pid: str, frame_1based: int, spatial_fallback: tuple[int, int, int]) -> np.ndarray:
        gt_path = case / f"{pid}_frame{frame_1based:02d}_gt.nii.gz"
        if gt_path.exists():
            gt = load_nifti(gt_path)
            if gt.ndim == 3:
                return _map_acdc_labels(xyz_to_dhw(gt.astype(np.int64)))
            raise ValueError(f"Unexpected GT shape {gt.shape}")
        return np.zeros(spatial_fallback, dtype=np.int64)

    @staticmethod
    def clinical_feature_names(clinical_dim: int) -> tuple[str, ...]:
        """Names of clinical input channels (never includes phenotype/Group)."""
        if clinical_dim <= 0:
            return ()
        base = ACDC_CLINICAL_FEATURES
        if clinical_dim <= len(base):
            return base[:clinical_dim]
        return base + tuple(f"pad_{i}" for i in range(clinical_dim - len(base)))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
            raw = meta.data.astype(np.float32)
            spacing_dhw = meta.spacing_dhw
            affine = meta.affine
            volume = acdc_xyzt_to_ctdhw(raw)
        else:
            frame_path = case / f"{pid}_frame{ed_1:02d}.nii.gz"
            meta = load_nifti(frame_path, with_meta=True)
            sl = meta.data.astype(np.float32)
            spacing_dhw = meta.spacing_dhw
            affine = meta.affine
            if sl.ndim == 3:
                dhw = xyz_to_dhw(sl)
                volume = dhw[np.newaxis, np.newaxis, ...]
            else:
                raise ValueError(f"Unexpected ACDC frame shape {sl.shape} in {frame_path}")

        orig_t = int(volume.shape[1])
        ed_orig = ed_1 - 1
        es_orig = es_1 - 1
        seg_ed = self._load_gt_frame(case, pid, ed_1, volume.shape[-3:])
        seg_es = self._load_gt_frame(case, pid, es_1, volume.shape[-3:])

        volume_t = torch.from_numpy(np.ascontiguousarray(volume))
        seg_ed_t = torch.from_numpy(np.ascontiguousarray(seg_ed))
        seg_es_t = torch.from_numpy(np.ascontiguousarray(seg_es))

        # Anchor-preserving temporal subsample: original ED/ES frames always kept.
        # Prefer full-T + temporal mask when GPU allows (set num_frames=None / == orig_t).
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

        # Full temporal GT: unlabeled frames = -1 (ignore_index)
        seg_seq = torch.full((new_t, *seg_ed_t.shape), -1, dtype=torch.long)
        seg_seq[ed_idx] = seg_ed_t.long()
        seg_seq[es_idx] = seg_es_t.long()
        valid = torch.zeros(new_t, dtype=torch.bool)
        valid[ed_idx] = True
        valid[es_idx] = True

        # P0-1: clinical inputs are height/weight/nb only — never phenotype/Group.
        height = float(info.get("Height", 0) or 0)
        weight = float(info.get("Weight", 0) or 0)
        nb = float(info.get("NbFrame", orig_t) or orig_t)
        clinical_base = np.array([height, weight, nb], dtype=np.float32)
        if self.clinical_dim <= 0:
            clinical = torch.zeros(0)
        elif self.clinical_dim <= 3:
            clinical = torch.from_numpy(clinical_base[: self.clinical_dim].copy())
        else:
            extra = np.zeros(self.clinical_dim - 3, dtype=np.float32)
            clinical = torch.from_numpy(np.concatenate([clinical_base, extra]))

        sample: dict[str, torch.Tensor] = {
            "volume": volume_t,
            "segmentation": seg_ed_t.long(),  # primary = ED
            "segmentation_sequence": seg_seq,
            "seg_valid_mask": valid,
            "seg_frame_indices": torch.tensor([ed_idx, es_idx], dtype=torch.long),
            "ed_index": torch.tensor(ed_idx, dtype=torch.long),
            "es_index": torch.tensor(es_idx, dtype=torch.long),
            "spacing": torch.tensor(spacing_dhw, dtype=torch.float32),
            "affine": torch.from_numpy(np.asarray(affine, dtype=np.float32)),
            "mace": torch.tensor(1.0 if phenotype == 1 else 0.0, dtype=torch.float32),
            "phenotype": torch.tensor(phenotype, dtype=torch.long),
            "minf": torch.tensor(1.0 if phenotype == 1 else 0.0, dtype=torch.float32),
            "time": torch.tensor(5.0 if phenotype != 1 else 2.0, dtype=torch.float32),
            "event": torch.tensor(1.0 if phenotype == 1 else 0.0, dtype=torch.float32),
            "case_id": torch.tensor(idx),
            "patient_id": pid,
        }
        if self.clinical_dim > 0:
            sample["clinical"] = clinical
            # Supervised targets stay outside clinical features.
            assert "phenotype" not in self.clinical_feature_names(self.clinical_dim)

        if self.train:
            clin = sample.get("clinical")
            # Co-transform every labeled phase (ED and ES), not only the ED slot.
            labeled_idxs = [i for i in range(new_t) if bool(valid[i])]
            primary_idx = ed_idx
            extra_idxs = [i for i in labeled_idxs if i != primary_idx]
            extra_masks = [sample["segmentation_sequence"][i].clone() for i in extra_idxs]
            if extra_masks:
                vol, seg_out, clin, extras = apply_train_transforms(
                    sample["volume"],
                    sample["segmentation"],
                    clin,
                    extra_masks=extra_masks,
                )
            else:
                vol, seg_out, clin = apply_train_transforms(
                    sample["volume"], sample["segmentation"], clin
                )
                extras = []
            sample["volume"] = vol
            sample["segmentation"] = seg_out
            sample["segmentation_sequence"] = sample["segmentation_sequence"].clone()
            sample["segmentation_sequence"][primary_idx] = seg_out
            for i, m in zip(extra_idxs, extras):
                sample["segmentation_sequence"][i] = m.long()
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
