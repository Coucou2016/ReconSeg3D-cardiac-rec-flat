"""NIfTI helpers shared by ACDC / MM-WHS / EMIDEC loaders."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

# NIfTI-1 datatype codes used by the numpy fallback (no nibabel required for tests).
_NIFTI_DT_INT16 = 4
_NIFTI_DT_INT32 = 8
_NIFTI_DT_FLOAT32 = 16
_NIFTI_DT_FLOAT64 = 64
_NIFTI_DT_UINT8 = 2


@dataclass(frozen=True)
class NiftiVolume:
    """Array plus affine / voxel spacing.

    ``spacing_xyz`` is the NIfTI pixdim order ``(sx, sy, sz)`` in mm.
    Use ``spacing_dhw`` after ``xyz_to_dhw`` / ACDC layout conversion.
    """

    data: np.ndarray
    affine: np.ndarray
    spacing_xyz: tuple[float, float, float]

    @property
    def spacing_dhw(self) -> tuple[float, float, float]:
        """Spacing for (D=Z, H=X, W=Y) layout: ``(sz, sx, sy)``."""
        sx, sy, sz = self.spacing_xyz
        return (float(sz), float(sx), float(sy))


def _open_maybe_gzip(path: Path, mode: str):
    if path.suffix == ".gz" or path.name.endswith(".nii.gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def _numpy_save_nifti(path: Path, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype("<f4", copy=False)
        datatype, bitpix = _NIFTI_DT_FLOAT32, 32
    elif arr.dtype == np.uint8:
        arr = arr.astype("<u1", copy=False)
        datatype, bitpix = _NIFTI_DT_UINT8, 8
    else:
        arr = arr.astype("<i2", copy=False)
        datatype, bitpix = _NIFTI_DT_INT16, 16
    ndim = arr.ndim
    if ndim < 1 or ndim > 7:
        raise ValueError(f"NIfTI fallback supports 1–7D, got ndim={ndim}")
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<h", header, 40, ndim)
    for i, size in enumerate(arr.shape):
        struct.pack_into("<h", header, 42 + i * 2, int(size))
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    for i in range(8):
        struct.pack_into("<f", header, 76 + i * 4, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[344:348] = b"n+1\x00"
    payload = bytes(header) + b"\x00\x00\x00\x00" + arr.tobytes(order="F")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_maybe_gzip(path, "wb") as f:
        f.write(payload)


def _numpy_load_nifti_meta(path: Path) -> NiftiVolume:
    with _open_maybe_gzip(path, "rb") as f:
        raw = f.read()
    if len(raw) < 352:
        raise ValueError(f"File too small to be NIfTI: {path}")
    ndim = struct.unpack_from("<h", raw, 40)[0]
    shape = tuple(struct.unpack_from("<h", raw, 42 + i * 2)[0] for i in range(max(int(ndim), 1)))
    datatype = struct.unpack_from("<h", raw, 70)[0]
    vox_offset = int(struct.unpack_from("<f", raw, 108)[0])
    pixdim = tuple(struct.unpack_from("<f", raw, 76 + i * 4)[0] for i in range(8))
    sx = float(pixdim[1]) if abs(pixdim[1]) > 1e-8 else 1.0
    sy = float(pixdim[2]) if abs(pixdim[2]) > 1e-8 else 1.0
    sz = float(pixdim[3]) if abs(pixdim[3]) > 1e-8 else 1.0
    dtype_map = {
        _NIFTI_DT_UINT8: np.dtype("<u1"),
        _NIFTI_DT_INT16: np.dtype("<i2"),
        _NIFTI_DT_INT32: np.dtype("<i4"),
        _NIFTI_DT_FLOAT32: np.dtype("<f4"),
        _NIFTI_DT_FLOAT64: np.dtype("<f8"),
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    data = np.frombuffer(raw, dtype=dtype_map[datatype], offset=max(vox_offset, 352))
    arr = np.reshape(data[: int(np.prod(shape))], shape, order="F").copy()
    affine = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
    return NiftiVolume(data=arr, affine=affine, spacing_xyz=(sx, sy, sz))


def load_nifti(path: str | Path, *, with_meta: bool = False) -> np.ndarray | NiftiVolume:
    """Load NIfTI data.

    Default returns ``np.ndarray`` (backward compatible). Pass ``with_meta=True``
    for a ``NiftiVolume`` with affine and ``spacing_xyz``.
    """
    path = Path(path)
    try:
        import nibabel as nib

        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        affine = np.asarray(img.affine, dtype=np.float64)
        zooms = img.header.get_zooms()
        sx = float(zooms[0]) if len(zooms) > 0 else 1.0
        sy = float(zooms[1]) if len(zooms) > 1 else 1.0
        sz = float(zooms[2]) if len(zooms) > 2 else 1.0
        vol = NiftiVolume(data=data, affine=affine, spacing_xyz=(sx, sy, sz))
    except ImportError:
        vol = _numpy_load_nifti_meta(path)
    return vol if with_meta else vol.data


def save_nifti(path: str | Path, array: np.ndarray, affine: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib

        img = nib.Nifti1Image(np.asarray(array), np.eye(4) if affine is None else affine)
        nib.save(img, str(path))
    except ImportError:
        _numpy_save_nifti(path, np.asarray(array))


def acdc_xyzt_to_ctdhw(arr: np.ndarray) -> np.ndarray:
    """ACDC-style (X, Y, Z, T) -> (C=1, T, D=Z, H=X, W=Y)."""
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (X,Y,Z,T), got shape {arr.shape}")
    vol = np.transpose(arr, (3, 2, 0, 1))
    return vol[np.newaxis, ...].astype(np.float32, copy=False)


def xyz_to_dhw(arr: np.ndarray) -> np.ndarray:
    """(X, Y, Z) -> (D=Z, H=X, W=Y)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D (X,Y,Z), got shape {arr.shape}")
    return np.transpose(arr, (2, 0, 1))


def scale_spacing_dhw(
    spacing: Sequence[float],
    src_size: Sequence[int],
    dst_size: Sequence[int],
) -> tuple[float, float, float]:
    """Scale (sz, sy, sx) when resizing (D, H, W)."""
    if len(spacing) != 3 or len(src_size) != 3 or len(dst_size) != 3:
        raise ValueError("spacing/src/dst must each have length 3")
    out = []
    for s, a, b in zip(spacing, src_size, dst_size):
        out.append(float(s) * (float(a) / float(max(b, 1))))
    return (out[0], out[1], out[2])


def resize_ctdhw(
    volume: torch.Tensor,
    spatial_size: tuple[int, int, int],
    *,
    is_label: bool = False,
) -> torch.Tensor:
    """Resize (C, T, D, H, W) spatial dims to ``spatial_size`` (D, H, W)."""
    if volume.ndim != 5:
        raise ValueError(f"Expected (C,T,D,H,W), got {tuple(volume.shape)}")
    d, h, w = spatial_size
    if tuple(volume.shape[-3:]) == (d, h, w):
        return volume
    c, t, _, _, _ = volume.shape
    flat = volume.reshape(c * t, 1, *volume.shape[-3:])
    if is_label:
        flat = F.interpolate(flat.float(), size=(d, h, w), mode="nearest")
        return flat.reshape(c, t, d, h, w).to(dtype=volume.dtype)
    flat = F.interpolate(flat, size=(d, h, w), mode="trilinear", align_corners=False)
    return flat.reshape(c, t, d, h, w)


def resize_dhw(
    vol: torch.Tensor,
    spatial_size: tuple[int, int, int],
    *,
    is_label: bool = False,
) -> torch.Tensor:
    """Resize (D, H, W) to ``spatial_size``."""
    if vol.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {tuple(vol.shape)}")
    x = vol.view(1, 1, *vol.shape).float()
    d, h, w = spatial_size
    if is_label:
        out = F.interpolate(x, size=(d, h, w), mode="nearest")
        return out.view(d, h, w).to(dtype=vol.dtype)
    out = F.interpolate(x, size=(d, h, w), mode="trilinear", align_corners=False)
    return out.view(d, h, w).to(dtype=vol.dtype)


def select_time_indices_keep_anchors(
    t: int,
    num_frames: int,
    anchors: Sequence[int] | None = None,
) -> list[int]:
    """
    Choose ``num_frames`` indices in ``[0, t)`` that **always include** ``anchors``.

    Preferred when GPU allows: keep full ``T`` and use a temporal mask instead.
    This path is the fallback when subsampling is required: original ED/ES (or
    other anchors) are forced into the selected set, then remaining slots are
    filled from a uniform linspace grid (deduped).
    """
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    if num_frames >= t:
        return list(range(t))

    anchor_set: list[int] = []
    for a in anchors or ():
        ai = int(a)
        if ai < 0 or ai >= t:
            raise ValueError(f"anchor {ai} out of range for T={t}")
        if ai not in anchor_set:
            anchor_set.append(ai)
    if len(anchor_set) > num_frames:
        raise ValueError(f"More unique anchors ({len(anchor_set)}) than num_frames ({num_frames})")

    grid = torch.linspace(0, t - 1, num_frames).round().long().tolist()
    selected: list[int] = []
    for idx in grid:
        ii = int(max(0, min(idx, t - 1)))
        if ii not in selected:
            selected.append(ii)
    for a in anchor_set:
        if a not in selected:
            selected.append(a)

    # Drop non-anchors until length == num_frames (prefer keeping anchors).
    while len(selected) > num_frames:
        drop_i = next((i for i, v in enumerate(selected) if v not in anchor_set), None)
        if drop_i is None:
            break
        selected.pop(drop_i)

    # Top up if linspace collisions left us short.
    if len(selected) < num_frames:
        for i in range(t):
            if i not in selected:
                selected.append(i)
            if len(selected) >= num_frames:
                break

    selected = sorted(selected[:num_frames])
    for a in anchor_set:
        if a not in selected:
            # Replace nearest non-anchor slot
            non_anchor = [i for i, v in enumerate(selected) if v not in anchor_set]
            if not non_anchor:
                raise RuntimeError("Cannot place anchors into selected indices")
            replace_at = min(non_anchor, key=lambda i: abs(selected[i] - a))
            selected[replace_at] = a
            selected = sorted(selected)
    return selected


def subsample_time_keep_anchors(
    volume: torch.Tensor,
    num_frames: int | None,
    anchors: Sequence[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """
    Subsample T of (C, T, D, H, W) while preserving ``anchors``.

    Returns ``(subsampled_volume, selected_indices)`` where indices refer to the
    original timeline. ``ed_idx`` / ``es_idx`` in the new sequence are
    ``selected_indices.index(orig_ed)`` etc.
    """
    if num_frames is None or volume.shape[1] == num_frames:
        return volume, list(range(volume.shape[1]))
    t = int(volume.shape[1])
    idx = select_time_indices_keep_anchors(t, num_frames, anchors=anchors)
    return volume[:, idx], idx


def subsample_time(volume: torch.Tensor, num_frames: int | None) -> torch.Tensor:
    """Uniformly subsample T of (C, T, D, H, W) to ``num_frames`` (legacy).

    Prefer ``subsample_time_keep_anchors`` when ED/ES must stay exact frames.
    """
    if num_frames is None or volume.shape[1] == num_frames:
        return volume
    t = volume.shape[1]
    if t < num_frames:
        idx = torch.linspace(0, t - 1, num_frames).round().long().clamp(0, t - 1)
        return volume[:, idx]
    idx = torch.linspace(0, t - 1, num_frames).round().long()
    return volume[:, idx]


def remap_frame_index(orig_idx: int, orig_t: int, new_t: int) -> int:
    """Map a 0-based frame index after uniform temporal subsampling (legacy).

    Deprecated for ED/ES: use ``select_time_indices_keep_anchors`` instead.
    """
    if orig_t <= 1 or new_t <= 1:
        return 0
    if orig_t == new_t:
        return int(max(0, min(orig_idx, new_t - 1)))
    grid = torch.linspace(0, orig_t - 1, new_t)
    return int((grid - float(orig_idx)).abs().argmin().item())
