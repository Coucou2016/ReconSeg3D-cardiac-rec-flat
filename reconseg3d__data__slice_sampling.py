"""
Simulate sparse short-axis (SA) stacks from dense 3D volumes.

Paper setting (original ReconSeg3D): target grid ~256×256×128 (H×W×D),
sample S ∈ [8, 16] slices along depth, in-plane rotation 1–5°, translation
in physical mm (or legacy pixels), additive noise; unselected slices zeroed.

This module supports smaller tensors for tests and CI (e.g. 32×32×16).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

# Paper-scale spatial grid as (D, H, W) matching this repo's tensor layout.
PAPER_SPATIAL_SIZE = (128, 256, 256)
DEFAULT_S_RANGE = (8, 16)
DEFAULT_ROT_DEG = (1.0, 5.0)
DEFAULT_TRANS_PX = (1, 5)
# Physical in-plane translation range in mm (preferred for publication sims).
DEFAULT_TRANS_MM = (1.0, 5.0)


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


def mm_to_px(trans_mm: float, spacing_yx: tuple[float, float]) -> tuple[float, float]:
    """Convert physical mm translation to pixel (dy, dx) using in-plane spacing."""
    sy, sx = spacing_yx
    sy = max(float(sy), 1e-6)
    sx = max(float(sx), 1e-6)
    return float(trans_mm) / sy, float(trans_mm) / sx


def sample_sparse_sa_stack(
    volume: np.ndarray | torch.Tensor,
    *,
    s_min: int = 8,
    s_max: int = 16,
    rot_deg: tuple[float, float] = DEFAULT_ROT_DEG,
    trans_px: tuple[int, int] | None = DEFAULT_TRANS_PX,
    trans_mm: tuple[float, float] | None = None,
    spacing_dhw: Sequence[float] | None = None,
    noise_std: float = 0.02,
    sampling_ratio: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
    """
    Sample S depth slices, perturb in-plane, zero the rest.

    Physical mode (preferred for publication):
        Pass ``trans_mm`` + ``spacing_dhw=(sz,sy,sx)`` so translations are in mm.
    Legacy mode:
        ``trans_px`` integer pixel range (used when ``trans_mm`` is None).

    ``sampling_ratio`` (optional): if set in (0, 1], overrides s_min/s_max with
    ``S = max(1, round(ratio * D))``.
    """
    is_torch = isinstance(volume, torch.Tensor)
    vol_np = volume.detach().cpu().numpy() if is_torch else np.asarray(volume)
    rng = rng if rng is not None else np.random.default_rng()

    ctdhw, layout = _as_ctdhw(vol_np.astype(np.float32, copy=False))
    _c, _t, d, h, w = ctdhw.shape

    if sampling_ratio is not None:
        ratio = float(sampling_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"sampling_ratio must be in (0, 1], got {ratio}")
        s_count = max(1, int(round(ratio * d)))
        s_count = min(s_count, d)
    else:
        s_hi = max(1, min(int(s_max), d))
        s_lo = max(1, min(int(s_min), s_hi))
        s_count = int(rng.integers(s_lo, s_hi + 1))

    indices = np.sort(rng.choice(d, size=s_count, replace=False))
    mask = np.zeros(d, dtype=bool)
    mask[indices] = True

    use_mm = trans_mm is not None
    if use_mm:
        if spacing_dhw is None:
            spacing_dhw = (1.0, 1.0, 1.0)
        _sz, sy, sx = (float(x) for x in spacing_dhw)
        tr_lo, tr_hi = float(trans_mm[0]), float(trans_mm[1])
    else:
        tr_lo, tr_hi = (int(trans_px[0]), int(trans_px[1])) if trans_px else (1, 5)

    sparse = np.zeros_like(ctdhw)
    rot_lo, rot_hi = rot_deg
    for zi in indices:
        angle = float(rng.uniform(rot_lo, rot_hi))
        if rng.random() < 0.5:
            angle = -angle
        if use_mm:
            mag = float(rng.uniform(tr_lo, tr_hi))
            sign_y = 1.0 if rng.random() < 0.5 else -1.0
            sign_x = 1.0 if rng.random() < 0.5 else -1.0
            dy = sign_y * mag / max(sy, 1e-6)
            dx = sign_x * mag / max(sx, 1e-6)
        else:
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
    trans_px: tuple[int, int] | None = DEFAULT_TRANS_PX,
    trans_mm: tuple[float, float] | None = None,
    spacing_dhw: Sequence[float] | None = None,
    noise_std: float = 0.02,
    sampling_ratio: float | None = None,
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
        trans_mm=trans_mm,
        spacing_dhw=spacing_dhw,
        noise_std=noise_std,
        sampling_ratio=sampling_ratio,
        rng=rng,
    )
    assert isinstance(sparse, torch.Tensor)
    mask_t = torch.from_numpy(mask.astype(np.float32))
    return sparse, mask_t, target


def slice_sampling_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Read slice-sampling hyperparameters from a data config mapping."""
    s_range = data_cfg.get("slice_s_range", list(DEFAULT_S_RANGE))
    rot = data_cfg.get("slice_rot_deg", list(DEFAULT_ROT_DEG))
    out: dict[str, Any] = {
        "s_min": int(s_range[0]),
        "s_max": int(s_range[1]),
        "rot_deg": (float(rot[0]), float(rot[1])),
        "noise_std": float(data_cfg.get("slice_noise_std", 0.02)),
    }
    if "slice_sampling_ratio" in data_cfg and data_cfg["slice_sampling_ratio"] is not None:
        out["sampling_ratio"] = float(data_cfg["slice_sampling_ratio"])
    if "slice_trans_mm" in data_cfg and data_cfg["slice_trans_mm"] is not None:
        tr = data_cfg["slice_trans_mm"]
        out["trans_mm"] = (float(tr[0]), float(tr[1]))
        out["trans_px"] = None
        if "slice_spacing_dhw" in data_cfg:
            sp = data_cfg["slice_spacing_dhw"]
            out["spacing_dhw"] = (float(sp[0]), float(sp[1]), float(sp[2]))
    else:
        trans = data_cfg.get("slice_trans_px", list(DEFAULT_TRANS_PX))
        out["trans_px"] = (int(trans[0]), int(trans[1]))
        out["trans_mm"] = None
    return out
