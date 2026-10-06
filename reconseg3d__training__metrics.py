"""Evaluation metrics with safe edge-case handling."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reconseg3d.models.motion import (
    cycle_consistency_loss,
    inverse_consistency_loss,
    jacobian_stats,
    volume_curve_loss,
    warp_consistency_loss,
    warp_volume,
)

SEG_NAME = {1: "lv", 2: "rv", 3: "myo", 4: "scar"}
HD95_MAX_SURFACE = 4000


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC with guards for single-class or tiny batches."""
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_score = np.asarray(y_score).astype(np.float64).ravel()
    if y_true.size == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim != 3 or not mask.any():
        return np.zeros((0, 3), dtype=np.int32)
    pad = np.pad(mask, 1, mode="constant")
    inner = (
        pad[1:-1, 1:-1, 1:-1]
        & pad[:-2, 1:-1, 1:-1]
        & pad[2:, 1:-1, 1:-1]
        & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 2:, 1:-1]
        & pad[1:-1, 1:-1, :-2]
        & pad[1:-1, 1:-1, 2:]
    )
    surface = mask & ~inner
    if not surface.any():
        surface = mask
    return np.argwhere(surface)


def hd95_binary(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> float:
    """95th-percentile Hausdorff distance.

    When ``spacing`` is ``(sz, sy, sx)`` in mm (matching D,H,W), distances are
    physical. Default ``None`` uses voxel units. Surfaces larger than
    ``HD95_MAX_SURFACE`` are subsampled.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    ps = _surface_voxels(pred)
    gs = _surface_voxels(target)
    if len(ps) == 0 or len(gs) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if len(ps) > HD95_MAX_SURFACE:
        ps = ps[rng.choice(len(ps), HD95_MAX_SURFACE, replace=False)]
    if len(gs) > HD95_MAX_SURFACE:
        gs = gs[rng.choice(len(gs), HD95_MAX_SURFACE, replace=False)]
    scale = np.ones(3, dtype=np.float64)
    if spacing is not None:
        scale = np.asarray(spacing, dtype=np.float64).reshape(3)
    delta = (ps[:, None, :] - gs[None, :, :]).astype(np.float64) * scale.reshape(1, 1, 3)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    d_pg = dist.min(axis=1)
    d_gp = dist.min(axis=0)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Mean Dice per class; pred/target (B,D,H,W) int."""
    scores: dict[str, float] = {}
    for cls in range(num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()
        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item()
        dice = (2 * inter + eps) / (union + eps) if union > 0 else 1.0
        scores[f"dice_class_{cls}"] = dice
        name = SEG_NAME.get(cls)
        if name is not None:
            scores[f"dice_{name}"] = dice
    valid = [v for k, v in scores.items() if k.startswith("dice_class_") and (k != "dice_class_0" or num_classes == 1)]
    scores["dice_mean"] = float(np.mean(valid)) if valid else 0.0
    return scores


def hd95_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    spacing: Sequence[float] | None = None,
) -> dict[str, float]:
    """Mean HD95 over LV/RV/MYO (classes 1–3 present in ``num_classes``)."""
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[None]
        tgt_np = tgt_np[None]
    out: dict[str, float] = {}
    acc: dict[str, list[float]] = {}
    for b in range(pred_np.shape[0]):
        for cls in range(1, min(num_classes, 4)):
            val = hd95_binary(pred_np[b] == cls, tgt_np[b] == cls, spacing=spacing)
            name = SEG_NAME.get(cls, str(cls))
            acc.setdefault(name, []).append(val)
    all_vals: list[float] = []
    for name, vals in acc.items():
        finite = [v for v in vals if v == v]
        out[f"hd95_{name}"] = float(np.mean(finite)) if finite else float("nan")
        all_vals.extend(finite)
    out["hd95_mean"] = float(np.mean(all_vals)) if all_vals else float("nan")
    return out


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    ious = []
    for cls in range(1, num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    return float(np.mean(ious)) if ious else 0.0


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return 99.0
    if data_range is None:
        data_range = float((target.max() - target.min()).item()) or 1.0
    return float(10.0 * math.log10((data_range ** 2) / mse))


def ssim_global(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Global (window-free) SSIM **proxy**. Prefer ``ssim_3d`` for main tables.

    Stabilizers scale with ``data_range`` (Wang et al.). Reported as
    ``recon_ssim_proxy`` only — never aliased as ``recon_ssim``.
    """
    x = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    if data_range is None:
        data_range = float((y.max() - y.min()).item()) or 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float((num / den.clamp_min(1e-12)).item())


def _gaussian_kernel_1d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return g / g.sum().clamp_min(1e-12)


def _gaussian_kernel_3d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = _gaussian_kernel_1d(window_size, sigma, device, dtype)
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, window_size, window_size, window_size)


def ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    window_size: int = 7,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Local / windowed 3D SSIM (Wang-style) averaged over the volume.

    Accepts any tensor with trailing spatial dims ``(..., D, H, W)``; leading
    dims are flattened into the batch/channel axis for convolution. Falls back
    to ``ssim_global`` when any spatial dim is smaller than ``window_size``.
    """
    x = pred.float()
    y = target.float()
    if x.shape != y.shape:
        raise ValueError(f"ssim_3d shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim < 3:
        raise ValueError(f"ssim_3d expects at least 3D, got {tuple(x.shape)}")
    spatial = x.shape[-3:]
    leading = int(np.prod(x.shape[:-3])) if x.ndim > 3 else 1
    x5 = x.reshape(leading, 1, *spatial)
    y5 = y.reshape(leading, 1, *spatial)
    d, h, w = spatial
    ws = int(window_size)
    if min(d, h, w) < ws:
        return ssim_global(pred, target, data_range=data_range, k1=k1, k2=k2)
    if data_range is None:
        data_range = float((y5.max() - y5.min()).item()) or 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    kernel = _gaussian_kernel_3d(ws, sigma, x5.device, x5.dtype)
    pad = ws // 2

    def _conv(vol: torch.Tensor) -> torch.Tensor:
        return F.conv3d(vol, kernel, padding=pad)

    mu_x = _conv(x5)
    mu_y = _conv(y5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = _conv(x5 * x5) - mu_x2
    sigma_y2 = _conv(y5 * y5) - mu_y2
    sigma_xy = _conv(x5 * y5) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den.clamp_min(1e-12)
    return float(ssim_map.mean().item())


def chamber_voxel_counts(
    mask: torch.Tensor,
    *,
    class_index: int = 1,
) -> torch.Tensor:
    """LV (or class) voxel counts. ``mask``: (B,D,H,W) → (B,) counts."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.dtype == torch.bool:
        sel = mask.float()
    else:
        sel = (mask.long() == int(class_index)).float()
    return sel.sum(dim=(1, 2, 3))


def chamber_volume_ml(
    mask: torch.Tensor,
    spacing: Sequence[float] | torch.Tensor,
    *,
    class_index: int = 1,
) -> torch.Tensor:
    """Physical chamber volume in mL from a label mask and spacing (sz,sy,sx) mm.

    ``mask``: (B,D,H,W) int labels or bool. Returns (B,) volumes in mL
    (1 mm³ = 0.001 mL). ``spacing`` is required — without it, use
    ``chamber_voxel_counts`` / ``edv_vox`` keys instead of ``*_ml``.
    """
    if spacing is None:
        raise ValueError(
            "chamber_volume_ml requires physical spacing (sz,sy,sx) mm; "
            "use chamber_voxel_counts when spacing is unavailable"
        )
    counts = chamber_voxel_counts(mask, class_index=class_index)
    if torch.is_tensor(spacing):
        sp = spacing.detach().float().to(counts.device)
        if sp.ndim == 1:
            sp = sp.view(1, 3).expand(counts.shape[0], 3)
        elif sp.ndim == 2:
            if sp.shape[0] == 1 and counts.shape[0] > 1:
                sp = sp.expand(counts.shape[0], 3)
        else:
            raise ValueError(f"spacing must be (3,) or (B,3), got {tuple(sp.shape)}")
        voxel_ml = (sp[:, 0] * sp[:, 1] * sp[:, 2]) * 0.001
    else:
        sz, sy, sx = (float(x) for x in spacing)
        voxel_ml = sz * sy * sx * 0.001
        return counts * voxel_ml
    return counts * voxel_ml


def physical_edv_esv_ef(
    seg_ed: torch.Tensor,
    seg_es: torch.Tensor,
    spacing: Sequence[float] | torch.Tensor | None,
    *,
    lv_index: int = 1,
) -> dict[str, float]:
    """EDV/ESV and EF from ED/ES label volumes.

    With physical ``spacing`` (sz,sy,sx) mm → ``edv_ml`` / ``esv_ml`` / ``ef_percent``.
    Without spacing → ``edv_vox`` / ``esv_vox`` / ``ef_proxy`` only (never ``*_ml``).
    """
    if spacing is None:
        edv = chamber_voxel_counts(seg_ed, class_index=lv_index)
        esv = chamber_voxel_counts(seg_es, class_index=lv_index)
        ef = (edv - esv) / edv.clamp_min(1.0)
        return {
            "edv_vox": float(edv.mean().item()),
            "esv_vox": float(esv.mean().item()),
            "ef_proxy": float(ef.mean().item()),
        }
    edv = chamber_volume_ml(seg_ed, spacing, class_index=lv_index)
    esv = chamber_volume_ml(seg_es, spacing, class_index=lv_index)
    ef = (edv - esv) / edv.clamp_min(1e-6) * 100.0
    return {
        "edv_ml": float(edv.mean().item()),
        "esv_ml": float(esv.mean().item()),
        "ef_percent": float(ef.mean().item()),
    }


def ef_proxy_from_seg_sequence(seg_seq: torch.Tensor, lv_index: int = 1) -> float:
    """Voxel-count EF **proxy** (max−min)/max over T. Not physical mL/% EF.

    Prefer ``physical_edv_esv_ef`` with ED/ES phases + spacing for tables.
    ``seg_seq``: (B,K,T,D,H,W) logits or (B,T,D,H,W) labels.
    """
    if seg_seq.ndim == 6:
        labels = seg_seq.argmax(dim=1)
    else:
        labels = seg_seq
    counts = (labels == lv_index).float().sum(dim=(2, 3, 4))
    edv = counts.max(dim=1).values
    esv = counts.min(dim=1).values
    ef = (edv - esv) / edv.clamp_min(1.0)
    return float(ef.mean().item())


def aggregate_case_metrics(
    case_rows: list[dict[str, Any]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Aggregate per-case metrics → mean/SD + bootstrap 95% CI.

    Returns ``(summary, bootstrap_ci)`` where summary has ``{key}_mean`` /
    ``{key}_std`` and bootstrap_ci has ``{key}: {mean, lo, hi, n}``.
    """
    if not case_rows:
        return {}, {}
    keys = sorted(set().union(*(r.keys() for r in case_rows)) - {"case_id", "patient_id"})
    rng = np.random.default_rng(seed)
    summary: dict[str, float] = {"n_cases": float(len(case_rows))}
    boot: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.asarray(
            [float(r[key]) for r in case_rows if key in r and isinstance(r[key], (int, float)) and r[key] == r[key]],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, rng=rng)
        boot[key] = {"mean": mean, "lo": lo, "hi": hi, "n": float(vals.size)}
    return summary, boot


def weighted_mean_metrics(
    batch_metrics: list[dict[str, float]],
    batch_sizes: list[int],
) -> dict[str, float]:
    """Sample-weighted mean of per-batch metric dicts (patient-level friendly)."""
    if not batch_metrics:
        return {}
    keys = set().union(*(m.keys() for m in batch_metrics))
    out: dict[str, float] = {}
    total_n = float(sum(max(int(n), 0) for n in batch_sizes)) or 1.0
    for key in keys:
        acc = 0.0
        wsum = 0.0
        for m, n in zip(batch_metrics, batch_sizes):
            if key not in m:
                continue
            v = m[key]
            if isinstance(v, float) and v == v:
                w = float(max(int(n), 0))
                acc += v * w
                wsum += w
        if wsum > 0:
            out[key] = acc / wsum
    out["_n_samples"] = total_n
    return out


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell C-index. NaN if no comparable pairs."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()
    n = risk.size
    conc = 0.0
    total = 0.0
    for i in range(n):
        if event[i] <= 0:
            continue
        for j in range(n):
            if time[i] >= time[j]:
                continue
            total += 1.0
            if risk[i] > risk[j]:
                conc += 1.0
            elif risk[i] == risk[j]:
                conc += 0.5
    if total <= 0:
        return float("nan")
    return float(conc / total)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean and (1-alpha) CI. Returns (mean, lo, hi). Tiny-n safe."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = rng.choice(values, size=n, replace=True).mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def mace_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach()).cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred_bin = (probs >= threshold).astype(np.int32)
    tp = int(((pred_bin == 1) & (y == 1)).sum())
    tn = int(((pred_bin == 0) & (y == 0)).sum())
    fp = int(((pred_bin == 1) & (y == 0)).sum())
    fn = int(((pred_bin == 0) & (y == 1)).sum())
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / max(len(y), 1)
    return {
        "mace_auc": _safe_auc(y, probs),
        "mace_sensitivity": float(sens),
        "mace_specificity": float(spec),
        "mace_accuracy": float(acc),
    }


def compose_flows_between(
    flow_fwd: torch.Tensor,
    t_src: int,
    t_tgt: int,
    *,
    n_frames: int | None = None,
) -> torch.Tensor:
    """
    Compose adjacent forward flows from frame ``t_src`` to ``t_tgt``.

    ``flow_fwd``: closed ``(B, 3, T, ...)`` or legacy ``(B, 3, T-1, ...)``.
    Closing edge is not used on linear paths.
    """
    from reconseg3d.models.motion import compose_flow_between_indices

    if t_tgt < t_src:
        raise ValueError("compose_flows_between requires t_tgt >= t_src; use flow_bwd for reverse")
    return compose_flow_between_indices(flow_fwd, t_src, t_tgt, flow_bwd=None, n_frames=n_frames)


def propagate_label_with_flow(
    label: torch.Tensor,
    flow: torch.Tensor,
    *,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Warp an integer label volume with a pull displacement (nearest via one-hot)."""
    if label.ndim != 4:
        raise ValueError(f"label must be (B,D,H,W), got {tuple(label.shape)}")
    k = int(num_classes) if num_classes is not None else int(label.max().item()) + 1
    one_hot = F.one_hot(label.long().clamp_min(0), num_classes=max(k, 1)).permute(0, 4, 1, 2, 3).float()
    warped = warp_volume(one_hot, flow)
    return warped.argmax(dim=1)


def _batch_spacing(
    spacing: Sequence[float] | torch.Tensor | None,
    batch_idx: int,
) -> Sequence[float] | None:
    if spacing is None:
        return None
    if torch.is_tensor(spacing):
        t = spacing
        if t.ndim == 1:
            return tuple(float(x) for x in t.detach().cpu().tolist())
        if t.ndim == 2:
            return tuple(float(x) for x in t[batch_idx].detach().cpu().tolist())
        raise ValueError(f"spacing tensor must be (3,) or (B,3), got {tuple(t.shape)}")
    return spacing


def ed_es_label_propagation_metrics(
    seg_ed: torch.Tensor,
    seg_es: torch.Tensor,
    flow_fwd: torch.Tensor,
    ed_index: torch.Tensor | int,
    es_index: torch.Tensor | int,
    flow_bwd: torch.Tensor | None = None,
    num_classes: int = 4,
    spacing: Sequence[float] | torch.Tensor | None = None,
    compute_hd95: bool = True,
) -> dict[str, float]:
    """
    Warp ED GT mask to ES (and reverse) with predicted flows; report Dice (+ HD95).

    **Per-patient:** never reuse ``ed_index.reshape(-1)[0]`` for the whole batch.
    Each sample ``b`` uses its own ED/ES; patient-level scores are then averaged.

    Real ACDC subject-level tables remain **待补充** until licensed data are mounted;
    this API is unit-tested with synthetic known warps.
    """
    from reconseg3d.models.motion import compose_flow_between_indices

    if seg_ed.ndim == 3:
        seg_ed = seg_ed.unsqueeze(0)
        seg_es = seg_es.unsqueeze(0)
    b = int(seg_ed.shape[0])

    if isinstance(ed_index, int):
        ed_t = torch.full((b,), int(ed_index), dtype=torch.long, device=flow_fwd.device)
    else:
        ed_t = ed_index.reshape(-1).long().to(flow_fwd.device)
        if ed_t.numel() == 1 and b > 1:
            ed_t = ed_t.expand(b)
        if ed_t.numel() != b:
            raise ValueError(f"ed_index length {ed_t.numel()} != batch {b}")

    if isinstance(es_index, int):
        es_t = torch.full((b,), int(es_index), dtype=torch.long, device=flow_fwd.device)
    else:
        es_t = es_index.reshape(-1).long().to(flow_fwd.device)  # type: ignore[union-attr]
        if es_t.numel() == 1 and b > 1:
            es_t = es_t.expand(b)
        if es_t.numel() != b:
            raise ValueError(f"es_index length {es_t.numel()} != batch {b}")

    n_pairs = int(flow_fwd.shape[2])
    max_idx = int(max(int(ed_t.max().item()), int(es_t.max().item())))
    # Closed: n_pairs == T; open legacy: n_pairs == T-1 (max frame can equal n_pairs).
    n_frames = n_pairs + 1 if max_idx >= n_pairs else n_pairs

    patient_scores: list[dict[str, float]] = []
    for bi in range(b):
        ed_i = int(ed_t[bi].item())
        es_i = int(es_t[bi].item())
        fwd_b = flow_fwd[bi : bi + 1]
        bwd_b = flow_bwd[bi : bi + 1] if flow_bwd is not None else None
        seg_ed_b = seg_ed[bi : bi + 1].to(flow_fwd.device)
        seg_es_b = seg_es[bi : bi + 1].to(flow_fwd.device)
        sp = _batch_spacing(spacing, bi)

        flow_ed_to_es = compose_flow_between_indices(
            fwd_b, ed_i, es_i, flow_bwd=bwd_b, n_frames=n_frames
        )
        prop_es = propagate_label_with_flow(seg_ed_b, flow_ed_to_es, num_classes=num_classes)
        dice_fwd = dice_per_class(prop_es, seg_es_b.long(), num_classes)
        row: dict[str, float] = {"prop_ed2es_dice_mean": dice_fwd["dice_mean"]}
        for name in ("lv", "rv", "myo"):
            key = f"dice_{name}"
            if key in dice_fwd:
                row[f"prop_ed2es_{key}"] = dice_fwd[key]

        if bwd_b is not None and es_i != ed_i:
            flow_es_to_ed = compose_flow_between_indices(
                fwd_b, es_i, ed_i, flow_bwd=bwd_b, n_frames=n_frames
            )
            prop_ed = propagate_label_with_flow(seg_es_b, flow_es_to_ed, num_classes=num_classes)
            dice_bwd = dice_per_class(prop_ed, seg_ed_b.long(), num_classes)
            row["prop_es2ed_dice_mean"] = dice_bwd["dice_mean"]

        if compute_hd95:
            try:
                hd = hd95_per_class(prop_es, seg_es_b.long(), num_classes, spacing=sp)
                row["prop_ed2es_hd95_mean"] = hd["hd95_mean"]
            except Exception:
                row["prop_ed2es_hd95_mean"] = float("nan")
        patient_scores.append(row)

    out: dict[str, float] = {}
    keys = set().union(*(r.keys() for r in patient_scores)) if patient_scores else set()
    for key in keys:
        vals = [r[key] for r in patient_scores if key in r and r[key] == r[key]]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def compute_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_classes: int = 5,
    compute_hd95: bool = True,
    spacing: Sequence[float] | torch.Tensor | None = None,
) -> dict[str, float]:
    seg_pred = outputs["segmentation"].argmax(dim=1)
    seg_tgt = batch["segmentation"].long()
    if seg_tgt.device != seg_pred.device:
        seg_tgt = seg_tgt.to(seg_pred.device)

    if spacing is None and "spacing" in batch:
        spacing = batch["spacing"]

    metrics: dict[str, Any] = dice_per_class(seg_pred, seg_tgt, num_classes)
    metrics["iou_foreground"] = iou_per_class(seg_pred, seg_tgt, num_classes)
    if compute_hd95:
        try:
            # Per-sample spacing when (B,3); else shared.
            if torch.is_tensor(spacing) and spacing.ndim == 2 and spacing.shape[0] > 1:
                hd_acc: dict[str, list[float]] = {}
                for bi in range(seg_pred.shape[0]):
                    sp = _batch_spacing(spacing, bi)
                    part = hd95_per_class(seg_pred[bi : bi + 1], seg_tgt[bi : bi + 1], num_classes, spacing=sp)
                    for k, v in part.items():
                        hd_acc.setdefault(k, []).append(v)
                for k, vals in hd_acc.items():
                    finite = [v for v in vals if v == v]
                    metrics[k] = float(np.mean(finite)) if finite else float("nan")
            else:
                sp = _batch_spacing(spacing, 0) if spacing is not None else None
                metrics.update(hd95_per_class(seg_pred, seg_tgt, num_classes, spacing=sp))
        except Exception:
            metrics["hd95_mean"] = float("nan")

    recon = outputs.get("reconstruction")
    vol = batch.get("volume_target", batch["volume"])
    if recon is not None:
        vol = vol.to(device=recon.device, dtype=recon.dtype)
        metrics["recon_mae"] = F.l1_loss(recon, vol, reduction="mean").item()
        metrics["recon_psnr"] = psnr(recon, vol)
        # Global SSIM proxy only — never alias as recon_ssim.
        metrics["recon_ssim_proxy"] = ssim_global(recon, vol)
        try:
            metrics["ssim_3d"] = ssim_3d(recon, vol)
        except Exception:
            metrics["ssim_3d"] = float("nan")

    if "mace_logits" in outputs and "mace" in batch:
        metrics.update(mace_metrics(outputs["mace_logits"], batch["mace"].to(outputs["mace_logits"].device)))

    if "mace_logits" in outputs and "time" in batch and "event" in batch:
        risk = outputs["mace_logits"].detach().cpu().numpy()
        metrics["c_index"] = concordance_index(
            risk,
            batch["time"].detach().cpu().numpy(),
            batch["event"].detach().cpu().numpy(),
        )

    if "phenotype_logits" in outputs and "phenotype" in batch:
        logits_p = outputs["phenotype_logits"]
        pred_p = logits_p.argmax(dim=1).detach().cpu().numpy()
        y_p = batch["phenotype"].detach().cpu().numpy()
        metrics["phenotype_acc"] = float((pred_p == y_p).mean()) if y_p.size else float("nan")
        if logits_p.shape[1] > 1:
            probs = torch.softmax(logits_p.detach(), dim=-1)[:, 1].cpu().numpy()
            y_minf = (y_p == 1).astype(np.int32)
            metrics["phenotype_auc"] = _safe_auc(y_minf, probs)
        else:
            metrics["phenotype_auc"] = float("nan")

    seg_seq = outputs.get("seg_sequence")
    if seg_seq is not None:
        # Labeled proxy only — not physical EF.
        metrics["ef_proxy"] = ef_proxy_from_seg_sequence(seg_seq)
        try:
            metrics["vol_curve"] = float(volume_curve_loss(seg_seq).item())
        except Exception:
            metrics["vol_curve"] = float("nan")

    # Physical EDV/ESV/EF (mL / %) when ED/ES labels + spacing are available.
    if (
        "ed_index" in batch
        and "es_index" in batch
        and "segmentation_sequence" in batch
    ):
        try:
            seg_seq_gt = batch["segmentation_sequence"]
            if seg_seq_gt.ndim == 5:
                bsz = seg_seq_gt.shape[0]
                ed_t = batch["ed_index"].reshape(-1).long()
                es_t = batch["es_index"].reshape(-1).long()
                if ed_t.numel() == 1 and bsz > 1:
                    ed_t = ed_t.expand(bsz)
                if es_t.numel() == 1 and bsz > 1:
                    es_t = es_t.expand(bsz)
                seg_ed_vol = torch.stack(
                    [seg_seq_gt[i, int(ed_t[i])] for i in range(bsz)], dim=0
                )
                seg_es_vol = torch.stack(
                    [seg_seq_gt[i, int(es_t[i])] for i in range(bsz)], dim=0
                )
                # Prefer model ED/ES predictions when per-frame seg is present.
                if seg_seq is not None and seg_seq.ndim == 6:
                    pred_labels = seg_seq.argmax(dim=1)
                    pred_ed = torch.stack(
                        [pred_labels[i, int(ed_t[i])] for i in range(bsz)], dim=0
                    )
                    pred_es = torch.stack(
                        [pred_labels[i, int(es_t[i])] for i in range(bsz)], dim=0
                    )
                    phys = physical_edv_esv_ef(pred_ed, pred_es, spacing, lv_index=1)
                else:
                    phys = physical_edv_esv_ef(seg_ed_vol, seg_es_vol, spacing, lv_index=1)
                metrics.update(phys)
                # GT reference volumes (for bias tables when available).
                gt_phys = physical_edv_esv_ef(seg_ed_vol, seg_es_vol, spacing, lv_index=1)
                if "edv_ml" in gt_phys:
                    metrics["edv_ml_gt"] = gt_phys["edv_ml"]
                    metrics["esv_ml_gt"] = gt_phys["esv_ml"]
                    metrics["ef_percent_gt"] = gt_phys["ef_percent"]
                else:
                    metrics["edv_vox_gt"] = gt_phys["edv_vox"]
                    metrics["esv_vox_gt"] = gt_phys["esv_vox"]
                    metrics["ef_proxy_gt"] = gt_phys["ef_proxy"]
        except Exception:
            metrics["ef_percent"] = float("nan")

    flow = outputs.get("flow")
    flow_bwd = outputs.get("flow_bwd")
    if recon is not None and flow is not None and recon.shape[2] > 1:
        try:
            metrics["warp_error"] = float(warp_consistency_loss(recon, flow, flow_bwd).item())
        except Exception:
            metrics["warp_error"] = float("nan")
        if flow_bwd is not None:
            try:
                metrics["cycle_error"] = float(cycle_consistency_loss(recon, flow, flow_bwd).item())
            except Exception:
                metrics["cycle_error"] = float("nan")
            try:
                metrics["inv_error"] = float(inverse_consistency_loss(flow, flow_bwd).item())
            except Exception:
                metrics["inv_error"] = float("nan")
        try:
            metrics.update(jacobian_stats(flow))
        except Exception:
            metrics["jac_neg_ratio"] = float("nan")

    if (
        flow is not None
        and "ed_index" in batch
        and "es_index" in batch
        and "segmentation_sequence" in batch
    ):
        try:
            seg_seq_gt = batch["segmentation_sequence"]
            if seg_seq_gt.ndim == 5:
                # Per-patient ED/ES gather (never reshape(-1)[0] for whole batch).
                bsz = seg_seq_gt.shape[0]
                ed_t = batch["ed_index"].reshape(-1).long()
                es_t = batch["es_index"].reshape(-1).long()
                if ed_t.numel() == 1 and bsz > 1:
                    ed_t = ed_t.expand(bsz)
                if es_t.numel() == 1 and bsz > 1:
                    es_t = es_t.expand(bsz)
                seg_ed = torch.stack([seg_seq_gt[i, int(ed_t[i])] for i in range(bsz)], dim=0).to(flow.device)
                seg_es = torch.stack([seg_seq_gt[i, int(es_t[i])] for i in range(bsz)], dim=0).to(flow.device)
                prop = ed_es_label_propagation_metrics(
                    seg_ed,
                    seg_es,
                    flow,
                    ed_t,
                    es_t,
                    flow_bwd=flow_bwd,
                    num_classes=num_classes,
                    spacing=spacing,
                    compute_hd95=compute_hd95,
                )
                metrics.update(prop)
        except Exception:
            metrics["prop_ed2es_dice_mean"] = float("nan")

    for k, v in list(metrics.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics[k] = float("nan")
    return metrics
