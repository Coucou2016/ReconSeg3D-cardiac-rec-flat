"""Volume transforms for 4D cardiac tensors."""

from __future__ import annotations

import torch


def normalize_intensity(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean unit-variance per sample."""
    mean = volume.mean()
    std = volume.std().clamp_min(eps)
    return (volume - mean) / std


def random_flip_3d(
    volume: torch.Tensor,
    mask: torch.Tensor | None = None,
    p: float = 0.5,
    extra_masks: list[torch.Tensor] | None = None,
) -> tuple:
    """
    Random flip along D/H/W axes.

    When ``extra_masks`` is provided, the same flip decisions are applied to
    every mask so labeled phases (ED/ES) stay co-transformed with the volume.
    """
    vol_dims = [-3, -2, -1]
    mask_dims = [0, 1, 2]
    extras = list(extra_masks) if extra_masks is not None else None
    for vd, md in zip(vol_dims, mask_dims):
        if torch.rand(1).item() < p:
            volume = torch.flip(volume, dims=[vd])
            if mask is not None:
                mask = torch.flip(mask, dims=[md])
            if extras is not None:
                extras = [torch.flip(m, dims=[md]) for m in extras]
    if extras is not None:
        if mask is None:
            return volume, extras
        return volume, mask, extras
    if mask is None:
        return volume
    return volume, mask


def apply_train_transforms(
    volume: torch.Tensor,
    mask: torch.Tensor,
    clinical: torch.Tensor | None = None,
    extra_masks: list[torch.Tensor] | None = None,
) -> tuple:
    """
    Normalize + shared random flips.

    If ``extra_masks`` is given (e.g. ES GT), returns
    ``(volume, mask, clinical, flipped_extras)``; otherwise
    ``(volume, mask, clinical)``.
    """
    volume = normalize_intensity(volume)
    if extra_masks is not None:
        volume, mask, flipped = random_flip_3d(volume, mask, extra_masks=extra_masks)
        return volume, mask, clinical, flipped
    volume, mask = random_flip_3d(volume, mask)
    return volume, mask, clinical
