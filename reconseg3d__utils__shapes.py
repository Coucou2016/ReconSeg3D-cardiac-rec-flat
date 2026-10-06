"""Tensor shape contracts for 4D cardiac sequences."""

from __future__ import annotations

import torch

# Canonical layout: (B, C, T, D, H, W)
VOLUME_DIMS = 6
EXPECTED_LAYOUT = "BCTDHW"


def assert_volume_shape(
    x: torch.Tensor,
    *,
    min_t: int = 1,
    min_spatial: int = 4,
    in_channels: int | None = None,
) -> None:
    """Validate input volume is (B, C, T, D, H, W)."""
    if x.ndim != VOLUME_DIMS:
        raise ValueError(
            f"Volume must be 6D (B,C,T,D,H,W); got ndim={x.ndim}, shape={tuple(x.shape)}. "
            f"Convert (B,C,D,H,W,T) with x = x.permute(0, 1, 5, 2, 3, 4)."
        )
    b, c, t, d, h, w = x.shape
    if b < 1 or c < 1:
        raise ValueError(f"Invalid batch/channel: shape={tuple(x.shape)}")
    if t < min_t:
        raise ValueError(f"T (frames) must be >= {min_t}, got {t}")
    for name, size in zip("DHW", (d, h, w)):
        if size < min_spatial:
            raise ValueError(f"{name} must be >= {min_spatial}, got {size}")
    if in_channels is not None and c != in_channels:
        raise ValueError(f"Expected in_channels={in_channels}, got C={c}")


def validate_batch(
    batch: dict[str, torch.Tensor],
    *,
    in_channels: int = 1,
    clinical_dim: int = 0,
) -> None:
    """Validate dataloader batch keys and shapes."""
    required = ("volume", "segmentation", "mace")
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(f"Batch missing keys: {missing}")

    vol = batch["volume"]
    assert_volume_shape(vol, in_channels=in_channels)
    b, _, _, d, h, w = vol.shape

    seg = batch["segmentation"]
    if seg.ndim != 4 or seg.shape != (b, d, h, w):
        raise ValueError(f"segmentation must be (B,D,H,W); got {tuple(seg.shape)}")

    mace = batch["mace"]
    if mace.ndim != 1 or mace.shape[0] != b:
        raise ValueError(f"mace must be (B,); got {tuple(mace.shape)}")

    if clinical_dim > 0:
        clinical = batch.get("clinical")
        if clinical is None:
            raise ValueError("clinical_dim > 0 but batch has no 'clinical' tensor")
        if clinical.shape != (b, clinical_dim):
            raise ValueError(f"clinical must be (B,{clinical_dim}); got {tuple(clinical.shape)}")


def permute_dhw_t_to_ctdhw(x: torch.Tensor) -> torch.Tensor:
    """(B, C, D, H, W, T) -> (B, C, T, D, H, W)."""
    if x.ndim != 6:
        raise ValueError(f"Expected 6D tensor, got {x.ndim}D")
    return x.permute(0, 1, 5, 2, 3, 4).contiguous()
