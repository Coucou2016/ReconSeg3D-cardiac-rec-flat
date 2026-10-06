"""
Baseline adapters for fair small-resolution comparisons.

- Reconstruction-only / no-motion: use ReconSeg3D with predict_motion=False
- Compact VoxelMorph-style: in-repo lightweight CNN registration
- FlowReg: external adapter interface (do not vendor; see docs/BASELINES.md)

No invented clinical numbers — run only when weights/data are available.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from reconseg3d.models.blocks import ConvBlock3D
from reconseg3d.models.motion import MotionNet, scaling_and_squaring, warp_volume


class ReconOnlyBaseline(nn.Module):
    """Thin wrapper: reconstruction without motion (predict_motion=False)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        if hasattr(model, "predict_motion"):
            model.predict_motion = False

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)


class CompactVoxelMorph(nn.Module):
    """
    Compact VoxelMorph-style pairwise registration for small-res fair compare.

    Input: source/target volumes (B,1,D,H,W). Output: pull displacement (B,3,D,H,W)
    and optionally warped source. Not a full VoxelMorph reimplementation —
    intended for smoke / small-grid ablations only.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        use_svf: bool = False,
        svf_steps: int = 7,
    ) -> None:
        super().__init__()
        self.motion = MotionNet(
            in_channels=in_channels,
            base_channels=base_channels,
            use_svf=use_svf,
            svf_steps=svf_steps,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flow = self.motion.forward_pair(source, target)
        warped = warp_volume(source, flow)
        return {"flow": flow, "warped": warped}


class FlowRegAdapter:
    """
    Adapter interface for https://github.com/mathpluscode/FlowReg

    This repo does **not** vendor FlowReg. Implement ``load`` / ``predict`` after
    installing FlowReg in a separate env, then plug into comparison scripts.
    """

    repo_url = "https://github.com/mathpluscode/FlowReg"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def available(self) -> bool:
        try:
            import flowreg  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def load(self) -> None:
        if not self.available():
            raise ImportError(
                "FlowReg is not installed. Clone "
                f"{self.repo_url} and install per its README, then retry."
            )
        raise NotImplementedError(
            "FlowReg weights/API differ by release — wire your local install here "
            "without claiming numbers until a real eval run completes."
        )

    def predict(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._model is None:
            self.load()
        raise NotImplementedError("Implement after FlowReg is installed.")


def build_baseline(name: str, cfg: dict[str, Any] | None = None) -> nn.Module | FlowRegAdapter:
    """Factory: ``recon_only`` | ``voxelmorph`` | ``flowreg``."""
    cfg = cfg or {}
    model_cfg = cfg.get("model", cfg)
    name = name.lower().replace("-", "_")
    if name in ("recon_only", "no_motion", "reconstruction_only"):
        from reconseg3d.models.reconseg3d import build_model

        local = dict(cfg)
        local.setdefault("model", dict(model_cfg))
        local["model"] = {**local["model"], "predict_motion": False, "task": "reconstruction"}
        return ReconOnlyBaseline(build_model(local))
    if name in ("voxelmorph", "voxel_morph", "compact_voxelmorph"):
        return CompactVoxelMorph(
            in_channels=int(model_cfg.get("in_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 8)),
            use_svf=bool(model_cfg.get("use_svf", False)),
            svf_steps=int(model_cfg.get("svf_steps", 7)),
        )
    if name in ("flowreg", "flow_reg"):
        return FlowRegAdapter(checkpoint=model_cfg.get("checkpoint"))
    raise ValueError(f"Unknown baseline: {name}")
