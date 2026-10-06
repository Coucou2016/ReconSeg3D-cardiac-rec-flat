#!/usr/bin/env python
"""Create fake ACDC / MM-WHS / EMIDEC trees under data/ for local paper experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconseg3d.data.acdc import discover_acdc_patients, make_fake_acdc
from reconseg3d.data.emidec import discover_emidec_cases, make_fake_emidec
from reconseg3d.data.mmwhs import discover_mmwhs_cases, make_fake_mmwhs


def prepare_demo_data(
    data_root: str | Path = "data",
    force: bool = False,
    n_acdc: int = 8,
    n_mmwhs: int = 4,
    n_emidec: int = 4,
    spatial: tuple[int, int, int] = (16, 32, 32),
    n_frames: int = 8,
) -> dict[str, Path]:
    """Write demo layouts; skip existing trees unless ``force``."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    acdc = data_root / "acdc"
    if force or not discover_acdc_patients(acdc):
        make_fake_acdc(acdc, n_patients=n_acdc, spatial=spatial, n_frames=n_frames)
        print(f"wrote fake ACDC -> {acdc} ({n_acdc} patients)")
    else:
        print(f"keep existing ACDC -> {acdc} ({len(discover_acdc_patients(acdc))} patients)")
    out["acdc"] = acdc

    mmwhs = data_root / "mmwhs"
    if force or not discover_mmwhs_cases(mmwhs):
        make_fake_mmwhs(mmwhs, n_cases=n_mmwhs, spatial=spatial)
        print(f"wrote fake MM-WHS -> {mmwhs} ({n_mmwhs} cases)")
    else:
        print(f"keep existing MM-WHS -> {mmwhs} ({len(discover_mmwhs_cases(mmwhs))} cases)")
    out["mmwhs"] = mmwhs

    emidec = data_root / "emidec"
    if force or not discover_emidec_cases(emidec):
        make_fake_emidec(emidec, n_cases=n_emidec, spatial=spatial)
        print(f"wrote fake EMIDEC -> {emidec} ({n_emidec} cases)")
    else:
        print(f"keep existing EMIDEC -> {emidec} ({len(discover_emidec_cases(emidec))} cases)")
    out["emidec"] = emidec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fake demo cardiac datasets")
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    parser.add_argument("--force", action="store_true", help="Overwrite even if patients exist")
    parser.add_argument("--n-acdc", type=int, default=8)
    parser.add_argument("--n-mmwhs", type=int, default=4)
    parser.add_argument("--n-emidec", type=int, default=4)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--w", type=int, default=32)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    prepare_demo_data(
        data_root=args.data_root,
        force=args.force,
        n_acdc=args.n_acdc,
        n_mmwhs=args.n_mmwhs,
        n_emidec=args.n_emidec,
        spatial=(args.d, args.h, args.w),
        n_frames=args.frames,
    )


if __name__ == "__main__":
    main()
