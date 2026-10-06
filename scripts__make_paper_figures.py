#!/usr/bin/env python3
"""SciencePlots figures from local smoke / ablation metrics (demo only)."""
from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import scienceplots  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# English Nature-style: Times New Roman; CJK fallback (SimSun/STSong) for mixed labels.
plt.style.use(["science", "nature", "no-latex"])
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "SimSun", "STSong", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
    }
)


def _read_ablation_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def save_fig(fig: plt.Figure, stem: str) -> dict[str, Path]:
    paths = {}
    for ext in ("png", "pdf", "svg"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths[ext] = p
    plt.close(fig)
    return paths


def fig1_pipeline_schematic() -> dict[str, Path]:
    """Schematic (not quantitative): geometry-constrained 4D pipeline roles."""
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    boxes = [
        (0.2, 1.35, 1.7, 1.15, "Sparse SA\ncine stack"),
        (2.2, 1.35, 1.7, 1.15, "Per-frame\n3D recon"),
        (4.2, 1.35, 2.2, 1.15, "Geometry motion\nL_inv / smooth / Jac\nloop / ED-ref"),
        (6.7, 1.35, 1.7, 1.15, "ED/ES seg\n(+ optional heads)"),
    ]
    for x, y, w, h, t in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.3, edgecolor="0.2")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)
    for x0, x1 in [(1.9, 2.2), (3.9, 4.2), (6.4, 6.7)]:
        ax.annotate("", xy=(x1, 1.9), xytext=(x0, 1.9), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(
        5.0,
        0.55,
        "Auxiliary: image-cycle (intensity) · demoted volume-curve · HeartTTable-lite = fusion ablation only",
        ha="center",
        fontsize=9,
        style="italic",
        color="0.35",
    )
    ax.set_title("Fig. 1 | Geometry-constrained 4D recon–seg–motion pipeline", pad=10)
    return save_fig(fig, "fig1_pipeline")


def fig2_recon_ablation(rows: list[dict]) -> dict[str, Path]:
    focus = ["broadcast_baseline", "per_frame_no_motion", "per_frame_motion"]
    labels = ["Broadcast", "Per-frame\n(no motion)", "Per-frame\n+ motion"]
    psnr, mae = [], []
    for name in focus:
        row = next(r for r in rows if r["run"] == name)
        psnr.append(_f(row, "recon_psnr"))
        mae.append(_f(row, "recon_mae"))
    x = np.arange(len(focus))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    axes[0].bar(x, psnr, color="#4C72B0", width=0.65)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("a  Reconstruction PSNR")
    axes[1].bar(x, mae, color="#DD8452", width=0.65)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("b  Reconstruction MAE")
    fig.suptitle(
        "Fig. 2 | Smoke ablation recon metrics (fake/demo data; not clinical)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    return save_fig(fig, "fig2_recon_ablation")


def fig3_motion_metrics(rows: list[dict], geo: dict | None = None) -> dict[str, Path]:
    """Panel a: ablation warp/cycle; panel b: geometry logs from outputs/metrics.json when present."""
    focus = ["per_frame_no_motion", "per_frame_motion", "fusion_concat", "fusion_heart_ttable"]
    labels = ["PF no-mot", "PF + mot", "Concat", "HTT-lite"]
    warp, cycle = [], []
    for name in focus:
        row = next(r for r in rows if r["run"] == name)
        warp.append(_f(row, "warp_error") or 0.0)
        cycle.append(_f(row, "cycle_error") or 0.0)
    x = np.arange(len(focus))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    axes[0].bar(x - w / 2, warp, width=w, label="Warp L1", color="#4C72B0")
    axes[0].bar(x + w / 2, cycle, width=w, label="Image-cycle (aux)", color="#55A868")
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("Error (a.u.)")
    axes[0].set_title("a  Ablation intensity proxies")
    axes[0].legend(frameon=False, fontsize=9)

    if geo and all(k in geo for k in ("inv_error", "loss_loop", "jac_neg_ratio")):
        names = [r"$L_\mathrm{inv}$", r"$L_\mathrm{loop}$", r"Jac$^{-}$ ratio"]
        vals = [float(geo["inv_error"]), float(geo["loss_loop"]), float(geo["jac_neg_ratio"])]
        axes[1].bar(np.arange(3), vals, color=["#4C72B0", "#55A868", "#C44E52"], width=0.65)
        axes[1].set_xticks(np.arange(3), names)
        axes[1].set_ylabel("Logged value")
        axes[1].set_title("b  Geometry logs (DEMO)")
    else:
        ef = [(_f(next(r for r in rows if r["run"] == name), "ef_proxy") or 0.0) for name in focus]
        axes[1].bar(x, ef, color="#C44E52", width=0.65)
        axes[1].set_xticks(x, labels, rotation=15)
        axes[1].set_ylabel("EF proxy")
        axes[1].set_title("b  EF proxy (smoke)")
    fig.suptitle(
        "Fig. 3 | Motion / geometry metrics (DEMO smoke; not clinical)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    return save_fig(fig, "fig3_motion_metrics")


def fig4_seg_dice(rows: list[dict]) -> dict[str, Path]:
    focus = ["broadcast_baseline", "per_frame_motion", "fusion_concat", "fusion_heart_ttable"]
    labels = ["Broadcast", "PF+mot", "Concat", "HTT-lite"]
    metrics = ["dice_lv", "dice_rv", "dice_myo", "dice_mean"]
    metric_labels = ["LV", "RV", "MYO", "Mean"]
    data = np.zeros((len(metrics), len(focus)))
    for j, name in enumerate(focus):
        row = next(r for r in rows if r["run"] == name)
        for i, m in enumerate(metrics):
            data[i, j] = _f(row, m) or 0.0
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.6, float(data.max())))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(metric_labels)), metric_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dice")
    ax.set_title("Fig. 4 | Segmentation Dice heatmap (smoke / fake NIfTI; not clinical)")
    fig.tight_layout()
    return save_fig(fig, "fig4_seg_dice")


def fig5_claim_boundary() -> dict[str, Path]:
    """Honest claim map vs original npj paper (qualitative)."""
    categories = [
        "Sparse→dense recon\n(public)",
        "LV/RV/MYO seg\n(public)",
        "Motion-consistent\n4D (this repo)",
        "HeartTTable-lite\nfusion ablation",
        "Private AMI\n5y MACE 0.934",
    ]
    status = [1.0, 1.0, 1.0, 0.55, 0.0]  # supported / partial / out-of-scope
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#CCB974", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    y = np.arange(len(categories))
    ax.barh(y, status, color=colors, height=0.6)
    ax.set_yticks(y, categories)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Claim support in THIS codebase (1=in-scope public/smoke)")
    ax.set_title("Fig. 5 | Innovation claim boundary vs original npj Digit. Med. paper")
    ax.axvline(0.5, color="0.5", ls="--", lw=0.8)
    for yi, s, lab in zip(y, status, ["Public", "Public", "Main novelty", "Ablation only", "Out of scope"]):
        ax.text(s + 0.02, yi, lab, va="center", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "fig5_claim_boundary")


def png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    csv_path = ROOT / "outputs" / "ablations_smoke_v2" / "table.csv"
    rows = _read_ablation_csv(csv_path)
    geo_path = ROOT / "outputs" / "metrics.json"
    geo = json.loads(geo_path.read_text(encoding="utf-8")) if geo_path.exists() else None
    paths = {}
    paths.update({f"fig1_{k}": v for k, v in fig1_pipeline_schematic().items()})
    paths.update({f"fig2_{k}": v for k, v in fig2_recon_ablation(rows).items()})
    paths.update({f"fig3_{k}": v for k, v in fig3_motion_metrics(rows, geo=geo).items()})
    paths.update({f"fig4_{k}": v for k, v in fig4_seg_dice(rows).items()})
    paths.update({f"fig5_{k}": v for k, v in fig5_claim_boundary().items()})

    meta = {
        "source_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "geometry_metrics_json": "outputs/metrics.json" if geo else None,
        "disclaimer": "All quantitative panels are smoke/demo metrics; prefer ssim_3d / recon_ssim_proxy; do not cite as clinical performance.",
        "figures": {
            "fig1": "fig1_pipeline.png",
            "fig2": "fig2_recon_ablation.png",
            "fig3": "fig3_motion_metrics.png",
            "fig4": "fig4_seg_dice.png",
            "fig5": "fig5_claim_boundary.png",
        },
        "font": "Times New Roman (+ SimSun/STSong CJK fallback)",
        "style": "science + nature + no-latex (SciencePlots)",
    }
    paper_recon = ROOT / "outputs" / "paper_recon_smoke" / "metrics.json"
    paper_acdc = ROOT / "outputs" / "paper_acdc_smoke" / "metrics.json"
    meta["paper_recon_smoke"] = json.loads(paper_recon.read_text(encoding="utf-8"))
    meta["paper_acdc_smoke"] = json.loads(paper_acdc.read_text(encoding="utf-8"))
    if geo:
        meta["geometry_smoke"] = {
            k: geo.get(k)
            for k in (
                "inv_error",
                "loss_inv",
                "loss_loop",
                "loss_jac",
                "loss_smooth",
                "jac_neg_ratio",
                "jac_det_mean",
                "recon_psnr",
                "ssim_3d",
                "recon_ssim_proxy",
            )
        }
    (OUT / "figure_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    b64_map = {stem: png_b64(OUT / name) for stem, name in meta["figures"].items()}
    (OUT / "figures_b64.json").write_text(json.dumps(b64_map), encoding="utf-8")
    print(f"Wrote figures under {OUT}")
    for k, v in sorted(paths.items()):
        if str(v).endswith(".png"):
            print(" ", v.name)


if __name__ == "__main__":
    main()
