"""Demo data prep + metrics summarization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from reconseg3d.data.acdc import discover_acdc_patients
from reconseg3d.data.emidec import discover_emidec_cases
from reconseg3d.data.mmwhs import discover_mmwhs_cases

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_demo_data(tmp_path: Path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_demo_data import prepare_demo_data

    out = prepare_demo_data(
        data_root=tmp_path / "data",
        n_acdc=4,
        n_mmwhs=2,
        n_emidec=2,
        spatial=(8, 16, 16),
        n_frames=4,
    )
    assert len(discover_acdc_patients(out["acdc"])) == 4
    assert len(discover_mmwhs_cases(out["mmwhs"])) == 2
    assert len(discover_emidec_cases(out["emidec"])) == 2
    # idempotent without force
    prepare_demo_data(data_root=tmp_path / "data", force=False)
    assert len(discover_acdc_patients(out["acdc"])) == 4


def test_summarize_runs(tmp_path: Path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from summarize_runs import collect_runs, to_csv, to_markdown

    run_a = tmp_path / "broadcast_baseline"
    run_b = tmp_path / "per_frame_motion"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "metrics.json").write_text(
        json.dumps({"run_name": "broadcast_baseline", "recon_psnr": 20.5, "dice_lv": 0.7}),
        encoding="utf-8",
    )
    (run_b / "metrics.json").write_text(
        json.dumps(
            {
                "run_name": "per_frame_motion",
                "recon_psnr": 22.1,
                "warp_error": 0.1,
                "cycle_error": 0.05,
                "ef_proxy": 0.4,
            }
        ),
        encoding="utf-8",
    )
    rows = collect_runs(tmp_path)
    assert len(rows) == 2
    md = to_markdown(rows)
    assert "broadcast_baseline" in md
    assert "recon_psnr" in md
    csv_path = tmp_path / "table.csv"
    to_csv(rows, csv_path)
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "per_frame_motion" in text
