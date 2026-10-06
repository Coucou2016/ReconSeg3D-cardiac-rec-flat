"""Eval case CSV must be true per-patient rows (not duplicated batch means)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_script", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_slice_batch_keeps_patient_ids():
    mod = _load_eval_module()
    batch = {
        "volume": torch.randn(2, 1, 4, 8, 8, 8),
        "patient_id": ["pA", "pB"],
        "spacing": torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
        "ed_index": torch.tensor([0, 1]),
    }
    s0 = mod._slice_batch(batch, 0, 2)
    s1 = mod._slice_batch(batch, 1, 2)
    assert s0["patient_id"] == ["pA"]
    assert s1["patient_id"] == ["pB"]
    assert s0["volume"].shape[0] == 1
    assert float(s1["spacing"][0, 0]) == 2.0


def test_per_sample_metrics_attaches_ids():
    mod = _load_eval_module()
    batch = {"patient_id": ["a", "b"], "case_id": torch.tensor([10, 11])}
    row = mod._per_sample_metrics({"dice_mean": 0.9}, batch, 1)
    assert row["patient_id"] == "b"
    assert row["case_id"] == 11
    assert row["dice_mean"] == 0.9
