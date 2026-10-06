"""Diagnosis-stratified ACDC K-fold split helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reconseg3d.data.acdc import (
    ACDC_GROUP_NAMES,
    discover_acdc_patients,
    parse_info_cfg,
    phenotype_from_group,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = REPO_ROOT / "splits"


def patient_diagnosis(patient_dir: Path) -> str:
    info_path = patient_dir / "Info.cfg"
    if not info_path.is_file():
        return "NOR"
    info = parse_info_cfg(info_path)
    group = info.get("Group", "NOR").upper()
    # Normalize ARV → RV
    if group == "ARV":
        group = "RV"
    if group not in ACDC_GROUP_NAMES:
        # Map unknown to closest bucket via phenotype index.
        idx = phenotype_from_group(group)
        group = ACDC_GROUP_NAMES[min(idx, len(ACDC_GROUP_NAMES) - 1)]
    return group


def stratified_kfold_patients(
    patients: Sequence[Path],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Build diagnosis-stratified K-fold splits (NOR/MINF/DCM/HCM/RV).

    Each fold dict: ``{fold, train, val, test, diagnosis_counts}``.
    For fold ``k``, ``test`` = fold k, ``val`` = fold (k+1)%K, ``train`` = rest.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    by_diag: dict[str, list[str]] = defaultdict(list)
    for p in patients:
        by_diag[patient_diagnosis(p)].append(p.name)
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for diag in ACDC_GROUP_NAMES:
        names = list(by_diag.get(diag, []))
        rng.shuffle(names)
        for i, name in enumerate(names):
            folds[i % n_folds].append(name)
    # Stable sort within each fold for reproducibility.
    for f in folds:
        f.sort()

    out: list[dict[str, Any]] = []
    for k in range(n_folds):
        test = list(folds[k])
        val = list(folds[(k + 1) % n_folds])
        train = []
        for j in range(n_folds):
            if j == k or j == (k + 1) % n_folds:
                continue
            train.extend(folds[j])
        train = sorted(train)
        counts = {g: 0 for g in ACDC_GROUP_NAMES}
        for name in test:
            # Recover diagnosis from original grouping.
            for g, names in by_diag.items():
                if name in names:
                    counts[g] = counts.get(g, 0) + 1
                    break
        out.append(
            {
                "fold": k,
                "n_folds": n_folds,
                "seed": seed,
                "stratify_by": list(ACDC_GROUP_NAMES),
                "train": train,
                "val": val,
                "test": test,
                "diagnosis_counts_test": counts,
            }
        )
    return out


def _assert_folds_non_degenerate(
    folds: list[dict[str, Any]],
    *,
    n_patients: int,
    n_folds: int,
    allow_degenerate: bool,
) -> None:
    """Refuse empty train/val/test (or too-few patients) unless smoke mode."""
    # Need ≥1 patient in every fold bucket so train/val/test stay non-empty.
    # Under diagnosis stratification this typically means ≥ n_folds × n_groups.
    min_patients = n_folds * max(1, len(ACDC_GROUP_NAMES) // 2) if not allow_degenerate else n_folds
    degenerate_reasons: list[str] = []
    if n_patients < min_patients:
        degenerate_reasons.append(
            f"n_patients={n_patients} < minimum {min_patients} for non-degenerate "
            f"{n_folds}-fold (train/val/test) splits"
        )
    for fold in folds:
        for key in ("train", "val", "test"):
            ids = fold.get(key) or []
            if len(ids) == 0:
                degenerate_reasons.append(
                    f"fold {fold.get('fold')} has empty '{key}'"
                )
    if degenerate_reasons and not allow_degenerate:
        raise ValueError(
            "Refusing to write publication ACDC folds with empty splits or too few "
            "patients. Mount licensed ACDC (typically ≥100 training subjects) or pass "
            "allow_degenerate=True for CI/smoke placeholders only.\n  - "
            + "\n  - ".join(degenerate_reasons)
        )


def write_acdc_folds(
    root: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    n_folds: int = 5,
    seed: int = 42,
    fake_if_empty: bool = True,
    n_fake: int = 20,
    allow_degenerate: bool = False,
    synthetic_placeholder: bool | None = None,
) -> list[Path]:
    """Write ``splits/acdc_fold{0..n-1}.json``.

    By default refuses folds with empty train/val/test or too few patients.
    Set ``allow_degenerate=True`` only for CI/smoke placeholder generation.
    """
    from reconseg3d.data.acdc import make_fake_acdc

    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if root is None:
        root = REPO_ROOT / "data" / "acdc"
    root = Path(root)
    patients = discover_acdc_patients(root)
    used_fake = False
    if not patients and fake_if_empty:
        # Prefer enough fake patients so each fold bucket is non-empty under
        # diagnosis stratification (≈ n_folds patients per ACDC group).
        n_need = max(
            int(n_fake),
            n_folds if allow_degenerate else n_folds * len(ACDC_GROUP_NAMES),
        )
        make_fake_acdc(root, n_patients=n_need, spatial=(8, 16, 16), n_frames=8, seed=seed)
        patients = discover_acdc_patients(root)
        used_fake = True
    if not patients:
        raise FileNotFoundError(f"No ACDC patients under {root}")
    folds = stratified_kfold_patients(patients, n_folds=n_folds, seed=seed)
    _assert_folds_non_degenerate(
        folds,
        n_patients=len(patients),
        n_folds=n_folds,
        allow_degenerate=allow_degenerate,
    )
    if synthetic_placeholder is None:
        synthetic_placeholder = bool(used_fake or allow_degenerate)
    paths: list[Path] = []
    for fold in folds:
        payload = dict(fold)
        if synthetic_placeholder:
            payload["synthetic_placeholder"] = True
        path = out_dir / f"acdc_fold{fold['fold']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(path)
    note = (
        "CI/SMOKE PLACEHOLDER ONLY — not publication CV. "
        if synthetic_placeholder
        else "Diagnosis-stratified CV. "
    )
    note += (
        "Real ACDC subject tables remain 待补充 until licensed data are mounted."
        if synthetic_placeholder
        else "Regenerate after remounting licensed ACDC if subject IDs change."
    )
    manifest = {
        "dataset": "ACDC",
        "n_folds": n_folds,
        "seed": seed,
        "stratify_by": list(ACDC_GROUP_NAMES),
        "n_patients": len(patients),
        "synthetic_placeholder": bool(synthetic_placeholder),
        "files": [p.name for p in paths],
        "note": note,
    }
    (out_dir / "acdc_folds_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths


def load_fold_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Fold file must be a JSON object: {path}")
    for key in ("train", "val", "test"):
        if key not in data:
            raise ValueError(f"Fold file missing '{key}': {path}")
    return data


def assert_fold_train_nonempty(
    fold: dict[str, Any],
    *,
    path: str | Path | None = None,
    allow_fake_data: bool = True,
) -> None:
    """Hard-fail empty train lists in publication mode (``allow_fake_data=false``)."""
    train = fold.get("train") or []
    if len(train) == 0 and not allow_fake_data:
        loc = f" ({path})" if path is not None else ""
        raise RuntimeError(
            f"Fold file has empty train list{loc}. Committed splits/acdc_fold*.json "
            "are CI/smoke placeholders (see splits/README.md). Mount licensed ACDC, "
            "regenerate folds with scripts/make_acdc_folds.py, or set fold: null for "
            "a simple train/val ratio split. Do not train publication runs on empty "
            "fold-0 train."
        )


def resolve_fold_path(cfg: dict[str, Any], fold: int | None = None) -> Path | None:
    """Resolve fold JSON from data config. Returns None if folds unused."""
    data_cfg = cfg.get("data", cfg)
    fold_file = data_cfg.get("fold_file")
    if fold_file:
        p = Path(fold_file)
        if not p.is_file():
            p = REPO_ROOT / fold_file
        return p
    fold_idx = fold if fold is not None else data_cfg.get("fold")
    if fold_idx is None:
        return None
    splits_dir = Path(data_cfg.get("splits_dir", DEFAULT_SPLITS_DIR))
    if not splits_dir.is_absolute():
        splits_dir = REPO_ROOT / splits_dir
    return splits_dir / f"acdc_fold{int(fold_idx)}.json"
