# ACDC fold JSON (CI / smoke placeholders)

The committed `acdc_fold{0-4}.json` files and `acdc_folds_manifest.json` are
**synthetic CI/smoke placeholders only**.

They were generated from an 8-patient fake ACDC tree and are **not** valid
publication 5-fold splits:

- `n_patients: 8` with `n_folds: 5` yields empty `train` / `val` on some folds
  (notably fold 0 has `train: []`).
- Manifest flag: `"synthetic_placeholder": true`.

## Publication use

1. Mount licensed ACDC under `data/acdc`.
2. Regenerate folds:

```powershell
python scripts/make_acdc_folds.py --root data/acdc --out-dir splits --n-folds 5 --seed 42
```

`write_acdc_folds` refuses to write publication folds when any fold has empty
`train`/`val`/`test` (or too few patients), unless `--allow-degenerate` is set
for smoke-only regeneration.

3. Point publication configs at a real fold via `data.fold: 0..4` or
   `data.fold_file: splits/acdc_foldK.json`. Default publication configs use
   `fold: null` (simple train/val ratio split) until real folds exist.

With `allow_fake_data: false`, the loader **hard-fails** if a fold file has an
empty `train` list.
