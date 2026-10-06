# START HERE

This repository is a **deliberately flat snapshot**: **every file lives in the root directory, with no subfolders at all.** That is intentional — it lets ChatGPT or any other agent/tool read the complete contents in a single directory listing.

**Read [`FLAT_LAYOUT.md`](FLAT_LAYOUT.md) first** — it explains why the layout is flat, the `dir__subdir__file` naming rule, what is included, what is deliberately excluded, a suggested reading order, the honesty statement about data, and how to restore the original folder structure.

Machine-readable index of every file: [`FLAT_FILEMAP.json`](FLAT_FILEMAP.json) (flat filename → original repo-relative path → byte size).

Naming rule:

```
reconseg3d/models/motion.py   ->  reconseg3d__models__motion.py
docs/paper/MANUSCRIPT_DRAFT.md ->  docs__paper__MANUSCRIPT_DRAFT.md
tests/test_motion.py          ->  tests__test_motion.py
```

Canonical, runnable repository (normal folder structure):
<https://github.com/Coucou2016/ReconSeg3D-cardiac-rec>
