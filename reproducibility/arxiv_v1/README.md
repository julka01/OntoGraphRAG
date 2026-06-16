# arXiv v1 reproducibility bundle

This directory contains the cached artefacts used to build the arXiv v1 paper
tables and figures. It is a verification bundle, not an experiment runner: the
checks do not rebuild the KG, rerun retrieval, call model APIs, or regenerate
scientific results.

`MANIFEST.json` records each packaged file, its role, byte size, SHA256 hash,
and the git commit/dirty-tree status at bundle creation time. From the repository
root, run:

```bash
python3 scripts/check_arxiv_package.py
```

The script verifies manifest hashes, figure/table inputs, bounded manuscript
wording, and a local `tectonic` build of `paper/main_arxiv.tex`.
