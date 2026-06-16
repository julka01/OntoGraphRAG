# arXiv v1 reproducibility bundle

This directory contains the cached artefacts used to build the arXiv v1 paper
tables and figures. It is a verification bundle, not an experiment runner: the
checks do not rebuild the KG, rerun retrieval, call model APIs, or regenerate
scientific results.

The manuscript sources and compiled PDF are not bundled here; the canonical
paper is available on arXiv. When the live `paper/` directory is present
locally (it is gitignored), the checker additionally validates manuscript
wording, figure/table references, and runs a local `tectonic` build.

`MANIFEST.json` records each packaged file, its role, byte size, SHA256 hash,
and the git commit/dirty-tree status at bundle creation time. From the repository
root, run:

```bash
python3 scripts/check_arxiv_package.py
```

The script verifies manifest hashes and, when `paper/` exists locally, figure
and table inputs, bounded manuscript wording, and a local `tectonic` build of
`paper/main_arxiv.tex`.