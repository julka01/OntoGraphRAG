#!/usr/bin/env python3
"""Validate the arXiv v1 cached-artifact package.

This is a packaging check only. It verifies cached files and, when the live
``paper/`` sources are present locally, manuscript wording, figure/table
references, and a local TeX build. It must not run experiments, call APIs,
rebuild KGs, or regenerate scientific results.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "reproducibility" / "arxiv_v1"
MANIFEST = PACKAGE / "MANIFEST.json"
PAPER = REPO / "paper"

BANNED_SOURCE_PATTERNS = {
    "future-tense availability": re.compile(r"\bwill be released\b", re.I),
    "unqualified clinical certificate": re.compile(r"\bclinical certificate\b", re.I),
    "causal stress-test phrasing": re.compile(r"\bclearest mechanism\b", re.I),
    "deployment-strength certificate": re.compile(r"\bdeployable certificate\b", re.I),
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        fail(f"missing manifest: {MANIFEST.relative_to(REPO)}")
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"manifest is not valid JSON: {exc}")


def check_manifest_files(doc: dict) -> None:
    files = doc.get("files")
    if not isinstance(files, list) or not files:
        fail("manifest must contain a non-empty 'files' list")

    required_roles = {
        "selection_ids",
        "headline_results",
        "analysis_output",
        "gps_replay",
    }
    roles = {item.get("role") for item in files}
    missing_roles = sorted(required_roles - roles)
    if missing_roles:
        fail(f"manifest missing required roles: {missing_roles}")

    for item in files:
        rel = item.get("path")
        expected_hash = item.get("sha256")
        expected_size = item.get("size")
        if not rel or not expected_hash or expected_size is None:
            fail(f"manifest entry missing path/sha256/size: {item}")
        path = REPO / rel
        if not path.exists():
            fail(f"manifest file missing: {rel}")
        if path.stat().st_size != expected_size:
            fail(f"size mismatch for {rel}: {path.stat().st_size} != {expected_size}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            fail(f"sha256 mismatch for {rel}: {actual_hash} != {expected_hash}")


def paper_sources() -> list[Path]:
    return sorted(PAPER.glob("*.tex")) + [PAPER / "references.bib"]


def check_manuscript_wording() -> None:
    for path in paper_sources():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in BANNED_SOURCE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                fail(f"{label} found in {path.relative_to(REPO)}:{line}")


def strip_latex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        out = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            out.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def check_figure_inputs() -> None:
    text = "\n".join(
        strip_latex_comments(path.read_text(encoding="utf-8", errors="replace"))
        for path in PAPER.glob("*.tex")
    )

    input_refs = re.findall(r"\\input\{figures/([^}]+)\}", text)
    for ref in sorted(set(input_refs)):
        candidate = PAPER / "figures" / ref
        if candidate.suffix:
            if not candidate.exists():
                fail(f"missing figure/table input: {candidate.relative_to(REPO)}")
        elif not candidate.with_suffix(".tex").exists():
            fail(f"missing figure/table input: {candidate.with_suffix('.tex').relative_to(REPO)}")

    graphic_refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", text)
    for ref in sorted(set(graphic_refs)):
        candidate = PAPER / "figures" / ref
        if candidate.suffix:
            if not candidate.exists():
                fail(f"missing graphic: {candidate.relative_to(REPO)}")
        else:
            found = any(candidate.with_suffix(ext).exists() for ext in (".pdf", ".png", ".jpg", ".jpeg"))
            if not found:
                fail(f"missing graphic for figures/{ref} with pdf/png/jpg extension")


def run_tectonic() -> None:
    cmd = ["tectonic", "main_arxiv.tex"]
    proc = subprocess.run(cmd, cwd=PAPER, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        fail("tectonic build failed")


def main() -> None:
    doc = load_manifest()
    check_manifest_files(doc)

    if not PAPER.is_dir():
        print("arXiv package check passed (paper sources not present locally; skipped paper checks)")
        return

    check_manuscript_wording()
    check_figure_inputs()
    run_tectonic()
    print("arXiv package check passed")


if __name__ == "__main__":
    main()
