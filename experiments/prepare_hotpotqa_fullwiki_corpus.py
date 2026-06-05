#!/usr/bin/env python3
"""Prepare a shared HotpotQA FullWiki retrieval corpus.

The standard ``hotpot_dev_fullwiki_v1.json`` file stores a retrieved paragraph
bundle per question.  For KG-RAG retrieval experiments we sometimes want a
shared corpus contract instead: one corpus built from the union of paragraphs
associated with a fixed evaluation subset, with retrieval no longer filtered
to a question-local bundle.

This script creates that corpus and, by default, writes the matching persisted
question selection used by the experiment runner.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from subset_selection import (
    deterministic_subset_ids,
    selection_file_path,
    subset_identity,
)


def _load_hotpot(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _question_id(item: Dict[str, Any], idx: int) -> str:
    return str(item.get("_id", item.get("id", f"hotpotqa_{idx}")))


def _paragraph_text(title: str, sentences: Any) -> str:
    if isinstance(sentences, list):
        body = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
    else:
        body = str(sentences or "").strip()
    title = str(title or "").strip()
    return f"{title}. {body}".strip() if title else body


def _iter_question_paragraphs(
    item: Dict[str, Any],
    question_id: str,
) -> Iterable[Tuple[str, str, str]]:
    for passage_idx, passage in enumerate(item.get("context", []) or []):
        if not isinstance(passage, (list, tuple)) or len(passage) < 2:
            continue
        title = str(passage[0] or "").strip()
        text = _paragraph_text(title, passage[1])
        if not text:
            continue
        doc_key = f"{title}\n{text}"
        yield doc_key, title, text


def _write_selection(
    *,
    selection_path: Path,
    dataset_name: str,
    selected_ids: Sequence[str],
    available_count: int,
    num_samples: int | None,
    subset_seed: int,
) -> None:
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    identity = subset_identity(
        dataset_name=dataset_name,
        question_ids=selected_ids,
        num_samples=num_samples,
        subset_seed=subset_seed,
    )
    payload = {
        "version": 1,
        "dataset": dataset_name,
        "created_at": datetime.now().isoformat(),
        "selection_strategy": "all" if num_samples is None else "seeded_sample",
        "subset_seed": int(subset_seed),
        "requested_num_samples": num_samples,
        "available_question_count": int(available_count),
        "selection_count": len(selected_ids),
        "question_ids": list(selected_ids),
        **identity,
    }
    selection_path.write_text(json.dumps(payload, indent=2))


def prepare_corpus(
    *,
    hotpot_file: Path,
    output: Path,
    selection_output: Path,
    dataset_name: str,
    num_samples: int | None,
    subset_seed: int,
    overwrite: bool,
) -> Dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if selection_output.exists() and not overwrite:
        raise FileExistsError(
            f"Selection already exists: {selection_output}. Pass --overwrite to replace it."
        )

    records = _load_hotpot(hotpot_file)
    ordered_ids = [_question_id(item, idx) for idx, item in enumerate(records)]
    selected_ids = deterministic_subset_ids(ordered_ids, num_samples, subset_seed)
    selected_id_set = set(selected_ids)

    by_id = {_question_id(item, idx): item for idx, item in enumerate(records)}
    docs: Dict[str, Dict[str, Any]] = {}
    for question_id in selected_ids:
        item = by_id.get(question_id)
        if not item:
            continue
        for doc_key, title, text in _iter_question_paragraphs(item, question_id):
            doc = docs.setdefault(
                doc_key,
                {
                    "id": f"hotpot_fullwiki_doc_{len(docs)}",
                    "title": title,
                    "text": text,
                    "source_question_ids": [],
                    "source": "hotpot_dev_fullwiki_retrieved_context_union",
                },
            )
            doc["source_question_ids"].append(question_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for doc in docs.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    _write_selection(
        selection_path=selection_output,
        dataset_name=dataset_name,
        selected_ids=selected_ids,
        available_count=len(ordered_ids),
        num_samples=num_samples,
        subset_seed=subset_seed,
    )

    return {
        "dataset": dataset_name,
        "hotpot_file": str(hotpot_file),
        "output": str(output),
        "selection_output": str(selection_output),
        "available_questions": len(ordered_ids),
        "selected_questions": len(selected_ids),
        "corpus_documents": len(docs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a shared HotpotQA FullWiki corpus from selected FullWiki retrieved paragraphs."
    )
    parser.add_argument(
        "--hotpot-file",
        type=Path,
        default=Path("MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MIRAGE/rawdata/hotpotqa/fullwiki_corpus.jsonl"),
    )
    parser.add_argument("--num-samples", type=int, default=250)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="hotpotqa_fullwiki")
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=None,
        help="Defaults to results/selections/<dataset>__n<num>_seed<seed>.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selection_output = args.selection_output or selection_file_path(
        Path("results") / "selections",
        args.dataset_name,
        num_samples=args.num_samples,
        subset_seed=args.subset_seed,
    )
    summary = prepare_corpus(
        hotpot_file=args.hotpot_file,
        output=args.output,
        selection_output=selection_output,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        subset_seed=args.subset_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
