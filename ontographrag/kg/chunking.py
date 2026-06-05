"""
Text chunking utilities for the KG build pipeline.

Extracted from OntologyGuidedKGCreator so the logic can be tested in isolation
and reused by other builders without importing the full creator class.

Public API
----------
chunk_text(text, chunk_size, chunk_overlap, embedding_fn) -> List[dict]
    The canonical chunking function.  OntologyGuidedKGCreator._chunk_text
    now delegates to this.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional


def _make_text_splitter(chunk_size: int, chunk_overlap: int):
    """Return a splitter that prefers token-aware chunking but works offline."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        logging.warning(
            "chunk_text: token-aware splitter unavailable (%s); "
            "falling back to character-based chunking",
            e,
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Split *text* into overlapping token-bounded chunks.

    Uses ``RecursiveCharacterTextSplitter.from_tiktoken_encoder`` so splits
    respect paragraph → sentence → word boundaries before falling back to
    character cuts.  Chunk sizes are token-based (tiktoken cl100k_base).

    Parameters
    ----------
    text:
        Raw document text to split.
    chunk_size:
        Maximum tokens per chunk.
    chunk_overlap:
        Token overlap between consecutive chunks.
    embedding_fn:
        Optional callable ``(text: str) -> List[float]``.  When provided,
        embeddings are generated for each chunk and stored in the ``embedding``
        field.  Pass ``None`` to skip (embedding field will be ``None``).

    Returns
    -------
    List of chunk dicts with keys:
        text, chunk_id (int index), start_pos, end_pos, total_chunks, embedding

    Notes on start_pos / end_pos
    ----------------------------
    Positions are *true character offsets* in the original *text* string.
    We locate each chunk by searching forward from the previous match rather
    than accumulating ``len(chunk) - chunk_overlap``.  The subtraction would
    use token-unit overlap against character-length chunks, producing drifting
    synthetic offsets that disagree with any character-based position computed
    on the same text (e.g. ``_detect_section_headers`` uses ``m.start()``).
    """
    splitter = _make_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        raw_chunks = splitter.split_text(text)
    except Exception as e:
        logging.warning(
            "chunk_text: preferred splitter failed during split (%s); "
            "retrying with character-based chunking",
            e,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)
    formatted: List[Dict[str, Any]] = []
    # search_start advances after each match so we find the *next* occurrence
    # of a repeated substring rather than always re-matching from 0.
    search_start = 0

    for idx, chunk_text_ in enumerate(raw_chunks):
        found = text.find(chunk_text_, search_start)
        if found != -1:
            start_pos = found
            end_pos = found + len(chunk_text_)
            # Advance past this match; overlap means next chunk starts before end_pos
            search_start = found + 1
        else:
            # Splitter returned text not present verbatim (shouldn't happen but
            # guard rather than crash).  Fall back to previous end position.
            start_pos = search_start
            end_pos = start_pos + len(chunk_text_)
            logging.warning(
                "chunk_text: could not locate chunk %d in source text; "
                "start_pos/end_pos are approximate", idx
            )

        embedding: Optional[List[float]] = None
        if embedding_fn is not None:
            try:
                embedding = embedding_fn(chunk_text_)
            except Exception as e:
                logging.warning("Failed to generate embedding for chunk %d: %s", idx, e)

        formatted.append({
            "text": chunk_text_,
            "chunk_id": idx,
            "position": idx,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "total_chunks": total,
            "embedding": embedding,
        })

    return formatted
