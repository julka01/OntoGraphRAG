"""Compatibility exports for legacy graph-extraction imports.

The active extraction endpoint lives in :mod:`ontographrag.api.app` and the
KG builder implementation lives under :mod:`ontographrag.kg.builders`. This
module remains as a narrow import shim for older callers that imported graph
builder constants from ``ontographrag.kg.utils.extract_graph``.
"""

from .constants import (
    BUCKET_FAILED_FILE,
    BUCKET_UPLOAD,
    DELETE_ENTITIES_AND_START_FROM_BEGINNING,
    PROJECT_ID,
    QUERY_TO_DELETE_EXISTING_ENTITIES,
    QUERY_TO_GET_CHUNKS,
    QUERY_TO_GET_LAST_PROCESSED_CHUNK_POSITION,
    QUERY_TO_GET_LAST_PROCESSED_CHUNK_WITHOUT_ENTITY,
    QUERY_TO_GET_NODES_AND_RELATIONS_OF_A_DOCUMENT,
    START_FROM_BEGINNING,
    START_FROM_LAST_PROCESSED_POSITION,
)

__all__ = [
    "BUCKET_FAILED_FILE",
    "BUCKET_UPLOAD",
    "DELETE_ENTITIES_AND_START_FROM_BEGINNING",
    "PROJECT_ID",
    "QUERY_TO_DELETE_EXISTING_ENTITIES",
    "QUERY_TO_GET_CHUNKS",
    "QUERY_TO_GET_LAST_PROCESSED_CHUNK_POSITION",
    "QUERY_TO_GET_LAST_PROCESSED_CHUNK_WITHOUT_ENTITY",
    "QUERY_TO_GET_NODES_AND_RELATIONS_OF_A_DOCUMENT",
    "START_FROM_BEGINNING",
    "START_FROM_LAST_PROCESSED_POSITION",
]
