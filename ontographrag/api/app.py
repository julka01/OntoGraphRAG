from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Body, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional
import asyncio
import hashlib
import logging
import threading
import time
import os, uuid, sys, tempfile, io, json
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
import httpx

logger = logging.getLogger(__name__)

load_dotenv()

_API_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _API_DIR.parent.parent
_STATIC_DIR = _API_DIR / "static"
_SOURCE_UI_DIST_DIR = _PROJECT_ROOT / "frontend" / "dist"
_DEFAULT_WRITE_PROBE_DIR = _PROJECT_ROOT / "results"
_READY_WRITE_PROBE_DIR = Path(
    os.getenv("ONTOGRAPHRAG_WRITE_PROBE_DIR", str(_DEFAULT_WRITE_PROBE_DIR))
)
_DEBUG_ERRORS = str(os.getenv("ONTOGRAPHRAG_DEBUG_ERRORS", "0")).strip().lower() in {
    "1", "true", "yes", "on",
}


def _resolve_ui_dist_dir() -> Path:
    """Find packaged UI assets first, then source-checkout build assets."""
    env_dir = os.getenv("ONTOGRAPHRAG_UI_DIST_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend([
        _STATIC_DIR,
        _SOURCE_UI_DIST_DIR,
        Path.cwd() / "frontend" / "dist",
    ])
    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate
    return candidates[0] if candidates else _STATIC_DIR


_UI_DIST_DIR = _resolve_ui_dist_dir()

from ontographrag.providers.model_providers import (
    get_provider as get_llm_provider,
    LangChainRunnableAdapter,
    TemperatureLockedProvider,
)
from experiments.answer_formatting import (
    build_answer_instructions,
    normalize_answer_to_contract,
)
from ontographrag.api.runtime_helpers import (
    build_support_guardrail_verdict,
    build_readiness_report,
    configured_provider_names_from_env,
    filesystem_write_probe_ok,
    guardrail_forces_abstention,
    parse_request_timeout_seconds,
)
from ontographrag.kg.builders.enhanced_kg_creator import UnifiedOntologyGuidedKGCreator
from ontographrag.rag.answer_guardrails import RUNTIME_GUARDRAIL_ABSTENTION
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem

_REQUEST_TIMEOUT_SECONDS = parse_request_timeout_seconds(
    os.getenv("ONTOGRAPHRAG_REQUEST_TIMEOUT_SECONDS", "120")
)

# Module-level singleton with lock to prevent race conditions at startup
_rag_system: EnhancedRAGSystem = None
_rag_system_lock = threading.Lock()
_rag_system_embedding: str | None = None

_DEFAULT_TASK_TYPE_BY_DATASET = {
    "pubmedqa": "binary",
    "realmedqa": "free_text",
    "hotpotqa": "free_text",
    "2wikimultihopqa": "free_text",
    "musique": "free_text",
    "multihoprag": "free_text",
}


def _normalize_dataset_and_task(dataset_name: Optional[str], task_type: Optional[str]) -> tuple[str, str]:
    dataset = str(dataset_name or "").strip().lower()
    task = str(task_type or "").strip().lower()
    if dataset and not task:
        task = _DEFAULT_TASK_TYPE_BY_DATASET.get(dataset, "")
    return dataset, task


def get_rag_system(embedding_model: str | None = None) -> EnhancedRAGSystem:
    global _rag_system, _rag_system_embedding
    resolved = (embedding_model or "").lower() or None
    with _rag_system_lock:
        if _rag_system is None or (resolved and resolved != _rag_system_embedding):
            _rag_system = EnhancedRAGSystem(embedding_model=resolved)
            _rag_system_embedding = resolved
    return _rag_system


def invalidate_rag_system(reason: str = "") -> None:
    """Drop the singleton retrieval system after KG mutations."""
    global _rag_system, _rag_system_embedding
    with _rag_system_lock:
        if _rag_system is not None:
            clear_fn = getattr(_rag_system, "clear_retrieval_caches", None)
            if callable(clear_fn):
                try:
                    clear_fn()
                except Exception as exc:
                    logger.warning("Failed to clear retrieval caches before reset: %s", exc)
        _rag_system = None
        _rag_system_embedding = None
    if reason:
        logger.info("Invalidated RAG system singleton: %s", reason)

from ontographrag.kg.csv_processor import MedicalReportCSVProcessor

# Configuration constants for input validation
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_FILE_EXTENSIONS = {'.pdf', '.txt', '.csv', '.json', '.xml'}
ALLOWED_ONTOLOGY_EXTENSIONS = {'.owl', '.rdf', '.ttl', '.xml', '.json'}

def validate_file_upload(file: UploadFile, max_size_bytes: int = MAX_FILE_SIZE_BYTES, allowed_extensions: set = None) -> None:
    """
    Validate file upload for size and extension.
    Raises HTTPException if validation fails.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check file extension
    if allowed_extensions:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}"
            )
    
    # Check file size (peek at beginning)
    # Note: For full validation, we'd need to read the entire file which we do in the endpoint
    # This is a preliminary check that can be enhanced with streaming size validation
    logger.info(f"Validating file: {file.filename}")

def validate_ontology_schema(ontology_bytes: bytes, filename: str) -> list[str]:
    """Validate an ontology file before it is handed to the KG builder.

    Returns a list of human-readable error strings.  An empty list means valid.
    Handles both JSON and OWL inputs; silently skips OWL (structural validation
    beyond format-detection is left to the XML parser).
    """
    errors: list[str] = []
    ext = os.path.splitext(filename)[1].lower()

    # Only deep-validate JSON; OWL validation is handled by ElementTree at load time
    if ext != '.json':
        return errors

    try:
        raw = json.loads(ontology_bytes.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return [f"Invalid JSON: {e}"]

    raw_classes = raw.get('classes') or raw.get('entity_types') or []
    raw_rels = raw.get('relationships') or raw.get('relationship_types') or []

    valid_cardinalities = {'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many', None, ''}
    class_ids: set[str] = set()
    rel_ids: set[str] = set()
    prop_types: dict[str, str] = {}  # property name → first-seen type

    # ---- validate entity types ----
    for i, cls in enumerate(raw_classes):
        if not isinstance(cls, dict):
            errors.append(f"classes[{i}]: expected object, got {type(cls).__name__}")
            continue
        eid = cls.get('id') or cls.get('name') or ''
        if not eid:
            errors.append(f"classes[{i}]: missing 'id'")
            continue
        if eid in class_ids:
            errors.append(f"Duplicate entity type id: '{eid}'")
        class_ids.add(eid)

        has_identifier = False
        for j, p in enumerate(cls.get('properties') or []):
            pname = (p.get('name') or p.get('id') or '') if isinstance(p, dict) else ''
            if not pname:
                errors.append(f"classes[{i}].properties[{j}]: missing 'name'")
                continue
            ptype = (p.get('type') or 'string').lower()

            # Conflicting property types across reuse
            key = f"{eid}.{pname}"
            if key in prop_types and prop_types[key] != ptype:
                errors.append(
                    f"classes[{i}].properties '{pname}': type conflict "
                    f"({prop_types[key]!r} vs {ptype!r})"
                )
            prop_types[key] = ptype

            if p.get('identifier'):
                has_identifier = True

            # Enum must have values
            if ptype == 'enum' and not (p.get('enum_values') or p.get('values')):
                errors.append(f"classes[{i}].properties '{pname}': type=enum but no enum_values")

        if not has_identifier:
            errors.append(
                f"classes[{i}] '{eid}': missing identifier property "
                f"(mark one property with identifier=true)"
            )

    # ---- validate relationship types ----
    for i, rel in enumerate(raw_rels):
        if not isinstance(rel, dict):
            errors.append(f"relationships[{i}]: expected object, got {type(rel).__name__}")
            continue
        rid = rel.get('id') or rel.get('name') or rel.get('type') or ''
        if not rid:
            errors.append(f"relationships[{i}]: missing 'id'")
            continue
        if rid in rel_ids:
            errors.append(f"Duplicate relationship id: '{rid}'")
        rel_ids.add(rid)

        # Dangling domain / range
        domain = rel.get('from') or rel.get('domain') or ''
        range_ = rel.get('to') or rel.get('range') or ''
        if domain and class_ids and domain not in class_ids:
            errors.append(f"relationships[{i}] '{rid}': domain '{domain}' not in entity types")
        if range_ and class_ids and range_ not in class_ids:
            errors.append(f"relationships[{i}] '{rid}': range '{range_}' not in entity types")

        # Self-referential edges (warn only — some schemas legitimately allow them)
        if domain and domain == range_:
            logger.debug("Relationship '%s' is self-referential (%s → %s)", rid, domain, range_)

        # Cardinality
        card = rel.get('cardinality') or ''
        if card and card not in valid_cardinalities:
            errors.append(
                f"relationships[{i}] '{rid}': invalid cardinality '{card}' "
                f"(expected one_to_one | one_to_many | many_to_one | many_to_many)"
            )

    return errors


# Import from local kg_utils
from ontographrag.kg.utils.graph_query import get_graphDB_driver

# Retain actual langchain_experimental if available
# import importlib
# if "langchain_experimental" not in sys.modules:
#     importlib.import_module("langchain_experimental")

# Core graph imports moved inside endpoint functions

# ---------------------------------------------------------------------------
# Document text extraction — tiered OCR strategy (inspired by MOSAICX)
# Tier 1: PyMuPDF — fast, zero extra deps, works on digitally-created PDFs
# Tier 2: Surya   — layout-aware OCR, activates when PyMuPDF yield is poor
#                   (< MIN_CHARS_PER_PAGE chars/page on average), meaning the
#                   document is likely a scan. Surya is optional; if not
#                   installed the pipeline warns and accepts the thin output.
# ---------------------------------------------------------------------------
_MIN_CHARS_PER_PAGE = 80  # below this avg we treat the PDF as a scan

def _extract_text_from_bytes(data: bytes, filename: str) -> tuple[str, str]:
    """
    Extract text from raw file bytes.

    Returns (text_content, ocr_method) where ocr_method is one of:
      'pymupdf', 'surya', 'plaintext'
    Raises HTTPException on unrecoverable errors.
    """
    import fitz  # PyMuPDF — always available (in requirements.txt)

    ext = os.path.splitext(filename)[1].lower()

    if ext != '.pdf':
        try:
            return data.decode('utf-8'), 'plaintext'
        except UnicodeDecodeError:
            return data.decode('latin-1', errors='ignore'), 'plaintext'

    # --- Tier 1: PyMuPDF ---
    try:
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
        text_pymupdf = "\n".join(pages_text)
        page_count = max(len(pages_text), 1)
        avg_chars = len(text_pymupdf.strip()) / page_count

        if avg_chars >= _MIN_CHARS_PER_PAGE:
            return text_pymupdf, 'pymupdf'

        logger.warning(
            "PyMuPDF extracted only %.0f chars/page — PDF looks like a scan; trying Surya OCR",
            avg_chars,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

    # --- Tier 2: Surya OCR (optional dependency) ---
    try:
        from surya.recognition import batch_recognition  # type: ignore
        from surya.detection import batch_text_detection  # type: ignore
        from surya.model.detection.model import load_model as load_det_model  # type: ignore
        from surya.model.recognition.model import load_model as load_rec_model  # type: ignore
        from surya.model.recognition.processor import load_processor  # type: ignore
        from PIL import Image  # type: ignore

        logger.info("Surya OCR available — running OCR on scanned PDF")
        surya_pages: list[str] = []
        pdf_doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        images = []
        for page in pdf_doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        pdf_doc.close()

        det_model, det_processor = load_det_model(), load_processor()
        rec_model, rec_processor = load_rec_model(), load_processor()

        det_results = batch_text_detection(images, det_model, det_processor)
        rec_results = batch_recognition(images, det_results, rec_model, rec_processor, langs=[["en"]] * len(images))

        for page_result in rec_results:
            surya_pages.append(" ".join(line.text for line in page_result.text_lines))

        text_surya = "\n".join(surya_pages)
        if text_surya.strip():
            logger.info("Surya OCR produced %d chars across %d pages", len(text_surya), len(surya_pages))
            return text_surya, 'surya'
        logger.warning("Surya OCR returned empty text — falling back to PyMuPDF output")
    except ImportError:
        logger.warning("Surya not installed (pip install surya-ocr) — using PyMuPDF output for scan")
    except Exception as surya_err:
        logger.warning("Surya OCR failed (%s) — using PyMuPDF output", surya_err)

    # Fall back to whatever PyMuPDF gave us (may be sparse)
    if text_pymupdf.strip():
        return text_pymupdf, 'pymupdf_fallback'
    raise HTTPException(status_code=400, detail="PDF contains no extractable text and Surya OCR is not available")


# Rate limiter (keyed on client IP)
limiter = Limiter(key_func=get_remote_address)

# Configure FastAPI
app = FastAPI(
    title="OntographRAG",
    description=(
        "Turn unstructured documents into schema-consistent knowledge graphs. "
        "Query them with hybrid vector + graph RAG. "
        "Measure answer confidence with output, grounding, and structural diagnostics."
    ),
    version="1.0.0",
)

# CORS — tighten ALLOWED_ORIGINS via env var in production
_cors_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/assets", StaticFiles(directory=str(_UI_DIST_DIR / "assets"), check_dir=False), name="assets")
app.mount("/static", StaticFiles(directory=str(_UI_DIST_DIR), check_dir=False), name="static")

# Optional API key authentication — only enforced when APP_API_KEY is set in the environment.
# In development (no APP_API_KEY) all requests pass through.
_APP_API_KEY = os.getenv("APP_API_KEY")
_OPEN_AUTH_PATHS = {"/", "/health", "/ready"}
_OPEN_AUTH_PREFIXES = ("/static",)

def require_api_key(request: Request) -> None:
    if _APP_API_KEY:
        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if key != _APP_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """
    Enforce API-key auth for server endpoints when APP_API_KEY is configured.

    The GUI shell ("/") and static assets remain public so the browser can load.
    Actual API calls must supply X-API-Key or ?api_key=...
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started_at = time.perf_counter()
    path = request.url.path
    if _APP_API_KEY and path not in _OPEN_AUTH_PATHS and not path.startswith(_OPEN_AUTH_PREFIXES):
        require_api_key(request)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - started_at) * 1000))
    return response


def _request_id_for(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    return request_id or uuid.uuid4().hex


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = _request_id_for(request)
    return JSONResponse(
        status_code=exc.status_code,
        headers={"X-Request-ID": request_id},
        content={
            "status": "error",
            "detail": exc.detail,
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = _request_id_for(request)
    return JSONResponse(
        status_code=422,
        headers={"X-Request-ID": request_id},
        content={
            "status": "error",
            "detail": "Request validation failed",
            "errors": jsonable_encoder(exc.errors()),
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = _request_id_for(request)
    logger.exception("Unhandled API exception [%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        headers={"X-Request-ID": request_id},
        content={
            "status": "error",
            "detail": str(exc) if _DEBUG_ERRORS else "Internal server error",
            "request_id": request_id,
        },
    )

# Global storage for current graph data
current_graph_data = None

# KG build progress log (ring buffer of last 200 lines)
_kg_progress: deque = deque(maxlen=200)
_kg_building: bool = False

def _log_progress(line: str) -> None:
    """Append a line to the KG build progress log."""
    _kg_progress.append(line)


@app.get("/kg_progress_stream")
async def kg_progress_stream(request: Request):
    """SSE endpoint — streams KG build progress to the browser."""
    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break
            lines = list(_kg_progress)
            if len(lines) > last_idx:
                for line in lines[last_idx:]:
                    yield f"data: {json.dumps({'line': line})}\n\n"
                last_idx = len(lines)
            if not _kg_building and last_idx >= len(lines) and last_idx > 0:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            await asyncio.sleep(0.4)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Tracks whether the Neo4j connection was successfully verified at startup.
# Endpoints that require Neo4j call require_neo4j() which raises 503 when False.
neo4j_ready: bool = False


def require_neo4j() -> None:
    """Raise 503 if Neo4j was not available at startup."""
    if not neo4j_ready:
        raise HTTPException(
            status_code=503,
            detail="Neo4j database is unavailable. Check connection settings and restart the server.",
        )


@app.on_event("startup")
def check_neo4j_connection():
    global neo4j_ready
    try:
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            session.run("RETURN 1").single()
        neo4j_ready = True
        logger.info("Neo4j connection check passed")
    except Exception as e:
        logger.warning("Neo4j health check failed: %s — Neo4j-dependent endpoints will return 503", e)
        # Allow the app to start so health/static endpoints still work

# ========== Named KG Management Endpoints ==========

@app.post("/kg/create")
async def create_kg(
    kg_name: str = Form(...),
    description: str = Form(None),
    data_source: str = Form(None)
):
    """
    Create a new named Knowledge Graph.
    """
    require_neo4j()
    try:
        from ontographrag.kg.loaders.graph_db_access import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph
        
        # Create Neo4jGraph (langchain) instead of driver
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )
        
        db_access = graphDBdataAccess(graph)
        
        result = db_access.create_kg(
            kg_name=kg_name,
            description=description,
            data_source=data_source
        )
        
        return JSONResponse(content={
            "status": "success",
            "kg": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create KG: {str(e)}")


@app.get("/kg/list")
async def list_kgs():
    """
    List all named Knowledge Graphs by querying Document nodes with kgName property.
    """
    require_neo4j()
    try:
        from ontographrag.kg.utils.graph_query import get_graphDB_driver
        
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            # Query for distinct kgName values from Document nodes
            result = session.run("""
                MATCH (d:Document)
                WHERE d.kgName IS NOT NULL AND d.kgName <> ''
                RETURN DISTINCT d.kgName AS kgName, count(d) AS documentCount, max(d.updatedAt) AS lastUpdated
                ORDER BY d.kgName
            """)
            
            kgs = []
            for record in result:
                kgs.append({
                    "name": record["kgName"],
                    "kg_name": record["kgName"],
                    "document_count": record["documentCount"],
                    "last_updated": record["lastUpdated"].isoformat() if record["lastUpdated"] else None
                })
        
        return JSONResponse(content={
            "status": "success",
            "kgs": kgs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list KGs: {str(e)}")


@app.get("/kg/{kg_name}")
async def get_kg(kg_name: str):
    """
    Get details of a specific Knowledge Graph.
    """
    try:
        from ontographrag.kg.loaders.graph_db_access import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )

        db_access = graphDBdataAccess(graph)
        kg = db_access.get_kg(kg_name)
        
        if not kg:
            raise HTTPException(status_code=404, detail=f"KG '{kg_name}' not found")
        
        # Also get stats
        stats = db_access.get_kg_stats(kg_name)
        
        return JSONResponse(content={
            "status": "success",
            "kg": kg,
            "stats": stats[0] if stats else {}
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KG: {str(e)}")


@app.delete("/kg/{kg_name}")
async def delete_kg(kg_name: str, delete_entities: bool = Query(True)):
    """
    Delete a named Knowledge Graph.
    """
    try:
        from ontographrag.kg.loaders.graph_db_access import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )

        db_access = graphDBdataAccess(graph)
        deleted_count = db_access.delete_kg_by_name(kg_name, delete_entities)
        invalidate_rag_system(f"KG deleted: {kg_name}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Deleted KG '{kg_name}' with {deleted_count} documents",
            "deleted_documents": deleted_count
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete KG: {str(e)}")


@app.get("/kg/{kg_name}/entities")
async def get_kg_entities(kg_name: str, limit: int = 100):
    """
    Get entities from a specific Knowledge Graph.
    """
    try:
        from ontographrag.kg.loaders.graph_db_access import graphDBdataAccess
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE")
        )

        db_access = graphDBdataAccess(graph)
        entities = db_access.get_kg_entities(kg_name, limit)
        
        return JSONResponse(content={
            "status": "success",
            "kg_name": kg_name,
            "entities": entities,
            "count": len(entities)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KG entities: {str(e)}")


@app.get("/health/neo4j")
def neo4j_health():
    try:
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            record = session.run("RETURN count(*) AS c").single()
        return {"status": "ok", "nodeCount": record["c"]}
    except Exception as e:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neo4j health check failed: {e}"
        )

@app.get("/health")
def health():
    """Lightweight liveness probe for container/platform healthchecks."""
    return {"status": "ok", "service": "ontographrag"}


@app.get("/ready")
async def ready():
    """Fast readiness probe for orchestration and production health checks."""
    rag_runtime_ready = False
    rag_runtime_detail = ""
    if neo4j_ready:
        try:
            await asyncio.to_thread(get_rag_system)
            rag_runtime_ready = True
            rag_runtime_detail = "RAG runtime initialized"
        except Exception as e:
            rag_runtime_detail = str(e)
    else:
        rag_runtime_detail = "Neo4j unavailable at startup"

    write_ok, write_detail = filesystem_write_probe_ok(_READY_WRITE_PROBE_DIR)
    report = build_readiness_report(
        neo4j_ready=neo4j_ready,
        rag_runtime_ready=rag_runtime_ready,
        write_probe_ok=write_ok,
        write_probe_detail=write_detail,
        configured_providers=configured_provider_names_from_env(os.environ),
    )
    for check in report["checks"]:
        if check["name"] == "rag_runtime":
            check["detail"] = rag_runtime_detail
            break
    return JSONResponse(status_code=200 if report["ready"] else 503, content=report)

@app.get("/")
async def root():
    index_path = _UI_DIST_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "UI assets are missing. Build the frontend, install a packaged "
                "wheel, or set ONTOGRAPHRAG_UI_DIST_DIR."
            ),
        )
    return FileResponse(index_path)

@app.post("/load_kg_from_file")
async def load_kg_from_file(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo"),
    kg_name: str = Form(None),
    neo4j_uri: str = Form(None),
    neo4j_user: str = Form(None),
    neo4j_password: str = Form(None),
    neo4j_database: str = Form(None),
):
    """
    Build a KG from the uploaded file using OntologyGuidedKGCreator and return
    the resulting nodes and relationships from Neo4j.
    """
    require_neo4j()
    try:
        data = await file.read()
        text_content, _ = _extract_text_from_bytes(data, file.filename)
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="File contains no readable text content")

        _neo4j_uri      = neo4j_uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        _neo4j_user     = neo4j_user     or os.getenv("NEO4J_USERNAME",  "neo4j")
        _neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD",  "password")
        _neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE",  "neo4j")
        _kg_name        = kg_name or f"kg_{uuid.uuid4()}"

        llm = TemperatureLockedProvider(get_llm_provider(provider, model), temperature=0.0)
        kg_creator = UnifiedOntologyGuidedKGCreator(
            neo4j_uri=_neo4j_uri,
            neo4j_user=_neo4j_user,
            neo4j_password=_neo4j_password,
            neo4j_database=_neo4j_database,
        )
        await asyncio.to_thread(
            kg_creator.generate_knowledge_graph,
            text_content, llm, file.filename, model, None, _kg_name, None, None,
        )
        invalidate_rag_system(f"KG built: {_kg_name}")

        # Read back the nodes scoped to this KG from Neo4j
        driver = get_graphDB_driver(_neo4j_uri, _neo4j_user, _neo4j_password, _neo4j_database)
        nodes = []
        with driver.session(database=_neo4j_database) as session:
            for record in session.run(
                "MATCH (n:__Entity__ {kgName: $kg_name}) RETURN n", {"kg_name": _kg_name}
            ):
                node = record["n"]
                props = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dict(node).items()}
                nodes.append({"id": node.element_id, "labels": list(node.labels), "properties": props})
        relationships = []
        with driver.session(database=_neo4j_database) as session:
            for record in session.run(
                "MATCH (s:__Entity__ {kgName: $kg_name})-[r]->(t:__Entity__ {kgName: $kg_name}) RETURN r",
                {"kg_name": _kg_name},
            ):
                rel = record["r"]
                props = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dict(rel).items()}
                relationships.append({
                    "id": rel.element_id,
                    "type": rel.type,
                    "start": rel.start_node.element_id,
                    "end": rel.end_node.element_id,
                    "properties": props,
                })
        driver.close()
        return JSONResponse(content=jsonable_encoder({
            "kg_id": str(uuid.uuid4()),
            "kg_name": _kg_name,
            "graph_data": {"nodes": nodes, "relationships": relationships},
        }))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_graph")
async def extract_graph(
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo"),
    kg_name: str = Form(None),
    neo4j_uri: str = Form(None),
    neo4j_user: str = Form(None),
    neo4j_password: str = Form(None),
    neo4j_database: str = Form(None),
):
    """
    Extract a KG from the uploaded file using OntologyGuidedKGCreator and
    return the extraction result (nodes, relationships, metadata) without
    persisting to Neo4j.  Nodes are returned in visualization format
    (label, properties, color, size, font, title fields); use node.properties.name
    to access the raw entity name.
    """
    require_neo4j()
    try:
        data = await file.read()
        text_content, _ = _extract_text_from_bytes(data, file.filename)
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="File contains no readable text content")

        _neo4j_uri      = neo4j_uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        _neo4j_user     = neo4j_user     or os.getenv("NEO4J_USERNAME",  "neo4j")
        _neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD",  "password")
        _neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE",  "neo4j")
        _kg_name        = kg_name or f"kg_{uuid.uuid4()}"

        llm = TemperatureLockedProvider(get_llm_provider(provider, model), temperature=0.0)
        kg_creator = UnifiedOntologyGuidedKGCreator(
            neo4j_uri=_neo4j_uri,
            neo4j_user=_neo4j_user,
            neo4j_password=_neo4j_password,
            neo4j_database=_neo4j_database,
        )
        # Pass file_name=None so generate_knowledge_graph skips Neo4j persistence —
        # this is a preview/extraction endpoint, not a storage endpoint.
        kg = await asyncio.to_thread(
            kg_creator.generate_knowledge_graph,
            text_content, llm, None, model, None, _kg_name, None, None,
        )
        return JSONResponse(content=jsonable_encoder({
            "kg_name": _kg_name,
            "fileName": file.filename,
            "nodeCount": kg.get("metadata", {}).get("total_entities", 0),
            "relationshipCount": kg.get("metadata", {}).get("total_relationships", 0),
            "status": "Completed",
            "model": model,
            "nodes": kg.get("nodes", []),
            "relationships": kg.get("relationships", []),
            "metadata": kg.get("metadata", {}),
        }))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_ontology_guided_kg")
@limiter.limit("5/minute")
async def create_ontology_guided_kg(  # noqa: C901
    request: Request,
    file: UploadFile = File(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo"),
    embedding_model: str = Form("sentence_transformers"),
    ontology_file: Optional[UploadFile] = File(None),
    max_chunks: int = Form(None),
    kg_name: str = Form(None),
    neo4j_uri: str = Form(None),
    neo4j_user: str = Form(None),
    neo4j_password: str = Form(None),
    neo4j_database: str = Form(None),
    enable_coreference_resolution: bool = Form(False),
):
    """
    Create knowledge graph with optional ontology guidance.
    If ontology file is provided, uses it to ensure consistent entity types and relationships.
    If no ontology is provided, performs basic LLM-based entity extraction.
    """
    global _kg_building, _kg_progress
    if max_chunks is not None and (max_chunks < 1 or max_chunks > 500):
        raise HTTPException(status_code=422, detail="max_chunks must be between 1 and 500")
    require_neo4j()
    ontology_path = None  # declared here so the finally block can always reference it
    _kg_progress.clear()
    _kg_building = True
    try:
        # Read file content with proper encoding handling
        _log_progress(f"📄 Reading file: {file.filename}")
        data = await file.read()

        # SHA-256 content deduplication — skip re-extraction for identical documents
        doc_hash = hashlib.sha256(data).hexdigest()
        _log_progress("🔎 Checking for duplicate document…")
        try:
            _neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            _neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
            _neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            _neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
            _dup_driver = get_graphDB_driver(_neo4j_uri, _neo4j_user, _neo4j_password, _neo4j_database)
            try:
                with _dup_driver.session() as _session:
                    _dup_result = _session.run(
                        "MATCH (d:Document {contentHash: $hash, kgName: $kg_name}) RETURN d.kgName AS kgName, d.fileName AS fileName LIMIT 1",
                        {"hash": doc_hash, "kg_name": kg_name}
                    ).single()
            finally:
                _dup_driver.close()
            if _dup_result:
                _existing_kg_name = _dup_result["kgName"]
                _log_progress(f"♻️  Duplicate detected — reusing existing KG '{_existing_kg_name}'")
                logger.info("Duplicate document (SHA-256 %s) — returning existing KG %s", doc_hash[:12], _existing_kg_name)
                from ontographrag.kg.loaders.kg_loader import KGLoader
                _dup_loader = KGLoader()
                _dup_kg = _dup_loader.load_from_neo4j(
                    uri=_neo4j_uri,
                    user=_neo4j_user,
                    password=_neo4j_password,
                    kg_label=_existing_kg_name
                ) if _existing_kg_name else None
                _kg_building = False
                return JSONResponse(content={
                    "kg_id": str(uuid.uuid4()),
                    "kg_name": _existing_kg_name,
                    "graph_data": _dup_kg,
                    "method": "deduplicated",
                    "doc_hash": doc_hash,
                    "deduplicated": True,
                    "message": f"Document already ingested (SHA-256 match). Returning existing KG '{_existing_kg_name}'."
                })
        except Exception as _dup_err:
            # Non-fatal: if duplicate check fails, proceed with normal extraction
            logger.warning("Deduplication check failed (proceeding normally): %s", _dup_err)

        # Determine file type and extract text (tiered OCR strategy)
        text_content, ocr_method = _extract_text_from_bytes(data, file.filename)
        _log_progress(f"📝 Text extracted via {ocr_method} · {len(text_content)} chars")

        if len(text_content.strip()) == 0:
            raise HTTPException(status_code=400, detail="File contains no readable text content")

        logger.info("Creating KG with model: %s from provider: %s", model, provider)
        logger.info("File: %s, ocr=%s, size: %d bytes, text: %d chars", file.filename, ocr_method, len(data), len(text_content))
        _log_progress(f"🎯 Using {provider}/{model} · {len(text_content)} chars")

        # Get LLM provider (use defaults matching test if not specified)
        provider = provider or "openrouter"
        model = model or "openai/gpt-oss-120b:free"
        llm = TemperatureLockedProvider(get_llm_provider(provider, model), temperature=0.0)

        # Handle ontology file if provided
        ontology_path = None
        if ontology_file:
            logger.debug("Ontology file: %s (%s)", ontology_file.filename, getattr(ontology_file, 'content_type', 'unknown'))
            ontology_data = await ontology_file.read()
            ontology_filename = f"ontology_{uuid.uuid4()}{os.path.splitext(os.path.basename(ontology_file.filename))[1]}"
            ontology_path = os.path.join(tempfile.gettempdir(), ontology_filename)
            with open(ontology_path, "wb") as tmpf:
                tmpf.write(ontology_data)
            logger.info("Ontology saved to %s (%d bytes)", ontology_path, len(ontology_data))

            # Schema validation — fail early with readable errors
            ontology_errors = validate_ontology_schema(ontology_data, ontology_file.filename)
            if ontology_errors:
                raise HTTPException(
                    status_code=422,
                    detail={"message": "Ontology validation failed", "errors": ontology_errors},
                )
        else:
            logger.debug("No ontology file provided")

        # Generate unique KG name if not provided
        if not kg_name:
            kg_name = f"kg_{str(uuid.uuid4())}"

        logger.info("Initializing OntologyGuidedKGCreator (ontology_path=%s)", ontology_path)

        # Initialize ontology-guided KG creator (with defaults matching test)
        # Use provided neo4j credentials or fall back to environment variables
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        kg_creator = UnifiedOntologyGuidedKGCreator(
            chunk_size=2000,  # Larger chunks for better patient report context
            chunk_overlap=300,
            ontology_path=ontology_path,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            embedding_model=embedding_model or "sentence_transformers",
            enable_coreference_resolution=enable_coreference_resolution,
            strict_ontology=bool(ontology_path),
        )

        logger.debug(
            "KG creator: ontology_path=%s, classes=%d, relationships=%d",
            kg_creator.ontology_path,
            len(kg_creator.ontology_classes),
            len(kg_creator.ontology_relationships),
        )

        # Generate KG with ontology guidance (or without if no ontology provided)
        # Run in a thread so the blocking LLM/Neo4j calls don't block the event loop.
        _log_progress("🔍 Extracting entities and relationships…")
        kg = await asyncio.to_thread(
            kg_creator.generate_knowledge_graph,
            text_content, llm, file.filename, model, max_chunks, kg_name, None, doc_hash, provider,
        )
        invalidate_rag_system(f"KG built: {kg_name}")

        # Log results like test script
        entities = kg.get('metadata', {}).get('total_entities', 0)
        relationships = kg.get('metadata', {}).get(
            'stored_relationships',
            kg.get('metadata', {}).get('total_relationships', 0),
        )
        stored = kg.get('metadata', {}).get('stored_in_neo4j', False)

        logger.info("KG results: %d entities, %d relationships, stored=%s", entities, relationships, stored)
        _log_progress(f"📊 Extracted {entities} entities, stored {relationships} relationships")

        # Reload KG from Neo4j to ensure ontology labels are properly displayed
        loaded_kg = None
        if stored:
            logger.info("Reloading KG from Neo4j to apply ontology labels")
            from ontographrag.kg.loaders.kg_loader import KGLoader

            kg_loader = KGLoader()
            reload_success = False
            if kg_name:
                # Load by KG name (ontology label) - now includes Document nodes
                loaded_kg = kg_loader.load_from_neo4j(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                    kg_label=kg_name
                )
                if loaded_kg and loaded_kg.get('status') == 'success':
                    reload_success = True
            else:
                # Load all entities (excluding system nodes)
                loaded_kg = kg_loader.load_from_neo4j(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                if loaded_kg and loaded_kg.get('status') == 'success':
                    reload_success = True

            if reload_success:
                loaded_entities = loaded_kg.get('loaded_nodes', 0) if 'loaded_nodes' in loaded_kg else len(loaded_kg.get('nodes', []))
                loaded_relationships = loaded_kg.get('loaded_relationships', 0) if 'loaded_relationships' in loaded_kg else len(loaded_kg.get('relationships', []))
                logger.info("Reloaded %d nodes, %d relationships from Neo4j", loaded_entities, loaded_relationships)
            else:
                logger.warning("Failed to reload KG from Neo4j, using initial KG data")
                loaded_kg = None

        # Use reloaded KG data if available and valid, otherwise use initial KG
        final_kg_data = loaded_kg if loaded_kg and loaded_kg.get('status') == 'success' else kg

        method = "ontology_guided" if ontology_path else "basic_llm"
        determinism_improvements = [
            "fixed_chunk_size",
            "temperature=0_for_all_LLMs",
            "node_label_fix",
            "neo4j_reload" if loaded_kg else None
        ]
        determinism_improvements = [x for x in determinism_improvements if x is not None]
        if ontology_path:
            determinism_improvements.append("ontology_constraints_applied")

        return JSONResponse(content={
            "kg_id": str(uuid.uuid4()),
            "kg_name": kg_name,
            "graph_data": final_kg_data,
            "method": method,
            "ocr_method": ocr_method,
            "ontology_file": ontology_file.filename if ontology_file else None,
            "determinism_improvements": determinism_improvements
        })

    except HTTPException:
        raise
    except Exception as e:
        # Fail closed: /create_ontology_guided_kg implies the KG was persisted.
        # Returning local-only data would blur "built" vs "persisted" for callers.
        # If you need a preview/extract-only path use /extract_graph instead.
        raise HTTPException(status_code=500, detail=f"Ontology-guided KG creation failed: {str(e)}")
    finally:
        _kg_building = False
        # Always clean up the ontology temp file, regardless of success or failure.
        if ontology_path:
            try:
                os.unlink(ontology_path)
            except OSError:
                pass

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, body: dict = Body(..., max_length=65536)):
    """
    Enhanced KG-focused RAG chat that ensures responses come from KG alone.
    Supports optional kg_name parameter to filter retrieval to a specific named KG.
    """
    require_neo4j()
    try:
        request_id = _request_id_for(request)
        question = body.get("question", "")
        if not question or not isinstance(question, str):
            raise HTTPException(status_code=422, detail="Missing question")
        if len(question) > 4096:
            raise HTTPException(status_code=422, detail="Question too long (max 4096 chars)")

        docs = body.get("document_names", [])
        if docs is not None and not isinstance(docs, list):
            raise HTTPException(status_code=422, detail="document_names must be a list of strings")
        session = body.get("session_id", "default_session")
        mode = body.get("mode", "default")
        provider = body.get("provider_rag", "openrouter")
        model = body.get("model_rag", "openai/gpt-oss-120b:free")
        kg_name = body.get("kg_name", None)
        dataset_name, task_type = _normalize_dataset_and_task(
            body.get("dataset_name"),
            body.get("task_type"),
        )
        explicit_answer_instructions = str(body.get("answer_instructions", "") or "").strip()
        answer_instructions = explicit_answer_instructions or (
            build_answer_instructions(dataset_name, task_type)
            if dataset_name and task_type
            else ""
        )
        _guardrail_default = str(os.getenv("ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL", "1")).strip().lower() in {
            "1", "true", "yes", "on",
        }
        runtime_guardrail = body.get("runtime_guardrail", _guardrail_default)
        if isinstance(runtime_guardrail, str):
            runtime_guardrail = runtime_guardrail.strip().lower() in {"1", "true", "yes", "on"}
        runtime_guardrail_mode = body.get(
            "runtime_guardrail_mode",
            os.getenv("ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL_MODE", "retry_then_abstain"),
        )

        # Validate kg_name exists before querying (avoids confusing empty-result errors)
        if kg_name:
            _driver = get_graphDB_driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
                os.getenv("NEO4J_DATABASE", "neo4j"),
            )
            with _driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as _s:
                _exists = _s.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN count(d) AS c",
                    {"kg_name": kg_name}
                ).single()
            _driver.close()
            if (_exists or {}).get("c", 0) == 0:
                raise HTTPException(status_code=404, detail=f"Knowledge graph '{kg_name}' not found")

        # Read embedding model stored on the KG's Document node
        embedding_model = None
        if kg_name:
            _drv = get_graphDB_driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
                os.getenv("NEO4J_DATABASE", "neo4j"),
            )
            with _drv.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as _s:
                _doc = _s.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN d.embeddingModel AS emb LIMIT 1",
                    {"kg_name": kg_name},
                ).single()
            _drv.close()
            if _doc and _doc["emb"]:
                embedding_model = _doc["emb"]

        rag_system = get_rag_system(embedding_model=embedding_model)

        # Get LLM provider
        llm = LangChainRunnableAdapter(get_llm_provider(provider, model), model)

        # Generate response; cap at 120 s to prevent thread pool exhaustion on hung LLMs.
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    rag_system.generate_response,
                    question,
                    llm,
                    docs,
                    kg_name=kg_name,
                    answer_instructions=answer_instructions,
                    # Measure the raw candidate first; support-based abstention is
                    # applied below so structural/grounding scores stay invariant.
                    runtime_guardrail=False,
                    runtime_guardrail_mode=runtime_guardrail_mode,
                ),
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=(
                    "LLM response timed out — try again or choose a faster model "
                    f"(timeout={_REQUEST_TIMEOUT_SECONDS}s)"
                ),
            )

        if "response" in result and dataset_name and task_type:
            normalized_response = normalize_answer_to_contract(
                dataset_name,
                task_type,
                result.get("response", ""),
            )
            if normalized_response != result.get("response", ""):
                result["response_raw"] = result.get("response", "")
                result["response"] = normalized_response

        # Compute GPS confidence score (best structural metric): fraction of answer
        # entities reachable from question entities in the KG.  GPS = 1 − support,
        # so confidence = 1 − GPS = support.  Only computed when a KG is loaded.
        kg_confidence: Optional[float] = None
        if kg_name and "response" in result:
            try:
                from experiments.uncertainty_metrics import compute_graph_path_support
                _gps = compute_graph_path_support(
                    question=question,
                    answer=result["response"],
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
                    neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
                    neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
                    kg_name=kg_name,
                )
                # GPS is uncertainty (0 = fully supported); convert to confidence
                kg_confidence = round(1.0 - _gps, 3) if _gps is not None else None
            except Exception as _conf_err:
                logger.debug("GPS confidence computation skipped: %s", _conf_err)

        grounding_support = result.get("context", {}).get("grounding_quality")
        support_guardrail = build_support_guardrail_verdict(
            enabled=bool(runtime_guardrail),
            mode=str(runtime_guardrail_mode or ""),
            structural_support=kg_confidence,
            grounding_support=grounding_support,
            confidence=result.get("confidence", 0.0),
        )
        result["guardrail"] = support_guardrail
        if (
            "response" in result
            and guardrail_forces_abstention(support_guardrail)
            and result.get("response") != RUNTIME_GUARDRAIL_ABSTENTION
        ):
            result["response_candidate"] = result.get("response")
            result["response"] = RUNTIME_GUARDRAIL_ABSTENTION

        # Format response to match expected structure
        if "error" in result:
            return JSONResponse(content={
                "session_id": session,
                "request_id": request_id,
                "message": f"KG Error: {result['error']}",
                "info": {
                    "sources": [],
                    "model": model,
                    "nodedetails": [],
                    "total_tokens": None,
                    "response_time": None,
                    "mode": mode,
                    "entities": result.get("entities", []),
                    "metric_details": {},
                    "kg_only": True,
                    "kg_stats": result.get("context", {}).get("kg_stats", {})
                },
                "user": "chatbot"
            })

        # Convert enhanced response to standard format
        return JSONResponse(content={
            "session_id": session,
            "request_id": request_id,
            "message": result["response"],
            "info": {
                "sources": result["sources"],
                "model": model,
                "nodedetails": {
                    "chunkdetails": result.get("context", {}).get("chunks", []),
                    "entitydetails": result.get("context", {}).get("entities", {}),
                    "communitydetails": []
                },
                "total_tokens": None,
                "response_time": None,
                "mode": mode,
                "entities": {
                    "entityids": result.get("entities", []),
                    "relationshipids": [r.get("key", "") for r in result.get("relationships", [])],
                    "used_entities": result.get("used_entities", []),  # Nodes highlighted in KG visualization
                    "reasoning_edges": result.get("reasoning_edges", [])  # Edges forming the reasoning path
                },
                "metric_details": {
                    "question": question,
                    "contexts": [chunk["text"] for chunk in result.get("context", {}).get("chunks", [])],
                    "answer": result.get("response_candidate") or result["response"],
                    "displayed_answer": result["response"],
                    "answer_raw": result.get("response_raw"),
                },
                "kg_only": True,
                "chunk_count": result.get("chunk_count", 0),
                "entity_count": result.get("entity_count", 0),
                "relationship_count": result.get("relationship_count", 0),
                "confidence": result.get("confidence", 0.0),
                "kg_confidence": kg_confidence,   # Graph Path Support (GPS) structural confidence score
                "structural_support": kg_confidence,
                "grounding_support": grounding_support,
                "guardrail": result.get("guardrail", {}),
                "dataset_name": dataset_name or None,
                "task_type": task_type or None,
                "answer_instructions": answer_instructions or None,
            },
            "user": "chatbot"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG-RAG Error: {str(e)}")

@app.get("/models/{provider}")
def list_models(provider: str):
    """
    Return available models for provider.
    """
    model_map = {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "openrouter": [
            "openai/gpt-4o-mini",
            "openai/gpt-oss-120b:free",
            "meta-llama/llama-3.3-8b-instruct:free",
            "deepseek/deepseek-chat-v3.1:free",
            "x-ai/grok-4-fast:free",
        ],
        "ollama": [],  # populated dynamically below
    }

    if provider.lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        return {"models": [], "warning": "OPENAI_API_KEY not set"}

    if provider.lower() == "ollama":
        try:
            import ollama
            tags = ollama.list()
            model_map["ollama"] = [m.model for m in tags.models]
        except Exception:
            pass

    return {"models": model_map.get(provider.lower(), [])}

@app.post("/clear_kg")
async def clear_kg(kg_name: str = Form(None)):
    """
    Delete a knowledge graph from Neo4j.

    When *kg_name* is provided, only nodes (``__Entity__`` and ``Document``)
    whose ``kgName`` property matches are removed together with their
    relationships.  When omitted, the entire database is wiped (all nodes,
    relationships, constraints, and indexes).
    """
    require_neo4j()
    try:
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )

        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            if kg_name:
                # Scoped delete: only nodes belonging to this KG
                logger.info("Deleting KG '%s' from Neo4j", kg_name)

                result = session.run(
                    "MATCH (n {kgName: $kg_name}) DETACH DELETE n RETURN count(n) AS deleted_count",
                    {"kg_name": kg_name},
                )
                record = result.single()
                deleted_count = record["deleted_count"] if record else 0
                logger.info("Deleted %d nodes for KG '%s'", deleted_count, kg_name)
                invalidate_rag_system(f"KG cleared: {kg_name}")

                return JSONResponse(content={
                    "message": f"KG '{kg_name}' deleted successfully! Removed {deleted_count} nodes.",
                    "status": "cleared",
                    "kg_name": kg_name,
                    "nodes_deleted": deleted_count,
                })

            # Full wipe: remove everything
            logger.info("Clearing entire Neo4j knowledge graph")

            # Drop constraints
            try:
                constraints = [record["name"] for record in session.run("SHOW CONSTRAINTS")]
                for name in constraints:
                    safe = name.replace("`", "")
                    try:
                        session.run(f"DROP CONSTRAINT `{safe}`")
                        logger.debug("Dropped constraint: %s", safe)
                    except Exception as e:
                        logger.warning("Could not drop constraint %s: %s", safe, e)
            except Exception as e:
                logger.warning("Error listing constraints: %s", e)

            # Drop indexes
            try:
                indexes = [
                    record["name"] for record in session.run("SHOW INDEXES")
                    if record["type"] != "LOOKUP"
                ]
                for name in indexes:
                    safe = name.replace("`", "")
                    try:
                        session.run(f"DROP INDEX `{safe}`")
                        logger.debug("Dropped index: %s", safe)
                    except Exception as e:
                        logger.warning("Could not drop index %s: %s", safe, e)
            except Exception as e:
                logger.warning("Error listing indexes: %s", e)

            # Delete all relationships first, then nodes
            session.run("MATCH ()-[r]-() DELETE r")
            result = session.run("MATCH (n) DELETE n RETURN count(n) as deleted_count")
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0
            logger.info("Cleared %d nodes and all relationships", deleted_count)

            try:
                session.run("CALL db.resample.index.all()")
            except Exception:
                pass  # APOC not available — non-fatal

        logger.info("Neo4j knowledge graph cleared successfully")
        invalidate_rag_system("KG cleared: full database wipe")

        return JSONResponse(content={
            "message": f"Knowledge graph cleared successfully! Deleted {deleted_count} nodes and all relationships.",
            "status": "cleared",
            "nodes_deleted": deleted_count,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error clearing KG")
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge graph: {str(e)}")



@app.post("/save_kg_to_neo4j")
async def save_kg_to_neo4j(
    kg_id: str = Form(...),
    uri: str = Form(...),
    user: str = Form(...),
    password: str = Form(...)
):
    """
    Save knowledge graph data to Neo4j database.
    """
    require_neo4j()
    try:
        global current_graph_data

        # Check if we have graph data to save
        if current_graph_data is None:
            raise HTTPException(status_code=400, detail="No graph data available to save. Load a KG first.")

        driver = get_graphDB_driver(uri, user, password, os.getenv("NEO4J_DATABASE"))

        nodes_saved = 0
        relationships_saved = 0

        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            # Clear existing data first (optional - you might want to keep this or make it configurable)
            try:
                session.run("MATCH (n) DETACH DELETE n")
                logger.debug("Cleared existing Neo4j database")
            except Exception as e:
                logger.debug(f"Warning: Could not clear existing data: {e}")

            # Save nodes
            for node in current_graph_data.get('nodes', []):
                labels = node.get('labels', [])
                if not labels:
                    labels = ['Entity']  # Default label

                # Build Cypher MERGE query
                label_str = ':'.join(f'`{label}`' for label in labels)
                properties = {}

                # Copy node properties, excluding internal ones
                for k, v in node.get('properties', {}).items():
                    if k not in ['embedding', 'element_id'] and v is not None:
                        properties[k] = v

                # Add id if it exists
                if node.get('properties', {}).get('id'):
                    properties['id'] = node['properties']['id']
                elif node.get('id'):
                    properties['id'] = str(node['id'])

                # Build parameterized MERGE query
                prop_str = ', '.join(f'`{k}`: ${k}' for k, v in properties.items())
                if prop_str:
                    merge_query = f"MERGE (n:{label_str} {{ {prop_str} }})"
                else:
                    # Fallback for nodes with no properties - use id if available
                    node_id = properties.get('id', str(node.get('id', str(uuid.uuid4()))))
                    merge_query = f"MERGE (n:{label_str} {{ id: '{node_id}' }})"

                try:
                    param_dict = {k: v for k, v in properties.items()}
                    session.run(merge_query, param_dict)
                    nodes_saved += 1
                except Exception as e:
                    logger.debug(f"Error saving node {node.get('id', 'unknown')}: {e}")
                    continue

            # Save relationships
            for rel in current_graph_data.get('relationships', []):
                start_id = rel.get('start') or rel.get('from')
                end_id = rel.get('end') or rel.get('to')
                rel_type = rel.get('type', 'RELATED_TO')

                if not start_id or not end_id:
                    continue

                # Build relationship query
                properties = {}
                for k, v in rel.get('properties', {}).items():
                    if v is not None:
                        properties[k] = v

                prop_str = ', '.join(f'`{k}`: ${k}' for k, v in properties.items())
                rel_prop = f" {{{prop_str}}}" if prop_str else ""

                match_query = f"""
                MATCH (a), (b)
                WHERE id(a) = $start_id AND id(b) = $end_id
                MERGE (a)-[r:`{rel_type}`{rel_prop}]->(b)
                """

                try:
                    param_dict = {k: v for k, v in properties.items()}
                    param_dict["start_id"] = start_id
                    param_dict["end_id"] = end_id
                    session.run(match_query, param_dict)
                    relationships_saved += 1
                except Exception as e:
                    logger.debug(f"Error saving relationship {start_id}-{rel_type}->{end_id}: {e}")
                    continue

        return JSONResponse(content={
            "message": "Knowledge graph saved to Neo4j successfully",
            "kg_id": kg_id,
            "nodes_saved": nodes_saved,
            "relationships_saved": relationships_saved
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save KG to Neo4j: {str(e)}")

@app.post("/load_kg_from_neo4j")
async def load_kg_from_neo4j(
    limit: int = Form(1000),
    sample_mode: bool = Form(False),
    load_complete: bool = Form(False),
    kg_label: str = Form(None),
):
    """
    Load the entire KG from Neo4j with optional sampling and filtering.
    """
    require_neo4j()
    try:
        driver = get_graphDB_driver(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("NEO4J_DATABASE"),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            def _serialize_props(raw_props: dict) -> dict:
                props = {}
                for k, v in raw_props.items():
                    if hasattr(v, "isoformat"):
                        props[k] = v.isoformat()
                    else:
                        props[k] = v
                return props

            _seen_node_ids: set = set()

            def _append_node(nodes_list: list, node_obj) -> None:
                if node_obj is None:
                    return
                nid = node_obj.id
                if nid in _seen_node_ids:
                    return
                _seen_node_ids.add(nid)
                nodes_list.append({
                    "id": nid,
                    "labels": list(node_obj.labels),
                    "properties": _serialize_props(dict(node_obj))
                })

            def _append_relationship(rels_list: list, rel_obj, start_node, end_node) -> None:
                if rel_obj is None or start_node is None or end_node is None:
                    return
                rels_list.append({
                    "id": rel_obj.id,
                    "type": rel_obj.type,
                    "start": start_node.id,
                    "end": end_node.id,
                    "properties": _serialize_props(dict(rel_obj))
                })

            # If kg_label is provided, check if it matches a KG name (Document.kgName)
            kg_name_match = 0
            if kg_label:
                kg_name_match = session.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN count(d) AS c",
                    {"kg_name": kg_label}
                ).single()["c"]

            if kg_label and kg_name_match:
                # Load by KG name — only __Entity__ + Document nodes, entity-to-entity edges.
                # Chunk and Mention infrastructure nodes are intentionally excluded from the
                # visualization so the graph matches what the creation flow renders.
                nodes = []
                relationships = []

                # Document node for this KG
                doc_records = session.run(
                    "MATCH (d:Document {kgName: $kg_name}) RETURN d",
                    {"kg_name": kg_label}
                )
                for record in doc_records:
                    _append_node(nodes, record["d"])

                # Load entity nodes scoped to this KG using the kgName property on each entity.
                # Previously used a Document←Chunk→Entity path, which silently excluded any
                # entity that wasn't linked to a chunk (mention-linking miss) → wrong counts.
                entity_nodes = []
                entity_ids = []
                order_clause = "ORDER BY rand()" if sample_mode else ""
                entity_limit = f"LIMIT {limit}" if not load_complete else ""
                entity_records = session.run(
                    f"MATCH (e:__Entity__ {{kgName: $kg_name}})"
                    f" RETURN DISTINCT e {order_clause} {entity_limit}",
                    {"kg_name": kg_label}
                )
                for record in entity_records:
                    entity = record.get("e")
                    if entity is None:
                        continue
                    entity_nodes.append(entity)
                    entity_ids.append(entity.id)
                    _append_node(nodes, entity)

                # Warn if any loaded entity has a kgName that differs from the requested KG
                if entity_ids:
                    shared_check = session.run(
                        "MATCH (e:__Entity__) "
                        "WHERE id(e) IN $eids AND e.kgName <> $kg_name "
                        "RETURN count(DISTINCT e) AS shared_count",
                        {"eids": entity_ids, "kg_name": kg_label}
                    ).single()
                    shared_count = (shared_check or {}).get("shared_count", 0)
                    if shared_count:
                        logging.warning(
                            "%d entities in KG '%s' are also referenced by other KGs — visualization may include shared entities",
                            shared_count, kg_label
                        )

                # Entity-to-entity relationships only
                if entity_ids:
                    entity_rel_records = session.run(
                        "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
                        "WHERE id(a) IN $entity_ids AND id(b) IN $entity_ids "
                        "RETURN r, a AS start, b AS end",
                        {"entity_ids": entity_ids}
                    )
                    for record in entity_rel_records:
                        _append_relationship(relationships, record["r"], record["start"], record["end"])
                    if sample_mode and total_rels > len(relationships):
                        logging.info(
                            "Sampled %d/%d entities — loaded %d/%d relationships (remainder involve unloaded nodes)",
                            len(entity_ids), total_nodes, len(relationships), total_rels
                        )

                # Read KG creation settings from Document node
                kg_settings = None
                doc_record = session.run(
                    "MATCH (d:Document {kgName: $kg_name}) "
                    "RETURN d.provider AS provider, d.model AS model, "
                    "d.embeddingModel AS embeddingModel, d.maxChunks AS maxChunks LIMIT 1",
                    {"kg_name": kg_label},
                ).single()
                if doc_record:
                    kg_settings = {
                        "provider": doc_record["provider"],
                        "model": doc_record["model"],
                        "embeddingModel": doc_record["embeddingModel"],
                        "maxChunks": doc_record["maxChunks"],
                    }

                stats = {
                    "total_nodes_in_db": total_nodes,
                    "total_relationships_in_db": total_rels,
                    "loaded_nodes": len(nodes),
                    "loaded_relationships": len(relationships),
                    "sample_mode": sample_mode,
                    "complete_import": load_complete,
                }
                return JSONResponse(content={
                    "kg_id": str(uuid.uuid4()),
                    "kg_name": kg_label,
                    "graph_data": {"nodes": nodes, "relationships": relationships},
                    "stats": stats,
                    "kg_settings": kg_settings,
                })

            # Fallback: no kg_label match — load all __Entity__ nodes and their
            # entity-to-entity relationships only (exclude Document/Chunk/Mention).
            order_clause = "ORDER BY rand()" if sample_mode else ""
            limit_clause = f"LIMIT {limit}" if not load_complete else ""
            node_query = f"MATCH (n:__Entity__) RETURN n {order_clause} {limit_clause}"

            nodes = []
            for record in session.run(node_query):
                _append_node(nodes, record["n"])

            loaded_node_ids = [node["id"] for node in nodes]  # application-level id strings, not Neo4j internal IDs

            relationships = []
            query = "MATCH (n:__Entity__)-[r]->(m:__Entity__) WHERE n.id IN $node_ids AND m.id IN $node_ids RETURN r, n AS start, m AS end"
            rel_params = {"node_ids": loaded_node_ids}

            for record in session.run(query, rel_params):
                _append_relationship(relationships, record["r"], record["start"], record["end"])

        stats = {
            "total_nodes_in_db": total_nodes,
            "total_relationships_in_db": total_rels,
            "loaded_nodes": len(nodes),
            "loaded_relationships": len(relationships),
            "sample_mode": sample_mode,
            "complete_import": load_complete,
        }
        return JSONResponse(content={
            "kg_id": str(uuid.uuid4()),
            "kg_name": kg_label,
            "graph_data": {"nodes": nodes, "relationships": relationships},
            "stats": stats,
            "kg_settings": None,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_csv")
async def validate_csv(csv_file: UploadFile = File(...)):
    """
    Validate CSV file format and structure for medical reports.
    """
    csv_path = None
    try:
        logger.info("Validating CSV file: %s", csv_file.filename)

        # Save uploaded file temporarily
        data = await csv_file.read()
        tmp_dir = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, f"validate_{uuid.uuid4()}.csv")

        await asyncio.to_thread(lambda: open(csv_path, "wb").write(data))

        # Initialize CSV processor
        processor = MedicalReportCSVProcessor(delimiter='|')

        # Validate format (blocking I/O — run in thread)
        validation_result = await asyncio.to_thread(processor.validate_csv_format, csv_path)

        return JSONResponse(content={
            "is_valid": validation_result.get("is_valid", False),
            "delimiter": validation_result.get("delimiter", "|"),
            "num_columns": validation_result.get("num_columns", 0),
            "num_rows": validation_result.get("num_rows", 0),
            "field_mappings_count": len(validation_result.get("field_mappings", {})),
            "columns": validation_result.get("columns", []),
            "field_mappings": validation_result.get("field_mappings", {}),
            "errors": validation_result.get("validation_errors", [])
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CSV validation error")
        raise HTTPException(status_code=500, detail=f"CSV validation failed: {str(e)}")
    finally:
        if csv_path:
            try:
                os.unlink(csv_path)
            except OSError:
                pass

@app.post("/bulk_process_csv")
async def bulk_process_csv(
    csv_file: UploadFile = File(...),
    text_column: str = Form("full_report_text", description="CSV column containing the document text"),
    id_column: str = Form(None, description="CSV column to use as document ID (defaults to row index)"),
    kg_name: str = Form(None, description="Name for the resulting knowledge graph (auto-generated if omitted)"),
    batch_size: int = Form(50, description="Number of documents to process per batch"),
    start_row: int = Form(0, description="Starting row number (0-based)"),
    max_chunks: int = Form(20, description="Maximum number of chunks to process per document (for testing)")
):
    """
    Process documents from any CSV in bulk batches, guided by the loaded ontology.

    The ontology defines what entities and relationships to extract.
    Only `text_column` needs to match a column in your CSV.
    """
    require_neo4j()
    csv_path = None
    try:
        logger.info("Starting bulk CSV processing: %s (batch=%d, start=%d)", csv_file.filename, batch_size, start_row)

        # Save uploaded file temporarily
        data = await csv_file.read()
        tmp_dir = tempfile.gettempdir()
        csv_path = os.path.join(tmp_dir, f"bulk_{uuid.uuid4()}.csv")

        await asyncio.to_thread(lambda: open(csv_path, "wb").write(data))

        # Initialize enhanced KG creator for bulk processing
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            raise HTTPException(status_code=500, detail="NEO4J_PASSWORD environment variable is not set")

        llm = TemperatureLockedProvider(
            get_llm_provider("openrouter", "openai/gpt-oss-120b:free"),
            temperature=0.0,
        )

        kg_creator = UnifiedOntologyGuidedKGCreator(
            chunk_size=2000,
            chunk_overlap=300,
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            embedding_model="sentence_transformers",
            max_chunks=max_chunks
        )

        # Process CSV in bulk.
        # Run in a thread so the blocking LLM/Neo4j calls don't block the event loop.
        resolved_kg_name = kg_name or f"bulk_{str(uuid.uuid4())[:8]}"
        bulk_result = await asyncio.to_thread(
            kg_creator.bulk_process_documents,
            csv_path=csv_path,
            text_column=text_column,
            id_column=id_column or None,
            start_row=start_row,
            batch_size=batch_size,
            llm=llm,
            kg_name=resolved_kg_name,
        )
        invalidate_rag_system(f"KG bulk build: {resolved_kg_name}")

        metadata = bulk_result.get("metadata", {})
        kg_id = str(uuid.uuid4())

        return JSONResponse(content={
            "kg_id": kg_id,
            "kg_name": resolved_kg_name,
            "message": f"Successfully processed {metadata.get('total_documents_processed', 0)} documents from CSV",
            "total_documents_processed": metadata.get("total_documents_processed", 0),
            "total_kgs": metadata.get("total_knowledge_graphs", 0),
            "batch_size": batch_size,
            "start_row": start_row,
            "csv_validation": metadata.get("csv_validation", {}),
            "bulk_processing_info": metadata.get("bulk_processing_info", {}),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Bulk CSV processing error")
        raise HTTPException(status_code=500, detail=f"Bulk CSV processing failed: {str(e)}")
    finally:
        if csv_path:
            try:
                os.unlink(csv_path)
            except OSError:
                pass

@app.get("/static/medical_reports_template.csv")
async def serve_csv_template():
    """
    Serve the medical reports CSV template for download.
    """
    from starlette.responses import StreamingResponse
    template_path = None
    try:
        processor = MedicalReportCSVProcessor()

        tmp_dir = tempfile.gettempdir()
        template_path = os.path.join(tmp_dir, f"template_{uuid.uuid4()}.csv")

        await asyncio.to_thread(processor.create_csv_template, template_path, num_sample_rows=3)

        # Read content into memory, then delete the temp file immediately — no threading.Timer needed.
        content = await asyncio.to_thread(lambda: open(template_path, "rb").read())

        headers = {
            "Content-Disposition": 'attachment; filename="medical_reports_template.csv"',
            "Content-Type": "text/csv",
        }

        return StreamingResponse(io.BytesIO(content), headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error serving CSV template")
        raise HTTPException(status_code=500, detail=f"Could not generate CSV template: {str(e)}")
    finally:
        if template_path:
            try:
                os.unlink(template_path)
            except OSError:
                pass


def _probe_openai_models() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    response = httpx.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=8.0,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    return f"reachable ({len(data)} models visible)"


def _probe_openrouter_models() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=8.0,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    return f"reachable ({len(data)} models visible)"


def _probe_deepseek_models() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    response = httpx.get(
        "https://api.deepseek.com/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=8.0,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    return f"reachable ({len(data)} models visible)"


def _probe_gemini_models() -> str:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model = next(iter(genai.list_models()), None)
    if model is None:
        return "reachable (no public models returned)"
    model_name = getattr(model, "name", "unknown-model")
    return f"reachable (sample model: {model_name})"


def _probe_ollama_models() -> str:
    import ollama

    tags = ollama.list()
    models = [m.model for m in getattr(tags, "models", [])]
    if not models:
        return "reachable (no local models pulled yet)"
    return f"reachable ({len(models)} local models, sample: {models[0]})"


async def run_doctor_checks(
    probe_models: bool = False,
    write_probe_dir: Optional[str] = None,
) -> dict:
    import importlib
    from datetime import datetime, timezone

    checks: list[dict] = []
    overall = "ok"

    def _add(name: str, status: str, detail: str = "") -> None:
        nonlocal overall
        checks.append({"check": name, "status": status, "detail": detail})
        if status == "fail":
            overall = "fail"
        elif status == "warn" and overall == "ok":
            overall = "warn"

    # 1. Static assets / package layout
    if (_UI_DIST_DIR / "index.html").exists():
        _add("static_assets", "ok", f"UI assets found in {_UI_DIST_DIR}")
    else:
        _add(
            "static_assets",
            "fail",
            f"Missing UI assets. Checked {_UI_DIST_DIR}; set ONTOGRAPHRAG_UI_DIST_DIR if needed.",
        )

    # 2. Neo4j connectivity
    try:
        _neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        _neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        _neo4j_pw = os.getenv("NEO4J_PASSWORD", "")
        _neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
        driver = get_graphDB_driver(_neo4j_uri, _neo4j_user, _neo4j_pw, _neo4j_db)
        with driver.session(database=_neo4j_db) as session:
            rec = session.run("MATCH (n) RETURN count(n) AS c").single()
            node_count = rec["c"] if rec else 0
        driver.close()
        _add("neo4j", "ok", f"Connected to {_neo4j_uri} ({_neo4j_db}) — {node_count} nodes")
    except Exception as e:
        _add("neo4j", "fail", str(e))

    # 3. Embedding model load + sample vector
    try:
        from ontographrag.kg.utils.common_functions import load_embedding_model

        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
        emb_fn, emb_dim = await asyncio.to_thread(load_embedding_model, embedding_provider)
        test_vec = await asyncio.to_thread(emb_fn.embed_query, "ontographrag readiness probe")
        _add(
            "embedding_model",
            "ok",
            f"{embedding_provider} dim={emb_dim}, sample vector len={len(test_vec)}",
        )
    except Exception as e:
        _add("embedding_model", "fail", str(e))

    # 4. OCR stack
    try:
        import fitz

        _add("pymupdf", "ok", f"fitz version {fitz.version[0]}")
    except ImportError:
        _add("pymupdf", "fail", "PyMuPDF not installed — PDF ingestion will fail")

    try:
        importlib.import_module("surya")
        _add("surya_ocr", "ok", "surya installed — scanned PDF fallback available")
    except ImportError:
        _add("surya_ocr", "warn", "surya not installed — scan fallback unavailable")

    # 5. Ontology parsing readiness
    try:
        import owlready2  # noqa: F401

        probe_ontology = {
            "entity_types": [
                {
                    "id": "Document",
                    "properties": [
                        {"name": "id", "type": "string", "identifier": True},
                        {"name": "title", "type": "string"},
                    ],
                }
            ],
            "relationship_types": [
                {"id": "REFERENCES", "from": "Document", "to": "Document", "cardinality": "many_to_many"}
            ],
        }
        errors = validate_ontology_schema(
            json.dumps(probe_ontology).encode("utf-8"),
            "doctor_probe.json",
        )
        if errors:
            _add("ontology_parse", "fail", "; ".join(errors))
        else:
            _add("ontology_parse", "ok", "JSON ontology validation and OWL loader dependency available")
    except ImportError:
        _add("ontology_parse", "fail", "owlready2 not installed — ontology-guided extraction will fail")
    except Exception as e:
        _add("ontology_parse", "fail", str(e))

    # 6. Filesystem write permissions
    try:
        probe_root = Path(write_probe_dir) if write_probe_dir else _DEFAULT_WRITE_PROBE_DIR
        probe_root.mkdir(parents=True, exist_ok=True)
        probe_file = probe_root / f".ontographrag_write_probe_{os.getpid()}"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        _add("write_permissions", "ok", f"Can write to {probe_root}")
    except Exception as e:
        _add("write_permissions", "fail", str(e))

    # 7. Provider configuration + optional connectivity probes
    provider_envs = [
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
    ]
    configured_providers = [name for name, env_var in provider_envs if os.getenv(env_var)]
    for name, env_var in provider_envs:
        if os.getenv(env_var):
            _add(f"{name}_config", "ok", f"{env_var} is set")
        else:
            _add(f"{name}_config", "warn", f"{env_var} not set")

    try:
        import ollama

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        await asyncio.to_thread(ollama.list)
        configured_providers.append("ollama")
        _add("ollama_config", "ok", f"Ollama reachable at {host}")
    except Exception as e:
        _add("ollama_config", "warn", f"Ollama unavailable: {e}")

    if not configured_providers:
        _add("model_provider_any", "fail", "No configured or reachable model provider found")
    elif not probe_models:
        _add(
            "model_connectivity",
            "warn",
            "Quick doctor mode — provider presence checked, active model probe skipped",
        )
    else:
        model_probe_map = {
            "openai": _probe_openai_models,
            "openrouter": _probe_openrouter_models,
            "gemini": _probe_gemini_models,
            "deepseek": _probe_deepseek_models,
            "ollama": _probe_ollama_models,
        }
        for provider_name in configured_providers:
            probe_fn = model_probe_map.get(provider_name)
            if not probe_fn:
                _add(f"{provider_name}_connectivity", "warn", "No probe available")
                continue
            try:
                detail = await asyncio.to_thread(probe_fn)
                _add(f"{provider_name}_connectivity", "ok", detail)
            except Exception as e:
                _add(f"{provider_name}_connectivity", "warn", str(e))

    # 8. API security / CORS
    if os.getenv("APP_API_KEY"):
        _add("api_key_auth", "ok", "APP_API_KEY set — endpoint auth enabled")
    else:
        _add("api_key_auth", "warn", "APP_API_KEY not set — API is open in current environment")

    origins = os.getenv("ALLOWED_ORIGINS", "*")
    if origins == "*":
        _add("cors", "warn", "ALLOWED_ORIGINS=* — wide-open CORS (fine for local dev)")
    else:
        _add("cors", "ok", f"ALLOWED_ORIGINS restricted to: {origins}")

    summary = {
        "ok": sum(1 for c in checks if c["status"] == "ok"),
        "warn": sum(1 for c in checks if c["status"] == "warn"),
        "fail": sum(1 for c in checks if c["status"] == "fail"),
    }
    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe_models": probe_models,
        "summary": summary,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# /doctor — infrastructure health check (inspired by `mosaicx doctor`)
# ---------------------------------------------------------------------------

@app.get("/doctor")
async def doctor(
    probe_models: bool = Query(False, description="Actively probe configured model providers"),
    write_probe_dir: Optional[str] = Query(None, description="Directory used for the write-permission probe"),
):
    return JSONResponse(content=await run_doctor_checks(probe_models=probe_models, write_probe_dir=write_probe_dir))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "ontographrag.api.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8004")),
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"},
    )
