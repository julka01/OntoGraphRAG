# OntographRAG

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)

Turn unstructured documents into schema-consistent knowledge graphs, ask grounded
questions over them, and measure what to trust.

OntographRAG is an ontology-guided KG-RAG system. It builds Neo4j-backed knowledge
graphs from raw text, retrieves over both graph structure and chunk vectors, and
exposes answer-grounding, provenance, and uncertainty signals. Unlike free-form
GraphRAG, extraction is constrained to a schema you supply, so the same concept
lands in the same type across every document.

![OntographRAG UI](assets/readme/ontographrag-ui-example.png)

## Requirements

- Python 3.11+
- Neo4j 5.0+ (Docker or local install)
- At least one LLM provider key (OpenRouter, OpenAI, Gemini, DeepSeek,
  HuggingFace) or a local Ollama instance
- Node.js 18+ and npm — only for source checkouts that rebuild the React
  frontend; pip wheels ship with packaged UI assets
- 8 GB RAM minimum (16 GB recommended for large documents)

## Quick start

```bash
# 1. Install
python -m pip install "ontographrag @ git+https://github.com/julka01/OntographRAG.git"

# 2. Start Neo4j
docker run -d --name ontographrag-neo4j \
  -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

# 3. Configure (copy .env.example to .env and fill in keys)
export NEO4J_URI=bolt://localhost:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=password
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY, GEMINI_API_KEY, ...

# 4. Check readiness and launch the web app
ontograph doctor
ontograph serve                  # → http://localhost:8004
```

See [`.env.example`](.env.example) and the [Configuration](#configuration)
section below for all variables.

For a source checkout (development, frontend changes, or benchmarks):

```bash
git clone https://github.com/julka01/OntographRAG.git && cd OntographRAG
uv sync && source .venv/bin/activate
cd frontend && npm install && npm run build && cd ..   # build the UI
docker compose up -d neo4j
python -m ontographrag.cli serve   # → http://localhost:8004/docs
```

## Command-line interface

`ontograph` (entry point for `ontographrag.cli:main`) exposes the full
product surface:

| Command | Purpose |
|---------|---------|
| `ontograph serve [--port 8004]` | Start the web app + REST API in one process |
| `ontograph doctor` | Readiness check: Neo4j connectivity, provider keys, UI assets |
| `ontograph ingest report.pdf --kg-name demo` | Build a named KG from a document |
| `ontograph ask "question" --kg-name demo` | Ask a grounded question against a named KG |
| `ontograph explore list` / `ontograph explore show <kg>` | List saved graphs / show one graph's stats |
| `ontograph datasets` | List supported benchmark datasets and their expected local paths |
| `ontograph prepare <dataset>` | Download/prepare one benchmark dataset into `MIRAGE/rawdata/` |
| `ontograph prepare-bioasq-corpus` | Build the shared PubMed abstract corpus required by BioASQ |
| `ontograph evaluate --datasets ... [flags]` | Run the benchmark suite (wraps `experiments/experiment.py`; same flags) |
| `ontograph runtime-regression` | Smoke-test the live server end to end (build, ask, probe) |

`ontograph ingest`, `ask`, and `explore` are thin wrappers around the server
endpoints; point them at a remote server with `--server` and authenticate with
`--api-key` or `ONTOGRAPHRAG_API_KEY`.

## API

OntographRAG serves the GUI and the FastAPI server from the same process
(default port **8004**; change with `ontograph serve --port ...`). The live
API docs are the source of truth:

- Swagger UI: `http://localhost:8004/docs`
- OpenAPI schema: `http://localhost:8004/openapi.json`

### Authentication

If `APP_API_KEY` is set, API requests must send the key in the `X-API-Key`
header or as `?api_key=...`. Health endpoints and static assets stay public so
the web app can still load. Unset = open (development mode).

### Knowledge graph — build & query

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/create_ontology_guided_kg` | Build an ontology-guided KG from a file upload (rate limit: 5/min per IP) |
| `POST` | `/extract_graph` | Extract a raw KG (no ontology) from a file |
| `POST` | `/load_kg_from_file` | Load a graph from file into Neo4j |
| `GET`  | `/kg_progress_stream` | Server-Sent Events stream of KG build progress |

`POST /create_ontology_guided_kg` multipart fields: `file` (PDF/TXT/CSV/JSON/XML,
≤ 50 MB), `provider`, `model`, `embedding_model`, optional `ontology_file`
(.owl/.rdf/.ttl/.xml), `max_chunks`, `kg_name`, and
`enable_coreference_resolution`.

### Named KG management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/kg/create` | Create a named KG record |
| `GET`    | `/kg/list` | List all KGs with document counts |
| `GET`    | `/kg/{kg_name}` | Stats for a specific KG |
| `GET`    | `/kg/{kg_name}/entities` | List entities in a KG |
| `DELETE` | `/kg/{kg_name}` | Delete a KG |

### Neo4j management & health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/save_kg_to_neo4j` | Persist an in-memory KG to Neo4j |
| `POST` | `/load_kg_from_neo4j` | Load a KG from Neo4j by name |
| `POST` | `/clear_kg` | Delete all nodes and relationships |
| `GET`  | `/health`, `/ready`, `/doctor` | Liveness, readiness, and full diagnostic checks |
| `GET`  | `/health/neo4j` | Neo4j connectivity check |

### Chat / RAG

`POST /chat` (rate limit: 30/min per IP). JSON body: `question` (required),
`provider_rag`, `model_rag`, optional `kg_name`, `document_names`,
`session_id`. The response carries the answer plus an `info` block with
`sources`, chunk/entity/relationship counts, and the two per-answer trust
signals the UI surfaces:

- `structural_support` — graph-path support for the answer
- `grounding_support` — how well the retrieved evidence supports the answer

The legacy `confidence` field is still returned for compatibility but is no
longer the primary app-facing summary.

### CSV bulk processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/validate_csv` | Validate a CSV before bulk processing |
| `POST` | `/bulk_process_csv` | Build KGs from all rows of a CSV (`text_column`, optional `id_column`, `start_row`, `batch_size`) |
| `GET`  | `/static/medical_reports_template.csv` | Downloadable CSV template |

### Models

`GET /models/{provider}` — list available models for a provider.

### cURL examples

```bash
# Build a KG with an ontology
curl -X POST http://localhost:8004/create_ontology_guided_kg \
  -F "file=@report.pdf" \
  -F "kg_name=demo" \
  -F "provider=openrouter" \
  -F "model=openai/gpt-4o-mini" \
  -F "ontology_file=@schema.owl"

# Stream build progress (SSE)
curl -N http://localhost:8004/kg_progress_stream

# Ask a question against a named KG
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "kg_name": "demo",
    "provider_rag": "openrouter",
    "model_rag": "openai/gpt-4o-mini"
  }'

# List KGs / health
curl http://localhost:8004/kg/list
curl http://localhost:8004/health/neo4j
```

## Web UI

The web interface is a React + TypeScript single-page app (`frontend/`,
Vite) served at `http://localhost:8004`. Pip wheels include the built UI
under `ontographrag/api/static/`; source checkouts fall back to
`frontend/dist/`, so run `cd frontend && npm install && npm run build` once
after clone or after frontend changes.

**Knowledge graph panel**

- **Build KG** — upload a document (PDF, TXT, CSV, JSON, XML ≤ 50 MB), choose
  provider/model, optionally attach an ontology file. Extraction progress
  streams to the UI in real time via SSE and the graph loads automatically.
- **Graph visualisation** — interactive force-directed network; node size
  scales with degree; click a node for its detail panel (type, properties,
  connected nodes).
- **Search & filter** — search dims non-matching nodes and shows match counts;
  per-type checkboxes filter with node/edge counts.
- **Named KG management** — create, list, and switch between saved graphs.

**Chat panel**

- Ask questions against the active KG; answers cite source chunks.
- **Trust pills** — each response surfaces the two lightweight support
  signals inline: *Structural* (graph-path support) and *Grounding*
  (evidence-entailment support).
- Entities used in the answer are highlighted in the graph; chat history is
  persisted in `localStorage`.

## What it does

- **Ontology-guided extraction** — bring an OWL/RDF/JSON schema; every entity and
  relationship is validated against it, eliminating synonym explosion and type
  drift across a corpus.
- **Routed hybrid retrieval** — entity-first linking, provenance-aware graph
  expansion, PPR-style chunk scoring, retriever-first fallback, and a clean vector
  fallback under one interface shared with a vanilla-RAG baseline.
- **Trust signals** — structural (graph-path) and grounding (evidence-entailment)
  support are surfaced per answer; the full uncertainty suite is computed in the
  evaluation pipeline.
- **Provider-agnostic** — OpenRouter, OpenAI, Gemini, DeepSeek, HuggingFace, and
  local Ollama, selectable per request.

## Benchmarks & experiment tracking

To reproduce the benchmarks (vanilla RAG vs KG-RAG, uncertainty suite,
retrieval lock-in study), see **[experiments/README.md](experiments/README.md)**
for the runner, datasets, download paths, and the live flag list.
`ontograph datasets` and `ontograph prepare <dataset>` automate dataset
download; `ontograph evaluate` wraps the runner.

**Weights & Biases integration** — every benchmark run
(`ontograph evaluate` / `python -m experiments.experiment`) initialises a W&B
run automatically: entity from `WANDB_ENTITY`, project
`mirage-kg-evaluation`, run name = run id. Authenticate once with
`wandb login` or set `WANDB_API_KEY`; set `WANDB_MODE=offline` (or
`disabled`) to run without an account. Each run logs the manifest (dataset,
model, seed, git commit, embedding provider, evaluation mode), per-question
tables, per-config AUROC/AUREC summary tables, and metric charts.
Results are also always written locally under `results/runs/<run_id>/`
(per-question JSON, manifest, summaries) regardless of W&B mode.

## Configuration

Copy `.env.example` to `.env` and fill in your values. Core variables:

```bash
# ── Neo4j ─────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# ── LLM providers (at least one required) ─────────────────────────────────
OPENROUTER_API_KEY=...        # recommended; free-tier models available
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
HF_API_TOKEN=...
OLLAMA_HOST=http://localhost:11434   # local models
LLM_TIMEOUT_SECONDS=120

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=sentence_transformers   # local default; or openai
EMBEDDING_MODEL=                           # override the default MiniLM model
OPENAI_EMBEDDING_MODEL=                    # used when EMBEDDING_PROVIDER=openai

# ── Weights & Biases (experiment tracking) ─────────────────────────────────
WANDB_API_KEY=                # or `wandb login`; auth for run logging
WANDB_ENTITY=                 # W&B entity/team (default: julka01)
WANDB_MODE=online             # online | offline | disabled

# ── Security (production) ──────────────────────────────────────────────────
APP_API_KEY=                  # set to enforce X-API-Key auth on the server
ALLOWED_ORIGINS=*             # comma-separated CORS origins

# ── Server ──────────────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8004
```

| Variable group | Examples | Effect |
|----------------|----------|--------|
| Retrieval chunking | `RETRIEVAL_CHUNK_SIZE` (default 256), `RETRIEVAL_CHUNK_OVERLAP` (default 64) | Size/overlap of retrieval sub-chunks at KG build time |
| Retrieval behaviour | `ONTOGRAPHRAG_RETRIEVAL_PROFILE`, `ONTOGRAPHRAG_QUERY_FUSION`, `ONTOGRAPHRAG_RERANKER*`, `ONTOGRAPHRAG_LATE_INTERACTION*` | Retrieval profile, query fusion, cross-encoder reranking, late-interaction retrieval |
| Answer guardrails | `ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL`, `..._MODE` | Runtime answer-quality guardrail toggle/mode |
| KG build features | `KG_ENABLE_SOFT_ENTITY_LINKING`, `KG_ENABLE_UMLS_LINKING`, `KG_ENABLE_CLAIM_EXTRACTION`, `KG_ENABLE_SELF_REFLECTION`, `KG_ENABLE_GRAPH_SUMMARIES`, `KG_ENABLE_FRAGMENTATION_REPAIR`, `KG_ENABLE_CROSS_PASSAGE_RELATION_RECOVERY`, `KG_ENABLE_LOW_CONFIDENCE_TRIPLE_REVERIFY`, `KG_ENABLE_ANCHOR_CONSTRAINED_EXTRACTION`, `KG_ENABLE_ANCHOR_COVERAGE_SUPPLEMENT` | Optional extraction/enrichment passes during KG construction |
| KG build tuning | `KG_SELF_CONSISTENCY_N`, `KG_FEW_SHOT_EXAMPLE_COUNT`, `KG_RELATION_PROMPT_ENTITY_CAP`, `KG_CROSS_CHUNK_RELATION_WINDOW`, `KG_CROSS_PASSAGE_RELATION_WINDOW`, `KG_CROSS_SECTION_RELATION_WINDOW`, `KG_UMLS_SPACY_MODEL` | Extraction prompt and window parameters |
| CLI client | `ONTOGRAPHRAG_API_KEY` | API key used by `ontograph ingest/ask/explore` against a secured server |

Chunking, retrieval thresholds, and benchmark sweeps are otherwise controlled
by constructor defaults and CLI flags rather than env vars; for the live
benchmark flags, [experiments/README.md](experiments/README.md) is the source
of truth.

### Providers

| Provider | Env var | Notes |
|----------|---------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | Recommended; free-tier models available |
| `openai` | `OPENAI_API_KEY` | GPT-4o family |
| `gemini` | `GEMINI_API_KEY` | Gemini Pro, Flash |
| `deepseek` | `DEEPSEEK_API_KEY` | DeepSeek Chat/Coder |
| `huggingface` | `HF_API_TOKEN` | HuggingFace Inference API |
| `ollama` | — | Local models; set `OLLAMA_HOST` if non-default |

## Architecture

**KG build pipeline**

1. **Ingest** — file uploaded; PDF text extracted, plaintext decoded
2. **Chunk** — deterministic overlapping windows (large extraction chunks +
   small retrieval sub-chunks)
3. **Ontology load** — custom `.owl`/`.ttl` parsed, or free-form extraction if
   none supplied
4. **LLM extraction** — each chunk processed with an ontology-constrained
   prompt; entities and relationships returned as structured JSON
5. **Cross-chunk extraction** — adjacent chunk pairs get a second pass for
   span-overflow relations
6. **Entity harmonisation** — duplicates and synonyms merged; alias surfaces
   retained; the most specific compatible type wins
7. **Relationship provenance** — edges stamped with chunk, passage, and
   question-local provenance
8. **Specificity stats** — entities receive `node_specificity` so generic hubs
   are down-weighted at retrieval time
9. **Embed** — chunks and entities embedded; entity vectors name-centred for
   clean short-mention lookup
10. **Write** — nodes, relationships, embeddings, and provenance stored in
    Neo4j, tagged by `kgName`; progress streamed via SSE

**RAG query pipeline**

1. **Entity-first seeding** — named mentions extracted from the question;
   symbolic alias matching plus per-entity ANN anchors retrieval
2. **Question-local traversal** — provenance-aware traversal stays local to
   the KG scope (and question bundle on benchmark datasets)
3. **Graph scoring** — entity neighbourhoods ranked with PPR-style support
   flow
4. **Retriever-first graph expansion** — weak anchoring triggers dense
   retrieval that seeds a second graph-expansion pass
5. **Fallback retrieval** — weak graph signal falls back to vector retrieval
   rather than forcing a brittle graph answer
6. **Evidence organisation** — graph paths and supporting passages grouped
   into chain-style evidence blocks
7. **Answer synthesis** — the LLM answers from the evidence block; the app
   surfaces the Structural and Grounding support signals

**Module layout**

```
ontographrag/
├── api/app.py               # FastAPI application, all endpoints; serves UI assets
├── cli.py                   # `ontograph` CLI (serve, ingest, ask, evaluate, ...)
├── kg/
│   ├── builders/            # Ontology-guided extraction, harmonisation, enrichment, storage
│   ├── loaders/             # Neo4j data access and KG load/save
│   ├── chunking.py          # Hierarchical chunking
│   ├── csv_processor.py     # CSV validation and bulk processing
│   └── utils/               # Shared helpers and constants
├── rag/
│   ├── systems/             # enhanced_rag_system (KG-RAG) and vanilla_rag_system (baseline)
│   ├── answer_guardrails.py # Runtime answer-quality guardrails
│   ├── reranking.py         # Cross-encoder reranking
│   └── retrieval_sampling.py
├── schemas/models.py        # Pydantic models: Chunk, Entity, Relationship, KGContext, ...
└── providers/model_providers.py  # LLM + embedding provider abstractions

experiments/                 # Benchmark runner + uncertainty suite (see experiments/README.md)
frontend/                    # React + TypeScript web UI (Vite)
shared/                      # Shared text/embedding utilities
MIRAGE/rawdata/              # Local ignored benchmark-data workspace (downloaded separately)
```

**Key specs**

| Component | Detail |
|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim), runs locally on CPU by default |
| Vector similarity | Cosine; default retrieval threshold 0.08 |
| Retrieval sub-chunks | 256 chars, 64 overlap (env-overridable) |
| Graph database | Neo4j 5.0+ with vector indexes |
| File upload limit | 50 MB |
| Rate limits | Chat 30 req/min per IP; KG build 5 req/min per IP |

## Docker

```bash
docker compose up -d neo4j   # Neo4j only (recommended for development)
docker compose up -d         # Full stack (Neo4j + API server)
docker compose logs -f       # Logs
docker compose down          # Stop

# Neo4j Browser → http://localhost:7474 (bolt://localhost:7687)
```

## Documentation

| Topic | Reference |
|-------|-----------|
| KG construction pipeline | [KG_GENERATION_PIPELINE.md](KG_GENERATION_PIPELINE.md) |
| Evaluation & uncertainty metrics | [EVALUATION_METRICS.md](EVALUATION_METRICS.md) |
| Uncertainty metric formulations | [UNCERTAINTY_METRICS.md](UNCERTAINTY_METRICS.md) |
| Benchmark runner, datasets, and flags | [experiments/README.md](experiments/README.md) |
| REST API | This README plus `http://localhost:8004/docs` (interactive) |

## Citation

If you use OntographRAG in your research, please cite:

```bibtex
@article{julka2026ontographrag,
  title   = {When Confidence Follows the Wrong Path: Decomposing Answer,
             Evidence, and Graph Support in Knowledge-Graph RAG},
  author  = {Julka, Sahib},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

<!-- TODO: replace arXiv:XXXX.XXXXX with the assigned identifier on release. -->

## License

MIT — see [LICENSE](LICENSE). Issues and feature requests:
[GitHub Issues](https://github.com/julka01/OntographRAG/issues).
