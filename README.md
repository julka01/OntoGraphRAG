# OntoGraphRAG

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)
[![PyPI](https://img.shields.io/pypi/v/ontographrag?label=PyPI&color=blue)](https://pypi.org/project/ontographrag/)

Turn unstructured documents into schema-consistent knowledge graphs, ask grounded
questions over them, and measure what to trust.

OntoGraphRAG is an ontology-guided KG-RAG system. It builds Neo4j-backed knowledge
graphs from raw text, retrieves over both graph structure and chunk vectors, and
exposes answer-grounding, provenance, and uncertainty signals. Unlike free-form
GraphRAG, extraction is constrained to a schema you supply, so the same concept
lands in the same type across every document.

![OntoGraphRAG UI](assets/readme/ontographrag-ui-example.png)

## Quick start

Requires Python 3.11+, Neo4j 5.0+, and one LLM provider key (or local Ollama).
Node.js 18+ is only needed for source checkouts that rebuild the frontend.

```bash
# 1. Install from PyPI
python -m pip install ontographrag

# Or install the latest source snapshot from GitHub
python -m pip install "ontographrag @ git+https://github.com/julka01/OntoGraphRAG.git"

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

Source checkout (development, frontend changes, or benchmarks):

```bash
git clone https://github.com/julka01/OntoGraphRAG.git && cd OntoGraphRAG
uv sync && source .venv/bin/activate
cd frontend && npm install && npm run build && cd ..
docker compose up -d neo4j
python -m ontographrag.cli serve
```

## CLI

| Command | Purpose |
|---------|---------|
| `ontograph serve [--port 8004]` | Start the web app + REST API |
| `ontograph doctor` | Readiness check: Neo4j, provider keys, UI assets |
| `ontograph ingest report.pdf --kg-name demo` | Build a named KG from a document |
| `ontograph ask "question" --kg-name demo` | Ask a grounded question |
| `ontograph explore list` / `show <kg>` | List saved graphs / show one graph's stats |
| `ontograph datasets` | List benchmark datasets and expected local paths |
| `ontograph prepare <dataset>` | Download/prepare one benchmark dataset |
| `ontograph prepare-bioasq-corpus` | Build the shared PubMed corpus for BioASQ |
| `ontograph evaluate --datasets ... [flags]` | Run the benchmark suite ([flags](experiments/README.md)) |
| `ontograph runtime-regression` | End-to-end smoke test against a live server |

`ingest`, `ask`, and `explore` are thin wrappers around the server endpoints;
use `--server` and `--api-key` (or `ONTOGRAPHRAG_API_KEY`) for remote/secured
servers.

## API

Served with the GUI on port **8004**. Interactive docs at
`http://localhost:8004/docs`, schema at `/openapi.json`. If `APP_API_KEY` is
set, requests need `X-API-Key: <key>` (or `?api_key=`); health endpoints and
static assets stay public.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/create_ontology_guided_kg` | Build an ontology-guided KG from a file upload (5/min per IP) |
| `POST` | `/extract_graph` | Extract a raw KG (no ontology) from a file |
| `GET`  | `/kg_progress_stream` | SSE stream of KG build progress |
| `POST` | `/chat` | Grounded QA, optionally scoped to `kg_name` (30/min per IP) |
| `POST` | `/kg/create` · `GET /kg/list` · `GET /kg/{kg}` · `GET /kg/{kg}/entities` · `DELETE /kg/{kg}` | Named KG management |
| `POST` | `/save_kg_to_neo4j` · `/load_kg_from_neo4j` · `/load_kg_from_file` · `/clear_kg` | Neo4j persistence |
| `POST` | `/validate_csv` · `/bulk_process_csv` | CSV bulk processing (template at `/static/medical_reports_template.csv`) |
| `GET`  | `/models/{provider}` | List models for a provider |
| `GET`  | `/health` · `/health/neo4j` · `/ready` · `/doctor` | Health and readiness |

`POST /create_ontology_guided_kg` takes multipart `file` (PDF/TXT/CSV/JSON/XML,
≤ 50 MB), `provider`, `model`, `embedding_model`, optional `ontology_file`
(.owl/.rdf/.ttl/.xml), `max_chunks`, `kg_name`, `enable_coreference_resolution`.

`POST /chat` takes JSON `question`, `provider_rag`, `model_rag`, optional
`kg_name`, `document_names`, `session_id`. The response `info` block carries
`sources`, chunk/entity/relationship counts, and the two per-answer trust
signals: `structural_support` (graph-path support) and `grounding_support`
(evidence entailment). The legacy `confidence` field remains for compatibility.

```bash
curl -X POST http://localhost:8004/create_ontology_guided_kg \
  -F "file=@report.pdf" -F "kg_name=demo" \
  -F "provider=openrouter" -F "model=openai/gpt-4o-mini"

curl -N http://localhost:8004/kg_progress_stream    # build progress (SSE)

curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "kg_name": "demo",
       "provider_rag": "openrouter", "model_rag": "openai/gpt-4o-mini"}'
```

## Web UI

React + TypeScript SPA (`frontend/`, Vite) served at `http://localhost:8004`.
Wheels ship built assets; source checkouts need `npm run build` once.

- **Build KG** — upload a document, pick provider/model, optionally attach an
  ontology; extraction progress streams live via SSE.
- **Graph view** — interactive force-directed network with node details,
  search (dims non-matches), and per-type filters.
- **Named KGs** — create, list, and switch between saved graphs.
- **Chat** — answers cite source chunks, highlight the entities used, and show
  inline **trust pills**: *Structural* (graph-path support) and *Grounding*
  (evidence entailment). History persists in `localStorage`.

## What it does

- **Ontology-guided extraction** — entities and relations validated against
  your OWL/RDF/JSON schema; no synonym explosion or type drift across a corpus.
- **Routed hybrid retrieval** — entity-first linking, provenance-aware graph
  expansion, PPR-style chunk scoring, retriever-first and vector fallbacks,
  one interface shared with a vanilla-RAG baseline.
- **Trust signals** — structural (graph-path) and grounding
  (evidence-entailment) support per answer; full uncertainty suite in the
  evaluation pipeline.
- **Provider-agnostic** — OpenRouter, OpenAI, Gemini, DeepSeek, HuggingFace,
  local Ollama; selectable per request.

## Benchmarks & experiment tracking

**[experiments/README.md](experiments/README.md)** covers the benchmark runner
(vanilla RAG vs KG-RAG, uncertainty suite, retrieval lock-in study), datasets,
and flags. `ontograph datasets` shows expected local paths, and
`ontograph prepare <dataset>` helps with local dataset setup (including direct
downloads where available); `ontograph evaluate` wraps the runner.

Every benchmark run logs to **Weights & Biases** automatically (entity
`WANDB_ENTITY`, project `mirage-kg-evaluation`): manifest config, per-question
tables, per-config AUROC/AUREC summaries, and metric charts. Authenticate with
`wandb login` or `WANDB_API_KEY`; set `WANDB_MODE=offline` (or `disabled`) to
run without an account. Local artefacts under `results/runs/<run_id>/` are
always written either way.

## Configuration

[`.env.example`](.env.example) documents the core variables: Neo4j connection,
provider keys, embeddings, W&B, API security, and server host/port. Advanced
behaviour is controlled by env-var families:

| Group | Variables | Effect |
|-------|-----------|--------|
| Retrieval chunking | `RETRIEVAL_CHUNK_SIZE` (256), `RETRIEVAL_CHUNK_OVERLAP` (64) | Retrieval sub-chunk size at KG build time |
| Retrieval behaviour | `ONTOGRAPHRAG_RETRIEVAL_PROFILE`, `ONTOGRAPHRAG_QUERY_FUSION`, `ONTOGRAPHRAG_RERANKER*`, `ONTOGRAPHRAG_LATE_INTERACTION*` | Retrieval profile, query fusion, reranking, late interaction |
| Answer guardrails | `ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL[_MODE]` | Runtime answer-quality guardrail |
| KG build features | `KG_ENABLE_*` (soft entity linking, UMLS linking, claim extraction, self-reflection, graph summaries, fragmentation repair, cross-passage recovery, triple re-verify, anchor passes) | Optional extraction/enrichment passes |
| KG build tuning | `KG_SELF_CONSISTENCY_N`, `KG_FEW_SHOT_EXAMPLE_COUNT`, `KG_RELATION_PROMPT_ENTITY_CAP`, `KG_CROSS_*_RELATION_WINDOW`, `KG_UMLS_SPACY_MODEL` | Extraction prompt and window parameters |
| CLI client | `ONTOGRAPHRAG_API_KEY` | API key for `ingest`/`ask`/`explore` against secured servers |

Provider keys: `openrouter` → `OPENROUTER_API_KEY` (free-tier models
available), `openai` → `OPENAI_API_KEY`, `gemini` → `GEMINI_API_KEY`,
`deepseek` → `DEEPSEEK_API_KEY`, `huggingface` → `HF_API_TOKEN`,
`ollama` → none (set `OLLAMA_HOST` if non-default). Benchmark sweeps and
retrieval thresholds are CLI-flag driven — see
[experiments/README.md](experiments/README.md).

## Architecture

**KG build**: ingest → chunk (extraction windows + retrieval sub-chunks) →
ontology load → per-chunk LLM extraction → cross-chunk relation pass →
entity harmonisation (synonym merge, most-specific type) → provenance
stamping → specificity stats (hub down-weighting) → embedding (name-centred
entity vectors) → Neo4j write, tagged by `kgName`, progress via SSE.

**RAG query**: entity-first seeding (alias match + per-entity ANN) →
question-local provenance-aware traversal → PPR-style graph scoring →
retriever-first graph expansion when anchoring is weak → vector fallback →
evidence organised into chain-style blocks → answer synthesis with
Structural/Grounding trust signals.

```
ontographrag/
├── api/app.py               # FastAPI app, all endpoints, serves UI assets
├── cli.py                   # `ontograph` CLI
├── kg/                      # builders (extraction/harmonisation), loaders (Neo4j),
│                            #   chunking, csv_processor, utils
├── rag/                     # systems (KG-RAG + vanilla baseline), guardrails,
│                            #   reranking, retrieval_sampling
├── schemas/models.py        # Pydantic models
└── providers/model_providers.py  # LLM + embedding providers
experiments/                 # benchmark runner + uncertainty suite
frontend/                    # React + TypeScript web UI
MIRAGE/rawdata/              # local ignored benchmark data (downloaded separately)
```

| Spec | Value |
|------|-------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim), local CPU by default |
| Vector similarity | Cosine, default threshold 0.08 |
| Retrieval sub-chunks | 256 chars / 64 overlap (env-overridable) |
| File upload limit | 50 MB |
| Rate limits | Chat 30/min, KG build 5/min (per IP) |

## Docker

```bash
docker compose up -d neo4j   # Neo4j only (development)
docker compose up -d         # Full stack (Neo4j + API)
# Neo4j Browser → http://localhost:7474 (bolt://localhost:7687)
```

## Documentation

| Topic | Reference |
|-------|-----------|
| KG construction pipeline | [KG_GENERATION_PIPELINE.md](KG_GENERATION_PIPELINE.md) |
| Evaluation & uncertainty metrics | [EVALUATION_METRICS.md](EVALUATION_METRICS.md) |
| Benchmark runner, datasets, flags | [experiments/README.md](experiments/README.md) |
| REST API | `http://localhost:8004/docs` (interactive) |

## Citation

```bibtex
@article{julka2026ontographrag,
  title   = {When Answer Agreement Fails:
             Retrieval-State Lock-In in Retrieval-Augmented Generation},
  author  = {Julka, Sahib},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

<!-- TODO: replace arXiv:XXXX.XXXXX with the assigned identifier on release. -->

## License

MIT — see [LICENSE](LICENSE). Issues:
[GitHub Issues](https://github.com/julka01/OntoGraphRAG/issues).
