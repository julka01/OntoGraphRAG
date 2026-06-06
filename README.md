# OntographRAG

[![Version](https://img.shields.io/badge/version-1.0.0--rc1-blue.svg)](https://github.com/julka01/OntographRAG)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-brightgreen.svg)](https://neo4j.com/)

**Turn unstructured documents into schema-consistent knowledge graphs. Explore them visually. Ask grounded questions. Evaluate what to trust.**

OntographRAG is an ontology-guided KG-RAG system for document intelligence. It builds Neo4j-backed knowledge graphs from raw text, retrieves over both graph structure and chunk vectors, and exposes answer-grounding, provenance, and uncertainty signals for downstream use. The research pipeline also includes strict entity-first retrieval profiles for studying retrieval lock-in: cases where a graph retriever supplies stable context, the model gives stable answers, and ordinary output-variance metrics can become uninformative.

## Example UI

![OntoGraphRAG UI example](assets/readme/ontographrag-ui-example.png)

The project is organized around **three interactive workflows plus one evaluation pipeline**:

1. **Ingest**: turn documents or benchmark corpora into named knowledge graphs.
2. **Explore**: inspect entities, relationships, provenance, and graph structure.
3. **Ask**: query the active graph with grounded RAG.
4. **Evaluate**: benchmark KG-RAG against vanilla RAG and compare uncertainty measures from the CLI.

Works across domains such as biomedical literature, legal documents, financial reports, and technical manuals. It is especially useful when schema consistency matters across many documents.

---

## Quick Start

### Pip users: install the package

For the frozen paper branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "ontographrag @ git+https://github.com/julka01/OntographRAG.git@paper-submission"
```

If you build a local wheel from this checkout:

```bash
uv build
python -m pip install dist/ontographrag-1.0.0rc1-py3-none-any.whl
```

The pip package installs the `ontograph` / `ontographrag` CLI and includes the current built web UI assets. You still need a running Neo4j instance:

```bash
docker run -d --rm --name ontographrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5

export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password

ontograph doctor
ontograph serve --port 8000
```

Open `http://localhost:8000`.

### App users: ingest, explore, ask

From a source checkout, the most reliable form is:
- `.venv/bin/python -m ontographrag.cli ...`

If you want the shorter `ontograph ...` command, run `uv sync` after pulling the latest changes so the console script is installed into your virtualenv.

```bash
# 1. Clone and install in editable/source mode
git clone https://github.com/julka01/OntographRAG.git
cd OntographRAG
uv sync
source .venv/bin/activate

# 2. Build the React frontend for source-checkout serving.
# Pip wheels include packaged UI assets, so this step is only needed here.
cd frontend && npm install && npm run build && cd ..

# 3. Start Neo4j
docker compose up -d neo4j

# 4. Check readiness
.venv/bin/python -m ontographrag.cli doctor

# 5. Start the app
.venv/bin/python -m ontographrag.cli serve
# or directly:
.venv/bin/uvicorn ontographrag.api.app:app --host 0.0.0.0 --port 8000

# 6. Open the GUI
# → http://localhost:8000
```

Happy path in the app:
- select a file
- optionally attach an ontology
- create a named KG
- inspect the graph
- ask questions against the active KG

### Benchmark users: evaluate

```bash
# 1. Clone and install
git clone https://github.com/julka01/OntographRAG.git
cd OntographRAG
uv sync
source .venv/bin/activate
docker compose up -d neo4j

# 2. Download benchmark datasets (see exact paths below)
mkdir -p MIRAGE/rawdata/{pubmedqa/data,hotpotqa,2wikimultihopqa,musique,multihoprag,realmedqa,bioasq/Task10BGoldenEnriched}

# PubMedQA — https://github.com/pubmedqa/pubmedqa
# Download test_set.json from the repo and place at:
# MIRAGE/rawdata/pubmedqa/data/test_set.json

# HotpotQA — https://hotpotqa.github.io/
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json \
  -O MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json
# Optional shared-corpus FullWiki variant for corpus-level retrieval:
python experiments/prepare_hotpotqa_fullwiki_corpus.py \
  --num-samples 250 --subset-seed 42 --overwrite

# 2WikiMultiHopQA — https://github.com/Alab-NII/2wikimultihop
# Download dev.json from the GitHub release and place at:
# MIRAGE/rawdata/2wikimultihopqa/dev.json

# MuSiQue — https://github.com/StonyBrookNLP/musique
# Download musique_ans_v1.0_dev.jsonl from the GitHub release and place at:
# MIRAGE/rawdata/musique/musique_ans_v1.0_dev.jsonl

# MultiHopRAG — https://github.com/yixuantt/MultiHop-RAG
# Download MultiHopRAG.json and corpus.json and place at:
# MIRAGE/rawdata/multihoprag/MultiHopRAG.json
# MIRAGE/rawdata/multihoprag/corpus.json

# RealMedQA — https://huggingface.co/datasets/k2141255/RealMedQA
# Download and place at: MIRAGE/rawdata/realmedqa/RealMedQA.json

# BioASQ — http://bioasq.org/participate/challenges (free registration required)
# Download Task10BGoldenEnriched and place at:
# MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json
# Then build the shared PubMed abstract corpus:
python experiments/prepare_bioasq_corpus.py \
  --bioasq-path MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json \
  --output MIRAGE/rawdata/bioasq/pubmed_abstracts.jsonl \
  --email you@example.com \
  --verbose

# 3. Run a cheap smoke test
python experiments/experiment.py \
  --datasets hotpotqa --num-samples 30 --subset-seed 42 --rebuild-kg --evaluation-mode accuracy_only

# 4. Run the current paper-facing comparison
# dense_floor runs vanilla RAG; kg_entity_first runs KG-RAG.
python experiments/experiment.py \
  --datasets realmedqa multihoprag hotpotqa \
  --num-samples 250 --subset-seed 42 --rebuild-kg --evaluation-mode full_metrics \
  --retrieval-study final_pair --kg-builder-profile full \
  --llm-provider openrouter --llm-model openai/gpt-4o-mini \
  --retrieval-temperature-values 0.0

# 5. Optional: strict entity-first KG stress test for context-collapse analysis
python experiments/experiment.py \
  --datasets realmedqa \
  --num-samples 230 --subset-seed 42 --evaluation-mode full_metrics \
  --retrieval-study strict_entity --kg-builder-profile full \
  --llm-provider openrouter --llm-model openai/gpt-4o-mini \
  --retrieval-temperature-values 0.0

# 6. Optional: BioASQ, once the shared PubMed abstract corpus exists.
# BioASQ is supported, but should be treated as a separate run unless included
# deliberately in the current result set.
python experiments/experiment.py \
  --datasets bioasq --num-samples 100 --subset-seed 42 --rebuild-kg --evaluation-mode full_metrics \
  --llm-provider openrouter --llm-model openai/gpt-4o-mini \
  --retrieval-temperature-values 0.0
```

---

## Why OntographRAG

Most GraphRAG tools (including [Microsoft's GraphRAG](https://github.com/microsoft/graphrag)) let an LLM freely decide what to extract, which often leads to type drift, duplicate entities, and schema inconsistency across documents. OntographRAG takes the opposite approach: **you define the schema, the system respects it.**

| | OntographRAG | Microsoft GraphRAG |
|---|---|---|
| **Schema control** | Bring your own OWL/RDF/JSON ontology; extraction is constrained to your types | LLM decides freely; no schema enforcement |
| **Graph storage** | Neo4j with named KGs, Cypher, vector indexes, and provenance | Parquet files in a local directory |
| **Retrieval** | Routed hybrid: entity-first linking, retriever-first graph expansion, vector fallback, and evidence organization | Community summarisation or entity search |
| **Trust signals** | App surfaces Structural and Grounding support; evaluation computes the full uncertainty suite | None by default |
| **Interfaces** | Web UI, REST API, CLI, experiments | CLI + Python library |

---

## Workflow Cheatsheet

```bash
# Ingest a document into a running server
.venv/bin/python -m ontographrag.cli ingest report.pdf --kg-name demo-kg

# Explore the available graphs
.venv/bin/python -m ontographrag.cli explore list
.venv/bin/python -m ontographrag.cli explore show demo-kg

# Ask a grounded question
.venv/bin/python -m ontographrag.cli ask "What are the main findings?" --kg-name demo-kg

# Evaluate benchmark runs
.venv/bin/python -m ontographrag.cli evaluate --datasets hotpotqa --num-samples 30 --subset-seed 42
```

---

## Use cases

### Clinical intelligence and population-level evidence
Supply a clinical ontology (SNOMED CT, ICD-10, HPO) and process patient notes, discharge summaries, or EHR exports in bulk. Because every patient's data is extracted into the *same schema*, the whole population becomes queryable as a single graph:

```cypher
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis {name: "Hypertension"})
      -[:CO_OCCURS_WITH]->(c:Diagnosis)
WHERE p.age > 60
RETURN c.name, count(*) AS frequency ORDER BY frequency DESC
```

### Research and knowledge synthesis
Process a corpus of papers, extract entities and relationships consistently across documents, and ask cross-paper questions that single documents cannot answer alone.

### Any domain with structured knowledge requirements
Legal (case law entities), finance (company relationships), engineering (component hierarchies). Supply the domain ontology; OntographRAG handles the rest.

---

## Key features

### 1. Ontology-guided KG construction
Supply a `.owl` / `.rdf` / `.ttl` ontology file and every extracted entity and relationship is validated against your schema. The same document processed twice produces the same graph shape. Across a corpus of documents, every entity lands in the same type hierarchy — enabling aggregation, comparison, and population-level queries that are impossible with free-form extraction.

This is the core differentiator. Without schema enforcement, LLMs produce synonym explosion ("myocardial infarction", "heart attack", "MI", "AMI" as four separate nodes), type drift (the same concept classified differently across documents), and graphs that can't be meaningfully queried at scale. The ontology collapses all of this into a consistent, traversable structure.

Without an ontology, extraction still works — the LLM infers types — but schema-constrained extraction is what unlocks population-level reasoning.

### 2. Neo4j as the graph store
Graphs are persisted in Neo4j with:
- **Vector indexes** (384-dim `all-MiniLM-L6-v2` by default) for semantic search over chunks
- **Named KGs** — multiple independent graphs in one database, scoped by name tag
- **Full Cypher access** — query or extend the graph with any Cypher statement

### 3. Routed hybrid retrieval
Queries do not rely on one brittle retrieval path. OntographRAG now uses a routed KG-RAG stack:
- **Entity-first retrieval**: symbolic matching plus per-entity ANN over entity embeddings
- **Graph expansion**: question-local traversal with provenance-aware edges
- **PPR-style scoring**: chunks are ranked by support flowing through the local entity subgraph, not only by hop count
- **Retriever-first graph expansion**: when entity anchoring is weak, dense passage retrieval seeds the graph instead
- **Vector fallback**: if graph signal is weak, the system falls back cleanly to vector retrieval rather than forcing a bad subgraph

This makes the retriever much closer to recent strong GraphRAG systems while preserving a single shared interface for vanilla RAG and KG-RAG.

### 4. Evidence organization for answer generation
Retrieved graph paths and supporting passages are organized into explicit reasoning chains before generation. In the app, the chat view surfaces two trust signals that are easy to interpret in practice:
- **Structural**: whether the answer is supported by graph paths
- **Grounding**: whether the retrieved evidence actually grounded the question

The live UI deliberately keeps these signals simple. The full uncertainty suite remains available in the evaluation pipeline.

### 5. Uncertainty quantification and hallucination detection

A dedicated evaluation pipeline (`experiments/uncertainty_metrics.py`) computes uncertainty metrics per answer in three families. The core challenge is not that all KG-RAG retrieval is deterministic. Rather, strict or highly stabilised entity-first graph retrieval can lock onto the same context across repeated samples. In that regime, standard output-variance metrics may report low uncertainty because every generation sees the same evidence, even when the evidence is incomplete or wrongly anchored. Structural and grounding diagnostics are included to expose that hidden retrieval state.

#### Output-side (prior work baselines + novel extensions)

| Metric | Formula | Ref |
|--------|---------|-----|
| `semantic_entropy` | NLI-cluster N responses with DeBERTa; $H = -\sum_c p_c \log p_c$ where $p_c$ = logsumexp of token log-probs per cluster | Farquhar et al., *Nature* 2024 |
| `discrete_semantic_entropy` | Same clustering; $p_c = n_c / N$ (count proportions, no log-probs) | Farquhar et al., *Nature* 2024 |
| `p_true` | Fraction of samples in the same NLI cluster as the most probable response; $p_{\text{true}} = \lvert\{i : c_i = c_0\}\rvert / N$ | Farquhar et al., *Nature* 2024 |
| `selfcheckgpt` | Pairwise NLI contradiction rate: contradictions / (2 × pairs) across all response pairs | Manakul et al., EMNLP 2023 |
| `sre_uq` | KME = weighted mean response embedding; Gaussian kernel $\kappa_r = e^{-d_r^2/2\sigma^2}$; perturbation sensitivity $= \lvert\Delta E_r + L_r\rvert$ averaged over top modes | Vipulanandan et al., ICLR 2026 |
| `vn_entropy` ⭐ | L2-normalise embeddings $V$; $\rho = VV^\top / N$; $S(\rho) = -\sum_i \lambda_i \log \lambda_i$ | This work |
| `sd_uq` ⭐ | Gram-Schmidt: $e_i = v_i - (v_i \cdot q)q$; SVD of centred residuals; $\text{SD-UQ} = \exp\!\left(\frac{1}{k}\sum_i \log(\lambda_i + \varepsilon)\right)$ | This work |

`vn_entropy` is a soft, parameter-free analogue of semantic entropy (no NLI, no threshold). `sd_uq` extends it by conditioning out the question direction, estimating $H(v \mid q\text{-direction})$ — the entropy of what responses add *beyond* the question.

#### Structural — KG-only *(this work)*

No LLM sampling required. Path queries filtered to edges with confidence >= 0.4. These metrics are not correctness oracles; they diagnose graph-state consistency, path support, competing evidence, and perturbation fragility.

| Metric | Formula | Intuition |
|--------|---------|----------|
| `graph_path_support` (GPS) ⭐ | Find Q-entities and A-entities by name; per-entity reachability query ≤ 3 hops; $\text{GPS} = 1 - \lvert\text{reachable}\rvert / \lvert\text{A-entities}\rvert$ | 0 = KG fully supports the answer path; 1 = answer has no structural grounding. Single-sample metric, does not collapse under context determinism. |

The full runner also computes `graph_path_disagreement`, `competing_answer_alternatives`, `evidence_vn_entropy`, `subgraph_informativeness`, and `subgraph_perturbation_stability`. The paper-facing analysis focuses on the core interpretable signals, while these additional diagnostics are retained for ablations and audit runs.

#### Grounding *(this work)*

NLI between retrieved chunks and the generated answer. Works even when all N samples receive identical context.

| Metric | Formula | Intuition |
|--------|---------|----------|
| `support_entailment_uncertainty` (SEU) ⭐ | DeBERTa NLI(chunk → answer) per chunk; $\text{support} = (n_{\text{entail}} - n_{\text{contradict}}) / N$; $\text{SEU} = (1 - \text{support}) / 2$ | 0 = all chunks entail the answer; 0.5 = neutral; 1 = all chunks contradict. Key signal for abstentions and wrong-hop answers on multi-hop benchmarks. |
| `evidence_conflict_uncertainty` (ECU) ⭐ | Count entail-contradict chunk pairs; $\text{ECU} = \text{conflict pairs} / \binom{N}{2}$ | Variance complement to SEU. High = some chunks support while others contradict — genuine evidentiary conflict. Particularly diagnostic on multi-hop questions. |

#### Selective prediction (AUROC / AUREC)

After each run the pipeline computes per metric:

- **AUROC** — discriminates correct from incorrect answers using the metric as a score. Higher = better (0.5 = random, 1.0 = perfect).
- **AUREC** — Area Under the Rejection-Error Curve. Reject most uncertain questions first; measure error rate on retained questions at each rejection level. Lower = better.

⭐ = novel contribution of this work.

### 6. Provider-agnostic LLM support
Every endpoint accepts a `provider` + `model` pair. Supported providers: OpenRouter (free tier available), OpenAI, Google Gemini, Ollama (local), DeepSeek, HuggingFace. Switch model per request with no code changes.

---

## Table of contents

- [Quick Start](#quick-start)
- [Workflow Cheatsheet](#workflow-cheatsheet)
- [Setup](#setup)
- [Web UI](#web-ui)
- [API Reference](#api-reference)
- [Experiments](#experiments)
- [Architecture](#architecture)
- [Utility scripts](#utility-scripts)
- [Docker](#docker)
- [Configuration](#configuration)

---

## Setup

### Requirements

- Python 3.11+
- Node.js 18+ and npm (for the React frontend)
- Neo4j 5.0+ (via Docker or local install)
- 8 GB RAM minimum (16 GB recommended for large documents)

### Installation

```bash
# Option A: pip install the frozen paper branch
python -m pip install "ontographrag @ git+https://github.com/julka01/OntographRAG.git@paper-submission"

# Option B: build and install a local wheel
uv build
python -m pip install dist/ontographrag-1.0.0rc1-py3-none-any.whl

# Option C: source checkout for development
# Python dependencies
uv sync          # recommended
# or
pip install -r requirements.txt

# React frontend for source checkouts. Pip wheels include packaged UI assets.
cd frontend && npm install && npm run build && cd ..
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# ── Neo4j ───────────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# ── LLM providers (at least one required) ───────────────────────────────────
OPENROUTER_API_KEY=your-key   # free-tier models available
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
HF_API_TOKEN=...
OLLAMA_HOST=http://localhost:11434

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=sentence_transformers   # recommended runtime default; or openai

# ── Weights & Biases (optional — experiment tracking) ────────────────────────
WANDB_API_KEY=             # if set, experiment runs are logged to W&B automatically
WANDB_PROJECT=ontographrag # default project name

# ── Security (production) ────────────────────────────────────────────────────
APP_API_KEY=               # set to enforce API key auth on all endpoints
ALLOWED_ORIGINS=*          # comma-separated origins for CORS

# ── Server ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
LLM_TIMEOUT_SECONDS=120
```

Chunking, retrieval thresholds, and benchmark sweeps are now controlled primarily by constructor defaults and CLI flags rather than by top-level environment variables. For the live benchmark flags, use [experiments/README.md](experiments/README.md) as the source of truth.

---

## Web UI

The web interface is served at `http://localhost:8000`. It is a React + TypeScript single-page app (`frontend/`) built with Vite. Pip wheels include the built UI under `ontographrag/api/static/`. Source checkouts fall back to `frontend/dist/`, so run `cd frontend && npm install && npm run build` once after clone or after frontend changes.

### Knowledge graph panel

- **Build KG** — upload a document (PDF, TXT, CSV, JSON, XML ≤ 50 MB), choose provider/model, optionally attach an ontology file. Extraction progress streams to the UI in real time via SSE and the graph loads automatically on completion.
- **Graph visualisation** — interactive force-directed network. Node size scales with degree. Click a node to open its detail panel (type, properties, connected nodes).
- **Search** — dims non-matching nodes rather than hiding them; shows match count.
- **Filter** — per-type checkboxes with node/edge counts.
- **Named KG management** — create, list, and switch between multiple saved graphs.

### Chat panel

- Ask questions against the active knowledge graph; answers cite source chunks.
- **Trust pills** — the chat surface exposes two lightweight support signals inline with each response:
  - **Structural**: graph-path support for the answer
  - **Grounding**: how well the retrieved evidence supports the question
- Chat history persisted in `localStorage`.
- Highlighted nodes — entities used in the answer are highlighted in the graph.
- Thinking indicator while waiting for the LLM response.

---

## API Reference

Server runs on **port 8000**. Interactive docs at `http://localhost:8000/docs`.

> **Authentication**: set `APP_API_KEY` in `.env` to require `X-API-Key: <key>` on all requests. Unset = open (development mode).

### Knowledge graph — build & query

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/create_ontology_guided_kg` | Build an ontology-guided KG from a file upload |
| `POST` | `/extract_graph` | Extract a raw KG (no ontology) from a file |
| `POST` | `/load_kg_from_file` | Load a graph from file into Neo4j |
| `GET`  | `/kg_progress_stream` | SSE stream of KG build progress |

#### `POST /create_ontology_guided_kg`

Multipart form:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Document (PDF/TXT/CSV/JSON/XML, ≤ 50 MB) |
| `provider` | string | `openai` | LLM provider |
| `model` | string | `gpt-3.5-turbo` | Model name |
| `embedding_model` | string | `sentence_transformers` | Embedding backend for chunks and entities |
| `ontology_file` | file | optional | Custom ontology (.owl/.rdf/.ttl/.xml) |
| `max_chunks` | int | optional | Max text chunks to process (`1..500`) |
| `kg_name` | string | optional | Name tag for the resulting KG |
| `enable_coreference_resolution` | bool | `false` | Optional build-time coreference pass |

Response:
```json
{
  "kg_id": "uuid",
  "kg_name": "my-kg",
  "graph_data": { "nodes": [...], "relationships": [...] },
  "method": "ontology_guided"
}
```

#### `GET /kg_progress_stream`

Server-Sent Events. Connect with `EventSource`:
```
data: {"line": "✓ Extracted 42 entities from chunk 3/10"}
data: {"done": true}
```

---

### Named KG management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST`   | `/kg/create` | Create a named KG record |
| `GET`    | `/kg/list` | List all KGs with document counts |
| `GET`    | `/kg/{kg_name}` | Stats for a specific KG |
| `DELETE` | `/kg/{kg_name}` | Delete a KG |
| `GET`    | `/kg/{kg_name}/entities` | List entities in a KG |

---

### Neo4j management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/save_kg_to_neo4j` | Persist an in-memory KG to Neo4j |
| `POST` | `/load_kg_from_neo4j` | Load a KG from Neo4j by name |
| `POST` | `/clear_kg` | Delete all nodes and relationships |
| `GET`  | `/health/neo4j` | Connectivity check |

---

### Chat / RAG

#### `POST /chat`

Rate limited: 30 requests/minute per IP.

JSON body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Question to answer (max 4096 chars) |
| `provider_rag` | string | `openrouter` | LLM provider |
| `model_rag` | string | `openai/gpt-oss-120b:free` | Model name |
| `kg_name` | string | optional | Restrict retrieval to a specific KG |
| `document_names` | string[] | `[]` | Restrict to specific documents |
| `session_id` | string | `default_session` | Session identifier |

Response:
```json
{
  "session_id": "default_session",
  "message": "...",
  "info": {
    "sources": ["chunk_id_1"],
    "model": "openai/gpt-oss-120b:free",
    "chunk_count": 5,
    "entity_count": 12,
    "relationship_count": 8,
    "confidence": 0.87,
    "kg_confidence": 0.74,
    "structural_support": 0.74,
    "grounding_support": 0.81,
    "guardrail": {},
    "entities": { "used_entities": [...] }
  }
}
```

The UI uses `structural_support` and `grounding_support` as the main trust signals. The older `confidence` field is still returned for compatibility, but it is not the primary app-facing summary anymore.

---

### CSV bulk processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/validate_csv` | Validate a CSV before bulk processing |
| `POST` | `/bulk_process_csv` | Build KGs from all rows of a CSV |

#### `POST /bulk_process_csv`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | CSV file |
| `provider` | string | `openai` | LLM provider |
| `model` | string | `gpt-3.5-turbo` | LLM model |
| `text_column` | string | `full_report_text` | Column containing the text to process |
| `id_column` | string | optional | Column to use as document ID |
| `start_row` | int | `0` | First row to process |
| `batch_size` | int | `50` | Rows per batch |

---

### Models

`GET /models/{provider}` — lists available models for a provider.

---

### cURL examples

```bash
# Build a KG with ontology
curl -X POST http://localhost:8000/create_ontology_guided_kg \
  -F "file=@document.pdf" \
  -F "provider=openrouter" \
  -F "model=openai/gpt-4o-mini" \
  -F "ontology_file=@schema.owl" \
  -F "kg_name=my-kg"

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main concepts?", "kg_name": "my-kg", "provider_rag": "openrouter", "model_rag": "openai/gpt-4o-mini"}'

# Stream build progress
curl -N http://localhost:8000/kg_progress_stream

# List KGs
curl http://localhost:8000/kg/list

# Health check
curl http://localhost:8000/health/neo4j
```

---

## Experiments

The `experiments/` directory runs the current benchmark pipeline for vanilla RAG vs KG-RAG across biomedical and multi-hop QA datasets. It uses seeded deterministic subsets, dataset-scoped KGs, official-style answer `EM/F1` where supported, clean accuracy as the headline accuracy number, and the current uncertainty suite. See [experiments/README.md](experiments/README.md) for the live flag list and dataset caveats.

**Weights & Biases integration** — if `WANDB_API_KEY` is set in the environment, each run is automatically logged to W&B with per-question tables, per-metric AUROC/AUREC scores, and run metadata (dataset, model, seed, evaluation mode). No flags required; set the key and it activates. Results are also always written locally under `results/runs/<run_id>/` regardless of W&B.

```bash
# 30-question smoke test
python experiments/experiment.py \
  --datasets hotpotqa \
  --num-samples 30 \
  --subset-seed 42 \
  --rebuild-kg \
  --evaluation-mode accuracy_only

# Paper-facing final-pair comparison
python experiments/experiment.py \
  --datasets realmedqa multihoprag hotpotqa \
  --num-samples 250 \
  --subset-seed 42 \
  --rebuild-kg \
  --evaluation-mode full_metrics \
  --retrieval-study final_pair \
  --kg-builder-profile full \
  --llm-provider openrouter \
  --llm-model openai/gpt-4o-mini \
  --retrieval-temperature-values 0.0

# Strict entity-first KG stress test for retrieval lock-in / context collapse
python experiments/experiment.py \
  --datasets realmedqa \
  --num-samples 230 \
  --subset-seed 42 \
  --evaluation-mode full_metrics \
  --retrieval-study strict_entity \
  --kg-builder-profile full \
  --llm-provider openrouter \
  --llm-model openai/gpt-4o-mini \
  --retrieval-temperature-values 0.0
```

### Supported datasets

| Dataset | Task | Download |
|---------|------|----------|
| `pubmedqa` | Biomedical yes/no/maybe over source abstracts | [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) |
| `realmedqa` | Clinical recommendation QA over NICE guidance | [RealMedQA on Hugging Face](https://huggingface.co/datasets/k2141255/RealMedQA) |
| `hotpotqa` | Multi-hop Wikipedia QA | [HotpotQA](https://hotpotqa.github.io/) |
| `hotpotqa_fullwiki` | HotpotQA over a prepared shared FullWiki retrieved-context corpus | [HotpotQA](https://hotpotqa.github.io/) |
| `2wikimultihopqa` | Multi-hop Wikipedia QA | [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) |
| `musique` | Compositional multi-hop QA | [MuSiQue](https://github.com/StonyBrookNLP/musique) |
| `multihoprag` | Multi-hop RAG benchmark with shared corpus | [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) |
| `bioasq` | Biomedical factoid / yes-no QA | [bioasq.org](http://bioasq.org/participate/challenges) (free registration; needs shared-corpus prep) |

Place downloaded files under `MIRAGE/rawdata/` — see [experiments/README.md](experiments/README.md) for exact paths.

### Evaluation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples` | all | Questions per dataset |
| `--subset-seed` | `42` | Deterministic question-subset seed |
| `--entropy-samples` | `5` | Responses per question for uncertainty metrics |
| `--similarity-thresholds` | `[0.1]` | Cosine similarity cutoffs to sweep |
| `--max-chunks-values` | `[10]` | Retrieved chunk counts to sweep |
| `--llm-provider` | `openai` | LLM provider |
| `--llm-model` | `gpt-4o-mini` | Model |
| `--datasets` | `pubmedqa bioasq` | Datasets to run |
| `--rebuild-kg` | `False` | Rebuild the dataset KG |
| `--max-kg-contexts` | unset | Cap the passages indexed into the KG build |
| `--dataset-kg-scope` | `evaluation_subset` | Build KG from the selected subset or the full normalized dataset |
| `--allow-gold-evidence-contexts` | `False` | Controlled-evidence mode only; bypass corpus-safety guardrails |
| `--no-llm-judge` | `False` | Disable LLM-as-judge and use heuristic matching only |
| `--judge-provider` | generation provider | Separate provider for the correctness judge |
| `--judge-model` | generation model | Separate model for the correctness judge |
| `--temperature` | `1.0` | Generation temperature for uncertainty sampling |
| `--retrieval-temperature-values` | `[0.0]` | Final-stage retrieval sampling temperature sweep |
| `--retrieval-shortlist-factor` | `4` | Overfetch factor for retrieval-temperature sampling |
| `--retrieval-study` | unset | Built-in profiles: `small`, `final_pair`, or `strict_entity` |
| `--kg-builder-profile` | `auto` | `full` enables the strongest in-repo KG construction path; `lightweight` is for cheap sweeps |
| `--multi-temperature` | `False` | Also run T=0, 0.5, 1.0 output-side sweeps |
| `--evaluation-mode` | `full_metrics` | `accuracy_only` or `full_metrics` |
| `--output-dir` | `results` | Root directory for run artifacts |

Run artifacts are written under `results/runs/<run_id>/` and checkpoints under `results/checkpoints/`.

---

## Innovations

### Retrieval
- **Adjacent chunk expansion** — when retrieval uses the `retrieval_vector` index, seed element IDs are resolved to their parent `Chunk` before expanding to positional neighbours, so answers split across chunk boundaries are correctly reassembled.
- **Confidence-aware graph filtering** — traversal queries and PPR subgraph fetch now apply `coalesce(r.confidence, 1.0) >= 0.4` to skip low-confidence edges extracted during KG build.

### Uncertainty metrics
- **GPS (Graph Path Support)** — switched from full `[*1..N]` path enumeration (times out on dense graphs) to one query per answer entity with `LIMIT 1`, preserving the confidence filter that `shortestPath` cannot support. GPS now correctly returns non-zero values when answer entities are not reachable via high-confidence paths.
- **SEU (Support Entailment Uncertainty)** — now computed even when generation failed: retrieved chunks still exist and are evaluated against the expected answer as hypothesis. This turns SEU into a signal for *why* the model abstained (context didn't entail the answer vs. other failure modes). ECU receives the same fix.
- **SPS (Subgraph Perturbation Stability)** — entity caps tightened from 20 to 5 per query with 20 s timeout to prevent silent hangs.

---

## Architecture

### KG build pipeline

1. **Ingest** — file uploaded; PDF text extracted via PyMuPDF, plaintext decoded
2. **Chunk** — deterministic overlapping text windows are created for passage-level extraction
3. **Ontology load** — custom `.owl`/`.ttl` parsed (owlready2), or free-form extraction if none supplied
4. **LLM extraction** — each chunk is processed with an ontology-constrained prompt; entities and relationships are returned as structured JSON
5. **Cross-chunk extraction** — adjacent chunk pairs get a second pass for span-overflow relations that would otherwise be missed
6. **Entity harmonization** — duplicate and synonym entities are merged; alias surfaces are retained in `synonyms`; the most specific compatible type wins
7. **Relationship provenance** — edges are stamped with chunk-position, passage, and question-local provenance so later retrieval can stay passage-local when needed
8. **Specificity stats** — entities receive `passage_count` and `node_specificity = 1 / passage_count` so generic hubs can be down-weighted at retrieval time
9. **Embed** — chunks and entities are embedded; entity vectors are name-centered so short query mentions align cleanly at ANN lookup time
10. **Write** — nodes, relationships, embeddings, and provenance are stored in Neo4j; entities are tagged by `kgName`; progress streams via SSE

### RAG query pipeline

1. **Entity-first seeding** — when enabled, the system extracts named mentions from the question, runs symbolic alias matching plus per-entity ANN, and anchors retrieval on those entity seeds
2. **Question-local traversal** — provenance-aware graph traversal keeps path hops local to the current KG scope and, for bundle-style benchmarks, to the current question bundle
3. **Graph scoring** — local entity neighborhoods are ranked with PPR-style support flow rather than a fixed hop table alone
4. **Retriever-first graph expansion** — if entity anchoring is weak, dense passage retrieval seeds a second graph-expansion pass from chunk-linked entities
5. **Fallback retrieval** — if graph signal stays weak, the system falls back to vector retrieval and then text search rather than forcing a brittle graph answer
6. **Evidence organization** — graph paths and supporting passages are grouped into chain-style evidence blocks before generation
7. **Answer synthesis** — the LLM answers from the evidence block, while the app surfaces simplified Structural and Grounding support signals

### Module layout

```
frontend/                              # React + TypeScript web UI (Vite)
├── src/
│   ├── components/                    # Chat, graph, KG build, layout UI components
│   ├── hooks/                         # useChat, useGraph, useModels, useHealth, ...
│   ├── context/                       # AppContext, ThemeContext
│   └── types/                         # Shared TypeScript types
└── dist/                              # Source-checkout build assets; copied into ontographrag/api/static/ for wheels

ontographrag/
├── api/
│   ├── app.py                         # FastAPI application, all endpoints; serves packaged/source UI assets
│   └── static/                        # Built UI assets included in pip wheels
├── kg/
│   ├── builders/
│   │   ├── ontology_guided_kg_creator.py   # OntologyGuidedKGCreator — core extraction, harmonization, Neo4j write
│   │   └── enhanced_kg_creator.py          # UnifiedOntologyGuidedKGCreator — API-facing wrapper + CSV bulk ops
│   ├── chunking.py                    # Hierarchical chunking (large extraction chunks + small retrieval sub-chunks)
│   └── utils/
│       ├── common_functions.py        # Shared helpers (embedding, text normalization)
│       └── constants.py               # Default values and Neo4j label constants
├── rag/
│   ├── systems/
│   │   ├── enhanced_rag_system.py     # KG-RAG: entity-first + PPR scoring + RFGE + vector fallback
│   │   └── vanilla_rag_system.py      # Vanilla RAG: vector-only baseline with adjacent chunk expansion
│   ├── answer_guardrails.py           # Runtime answer quality guardrails
│   └── retrieval_sampling.py         # Retrieval temperature sampling helpers
├── schemas/
│   └── models.py                      # Pydantic models: Chunk, Entity, Relationship, KGContext, RetrievalResult
└── providers/
    └── model_providers.py             # LLM + embedding provider abstractions

experiments/
├── experiment.py                      # Main benchmark runner
├── uncertainty_metrics.py             # UQ suite (output, structural, grounding)
├── dataset_adapters.py                # Dataset normalization and corpus-role metadata
├── prepare_bioasq_corpus.py           # Build shared PubMed abstract corpus for BioASQ
└── visualize_results.py              # Plotting utilities
```

### Key specs

| Component | Detail |
|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim), runs locally on CPU |
| Vector similarity | Cosine, default threshold 0.1 |
| Chunk size | 1500 chars, 200 overlap |
| Graph database | Neo4j 5.0+ with vector indexes |
| Graph visualisation | React + force-directed graph (frontend) |
| File upload limit | 50 MB |
| Chat rate limit | 30 req/min per IP |
| KG build rate limit | 5 req/min per IP |

---

## Internal support modules

The supported product surface is the CLI, web app, and experiment runner. A few
root-level Python modules remain because the app imports them directly:

| Module | Purpose |
|--------|---------|
| `graphDB_dataAccess.py` | Low-level Neo4j data access layer used by the API |
| `csv_processor.py` | CSV validation and bulk-processing helpers for the app |
| `shared/common_fn.py` | Shared text and embedding utilities used across modules |

---

## Docker

```bash
# Neo4j only (recommended for development)
docker compose up -d neo4j

# Full stack (Neo4j + API server)
docker compose up -d

# Logs
docker compose logs -f

# Stop
docker compose down

# Neo4j Browser → http://localhost:7474
# Connect to bolt://localhost:7687
```

---

## Configuration reference

### LLM providers

| Provider | Env var | Notes |
|----------|---------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | Recommended; free-tier models available |
| `openai` | `OPENAI_API_KEY` | GPT-3.5, GPT-4, GPT-4o |
| `gemini` | `GEMINI_API_KEY` | Gemini Pro, Flash |
| `ollama` | — | Local models; set `OLLAMA_HOST` if non-default |
| `huggingface` | `HF_API_TOKEN` | HuggingFace Inference API |
| `deepseek` | `DEEPSEEK_API_KEY` | DeepSeek Chat/Coder |

### Embedding providers

| Provider | Typical use | Notes |
|----------|-------------|-------|
| `sentence_transformers` (runtime default) | local CPU/GPU embeddings | Uses the local MiniLM-family sentence-transformers path |
| `openai` | hosted embeddings | Requires `OPENAI_API_KEY` |
| `huggingface`, `vertexai` | advanced/provider-helper integrations | Supported by lower-level provider helpers, but the current runtime paths default to `sentence_transformers` or `openai` |

### Runtime knobs that are actually live via env vars

| Variable | Default | Effect |
|----------|---------|--------|
| `EMBEDDING_PROVIDER` | `sentence_transformers` | Runtime embedding backend for app / retrieval |
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL for local Ollama models |
| `APP_API_KEY` | unset | Enables API-key enforcement when present |
| `ALLOWED_ORIGINS` | `*` | CORS policy for the FastAPI server |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-request LLM timeout |

### Retrieval and evaluation defaults

These are no longer primarily env-var driven:
- KG build chunk windows and overlaps are code-level defaults in the builder / benchmark runner
- retrieval thresholds and chunk-count sweeps are CLI-controlled in [experiments/README.md](experiments/README.md)
- hybrid-retrieval behavior is controlled by retriever settings such as `retrieval_mode`, `use_rfge`, `use_ppr_scoring`, and `use_evidence_block`

---

## License

MIT. See [LICENSE](LICENSE) for details.

*Issues and feature requests: [GitHub Issues](https://github.com/julka01/OntographRAG/issues)*
