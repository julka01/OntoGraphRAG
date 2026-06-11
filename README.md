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

## Quick start

You need a running Neo4j instance and at least one LLM provider key.

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
ontograph serve --port 8000     # → http://localhost:8000
```

See [`.env.example`](.env.example) for all configuration variables.

## Usage

The packaged web app (Neo4j panel + chat) is the simplest entry point. From the
command line:

```bash
ontograph ingest report.pdf --kg-name demo          # build a named KG
ontograph ask "What are the main findings?" --kg-name demo
ontograph explore list                              # list saved graphs
```

A REST API is served alongside the app; interactive docs live at
`http://localhost:8000/docs`.

For a source checkout (development, frontend changes, or benchmarks):

```bash
git clone https://github.com/julka01/OntographRAG.git && cd OntographRAG
uv sync && source .venv/bin/activate
cd frontend && npm install && npm run build && cd ..   # build the UI
docker compose up -d neo4j
python -m ontographrag.cli serve
```

To reproduce the benchmarks (vanilla RAG vs KG-RAG, uncertainty suite, retrieval
lock-in study), see **[experiments/README.md](experiments/README.md)** for the
runner, datasets, and flags.

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

## Documentation

| Topic | Reference |
|-------|-----------|
| KG construction pipeline | [KG_GENERATION_PIPELINE.md](KG_GENERATION_PIPELINE.md) |
| Evaluation & uncertainty metrics | [EVALUATION_METRICS.md](EVALUATION_METRICS.md) |
| Benchmark runner, datasets, and flags | [experiments/README.md](experiments/README.md) |
| REST API | `http://localhost:8000/docs` (interactive) |

## Project layout

```
ontographrag/      installable package: api, kg builders, rag systems, providers
experiments/       benchmark runner and uncertainty suite — see experiments/README.md
frontend/          React + TypeScript web UI (Vite)
MIRAGE/            benchmark raw data and adapters
```

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
