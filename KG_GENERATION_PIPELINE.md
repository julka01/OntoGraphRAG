# KG Generation Pipeline — Detailed Reference

## Overview

The pipeline runs inside `OntologyGuidedKGCreator.generate_knowledge_graph()` and `store_knowledge_graph_with_embeddings()`. It has 7 sequential stages.

---

## Stage 1 — Text Chunking

**Code:** `_chunk_text(text)` → `TokenTextSplitter`

- Input text is split into overlapping token-window chunks (`chunk_size=1500`, `chunk_overlap=200` by default).
- Each chunk dict: `{text, chunk_id (index), start_pos, end_pos}`.
- If `max_chunks` is set and total chunks exceed it, only the first `max_chunks` chunks are kept.

**Log line:** `Created N chunks` / `Limiting processing to M chunks out of N total`

**Current run:** 2404 chunks created, limited to 1.

---

## Stage 2a — Per-Chunk LLM Extraction

**Code:** `_extract_entities_and_relationships_with_llm(chunk_text, llm, model_name)`

For each chunk, one LLM call is made. The prompt varies by whether an ontology is loaded:

### With ontology
The prompt includes the list of ontology classes and relationship types, instructing the LLM to classify entities against them.

### Without ontology (current run)
Free-form extraction. Prompt asks for entities of any type and relationships between them.

### Expected JSON output
```json
{
  "entities": [
    { "id": "entity name", "type": "EntityType", "properties": {"name": "...", "description": "..."} }
  ],
  "relationships": [
    { "source": "entity id", "target": "entity id", "type": "REL_TYPE", "properties": {"description": "..."} }
  ]
}
```

### JSON parsing pipeline
1. Strip markdown fences (` ```json `)
2. Find balanced `{ }` block
3. `json.loads()`
4. If that fails → `_partial_json_extract()` (regex recovery of `entities` and `relationships` arrays independently)

### ⚠️ KNOWN BUG — Response truncation silently loses all relationships

The LLM generates `entities` first, then `relationships` in the JSON. If the response hits `max_tokens`, it is truncated mid-JSON. The `_partial_json_extract` regex `"relationships"\s*:\s*(\[.*?\])` requires the `relationships` array to be complete (opening `[` AND closing `]`). A truncated array has no closing `]` → regex fails → **0 relationships recovered**, even though the LLM generated them.

**What happened in the current run:**
```
Raw LLM response length: 15474          ← response hit token limit
Incomplete JSON structure (brace_count=2)
Partial JSON recovery succeeded: 74 entities, 0 relationships  ← relationships truncated
```

**Fix:** Put `relationships` before `entities` in the prompt JSON schema. Relationships are shorter and fewer; they will be captured before the response is truncated.

### Entity filtering (after JSON parse)
- Each extracted entity is validated: must be a dict with `id` and `type`.
- If `type` is missing, `_classify_entity_with_ontology()` is called (3 strategies: embedding similarity → substring match → keyword heuristics → fallback `Concept`).
- String entities (bare strings instead of dicts) are wrapped into dicts with `_classify_entity_with_ontology`.

### Relationship filtering
- `source` and `target` are resolved against extracted entity IDs (case-insensitive).
- Relationships referencing unknown entities are dropped.
- `entity_ids` dict is built as `{id.lower(): id}` for the lookup.

**Log line:** `✓ Chunk N processed: X entities, Y relationships`

---

## Stage 2b — Cross-Chunk Relationship Extraction (Sliding Window)

**Code:** after the main loop, `_extract_relationships_only(combined_text, known_entities, llm, model_name)`

- Runs only if `len(chunks) > 1` AND `llm is not None`.
- For each adjacent pair `(chunk[i], chunk[i+1])`:
  - Entities appearing in combined text are pre-filtered (substring presence check).
  - If ≥ 2 such entities exist, one LLM call is made asking **only** for relationships between the provided entities.
  - Results appended to `all_relationships`.

**Current run:** Skipped — only 1 chunk was processed.

---

## Stage 3 — Entity Harmonization

**Code:** `_harmonize_entities(all_entities, return_id_map=True)`

### Grouping key
```python
normalized_key = _normalize_entity_text(entity['id'])
```
Entities are grouped by **normalized text only** (not by type). Same entity text with different LLM-assigned types merges into one group.

`_normalize_entity_text` applies:
1. lowercase + collapse whitespace
2. strip leading articles (`the`, `a`, `an`, `and`, `or`)
3. remove punctuation `,()\[\];:.`
4. hyphens/slashes → underscores
5. collapse/strip leading/trailing underscores

### Representative selection (per group)
Priority: `(type specificity, description length, name length)`
- Type specificity: 1 if type ∉ `{Concept, Entity, Unknown, Other}`, else 0.
- The entity with a specific ontology type, longest description, then longest name wins.

### entity_map
```python
entity_map[original_entity_id] = representative_entity
```
ALL variant names in a group are mapped to the same representative. Also lowercase versions.

### UUID generation
```python
uuid5(NAMESPACE_OID, _normalize_entity_text(entity['id']))
```
Deterministic: same normalized text → same UUID every run.

**Log lines:**
```
Starting harmonization of N raw entities
Harmonization complete: M entities (removed K duplicates)
```

---

## Stage 4 — Relationship Harmonization

**Code:** `_harmonize_relationships(all_relationships, id_to_representative)`

1. Builds `original_to_uuid`: every variant name → canonical UUID (exact + lowercase).
2. For each raw relationship `{source, target, type, properties}`:
   - Maps `source` and `target` names to UUIDs via `original_to_uuid`.
   - If either lookup fails → relationship dropped with warning log.
3. Deduplicates via `seen_relationships = {uuid_src:type:uuid_tgt}`.

**Common drop reason:** LLM wrote a relationship source/target that doesn't exactly match any extracted entity ID (after case normalization).

---

## Stage 5 — KG Dict Assembly

**Code:** `generate_knowledge_graph()` lines ~1200–1245

Produces the in-memory KG dict:
```python
kg = {
    "nodes": [
        {
            "id": f"{kg_name}_{entity_uuid}",   # e.g. "kg_abc123_<uuid5>"
            "label": entity['type'],
            "properties": {
                "name": entity['id'],            # canonical surface form
                "type": entity['type'],
                "original_id": entity['id'],
                ...
            }
        }
    ],
    "relationships": [
        {
            "from": f"{kg_name}_{source_uuid}",
            "to": f"{kg_name}_{target_uuid}",
            "type": canonicalized_rel_type,
            ...
        }
    ],
    "chunks": chunks   # original text chunks
}
```

Node `id` format: `{kg_name}_{uuid5}`. This is the stable key used in ALL Neo4j MATCH queries.

---

## Stage 6 — Neo4j Storage

**Code:** `store_knowledge_graph_with_embeddings(kg, file_name, ...)`

### 6a — Document node
```cypher
MERGE (d:Document {fileName: $fileName, kgName: $kgName})
SET d.kgVersion = ..., d.totalChunks = ..., ...
```

### 6b — Chunk nodes
```cypher
MERGE (c:Chunk {id: $chunk_id})   ← SHA1 of chunk text
SET c.text = ..., c.embedding = ..., c.fileName = ..., c.kgName = ...
MERGE (c)-[:PART_OF]->(d)
```

### 6c — Entity nodes
```cypher
MERGE (n:{EntityType}:__Entity__ {id: $id})   ← kg-scoped UUID
ON CREATE SET n.name, n.type, n.kgName, n.embedding, n.content_hash, ...
ON MATCH SET n.last_accessed, n.all_names += ..., n.original_ids += ...
```
- MERGE key is `n.id` (the kg-prefixed UUID). This guarantees the same entity in the same KG always maps to the same node.
- `n.kgName` is stored for direct scoped loading (avoids chunk-path dependency).

### 6d — Confidence filtering + Relationship storage

For each relationship in `kg['relationships']`:

1. **Sanitize type:** `_canonicalize_relationship_type()` — exact ontology match → difflib fuzzy (≥0.72) → sanitize string → `ASSOCIATED_WITH` fallback.
2. **Resolve names:** `_uuid_to_name[rel['from']]` → human-readable source/target names.
3. **Confidence check:** `_verify_triple_confidence(source_name, target_name, rel_type, chunks)`
   - Score 1.0: both names in same sentence
   - Score 0.7: both names in same chunk
   - Score 0.4: one name found anywhere
   - Score 0.1: neither name found
   - Threshold: **0.15** — only score=0.1 (hallucination) is rejected.
4. **Store:**
```cypher
MATCH (source:__Entity__ {id: $source_id})
MATCH (target:__Entity__ {id: $target_id})
MERGE (source)-[r:{REL_TYPE}]->(target)
SET r += $properties
```

### 6e — Mention linking (Chunk → Entity)

For each chunk × each node:
- Build candidate names from `all_names`, `name`, `original_id` properties.
- Search for each name in chunk text using adaptive word-boundary regex (`(?<!\w)name(?!\w)` or `\b`).
- If found:
```cypher
MATCH (c:Chunk {id: $chunk_id})
MATCH (e:__Entity__ {id: $entity_id})   ← matches by id, not content_hash
MERGE (m:Mention {id: $mention_id})
SET m.quote = ..., m.entityName = ...
MERGE (e)-[:MENTIONED_IN]->(m)
MERGE (m)-[:FROM_CHUNK]->(c)
MERGE (c)-[:HAS_ENTITY]->(e)
```

`HAS_ENTITY` edges are how the RAG system finds entities during retrieval. Entities whose names don't appear verbatim in any chunk text will not get chunk links — they are still in Neo4j and their relationships are stored, but they rely on `kgName`-based loading (Stage 7) rather than chunk-path traversal.

---

## Stage 7 — KG Loading (Read Path)

**Code:** `app.py /load_kg_from_neo4j` + `kg_loader.py`

### Node query (app.py)
```cypher
MATCH (e:__Entity__ {kgName: $kg_name}) RETURN DISTINCT e
```
Scoped directly by `kgName` property — does NOT depend on chunk links.

### Relationship query (app.py)
```cypher
MATCH (a:__Entity__)-[r]->(b:__Entity__)
WHERE id(a) IN $entity_ids AND id(b) IN $entity_ids
RETURN r, a AS start, b AS end
```
`$entity_ids` = Neo4j internal IDs of all loaded entities. Returns any relationship between two entities that belong to this KG.

### Node query (kg_loader.py)
```cypher
MATCH (e:__Entity__ {kgName: $kg_name}) RETURN e as node
UNION ALL
MATCH (d:Document {kgName: $kg_name}) RETURN d as node
```

### Relationship query (kg_loader.py)
```cypher
MATCH (e1:__Entity__ {kgName: $kg_name})-[r]->(e2:__Entity__ {kgName: $kg_name})
RETURN r, e1 as start, e2 as end
UNION ALL
MATCH (d:Document {kgName: $kg_name})<-[r:PART_OF]-(c:Chunk)
RETURN r, c as start, d as end
```

---

## Current Run — Root Cause of 0 Relationships

```
max_chunks = 1   →   only chunk 1 of 2404 processed
LLM extracted entities + relationships, but response was 15474 chars and TRUNCATED
_partial_json_extract recovered: 74 entities, 0 relationships
  → "relationships" array in JSON was cut off — no closing ] → regex failed
Cross-chunk pass: skipped (only 1 chunk)
Result: 74 entities stored, 0 relationships stored
```

**Fix:** In the LLM prompt, put `relationships` before `entities` in the JSON schema so relationships are written first and captured before truncation.

---

## Summary — Where Relationships Get Dropped

| Stage | Drop condition | Log signal |
|---|---|---|
| 2a extraction | LLM response truncated → `relationships` at end of JSON cut off | `Partial JSON recovery succeeded: N entities, 0 relationships` |
| 2a filtering | `source`/`target` not in extracted entity IDs | `Invalid relationship ... Skipping` |
| 4 harmonization | Source/target name not in `original_to_uuid` | `Dropping relationship — entity not found in map` |
| 6d confidence | Both entities absent from all chunk texts (score 0.1) | `Skipping hallucinated relationship (confidence=0.10)` |
| 6d storage | `MATCH (source {id: ...})` finds no node | `Failed to store relationship N: ...` |
| 7 loading | Entity has no `kgName` property (old data) | Silent — entity not returned by load query |
