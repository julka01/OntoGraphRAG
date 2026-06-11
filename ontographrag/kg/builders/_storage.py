import json
import re
import hashlib
from typing import Dict, Any, List, Optional
from langchain_neo4j import Neo4jGraph
import os
import logging
import math

class StorageMixin:
    """Neo4j persistence, property sanitisation, and vector index creation.

    Mixin for :class:`OntologyGuidedKGCreator`; method bodies are
    unchanged from the original monolithic implementation.
    """

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        # Ensure password is not None - use environment variable as fallback
        password = self.neo4j_password
        if password is None or password == "":
            password = os.getenv("NEO4J_PASSWORD", "password")

        # Set environment variables to ensure LangChain Neo4jGraph can read them
        os.environ["NEO4J_URI"] = self.neo4j_uri
        os.environ["NEO4J_USERNAME"] = self.neo4j_user
        os.environ["NEO4J_PASSWORD"] = password
        os.environ["NEO4J_DATABASE"] = self.neo4j_database

        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    @staticmethod
    def _is_neo4j_primitive(value: Any) -> bool:
        """Return True for Neo4j-safe primitive property values."""
        if isinstance(value, bool):
            return True
        if isinstance(value, (str, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        return False

    @classmethod
    def _coerce_neo4j_property_value(cls, value: Any) -> Any:
        """
        Convert arbitrary Python values into Neo4j-safe property values.

        Neo4j only accepts primitives or arrays of primitives as property values.
        Nested dicts/lists are serialized to JSON so richer metadata can still be
        preserved without breaking writes.
        """
        if value is None:
            return None
        if cls._is_neo4j_primitive(value):
            return value
        if isinstance(value, (list, tuple, set)):
            sequence = list(value)
            if all(cls._is_neo4j_primitive(item) for item in sequence):
                return list(sequence)
            return json.dumps(sequence, ensure_ascii=False, default=str)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)

    @classmethod
    def _sanitize_neo4j_properties(cls, properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Drop null/NaN values and coerce the rest to Neo4j-safe property values."""
        if not isinstance(properties, dict):
            return {}

        safe_properties: Dict[str, Any] = {}
        for key, value in properties.items():
            if value is None:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            coerced = cls._coerce_neo4j_property_value(value)
            if coerced is None:
                continue
            safe_properties[str(key)] = coerced
        return safe_properties

    def _store_graph_enrichment(
        self,
        graph,
        *,
        kg_name: str,
        file_name: str,
        enrichment: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Persist optional summaries, claims, and bridge edges to Neo4j."""
        if not enrichment:
            return {"component_summaries": 0, "claims": 0, "fragmentation_bridges": 0}

        graph_summary = enrichment.get("graph_summary") or {}
        component_summaries = list(enrichment.get("component_summaries") or [])
        claims = list(enrichment.get("claims") or [])
        fragmentation_bridges = list(enrichment.get("fragmentation_bridges") or [])

        chunk_id_by_position = {
            int(chunk.get("position", 0)): hashlib.sha1(
                f"{kg_name}:{file_name}:{chunk['position']}:{chunk['text']}".encode()
            ).hexdigest()
            for chunk in chunks or []
        }

        if graph_summary:
            graph.query(
                """
                MERGE (s:Summary {id: $summary_id})
                SET s.kgName = $kg_name,
                    s.scope = 'graph',
                    s.text = $text,
                    s.componentCount = $component_count,
                    s.createdAt = datetime()
                WITH s
                MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                MERGE (d)-[:HAS_GRAPH_SUMMARY]->(s)
                """,
                {
                    "summary_id": graph_summary["id"],
                    "kg_name": kg_name,
                    "file_name": file_name,
                    "text": graph_summary.get("text", ""),
                    "component_count": int(graph_summary.get("component_count", 0)),
                },
            )

        for summary in component_summaries:
            graph.query(
                """
                MERGE (s:Summary {id: $summary_id})
                SET s.kgName = $kg_name,
                    s.scope = 'component',
                    s.text = $text,
                    s.componentIndex = $component_index,
                    s.entityCount = $entity_count,
                    s.relationshipCount = $relationship_count,
                    s.createdAt = datetime()
                WITH s
                MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                MERGE (d)-[:HAS_COMPONENT_SUMMARY]->(s)
                """,
                {
                    "summary_id": summary["id"],
                    "kg_name": kg_name,
                    "file_name": file_name,
                    "text": summary.get("text", ""),
                    "component_index": int(summary.get("component_index", 0)),
                    "entity_count": int(summary.get("entity_count", 0)),
                    "relationship_count": int(summary.get("relationship_count", 0)),
                },
            )
            if graph_summary:
                graph.query(
                    """
                    MATCH (g:Summary {id: $graph_summary_id})
                    MATCH (s:Summary {id: $summary_id})
                    MERGE (g)-[:ABSTRACTS_COMPONENT]->(s)
                    """,
                    {
                        "graph_summary_id": graph_summary["id"],
                        "summary_id": summary["id"],
                    },
                )
            for entity_id in summary.get("entity_ids") or []:
                graph.query(
                    """
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (e:__Entity__ {id: $entity_id})
                    MERGE (s)-[:SUMMARIZES]->(e)
                    """,
                    {
                        "summary_id": summary["id"],
                        "entity_id": entity_id,
                    },
                )

        for claim in claims:
            graph.query(
                """
                MERGE (c:Claim {id: $claim_id})
                SET c.kgName = $kg_name,
                    c.text = $text,
                    c.polarity = $polarity,
                    c.confidence = $confidence,
                    c.componentIndex = $component_index,
                    c.createdAt = datetime()
                WITH c
                MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                MERGE (d)-[:HAS_CLAIM]->(c)
                """,
                {
                    "claim_id": claim["id"],
                    "kg_name": kg_name,
                    "file_name": file_name,
                    "text": claim.get("text", ""),
                    "polarity": claim.get("polarity", "affirmed"),
                    "confidence": float(claim.get("confidence", 0.0) or 0.0),
                    "component_index": int(claim.get("component_index", 0)),
                },
            )
            for entity_id in claim.get("entity_ids") or []:
                graph.query(
                    """
                    MATCH (c:Claim {id: $claim_id})
                    MATCH (e:__Entity__ {id: $entity_id})
                    MERGE (c)-[:ABOUT]->(e)
                    """,
                    {
                        "claim_id": claim["id"],
                        "entity_id": entity_id,
                    },
                )
            for position in claim.get("provenance_positions") or []:
                chunk_id = chunk_id_by_position.get(int(position))
                if not chunk_id:
                    continue
                graph.query(
                    """
                    MATCH (c:Claim {id: $claim_id})
                    MATCH (ch:Chunk {id: $chunk_id})
                    MERGE (c)-[:SUPPORTED_BY]->(ch)
                    """,
                    {
                        "claim_id": claim["id"],
                        "chunk_id": chunk_id,
                    },
                )

        for bridge in fragmentation_bridges:
            bridge_key = hashlib.sha1(
                f"{kg_name}:{bridge['source_id']}:{bridge['target_id']}:{bridge['reason']}".encode("utf-8")
            ).hexdigest()
            graph.query(
                """
                MATCH (source:__Entity__ {id: $source_id})
                MATCH (target:__Entity__ {id: $target_id})
                MERGE (source)-[r:SOFT_BRIDGE {id: $bridge_id}]->(target)
                SET r.kgName = $kg_name,
                    r.synthetic = true,
                    r.reason = $reason,
                    r.confidence = $confidence,
                    r.sharedAliasKeys = $shared_alias_keys,
                    r.createdAt = datetime()
                """,
                {
                    "bridge_id": bridge_key,
                    "kg_name": kg_name,
                    "source_id": bridge["source_id"],
                    "target_id": bridge["target_id"],
                    "reason": bridge.get("reason", "lexical_guard"),
                    "confidence": float(bridge.get("confidence", 0.0) or 0.0),
                    "shared_alias_keys": list(bridge.get("shared_alias_keys") or []),
                },
            )

        return {
            "component_summaries": len(component_summaries),
            "claims": len(claims),
            "fragmentation_bridges": len(fragmentation_bridges),
        }

    def _build_relationship_merge_query(
        self,
        sanitized_rel_type: str,
        *,
        include_condition: bool,
        include_quantitative: bool,
    ) -> str:
        """Build the relationship MERGE query without null-valued qualifiers.

        Neo4j rejects null property values inside MERGE patterns, so optional
        qualifiers must only participate in the identity key when present.
        """
        merge_key_parts = ["negated: $negated"]
        if include_condition:
            merge_key_parts.append("condition: $condition")
        if include_quantitative:
            merge_key_parts.append("quantitative: $quantitative")

        merge_key = ",\n                        ".join(merge_key_parts)
        return f"""
                    MATCH (source:__Entity__ {{id: $source_id}})
                    MATCH (target:__Entity__ {{id: $target_id}})
                    MERGE (source)-[r:{sanitized_rel_type} {{
                        {merge_key}
                    }}]->(target)
                    SET r += $properties,
                        r.provenancePositions =
                            reduce(acc = [], x IN coalesce(r.provenancePositions, []) + $provenance_positions |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.questionIds =
                            reduce(acc = [], x IN coalesce(r.questionIds, []) + $question_ids |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.passageKeys =
                            reduce(acc = [], x IN coalesce(r.passageKeys, []) + $passage_keys |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END)
                    """

    def store_knowledge_graph_with_embeddings(
        self,
        kg: Dict[str, Any],
        file_name: str,
        doc_metadata: dict = None,
        doc_hash: str = None,
        llm=None,
        model_name: str = "openai/gpt-oss-120b:free",
    ) -> bool:
        """
        Store the knowledge graph in Neo4j database with proper embedding support
        """
        try:
            # Try to create Neo4j connection - handle APOC issues gracefully
            try:
                graph = self._create_neo4j_connection()
            except Exception as conn_error:
                if "APOC" in str(conn_error) or "apoc" in str(conn_error):
                    logging.warning(f"APOC not available, skipping advanced KG storage: {conn_error}")
                    return False
                else:
                    raise conn_error

            # Pre-flight: remove any orphaned entities from a previous failed build
            # for this dataset (entities written before the Document node was committed).
            # This prevents unique-constraint violations on __Entity__.id when the LLM
            # assigns a different type label on a subsequent run.
            import uuid
            kg_name_value = kg['metadata'].get('kg_name') or file_name or "default"
            try:
                graph.query(
                    """
                    MATCH (e:__Entity__)
                    WHERE e.id STARTS WITH $prefix
                      AND NOT EXISTS {
                        MATCH (:Chunk)-[:HAS_ENTITY|MENTIONS]->(e)
                      }
                    DETACH DELETE e
                    """,
                    {"prefix": kg_name_value + "_"},
                )
                logging.info(f"[store_kg] Pre-flight cleanup done for '{kg_name_value}'")
            except Exception as _cleanup_err:
                logging.warning(f"[store_kg] Pre-flight cleanup failed (non-fatal): {_cleanup_err}")

            try:
                graph.query(
                    """
                    MATCH (q:Qualifier {kgName: $kg_name})
                    WHERE NOT EXISTS { MATCH ()-[]->(q) }
                       OR NOT EXISTS { MATCH (q)-[]->() }
                    DETACH DELETE q
                    """,
                    {"kg_name": kg_name_value},
                )
            except Exception as _qual_cleanup_err:
                logging.debug("[store_kg] Orphan qualifier cleanup failed (non-fatal): %s", _qual_cleanup_err)

            # Create document node with versioning
            kg_version = str(uuid.uuid4())
            schema_card = self._build_schema_card()
            schema_card_json = json.dumps(schema_card, ensure_ascii=False)
            doc_query = """
            MERGE (d:Document {fileName: $fileName, kgName: $kgName})
            SET d.kgVersion = $kgVersion,
                d.kgName = $kgName,
                d.createdAt = datetime(),
                d.updatedAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships,
                d.ontologyClasses = $ontologyClasses,
                d.ontologyRelationships = $ontologyRelationships,
                d.schemaEnforcementDroppedEntities = $schemaEnforcementDroppedEntities,
                d.schemaEnforcementDroppedRelationships = $schemaEnforcementDroppedRelationships,
                d.harmonizationRelationshipsDroppedUnmapped = $harmonizationRelationshipsDroppedUnmapped,
                d.harmonizationRelationshipsDroppedSchemaMismatch = $harmonizationRelationshipsDroppedSchemaMismatch,
                d.harmonizationRelationshipsDeduped = $harmonizationRelationshipsDeduped,
                d.harmonizationRelationshipContradictionGroups = $harmonizationRelationshipContradictionGroups,
                d.harmonizationRelationshipContradictionEdges = $harmonizationRelationshipContradictionEdges,
                d.contentHash = $contentHash,
                d.schemaCard = $schemaCard,
                d.schemaVersion = $schemaVersion,
                d.schemaHash = $schemaHash,
                d.embeddingModel = $embeddingModel,
                d.provider = $provider,
                d.model = $model,
                d.maxChunks = $maxChunks
            """
            graph.query(doc_query, {
                "kgVersion": kg_version,
                "kgName": kg_name_value,
                "fileName": file_name,
                "totalChunks": kg['metadata']['total_chunks'],
                "totalEntities": kg['metadata']['total_entities'],
                "totalRelationships": kg['metadata']['total_relationships'],
                "ontologyClasses": kg['metadata']['ontology_classes'],
                "ontologyRelationships": kg['metadata']['ontology_relationships'],
                "schemaEnforcementDroppedEntities": kg['metadata'].get('schema_enforcement_dropped_entities', 0),
                "schemaEnforcementDroppedRelationships": kg['metadata'].get('schema_enforcement_dropped_relationships', 0),
                "harmonizationRelationshipsDroppedUnmapped": kg['metadata'].get('harmonization_relationships_dropped_unmapped', 0),
                "harmonizationRelationshipsDroppedSchemaMismatch": kg['metadata'].get('harmonization_relationships_dropped_schema_mismatch', 0),
                "harmonizationRelationshipsDeduped": kg['metadata'].get('harmonization_relationships_deduped', 0),
                "harmonizationRelationshipContradictionGroups": kg['metadata'].get('harmonization_relationship_contradiction_groups', 0),
                "harmonizationRelationshipContradictionEdges": kg['metadata'].get('harmonization_relationship_contradiction_edges', 0),
                "contentHash": doc_hash or "",
                "schemaCard": schema_card_json,
                "schemaVersion": schema_card["schemaVersion"],
                "schemaHash": schema_card["schemaHash"],
                "embeddingModel": self.embedding_model,
                "provider": kg['metadata'].get('provider', 'openai'),
                "model": kg['metadata'].get('model', ''),
                "maxChunks": kg['metadata'].get('max_chunks_setting'),
            })

            # Store document-level metadata from source (e.g. CSV columns like SUBJECT_ID, HADM_ID)
            if doc_metadata:
                safe_meta = self._sanitize_neo4j_properties(doc_metadata)
                if safe_meta:
                    graph.query(
                        "MATCH (d:Document {fileName: $fileName, kgName: $kgName}) SET d += $meta",
                        {"fileName": file_name, "kgName": kg_name_value, "meta": safe_meta},
                    )

            # Create chunk nodes with embeddings.
            # Include kg_name in the hash so identical text in different KGs
            # gets distinct Chunk nodes and retrieval filters are not contaminated.
            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(f"{kg_name_value}:{file_name}:{chunk['position']}:{chunk['text']}".encode()).hexdigest()
                chunk_query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.kgName = $kg_name,
                    c.position = $position,
                    c.chunkLocalIndex = $chunk_local_index,
                    c.start_pos = $start_pos,
                    c.end_pos = $end_pos,
                    c.source = $source,
                    c.dataset = $dataset,
                    c.questionId = $question_id,
                    c.passageIndex = $passage_index,
                    c.embedding = $embedding
                """
                graph.query(chunk_query, {
                    "chunk_id": chunk_id,
                    "kg_name": kg_name_value,
                    "text": chunk['text'],
                    "position": chunk['position'],
                    "chunk_local_index": chunk.get('chunk_local_index', chunk.get('chunk_id', 0)),
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos'],
                    "source": chunk.get('source'),
                    "dataset": chunk.get('dataset'),
                    "question_id": chunk.get('question_id'),
                    "passage_index": chunk.get('passage_index'),
                    "embedding": chunk.get('embedding')
                })

                # Link chunk to document
                chunk_doc_query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                MERGE (c)-[:PART_OF]->(d)
                """
                graph.query(chunk_doc_query, {
                    "chunk_id": chunk_id,
                    "fileName": file_name,
                    "kgName": kg_name_value,
                })

                # Create retrieval-only subchunks sized for the embedding model.
                # These keep dense retrieval faithful without shrinking the
                # larger parent chunks used for KG extraction.
                retrieval_subchunks = self._build_retrieval_subchunks(
                    chunk,
                    parent_chunk_id=chunk_id,
                )
                for subchunk in retrieval_subchunks:
                    retrieval_chunk_query = """
                    MERGE (rc:RetrievalChunk {id: $retrieval_chunk_id})
                    SET rc.text = $text,
                        rc.kgName = $kg_name,
                        rc.position = $position,
                        rc.retrievalLocalIndex = $retrieval_local_index,
                        rc.parentChunkId = $parent_chunk_id,
                        rc.chunkLocalIndex = $chunk_local_index,
                        rc.start_pos = $start_pos,
                        rc.end_pos = $end_pos,
                        rc.source = $source,
                        rc.dataset = $dataset,
                        rc.questionId = $question_id,
                        rc.passageIndex = $passage_index,
                        rc.embedding = $embedding
                    """
                    graph.query(retrieval_chunk_query, {
                        "retrieval_chunk_id": subchunk["id"],
                        "kg_name": kg_name_value,
                        "text": subchunk["text"],
                        "position": subchunk["position"],
                        "retrieval_local_index": subchunk["retrieval_local_index"],
                        "parent_chunk_id": subchunk["parent_chunk_id"],
                        "chunk_local_index": subchunk["chunk_local_index"],
                        "start_pos": subchunk["start_pos"],
                        "end_pos": subchunk["end_pos"],
                        "source": subchunk.get("source"),
                        "dataset": subchunk.get("dataset"),
                        "question_id": subchunk.get("question_id"),
                        "passage_index": subchunk.get("passage_index"),
                        "embedding": subchunk.get("embedding"),
                    })
                    graph.query(
                        """
                        MATCH (rc:RetrievalChunk {id: $retrieval_chunk_id})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (rc)-[:RETRIEVES_FROM]->(c)
                        """,
                        {
                            "retrieval_chunk_id": subchunk["id"],
                            "chunk_id": chunk_id,
                        },
                    )

            # Create entity nodes with embeddings and ontology-based labels.
            # MERGE on node id (kg-prefixed UUID) so each KG's entities are independent.
            # content_hash is stored as a property for reference but is NOT the merge key,
            # which previously caused relationship storage to silently fail: a second KG
            # build would find the existing node by content_hash but leave n.id pointing at
            # the first KG's prefix, so MATCH (source {id: "kg2_uuid"}) never matched.
            for node in kg['nodes']:
                # Generate content-based deduplication hash (stored as property, not merge key)
                original_id = node.get('properties', {}).get('original_id', node['id'])
                normalized_content = self._normalize_entity_text(original_id)
                content_hash = hashlib.md5(f"{node['label']}:{normalized_content}".encode()).hexdigest()
                node['content_hash'] = content_hash

            for node in kg['nodes']:
                properties = node.get('properties', {})
                entity_type = node['label']  # This is the ontology class (Disease, Treatment, etc.)
                # Sanitize entity type for Cypher label compatibility and validate against whitelist
                cypher_safe_entity_type = re.sub(r'[^A-Za-z0-9_]', '_', entity_type).strip('_') or 'Concept'
                # Validate: label must start with a letter, max 64 chars, no injection risk
                if not re.match(r'^[A-Za-z][A-Za-z0-9_]{0,63}$', cypher_safe_entity_type):
                    logging.warning(f"Unsafe entity type '{entity_type}' → falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'
                # Blocklist structural/system labels that must not be applied to entity nodes
                _RESERVED_LABELS = {'Document', 'Chunk', 'Mention', 'Entity', '__Entity__',
                                    '__KGDocument__', 'Relationship', 'Node', 'Schema'}
                if cypher_safe_entity_type in _RESERVED_LABELS:
                    logging.warning(f"Entity type '{entity_type}' clashes with structural label — falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'

                # Generate embedding for entity if it doesn't have one.
                # Embed the entity name only (not description): the entity_vector
                # index is used for entity→entity ANN matching at query time, where
                # the probe vector is also a short entity-mention string.  Including
                # description here shifts the embedding into description-semantic space,
                # making name-level similarity unreliable (HippoRAG embeds names only).
                entity_embedding = node.get('embedding')
                if entity_embedding is None:
                    entity_text = properties.get('name', node['id'])
                    try:
                        entity_embedding = self.embedding_function.embed_query(entity_text)
                    except Exception as e:
                        logging.warning(f"Failed to generate embedding for entity {node['id']}: {e}")
                        entity_embedding = None

                # MERGE on __Entity__ only (not the specific type label) so that the
                # unique constraint on __Entity__.id is respected even when the LLM assigns
                # a different type label on a re-run.  The specific label is added with SET
                # after the MERGE, which is idempotent on existing nodes.
                node_query = f"""
                MERGE (n:__Entity__ {{id: $id}})
                ON CREATE SET
                    n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.ontology_class = $entity_type,
                    n.content_hash = $content_hash,
                    n.kgName = $kg_name,
                    n.all_names = $all_names,
                    n.original_ids = $original_ids,
                    n += $extra_properties,
                    n.created_at = datetime()
                ON MATCH SET
                    n.last_accessed = datetime(),
                    n.kgName = $kg_name,
                    n.type = $type,
                    n.ontology_class = $entity_type,
                    n.all_names = coalesce(n.all_names, []) + $all_names,
                    n.original_ids = coalesce(n.original_ids, []) + $original_ids,
                    n += $extra_properties
                SET n:{cypher_safe_entity_type}
                """
                extra_properties = self._sanitize_neo4j_properties(
                    {
                        k: v
                        for k, v in properties.items()
                        if k not in {"name", "description", "type", "all_names", "original_ids", "aliases"}
                    }
                )
                graph.query(node_query, {
                    "id": node['id'],
                    "content_hash": node['content_hash'],
                    "kg_name": kg_name_value,
                    "name": self._coerce_neo4j_property_value(properties.get('name', node['id'])),
                    "type": node['label'],
                    "description": self._coerce_neo4j_property_value(properties.get('description', '')),
                    "embedding": entity_embedding,
                    "entity_type": entity_type,
                    "all_names": self._coerce_neo4j_property_value(
                        list(set(
                            properties.get('all_names', [node['id']])
                            + [a for a in (properties.get('aliases') or []) if isinstance(a, str) and a.strip()]
                        ))
                    ),
                    "original_ids": self._coerce_neo4j_property_value(
                        list(set(properties.get('original_ids', [node['id']])))
                    ),
                    "extra_properties": extra_properties,
                })

            # Build prefixed-UUID → human-readable name lookup for confidence verification.
            # kg['nodes'][i]['id'] is the prefixed UUID (e.g. "kg_abc_<uuid>")
            # kg['nodes'][i]['properties']['name'] is the actual entity name text.
            # rel['from'] / rel['to'] use the same prefixed-UUID format.
            _uuid_to_name = {
                _n['id']: (_n.get('properties', {}).get('name') or _n['id'])
                for _n in kg.get('nodes', [])
                if _n.get('id')
            }

            # Create relationships with improved error handling
            relationships_stored = 0
            relationship_store_failures = 0
            relationships_skipped_low_confidence = 0
            relationships_skipped_schema_mismatch = 0
            relationships_reverified_kept = 0
            relationships_reverified_rejected = 0
            for idx, rel in enumerate(kg['relationships']):
                try:
                    # Filter out fields that are managed explicitly below to avoid duplicates:
                    # 'id' causes duplicate key issues; 'negated'/'condition'/'quantitative'
                    # are promoted to top-level properties_with_confidence below.
                    _exclude = {'id', 'negated', 'condition', 'quantitative'}
                    properties_filtered = {
                        k: v for k, v in rel.get('properties', {}).items()
                        if k not in _exclude
                    }
                    original_properties_filtered = dict(properties_filtered)

                    # Resolve UUIDs to entity names for evidence-grounded confidence check.
                    # Prefer explicit source_name/target_name properties; fall back to name lookup.
                    _src_id = rel.get('from') or rel.get('source', '')
                    _tgt_id = rel.get('to') or rel.get('target', '')
                    _src_node = next((n for n in kg.get('nodes', []) if n.get('id') == _src_id), {})
                    _tgt_node = next((n for n in kg.get('nodes', []) if n.get('id') == _tgt_id), {})
                    source_type = (
                        _src_node.get('label')
                        or (_src_node.get('properties') or {}).get('type')
                        or (_src_node.get('properties') or {}).get('ontology_class')
                    )
                    target_type = (
                        _tgt_node.get('label')
                        or (_tgt_node.get('properties') or {}).get('type')
                        or (_tgt_node.get('properties') or {}).get('ontology_class')
                    )

                    # Canonicalize relationship type against ontology using fuzzy matching
                    sanitized_rel_type = self._canonicalize_relationship_type(
                        rel['type'],
                        source_type=source_type,
                        target_type=target_type,
                    )
                    if not sanitized_rel_type:
                        relationships_skipped_schema_mismatch += 1
                        logging.info(
                            "Skipping relationship with no schema-compatible type: %s -[%s]-> %s",
                            rel.get('from'),
                            rel.get('type'),
                            rel.get('to'),
                        )
                        continue

                    # Use all known surface forms for the entity so canonical-name
                    # mismatches (e.g. "TBK1 kinase" → "TBK1") don't falsely score 0.
                    _src_all_names = _src_node.get('properties', {}).get('all_names') or []
                    _tgt_all_names = _tgt_node.get('properties', {}).get('all_names') or []
                    source_name = (rel.get('properties', {}).get('source_name')
                                   or _uuid_to_name.get(_src_id)
                                   or _src_id)
                    target_name = (rel.get('properties', {}).get('target_name')
                                   or _uuid_to_name.get(_tgt_id)
                                   or _tgt_id)
                    verification_chunks = kg.get('chunks', [])
                    provenance_positions = rel.get('provenance_positions') or []
                    if provenance_positions:
                        scoped_chunks = [
                            chunk for chunk in kg.get('chunks', [])
                            if chunk.get('position') in provenance_positions
                        ]
                        if scoped_chunks:
                            verification_chunks = scoped_chunks

                    restoration = self._verify_relationship_restoration(
                        rel,
                        verification_chunks,
                        source_name=source_name,
                        target_name=target_name,
                        relation_type=sanitized_rel_type,
                        source_aliases=_src_all_names,
                        target_aliases=_tgt_all_names,
                    )
                    properties_filtered["anchor_grounding"] = restoration["anchor_grounding"]
                    properties_filtered["restoration_status"] = restoration["status"]
                    properties_filtered["restoration_verified"] = restoration["verified"]
                    properties_filtered["restoration_grounded_components"] = restoration["grounded_components"]
                    properties_filtered["restoration_grounded_count"] = restoration["grounded_count"]

                    triple_confidence = self._verify_triple_confidence(
                        source_name,
                        target_name,
                        sanitized_rel_type,
                        verification_chunks,
                        source_aliases=_src_all_names,
                        target_aliases=_tgt_all_names,
                    )
                    has_upstream_anchor_grounding = any(
                        original_properties_filtered.get(key)
                        for key in (
                            "anchor_grounding",
                            "source_anchor_spans",
                            "target_anchor_spans",
                            "relation_anchor_spans",
                        )
                    )
                    if restoration["status"] == "full" and has_upstream_anchor_grounding:
                        triple_confidence = max(triple_confidence, 0.95)
                    elif restoration["status"] == "failed":
                        triple_confidence = min(triple_confidence, 0.1)
                    evidence_scope = self._relationship_evidence_scope(
                        verification_chunks,
                        triple_confidence,
                    )

                    llm_reverified = False
                    if (
                        self.enable_low_confidence_triple_reverification
                        and llm is not None
                        and triple_confidence <= self.low_confidence_reverify_threshold
                    ):
                        reverification = self._reverify_low_confidence_triple(
                            source_name=source_name,
                            target_name=target_name,
                            relationship_type=sanitized_rel_type,
                            verification_chunks=verification_chunks,
                            llm=llm,
                            model_name=model_name,
                        )
                        if reverification is False:
                            relationships_reverified_rejected += 1
                            relationships_skipped_low_confidence += 1
                            logging.info(
                                "Skipping low-confidence relationship after LLM reverification: %s -[%s]-> %s",
                                rel.get('from'),
                                sanitized_rel_type,
                                rel.get('to'),
                            )
                            continue
                        if reverification is True:
                            llm_reverified = True
                            relationships_reverified_kept += 1
                            triple_confidence = max(
                                triple_confidence,
                                self.low_confidence_reverify_threshold,
                                self.min_triple_confidence,
                            )

                    # Reject only clear hallucinations: neither entity found anywhere in the document.
                    # Score 0.1 = neither entity present; 0.4+ = at least one entity grounded in text.
                    # Threshold just above 0.1 avoids discarding relationships where one entity
                    # is confirmed (score 0.4) or entity names have minor surface-form mismatches.
                    if triple_confidence < self.min_triple_confidence:
                        relationships_skipped_low_confidence += 1
                        logging.info(
                            "Skipping hallucinated relationship (confidence=%.2f): %s -[%s]-> %s",
                            triple_confidence, rel.get('from'), sanitized_rel_type, rel.get('to'),
                        )
                        continue

                    # Pull negation and qualifiers extracted by LLM
                    negated    = bool(rel.get('negated', False))
                    condition  = rel.get('properties', {}).get('condition') or None
                    quantitative = rel.get('properties', {}).get('quantitative') or None

                    properties_with_confidence = {
                        **properties_filtered,
                        "confidence": triple_confidence,
                        "evidence_scope": evidence_scope,
                        "negated": negated,
                    }
                    if llm_reverified:
                        properties_with_confidence["llm_verified"] = True
                    if condition:
                        properties_with_confidence["condition"] = condition
                    if quantitative:
                        properties_with_confidence["quantitative"] = quantitative
                    properties_with_confidence = self._sanitize_neo4j_properties(
                        properties_with_confidence
                    )

                    # Keep negated in the MERGE key so that opposite claims
                    # (A INHIBITS B negated=false vs true) do not overwrite each other.
                    # Optional qualifiers only participate when present; Neo4j forbids
                    # null property values inside MERGE patterns.
                    rel_query = self._build_relationship_merge_query(
                        sanitized_rel_type,
                        include_condition=condition is not None,
                        include_quantitative=quantitative is not None,
                    )

                    edge_provenance = self._relationship_local_provenance(
                        rel,
                        kg.get("chunks", []),
                    )

                    logging.info(
                        "Creating relationship %d/%d: %s -[%s%s]-> %s (confidence=%.2f)",
                        idx + 1, len(kg['relationships']),
                        rel.get('from'),
                        "NOT " if negated else "",
                        sanitized_rel_type,
                        rel.get('to'), triple_confidence,
                    )

                    graph.query(rel_query, {
                        "source_id": rel.get('from'),
                        "target_id": rel.get('to'),
                        "negated": negated,
                        "condition": condition,
                        "quantitative": quantitative,
                        "properties": properties_with_confidence,
                        "provenance_positions": edge_provenance["provenance_positions"],
                        "question_ids": edge_provenance["question_ids"],
                        "passage_keys": edge_provenance["passage_keys"],
                    })

                    # Create QUALIFIED_BY nodes for significant qualifiers so they
                    # can be traversed independently and appear in path strings.
                    for q_type, q_value in [("condition", condition), ("quantitative", quantitative)]:
                        if not q_value:
                            continue
                        try:
                            qual_id = hashlib.sha1(
                                f"{rel.get('from')}|{sanitized_rel_type}|{rel.get('to')}|{q_type}|{q_value}".encode()
                            ).hexdigest()
                            qual_query = f"""
                            MATCH (source:__Entity__ {{id: $source_id}})
                            MATCH (target:__Entity__ {{id: $target_id}})
                            MERGE (q:Qualifier {{id: $qual_id}})
                            SET q.type = $q_type, q.value = $q_value,
                                q.kgName = $kg_name
                            MERGE (source)-[:{sanitized_rel_type}_QUALIFIED {{negated: $negated}}]->(q)
                            MERGE (q)-[:QUALIFIES]->(target)
                            """
                            graph.query(qual_query, {
                                "source_id": rel.get('from'),
                                "target_id": rel.get('to'),
                                "qual_id":   qual_id,
                                "q_type":    q_type,
                                "q_value":   self._coerce_neo4j_property_value(q_value),
                                "kg_name":   kg.get('metadata', {}).get('kg_name', ''),
                                "negated":   negated,
                            })
                        except Exception as _qe:
                            logging.debug("QUALIFIED_BY node creation failed (non-fatal): %s", _qe)

                    relationships_stored += 1

                except Exception as rel_error:
                    relationship_store_failures += 1
                    logging.error(f"Failed to store relationship {idx+1}: {rel} - Error: {rel_error}")
                    continue

            logging.info(f"Successfully stored {relationships_stored} out of {len(kg['relationships'])} relationships")

            extracted_relationships = len(kg['relationships'])
            attempted_relationships = max(
                0,
                extracted_relationships
                - relationships_skipped_low_confidence
                - relationships_skipped_schema_mismatch,
            )
            relationship_store_ratio = (
                relationships_stored / attempted_relationships
                if attempted_relationships
                else 1.0
            )
            kg.setdefault("metadata", {})
            kg["metadata"]["stored_relationships"] = relationships_stored
            kg["metadata"]["relationship_store_failures"] = relationship_store_failures
            kg["metadata"]["relationships_skipped_low_confidence"] = relationships_skipped_low_confidence
            kg["metadata"]["relationships_skipped_schema_mismatch"] = relationships_skipped_schema_mismatch
            kg["metadata"]["relationships_reverified_kept"] = relationships_reverified_kept
            kg["metadata"]["relationships_reverified_rejected"] = relationships_reverified_rejected
            kg["metadata"]["relationship_store_ratio"] = relationship_store_ratio

            graph.query(
                """
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                SET d.totalRelationships = $storedRelationships,
                    d.extractedRelationships = $extractedRelationships,
                    d.relationshipStoreFailures = $relationshipStoreFailures,
                    d.relationshipsSkippedLowConfidence = $relationshipsSkippedLowConfidence,
                    d.relationshipsSkippedSchemaMismatch = $relationshipsSkippedSchemaMismatch,
                    d.relationshipsReverifiedKept = $relationshipsReverifiedKept,
                    d.relationshipsReverifiedRejected = $relationshipsReverifiedRejected
                """,
                {
                    "fileName": file_name,
                    "kgName": kg_name_value,
                    "storedRelationships": relationships_stored,
                    "extractedRelationships": extracted_relationships,
                    "relationshipStoreFailures": relationship_store_failures,
                    "relationshipsSkippedLowConfidence": relationships_skipped_low_confidence,
                    "relationshipsSkippedSchemaMismatch": relationships_skipped_schema_mismatch,
                    "relationshipsReverifiedKept": relationships_reverified_kept,
                    "relationshipsReverifiedRejected": relationships_reverified_rejected,
                },
            )

            if relationship_store_failures > 0:
                # Treat isolated write misses on large graphs as degraded-but-usable.
                # We still fail fast on tiny graphs or when the failure rate suggests
                # a systemic storage problem rather than a one-off edge issue.
                relationship_failure_ratio = (
                    relationship_store_failures / attempted_relationships
                    if attempted_relationships
                    else 1.0
                )
                fail_small_graph = attempted_relationships < 100
                fail_systemic = relationship_failure_ratio > 0.005
                fail_complete_loss = attempted_relationships > 0 and relationships_stored == 0
                should_fail_build = fail_small_graph or fail_systemic or fail_complete_loss

                log_fn = logging.error if should_fail_build else logging.warning
                log_fn(
                    "Relationship storage incomplete for %s: stored=%d attempted=%d skipped_low_confidence=%d failures=%d failure_ratio=%.4f fatal=%s",
                    file_name,
                    relationships_stored,
                    attempted_relationships,
                    relationships_skipped_low_confidence,
                    relationship_store_failures,
                    relationship_failure_ratio,
                    should_fail_build,
                )
                kg["metadata"]["relationship_store_failure_ratio"] = relationship_failure_ratio
                kg["metadata"]["relationship_store_degraded"] = not should_fail_build
                if should_fail_build:
                    return False

            # Link entities to chunks via per-fact provenance Mention nodes
            # Pattern: (Entity)-[:MENTIONED_IN]->(Mention {quote, ...})-[:FROM_CHUNK]->(Chunk)
            def _mention_boundary(name: str) -> re.Pattern:
                """Adaptive boundary pattern — handles names ending in non-word chars like '(' or '-'."""
                prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
                suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
                return re.compile(prefix + re.escape(name) + suffix)

            for chunk in kg['chunks']:
                # Must use the same kg-scoped hash as the chunk CREATE step above.
                chunk_id = hashlib.sha1(f"{kg_name_value}:{file_name}:{chunk['position']}:{chunk['text']}".encode()).hexdigest()
                chunk_text_lower = chunk['text'].lower()

                for node in kg['nodes']:
                    properties = node.get('properties', {})
                    candidate_names = []
                    candidate_names.extend(properties.get('all_names', []) if isinstance(properties.get('all_names', []), list) else [])
                    candidate_names.append(properties.get('name', ''))
                    candidate_names.append(properties.get('original_id', ''))

                    # Keep meaningful normalized names only
                    normalized_names = [n.strip().lower() for n in candidate_names if isinstance(n, str) and len(n.strip()) > 2]
                    matched_name = next(
                        (n for n in normalized_names if _mention_boundary(n).search(chunk_text_lower)),
                        None
                    )
                    if matched_name:
                        # Extract a short quote: the sentence containing the matched name
                        sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
                        quote = next(
                            (s.strip() for s in sentences if matched_name in s.lower()),
                            chunk['text'][:200]
                        )[:500]  # cap at 500 chars
                        # Seed mention_id from node['id'] (the unique key), not content_hash.
                        # content_hash is no longer unique across KGs (Fix 8 changed MERGE to id).
                        mention_id = hashlib.sha256(
                            f"{node['id']}::{chunk_id}".encode()
                        ).hexdigest()
                        mention_query = """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:__Entity__ {id: $entity_id})
                        MERGE (m:Mention {id: $mention_id})
                        SET m.quote = $quote,
                            m.chunkIndex = $chunk_index,
                            m.chunkLocalIndex = $chunk_local_index,
                            m.chunkStart = $chunk_start,
                            m.chunkEnd = $chunk_end,
                            m.chunkSource = $chunk_source,
                            m.entityName = $entity_name,
                            m.createdAt = datetime()
                        MERGE (e)-[:MENTIONED_IN]->(m)
                        MERGE (m)-[:FROM_CHUNK]->(c)
                        MERGE (c)-[:HAS_ENTITY]->(e)
                        """
                        graph.query(mention_query, {
                            "chunk_id": chunk_id,
                            "entity_id": node['id'],
                            "mention_id": mention_id,
                            "quote": quote,
                            "chunk_index": chunk.get('position', chunk.get('chunk_id', 0)),
                            "chunk_local_index": chunk.get('chunk_local_index', chunk.get('chunk_id', 0)),
                            "chunk_start": chunk.get('start_pos', 0),
                            "chunk_end": chunk.get('end_pos', 0),
                            "chunk_source": chunk.get('source', ''),
                            "entity_name": properties.get('name', ''),
                        })

            enrichment_stats = self._store_graph_enrichment(
                graph,
                kg_name=kg_name_value,
                file_name=file_name,
                enrichment=kg.get("metadata", {}).get("graph_enrichment") or {},
                chunks=kg.get("chunks", []),
            )
            if enrichment_stats:
                kg["metadata"]["component_summary_count"] = enrichment_stats.get("component_summaries", 0)
                kg["metadata"]["claim_count"] = enrichment_stats.get("claims", 0)
                kg["metadata"]["fragmentation_bridge_count"] = enrichment_stats.get("fragmentation_bridges", 0)
                graph.query(
                    """
                    MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                    SET d.componentSummaryCount = $component_summary_count,
                        d.claimCount = $claim_count,
                        d.fragmentationBridgeCount = $fragmentation_bridge_count
                    """,
                    {
                        "file_name": file_name,
                        "kg_name": kg_name_value,
                        "component_summary_count": int(enrichment_stats.get("component_summaries", 0)),
                        "claim_count": int(enrichment_stats.get("claims", 0)),
                        "fragmentation_bridge_count": int(enrichment_stats.get("fragmentation_bridges", 0)),
                    },
                )

            # Create vector indexes for RAG
            self._create_vector_indexes(graph)

            logging.info(f"Successfully stored ontology-guided knowledge graph for {file_name}")
            return True

        except Exception as e:
            logging.error(f"Error storing knowledge graph: {e}")
            return False

    def _create_vector_indexes(self, graph):
        """
        Create vector indexes and unique constraints for RAG functionality
        """
        try:
            # Create unique constraint for entity IDs to prevent duplicates
            entity_constraint_query = """
            CREATE CONSTRAINT unique_entity_id IF NOT EXISTS
            FOR (e:__Entity__) REQUIRE e.id IS UNIQUE
            """
            graph.query(entity_constraint_query)

            # Create unique constraint for chunk IDs
            chunk_constraint_query = """
            CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
            graph.query(chunk_constraint_query)

            retrieval_chunk_constraint_query = """
            CREATE CONSTRAINT unique_retrieval_chunk_id IF NOT EXISTS
            FOR (rc:RetrievalChunk) REQUIRE rc.id IS UNIQUE
            """
            graph.query(retrieval_chunk_constraint_query)

            # Create composite uniqueness for dataset-scoped documents
            # (same fileName may exist in different kgName datasets)
            doc_constraint_query = """
            CREATE CONSTRAINT unique_document_filename_kgname IF NOT EXISTS
            FOR (d:Document) REQUIRE (d.fileName, d.kgName) IS UNIQUE
            """
            graph.query(doc_constraint_query)

            # Unique constraint for Mention nodes (entity × chunk pair)
            mention_constraint_query = """
            CREATE CONSTRAINT unique_mention_id IF NOT EXISTS
            FOR (m:Mention) REQUIRE m.id IS UNIQUE
            """
            graph.query(mention_constraint_query)

            summary_constraint_query = """
            CREATE CONSTRAINT unique_summary_id IF NOT EXISTS
            FOR (s:Summary) REQUIRE s.id IS UNIQUE
            """
            graph.query(summary_constraint_query)

            claim_constraint_query = """
            CREATE CONSTRAINT unique_claim_id IF NOT EXISTS
            FOR (c:Claim) REQUIRE c.id IS UNIQUE
            """
            graph.query(claim_constraint_query)

            # Create vector index for chunks
            chunk_index_query = f"""
            CREATE VECTOR INDEX vector IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(chunk_index_query)

            retrieval_chunk_index_query = f"""
            CREATE VECTOR INDEX retrieval_vector IF NOT EXISTS
            FOR (rc:RetrievalChunk) ON (rc.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(retrieval_chunk_index_query)

            # Create vector index for entities
            entity_index_query = f"""
            CREATE VECTOR INDEX entity_vector IF NOT EXISTS
            FOR (e:__Entity__) ON (e.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(entity_index_query)

            # Create keyword index for full-text search
            keyword_index_query = """
            CREATE FULLTEXT INDEX keyword IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """
            graph.query(keyword_index_query)

            retrieval_keyword_index_query = """
            CREATE FULLTEXT INDEX retrieval_keyword IF NOT EXISTS
            FOR (rc:RetrievalChunk) ON EACH [rc.text]
            """
            graph.query(retrieval_keyword_index_query)

            # Index on entity name for fast text-matching and multi-hop traversal
            # lookups (EnhancedRAGSystem._expand_entities_via_graph seeds from e.id)
            entity_id_index_query = """
            CREATE INDEX entity_id_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.id)
            """
            graph.query(entity_id_index_query)

            entity_name_index_query = """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.name)
            """
            graph.query(entity_name_index_query)

            # Composite index for chunk→document lookup used in kg_name filtering
            chunk_kg_index_query = """
            CREATE INDEX chunk_document_index IF NOT EXISTS
            FOR ()-[r:PART_OF]-() ON (r)
            """
            try:
                graph.query(chunk_kg_index_query)
            except Exception:
                pass  # Relationship indexes not supported on all Neo4j versions

            logging.info("Created constraints, vector, keyword, and entity lookup indexes for RAG")

        except Exception as e:
            logging.warning(f"Error creating constraints/indexes (may already exist): {e}")
