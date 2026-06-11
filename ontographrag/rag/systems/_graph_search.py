import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np

class GraphSearchMixin:
    """Entity-first retrieval, graph expansion, scoped support clauses, and PPR scoring.

    Mixin for :class:`EnhancedRAGSystem`; method bodies are unchanged
    from the original monolithic implementation.
    """

    @classmethod
    def _graph_traversal_seed_limit_for_kg(cls, kg_name: Optional[str]) -> int:
        kg_key = str(kg_name or "").strip().lower()
        return cls._GRAPH_TRAVERSAL_SEED_LIMIT_BY_KG.get(
            kg_key,
            cls._GRAPH_TRAVERSAL_SEED_LIMIT,
        )

    @classmethod
    def _graph_traversal_neighbor_limit_for_kg(
        cls,
        kg_name: Optional[str],
        requested_limit: int,
    ) -> int:
        kg_key = str(kg_name or "").strip().lower()
        dataset_limit = cls._GRAPH_TRAVERSAL_NEIGHBOR_LIMIT_BY_KG.get(
            kg_key,
            cls._GRAPH_TRAVERSAL_NEIGHBOR_LIMIT,
        )
        return max(1, min(requested_limit, dataset_limit))

    @classmethod
    def _graph_traversal_max_hops_for_kg(
        cls,
        kg_name: Optional[str],
        requested_max_hops: int,
    ) -> int:
        kg_key = str(kg_name or "").strip().lower()
        dataset_limit = cls._GRAPH_TRAVERSAL_MAX_HOPS_BY_KG.get(
            kg_key,
            requested_max_hops,
        )
        return max(1, min(requested_max_hops, dataset_limit))

    @staticmethod
    def _question_local_pair_support_clause(
        left_var: str,
        right_var: str,
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]] = None,
        relationship_var: Optional[str] = None,
    ) -> str:
        """
        Build a Cypher predicate requiring local support for an entity pair.

        On question-scoped datasets, entity nodes are still shared within the
        dataset KG. This clause keeps relation lookup tied to the current
        question bundle by requiring both entities to be supported in the same
        passage, or in adjacent chunks of that passage.
        """
        if not question_id and not document_names:
            return "true"

        left_filters = ""
        right_filters = ""
        if kg_name:
            left_filters += "\n              AND d1.kgName = $kg_name"
            right_filters += "\n              AND d2.kgName = $kg_name"
        if document_names:
            left_filters += "\n              AND d1.fileName IN $document_names"
            right_filters += "\n              AND d2.fileName IN $document_names"

        if not question_id:
            return f"""EXISTS {{
            MATCH ({left_var})<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
            WHERE true{left_filters}
        }} AND EXISTS {{
            MATCH ({right_var})<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
            WHERE true{right_filters}
        }}"""

        direct_edge_scope = ""
        if relationship_var:
            direct_edge_scope = f"$question_id IN coalesce({relationship_var}.questionIds, []) OR "

        return f"""{direct_edge_scope}EXISTS {{
            MATCH ({left_var})<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
            MATCH ({right_var})<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
            WHERE c1.questionId = $question_id
              AND c2.questionId = $question_id
              AND coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)
              AND abs(
                    coalesce(c1.chunkLocalIndex, c1.position)
                    - coalesce(c2.chunkLocalIndex, c2.position)
                  ) <= 1
                      {left_filters}
                      {right_filters}
        }}"""

    @staticmethod
    def _question_local_entity_support_clause(
        entity_var: str,
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]] = None,
    ) -> str:
        """Build a Cypher predicate requiring an entity seed to be locally supported."""
        conditions: List[str] = []
        if question_id:
            conditions.append("c.questionId = $question_id")
        if kg_name:
            conditions.append("d.kgName = $kg_name")
        if document_names:
            conditions.append("d.fileName IN $document_names")
        where_clause = " AND ".join(conditions) if conditions else "true"
        return f"""EXISTS {{
            MATCH ({entity_var})<-[:MENTIONS|HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
            WHERE {where_clause}
        }}"""

    @staticmethod
    def _question_local_path_support_clause(
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]] = None,
    ) -> str:
        """Build a Cypher predicate requiring question-local support for every path hop."""
        if not question_id and not document_names:
            return ""

        path_filters = ""
        if kg_name:
            path_filters += """
                      AND d1.kgName = $kg_name
                      AND d2.kgName = $kg_name"""
        if document_names:
            path_filters += """
                      AND d1.fileName IN $document_names
                      AND d2.fileName IN $document_names"""

        if not question_id:
            node_conditions: List[str] = []
            if kg_name:
                node_conditions.append("d.kgName = $kg_name")
            if document_names:
                node_conditions.append("d.fileName IN $document_names")
            node_where = " AND ".join(node_conditions) if node_conditions else "true"
            return f"""
              AND ALL(pathNode IN nodes(path) WHERE EXISTS {{
                    MATCH (pathNode)<-[:MENTIONS|HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
                    WHERE {node_where}
              }})"""

        return f"""
              AND ALL(idx IN range(0, length(path) - 1) WHERE (
                    $question_id IN coalesce(relationships(path)[idx].questionIds, [])
                    OR EXISTS {{
                    WITH nodes(path)[idx] AS a, nodes(path)[idx + 1] AS b
                    MATCH (a)<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
                    MATCH (b)<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
                    WHERE c1.questionId = $question_id
                      AND c2.questionId = $question_id
                      AND coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)
                      AND abs(
                            coalesce(c1.chunkLocalIndex, c1.position)
                            - coalesce(c2.chunkLocalIndex, c2.position)
                          ) <= 1
                      {path_filters}
              }}))"""

    def _fetch_relationships_for_entity_ids(
        self,
        graph,
        entity_ids: List[str],
        *,
        kg_name: Optional[str] = None,
        question_id: Optional[str] = None,
        document_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch entity-to-entity relationships, optionally scoped to one question bundle."""
        if not entity_ids:
            return []

        pair_scope = self._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name=kg_name,
            question_id=question_id,
            document_names=document_names,
            relationship_var="r",
        )
        relationship_query = f"""
        MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        WHERE e1.id IN $entity_ids
          AND e2.id IN $entity_ids
          AND coalesce(r.confidence, 1.0) >= 0.4
          AND {pair_scope}
        RETURN DISTINCT
            e1.id AS source,
            e1.name AS source_name,
            elementId(e1) AS source_element_id,
            e2.id AS target,
            e2.name AS target_name,
            elementId(e2) AS target_element_id,
            type(r) AS relationship_type,
            elementId(r) AS relationship_element_id,
            coalesce(r.negated, false) AS negated,
            r.condition AS condition,
            r.quantitative AS quantitative,
            coalesce(r.confidence, 1.0) AS confidence,
            coalesce(r.questionIds, []) AS question_ids,
            coalesce(r.passageKeys, []) AS passage_keys,
            coalesce(r.provenancePositions, []) AS provenance_positions
        """
        return graph.query(
            relationship_query,
            {
                "entity_ids": entity_ids,
                "kg_name": kg_name,
                "question_id": question_id,
                "document_names": document_names,
            },
        ) or []

    def _plan_relation_paths(
        self,
        graph,
        seed_entity_ids: List[str],
        query: str,
        *,
        kg_name: str = None,
        question_id: str = None,
        document_names: List[str] = None,
        top_k: int = 10,
        min_sim: float = 0.20,
    ) -> Optional[List[str]]:
        """RoG-style relation path planning.

        Enumerate all relation types reachable in 1-2 hops from seed entities,
        score each by cosine similarity between its label embedding and the query
        embedding, and return the top-K types whose similarity exceeds min_sim.

        Returns None if planning should be skipped (too few relation types found
        or all types are structural), which signals the caller to use open traversal.
        """
        if not seed_entity_ids:
            return None

        scope_filters = ""
        params: Dict[str, Any] = {"seed_ids": seed_entity_ids}
        if kg_name:
            params["kg_name"] = kg_name
            scope_filters += "\n  AND (a.kgName IS NULL OR a.kgName = $kg_name)"
        if question_id:
            params["question_id"] = question_id
            scope_filters += "\n  AND (a.questionId IS NULL OR a.questionId = $question_id)"

        rel_query = f"""
        MATCH (a:__Entity__)-[r]-(b:__Entity__)
        WHERE a.id IN $seed_ids
          {scope_filters}
          AND NOT type(r) IN {list(self._STRUCTURAL_REL_TYPES)}
        RETURN DISTINCT type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 40
        """
        try:
            rows = graph.query(rel_query, params) or []
        except Exception as _e:
            logging.debug("RoG relation planning query failed: %s", _e)
            return None

        rel_types = [row["rel_type"] for row in rows if row.get("rel_type")]
        if not rel_types:
            return None

        # Score each relation type label against the query by cosine similarity.
        query_emb = np.array(self._generate_query_embedding(query), dtype=np.float32)
        qnorm = np.linalg.norm(query_emb)
        if qnorm == 0:
            return None

        scored: List[Tuple[float, str]] = []
        for rel_type in rel_types:
            # Convert snake_case / ALL_CAPS label to readable words for embedding
            readable = rel_type.replace("_", " ").lower()
            try:
                rel_emb = np.array(self._generate_query_embedding(readable), dtype=np.float32)
            except Exception:
                continue
            rnorm = np.linalg.norm(rel_emb)
            if rnorm == 0:
                continue
            sim = float(np.dot(query_emb, rel_emb) / (qnorm * rnorm))
            scored.append((sim, rel_type))

        scored.sort(reverse=True)
        allowed = [rt for sim, rt in scored[:top_k] if sim >= min_sim]

        if not allowed:
            logging.debug("RoG planning: no relation types above sim threshold %.2f — using open traversal", min_sim)
            return None

        logging.info(
            "RoG relation planning: %d/%d types selected (top sim=%.3f): %s",
            len(allowed), len(rel_types),
            scored[0][0] if scored else 0.0,
            allowed[:5],
        )
        return allowed

    def _expand_entities_via_graph(
        self,
        graph,
        seed_entity_ids: List[str],
        kg_name: str = None,
        max_hops: int = 2,
        max_neighbors: int = 30,
        question_id: str = None,
        document_names: List[str] = None,
        allowed_rel_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-hop graph traversal from seed entities.

        Starting from the entities found in the initially retrieved chunks,
        this method walks the knowledge graph up to ``max_hops`` relationship
        hops and returns:
          - ``neighbors``: newly discovered entities (not in the seed set)
          - ``paths``: human-readable traversal paths for the LLM prompt,
              e.g. "Metformin --TREATS--> Type2Diabetes --HAS_COMPLICATION--> Nephropathy"

        Neighbors are ranked by:
          1. How many distinct seed entities connect to them (higher = more central)
          2. Minimum hop distance (closer = higher priority)
        """
        if not seed_entity_ids:
            return {"neighbors": {}, "paths": []}

        try:
            effective_max_hops = self._graph_traversal_max_hops_for_kg(kg_name, max_hops)
            effective_max_neighbors = self._graph_traversal_neighbor_limit_for_kg(
                kg_name,
                max_neighbors,
            )
            seed_limit = self._graph_traversal_seed_limit_for_kg(kg_name)
            if len(seed_entity_ids) > seed_limit:
                logging.info(
                    "Capping graph traversal seeds for kg '%s' from %d to %d",
                    kg_name,
                    len(seed_entity_ids),
                    seed_limit,
                )
                seed_entity_ids = seed_entity_ids[:seed_limit]

            params: Dict[str, Any] = {
                "seed_ids": seed_entity_ids,
                "max_neighbors": effective_max_neighbors,
            }

            seed_scope = "true"
            neighbor_scope = ""
            if kg_name or question_id or document_names:
                params["kg_name"] = kg_name
                params["question_id"] = question_id
                params["document_names"] = document_names
                seed_scope = self._question_local_entity_support_clause(
                    "seed",
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                neighbor_scope = self._question_local_entity_support_clause(
                    "neighbor",
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                neighbor_scope = f"\n              AND {neighbor_scope}"

            path_scope = self._question_local_path_support_clause(
                kg_name=kg_name,
                question_id=question_id,
                document_names=document_names,
            )

            # RoG-style: when an allowed relation type list is provided, constrain
            # traversal to only follow those relation types (in addition to the
            # standard structural-edge exclusion).
            rog_rel_filter = ""
            if allowed_rel_types:
                params["rog_allowed_rel_types"] = allowed_rel_types
                rog_rel_filter = "\n                AND type(r) IN $rog_allowed_rel_types"

            traversal_query = f"""
            MATCH (seed:__Entity__)
            WHERE seed.id IN $seed_ids
              AND {seed_scope}
            MATCH path = (seed)-[*1..{effective_max_hops}]-(neighbor:__Entity__)
            WHERE NOT neighbor.id IN $seed_ids
              AND neighbor.id IS NOT NULL
              AND ALL(r IN relationships(path) WHERE NOT type(r) IN ['HAS_ENTITY', 'FROM_CHUNK', 'PART_OF', 'MENTIONED_IN', 'MENTIONS', 'QUALIFIES']
                AND coalesce(r.confidence, 1.0) >= 0.4{rog_rel_filter})
              {neighbor_scope}
              {path_scope}
            WITH neighbor,
                 length(path) AS hops,
                 [n IN nodes(path) | coalesce(n.name, n.id)] AS node_names,
                 [n IN nodes(path) | n.id] AS node_ids,
                [r IN relationships(path) | {{
                     type: type(r),
                     negated: coalesce(r.negated, false),
                     condition: r.condition,
                     quantitative: r.quantitative,
                     confidence: coalesce(r.confidence, 1.0),
                     question_ids: coalesce(r.questionIds, []),
                     passage_keys: coalesce(r.passageKeys, []),
                     provenance_positions: coalesce(r.provenancePositions, [])
                 }}] AS rel_data,
                 seed.id AS seed_id
            WITH neighbor,
                 min(hops) AS min_hops,
                 count(DISTINCT seed_id) AS seed_connections,
                 collect(DISTINCT {{nodes: node_names, node_ids: node_ids, rels: rel_data}})[0..3] AS sample_paths
            RETURN
                 neighbor.id AS id,
                 neighbor.name AS name,
                 coalesce(neighbor.type, 'Entity') AS type,
                 elementId(neighbor) AS element_id,
                 min_hops,
                 seed_connections,
                 sample_paths
            ORDER BY seed_connections DESC, min_hops ASC
            LIMIT $max_neighbors
            """

            results = graph.query(traversal_query, params)

            neighbors: Dict[str, Any] = {}
            seen_paths: set = set()
            paths: List[Dict[str, Any]] = []

            for row in results:
                neighbor_id = row["id"]
                if not neighbor_id:
                    continue

                neighbors[neighbor_id] = {
                    "id": neighbor_id,
                    "name": row["name"] or neighbor_id,
                    "type": row["type"],
                    "element_id": row["element_id"],
                    "min_hops": row["min_hops"],
                    "seed_connections": row["seed_connections"],
                    "source": "graph_traversal",
                }

                # Build human-readable path strings for the prompt.
                # Store node_ids alongside the path string so iterative retrieval
                # can prune paths whose nodes are not in the retained entity set.
                for path_data in (row["sample_paths"] or []):
                    node_names = path_data.get("nodes", [])
                    node_ids = path_data.get("node_ids", [])
                    rel_data = path_data.get("rels", [])
                    if node_names and rel_data and len(node_names) == len(rel_data) + 1:
                        parts = [node_names[0]]
                        for rel, node in zip(rel_data, node_names[1:]):
                            if rel is None:
                                rel_label = "RELATED"
                            elif isinstance(rel, dict):
                                rel_label = self._format_relationship_label(rel)
                            else:
                                rel_label = str(rel) or "RELATED"
                            parts.extend([f"--{rel_label}-->", node])
                        path_str = " ".join(parts)
                        if path_str not in seen_paths:
                            path_positions: Set[int] = set()
                            path_question_ids: Set[str] = set()
                            path_passage_keys: Set[str] = set()
                            for rel in rel_data:
                                if not isinstance(rel, dict):
                                    continue
                                path_positions.update(
                                    int(pos)
                                    for pos in (rel.get("provenance_positions") or [])
                                    if isinstance(pos, (int, float))
                                )
                                path_question_ids.update(
                                    str(qid)
                                    for qid in (rel.get("question_ids") or [])
                                    if str(qid).strip()
                                )
                                path_passage_keys.update(
                                    str(key)
                                    for key in (rel.get("passage_keys") or [])
                                    if str(key).strip()
                                )
                            seen_paths.add(path_str)
                            paths.append({
                                "path": path_str,
                                "hops": row["min_hops"],
                                "node_ids": [nid for nid in node_ids if nid],
                                "provenance_positions": sorted(path_positions),
                                "question_ids": sorted(path_question_ids),
                                "passage_keys": sorted(path_passage_keys),
                            })

            logging.info(
                "Graph traversal: %d neighbors, %d paths from %d seeds (max_hops=%d)",
                len(neighbors), len(paths), len(seed_entity_ids), effective_max_hops,
            )
            return {"neighbors": neighbors, "paths": paths}

        except Exception as e:
            logging.warning("Graph traversal failed: %s", e)
            return {"neighbors": {}, "paths": []}

    @staticmethod
    def _build_symbolic_entity_lookup_query(entity_support_scope: str) -> str:
        """Build the Cypher query used by the symbolic entity matcher.

        Neo4j aliases defined in a WITH projection are not visible to other
        expressions in the same projection list. We therefore normalize
        ``entity_name`` in one WITH clause and compute token overlap in the
        next one; otherwise the symbolic matcher fails with a syntax error and
        the system unnecessarily falls back to vector search.
        """
        return f"""
        MATCH (e:__Entity__)
        WITH e,
             toLower(coalesce(e.name, '')) AS entity_name,
             toLower(coalesce(e.source_title, '')) AS source_title_lower
        WITH e, entity_name, source_title_lower,
             [tok IN $query_tokens WHERE entity_name CONTAINS tok] AS matched_tokens,
             CASE WHEN $query_text CONTAINS entity_name THEN 1 ELSE 0 END AS exact_in_query,
             CASE WHEN entity_name CONTAINS $query_text THEN 1 ELSE 0 END AS query_in_entity,
             CASE WHEN size(source_title_lower) >= 4
                       AND ($query_text CONTAINS source_title_lower
                            OR source_title_lower CONTAINS $query_text)
                  THEN 1 ELSE 0 END AS title_match
        WHERE size(entity_name) >= 4
          AND (
            exact_in_query = 1
            OR query_in_entity = 1
            OR size(matched_tokens) >= 2
            OR title_match = 1
            OR ANY(alias IN coalesce(e.all_names, [])
                   WHERE $query_text CONTAINS toLower(alias)
                      OR toLower(alias) CONTAINS $query_text)
            OR ANY(syn IN coalesce(e.synonyms, [])
                   WHERE $query_text CONTAINS toLower(syn)
                      OR toLower(syn) CONTAINS $query_text)
          )
          AND {entity_support_scope}
        RETURN e.id AS id, e.name AS name,
               coalesce(e.type, 'Entity') AS type,
               elementId(e) AS element_id,
               exact_in_query,
               query_in_entity,
               size(matched_tokens) AS token_overlap
        ORDER BY exact_in_query DESC,
                 query_in_entity DESC,
                 token_overlap DESC,
                 title_match DESC,
                 size(entity_name) DESC
        LIMIT $entity_lookup_limit
        """

    def _fetch_subgraph_edges(
        self,
        graph,
        entity_ids: List[str],
        kg_name: str = None,
        question_id: str = None,
        document_names: List[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Return all directed edges (src_id, tgt_id) between a set of entity IDs.

        Used to build the adjacency structure for PPR power iteration.
        Structural meta-edges (HAS_ENTITY, PART_OF, etc.) are excluded so only
        domain-semantic relationships are included in the random walk.
        """
        if not entity_ids:
            return []
        params: Dict[str, Any] = {
            "entity_ids": entity_ids,
            "kg_name": kg_name,
            "question_id": question_id,
            "document_names": document_names,
        }
        source_support = self._question_local_entity_support_clause(
            "a",
            kg_name=kg_name,
            question_id=question_id,
            document_names=document_names,
        )
        target_support = self._question_local_entity_support_clause(
            "b",
            kg_name=kg_name,
            question_id=question_id,
            document_names=document_names,
        )
        local_filters = ""
        if kg_name or question_id or document_names:
            local_filters = f"""
                  AND {source_support}
                  AND {target_support}"""
        qid_filter = ""
        if question_id:
            qid_filter = """
            AND ANY(pid IN coalesce(r.questionIds, []) WHERE pid = $question_id)"""
        try:
            rows = graph.query(
                f"""
                MATCH (a:__Entity__)-[r]->(b:__Entity__)
                WHERE a.id IN $entity_ids
                  AND b.id IN $entity_ids
                  AND NOT type(r) IN ['HAS_ENTITY', 'FROM_CHUNK', 'PART_OF',
                                      'MENTIONED_IN', 'MENTIONS', 'QUALIFIES']
                  AND coalesce(r.confidence, 1.0) >= 0.4
                  {local_filters}
                  {qid_filter}
                RETURN a.id AS src, b.id AS tgt, coalesce(r.confidence, 1.0) AS conf
                LIMIT 2000
                """,
                params,
            ) or []
            return [(row["src"], row["tgt"], float(row["conf"])) for row in rows if row["src"] and row["tgt"]]
        except Exception as exc:
            logging.debug("_fetch_subgraph_edges failed (%s); PPR will fall back to hop scores.", exc)
            return []

    def _ppr_entity_scores(
        self,
        seed_ids: List[str],
        all_entity_ids: List[str],
        edges: List[Tuple],
    ) -> Dict[str, float]:
        """
        Approximate Personalized PageRank via confidence-weighted power iteration.

        Each edge carries a confidence weight (0.4–1.0 from the KG).  The random
        surfer splits probability mass across neighbours proportional to edge weight
        rather than uniformly, so high-confidence relation paths receive stronger
        activation than low-confidence ones.

        Returns {entity_id: ppr_score}.  Scores sum to 1.0 over all entities.
        """
        n = len(all_entity_ids)
        if n == 0:
            return {}

        idx = {eid: i for i, eid in enumerate(all_entity_ids)}

        # Build weighted undirected adjacency: adj[i] = {j: weight, ...}
        adj: List[Dict[int, float]] = [{} for _ in range(n)]
        for edge in edges:
            src, tgt = edge[0], edge[1]
            weight = float(edge[2]) if len(edge) > 2 else 1.0
            i, j = idx.get(src), idx.get(tgt)
            if i is not None and j is not None and i != j:
                adj[i][j] = max(adj[i].get(j, 0.0), weight)
                adj[j][i] = max(adj[j].get(i, 0.0), weight)

        seed_indices = [idx[s] for s in seed_ids if s in idx]
        e = np.zeros(n)
        if seed_indices:
            for si in seed_indices:
                e[si] = 1.0 / len(seed_indices)

        v = e.copy()
        alpha = self._PPR_ALPHA
        for _ in range(self._PPR_STEPS):
            new_v = np.zeros(n)
            for i in range(n):
                if adj[i]:
                    total_weight = sum(adj[i].values())
                    for j, w in adj[i].items():
                        new_v[j] += v[i] * (w / total_weight)
                else:
                    # Dangling nodes redistribute through the restart distribution
                    # so probability mass is preserved and scores stay comparable
                    # across sparse and dense local subgraphs.
                    new_v += v[i] * e
            v = (1.0 - alpha) * e + alpha * new_v

        return {eid: float(v[idx[eid]]) for eid in all_entity_ids}

    def _entity_first_search(self, graph, query: str, max_chunks: int = 20, kg_name: str = None, max_hops: int = 2, question_id: str = None, llm=None, document_names: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Entity-first retrieval: find KG entities whose names appear in the question,
        walk the graph from those seeds, then retrieve the chunks those entities came from.

        Improvements over baseline:
        - Hop-weighted chunk scores: seed chunks score 1.0, 1-hop 0.8, 2-hop 0.6.
        - Grounding quality is computed and returned with every result.
        - Sufficiency check: if < 2 chunks are recovered, returns None so the caller
          falls back to vector search rather than serving thin, low-recall context.

        Returns a context dict on success, or None to signal the caller should fall back
        to vector/text similarity search.
        """
        try:
            # Step 1a: embedding-based entity linking — find KG entities whose
            # embedding is closest to the question embedding.  This catches
            # partial-name matches (e.g. "TBK1" → "TBK1 kinase") that substring
            # search misses.
            entity_support_scope = self._question_local_entity_support_clause(
                "e",
                kg_name=kg_name,
                question_id=question_id,
                document_names=document_names,
            )
            kg_scope_exist = f"\n                  AND {entity_support_scope}"

            # Step 1a: ANN entity linking.
            #
            # HippoRAG-style: when an LLM is available, extract named entity
            # mentions from the question first, then embed each short entity
            # string individually and look it up in the entity_vector index.
            # entity-string → entity-embedding similarity is much tighter than
            # full-question → entity-embedding similarity, so we get fewer
            # hub-entity false positives and can use a higher threshold (0.72).
            #
            # Fallback (no llm, or extraction returns nothing): embed the full
            # query and use the original 0.55 threshold, same as before.
            query_embedding = self._generate_query_embedding(query)

            # Check entity_vector index dimension before querying — mismatches cause
            # a hard exception that aborts the entire entity-first search.
            emb_matched = []
            try:
                _idx_info = graph.query(
                    "SHOW INDEXES YIELD name, options WHERE name = 'entity_vector' RETURN options"
                )
                _idx_dim = None
                if _idx_info:
                    _opts = _idx_info[0].get("options") or {}
                    _idx_dim = (_opts.get("indexConfig") or {}).get("vector.dimensions")
                if _idx_dim is not None and int(_idx_dim) != self.embedding_dimension:
                    logging.warning(
                        "Skipping embedding entity lookup: index has %d dims, model has %d dims. "
                        "Rebuild the KG with the current embedding model to enable this.",
                        _idx_dim, self.embedding_dimension,
                    )
                else:
                    emb_lookup_query = f"""
                    CALL db.index.vector.queryNodes('entity_vector', 15, $query_vector)
                    YIELD node AS e, score
                    WHERE score >= $ann_threshold
                      {kg_scope_exist}
                    RETURN e.id AS id, e.name AS name,
                           coalesce(e.type, 'Entity') AS type,
                           elementId(e) AS element_id,
                           score AS ann_score,
                           coalesce(e.node_specificity, 1.0) AS node_specificity,
                           coalesce(e.passage_count, 1) AS passage_count,
                           CASE WHEN $use_node_specificity
                                THEN score * coalesce(e.node_specificity, 1.0)
                                ELSE score
                           END AS weighted_score
                    ORDER BY weighted_score DESC
                    """
                    ann_scope_params: Dict[str, Any] = {
                        "kg_name": kg_name,
                        "use_node_specificity": self.use_node_specificity,
                    }
                    if question_id:
                        ann_scope_params["question_id"] = question_id
                    if document_names:
                        ann_scope_params["document_names"] = document_names

                    # --- Per-entity embedding pass (HippoRAG-style) ---
                    query_entities: List[str] = []
                    if llm is not None and self.use_per_entity_ann:
                        query_entities = self._extract_query_entities(query, llm)

                    if query_entities:
                        # One ANN call per extracted entity; union results, keep best score.
                        seen_by_id: Dict[str, Any] = {}
                        for entity_mention in query_entities:
                            entity_vec = self._generate_query_embedding(entity_mention)
                            rows = graph.query(
                                emb_lookup_query,
                                {
                                    **ann_scope_params,
                                    "query_vector": entity_vec,
                                    "ann_threshold": 0.72,
                                },
                            ) or []
                            for row in rows:
                                row = dict(row)
                                eid = row["id"]
                                if eid not in seen_by_id or row["weighted_score"] > seen_by_id[eid]["weighted_score"]:
                                    seen_by_id[eid] = row
                        emb_matched = list(seen_by_id.values())
                        logging.info(
                            "Entity-first ANN (per-entity): extracted %d query entities → %d KG candidates",
                            len(query_entities), len(emb_matched),
                        )
                    else:
                        # Fallback: embed the full query (original behaviour)
                        emb_matched = graph.query(
                            emb_lookup_query,
                            {
                                **ann_scope_params,
                                "query_vector": query_embedding,
                                "ann_threshold": 0.55,
                            },
                        ) or []
            except Exception as _emb_err:
                logging.warning("Embedding entity lookup skipped: %s", _emb_err)

            # Step 1b: substring / token matching as a complementary pass
            params: Dict[str, Any] = {"query_text": query.lower()}
            if kg_name:
                params["kg_name"] = kg_name
            if question_id:
                params["question_id"] = question_id
            if document_names:
                params["document_names"] = document_names

            # Build a conservative content-token set for symbolic lookup.
            # Requiring either an exact phrase or at least two content-token overlaps
            # sharply reduces spurious open-domain matches on dataset KGs.
            query_tokens = self._content_query_tokens(query)
            params["query_tokens"] = query_tokens
            params["entity_lookup_limit"] = self._ENTITY_LOOKUP_LIMIT

            entity_lookup_query = self._build_symbolic_entity_lookup_query(entity_support_scope)
            sub_matched = graph.query(entity_lookup_query, params) or []

            # Merge both passes, deduplicating by entity id and keeping the best-ranked
            # candidate features. Exact phrase matches outrank overlap matches, which
            # outrank embedding-only matches.
            merged_by_id: Dict[str, Dict[str, Any]] = {}
            for row in emb_matched:
                row = dict(row)
                row.setdefault("exact_in_query", 0)
                row.setdefault("query_in_entity", 0)
                row.setdefault("token_overlap", 0)
                row["_embedding_score"] = float(row.get("ann_score", row.get("score", 0.0)))
                row["_source_kind"] = "embedding"
                merged_by_id[row["id"]] = row

            for row in sub_matched:
                row = dict(row)
                row["_embedding_score"] = float(
                    merged_by_id.get(row["id"], {}).get("_embedding_score", 0.0)
                )
                row["_source_kind"] = "symbolic"
                existing = merged_by_id.get(row["id"])
                if existing is None:
                    merged_by_id[row["id"]] = row
                    continue
                existing_rank = (
                    int(existing.get("exact_in_query", 0)),
                    int(existing.get("query_in_entity", 0)),
                    int(existing.get("token_overlap", 0)),
                    float(existing.get("_embedding_score", 0.0)),
                )
                new_rank = (
                    int(row.get("exact_in_query", 0)),
                    int(row.get("query_in_entity", 0)),
                    int(row.get("token_overlap", 0)),
                    float(row.get("_embedding_score", 0.0)),
                )
                if new_rank > existing_rank:
                    merged_by_id[row["id"]] = row

            matched = sorted(
                merged_by_id.values(),
                key=lambda row: (
                    int(row.get("exact_in_query", 0)),
                    int(row.get("query_in_entity", 0)),
                    int(row.get("token_overlap", 0)),
                    float(row.get("_embedding_score", 0.0)),
                    len(str(row.get("name") or row.get("id") or "")),
                ),
                reverse=True,
            )[:self._MAX_ENTITY_SEEDS]

            matched_query_token_count = self._count_matched_query_tokens(
                query,
                [row.get("name") or row.get("id") for row in matched],
            )
            grounding_q = self._grounding_quality(query, matched_query_token_count)

            if not matched:
                logging.info("Entity-first: no entities found in question (grounding=0)")
                return None

            # Disambiguation confidence gate: if grounding is weak AND no entity
            # has a symbolic signal (exact substring or ≥2 token overlaps), the
            # anchor is embedding-only on a noisy graph — skip traversal and fall
            # back to vector search rather than confidently traversing from the
            # wrong node.
            _min_grounding = self._ENTITY_MATCH_MIN_GROUNDING_BY_KG.get(
                (kg_name or "").lower(),
                self._ENTITY_MATCH_MIN_GROUNDING,
            )
            if _min_grounding > 0.0:
                _has_symbolic = any(
                    int(r.get("exact_in_query", 0))
                    or int(r.get("query_in_entity", 0))
                    or int(r.get("token_overlap", 0)) >= 2
                    for r in matched
                )
                if grounding_q < _min_grounding and not _has_symbolic:
                    logging.info(
                        "Entity-first: low-confidence anchor "
                        "(grounding=%.2f < %.2f, no symbolic match) — "
                        "skipping graph traversal, falling back to vector",
                        grounding_q, _min_grounding,
                    )
                    return None

            logging.info(
                "Entity-first: matched %d entities covering %d query tokens (grounding=%.2f): %s",
                len(matched), matched_query_token_count, grounding_q, [r["name"] for r in matched]
            )

            # Step 2: build seed entity dict (hop 0)
            entities: Dict[str, Any] = {}
            for row in matched:
                eid = row["id"]
                if not eid:
                    continue
                entities[eid] = {
                    "id": eid,
                    "element_id": row["element_id"],
                    "type": row["type"],
                    "description": row["name"],
                    "mentioned_in_chunks": [],
                    "source": "entity_lookup",
                    "min_hops": 0,
                }

            if not entities:
                return None

            # Step 3: walk the graph from seed entities to discover neighbors.
            # RoG-style: plan which relation types are query-relevant before
            # expanding, so traversal is constrained to semantically useful edges.
            seed_ids = list(entities.keys())
            allowed_rel_types = None
            if self.use_rog_path_planning:
                allowed_rel_types = self._plan_relation_paths(
                    graph,
                    seed_ids,
                    query,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
            expansion = self._expand_entities_via_graph(
                graph,
                seed_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
                allowed_rel_types=allowed_rel_types,
            )
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in entities:
                    entities[nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Step 4: retrieve chunks, tracking which entity (and therefore which hop
            # depth) each chunk was reached through so we can assign hop-weighted scores.
            all_entity_ids = list(entities.keys())
            candidate_chunk_limit = min(max_chunks * 10, 200)
            chunk_params: Dict[str, Any] = {
                "entity_ids": all_entity_ids,
                "candidate_chunk_limit": candidate_chunk_limit,
            }
            kg_chunk_filter = ""
            if kg_name:
                kg_chunk_filter = "AND d.kgName = $kg_name"
                chunk_params["kg_name"] = kg_name
            if document_names:
                kg_chunk_filter += " AND d.fileName IN $document_names"
                chunk_params["document_names"] = document_names
            # Scope to the current question's passages when question_id is provided.
            # This prevents cross-question entity contamination on datasets where each
            # question has its own closed passage set (source_document / retrieval_bundle).
            if question_id:
                kg_chunk_filter += " AND c.questionId = $question_id"
                chunk_params["question_id"] = question_id

            # Collect the minimum hop depth of any entity linked to each chunk so
            # we can assign the most favourable score when a chunk is reachable via
            # multiple entities at different depths.
            chunk_query = f"""
            MATCH (c:Chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WHERE e.id IN $entity_ids
            MATCH (c)-[:PART_OF]->(d:Document)
            WHERE true {kg_chunk_filter}
            WITH c, d,
                 collect(DISTINCT e.id) AS linked_entity_ids,
                 count(DISTINCT e) AS linked_entity_count
            RETURN DISTINCT
                c.text AS text,
                c.id AS chunk_id,
                elementId(c) AS chunk_element_id,
                d.fileName AS document,
                d.kgName AS kg_name,
                c.position AS position,
                c.source AS source,
                c.questionId AS question_id,
                c.passageIndex AS passage_index,
                c.chunkLocalIndex AS chunk_local_index,
                linked_entity_ids,
                linked_entity_count
            ORDER BY linked_entity_count DESC, c.position ASC
            LIMIT $candidate_chunk_limit
            """
            chunk_results = graph.query(chunk_query, chunk_params)

            # Sufficiency check: fewer than 2 chunks usually means entity matching
            # found only a stray surface-form hit and the context will be too thin to
            # be useful.  Return None so the caller falls back to vector search.
            if not chunk_results or len(chunk_results) < 2:
                logging.info(
                    "Entity-first: only %d chunk(s) — below sufficiency threshold, falling back",
                    len(chunk_results) if chunk_results else 0,
                )
                return None

            # PPR scoring: fetch edges in the entity subgraph and run power iteration.
            # This replaces the fixed hop-score table with a principled random-walk
            # score that accounts for the full graph topology — entities connected to
            # more seeds via shorter, denser paths receive higher scores.
            all_entity_ids_for_ppr = list(entities.keys())
            seed_ids_for_ppr = [eid for eid, info in entities.items() if info.get("min_hops", 99) == 0]
            ppr_scores: Dict[str, float] = {}
            subgraph_edges: List[Tuple[str, str]] = []
            if self.use_ppr_scoring and seed_ids_for_ppr and len(all_entity_ids_for_ppr) > 1:
                subgraph_edges = self._fetch_subgraph_edges(
                    graph,
                    all_entity_ids_for_ppr,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                if subgraph_edges:
                    ppr_scores = self._ppr_entity_scores(
                        seed_ids_for_ppr, all_entity_ids_for_ppr, subgraph_edges
                    )
                    logging.info(
                        "Entity-first PPR: %d entities, %d edges, top score=%.4f",
                        len(all_entity_ids_for_ppr), len(subgraph_edges),
                        max(ppr_scores.values()) if ppr_scores else 0.0,
                    )

            # Normalise PPR scores to [0, 1] so they're comparable to hop scores.
            max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0

            chunks = []
            documents: set = set()
            for row in chunk_results:
                linked = row.get("linked_entity_ids") or []
                if ppr_scores and max_ppr > 0:
                    # PPR path: score = avg normalised PPR of covering entities
                    # + coverage bonus (n'·P from HippoRAG).
                    entity_ppr = [
                        ppr_scores.get(eid, 0.0) / max_ppr
                        for eid in linked if eid in entities
                    ]
                    if entity_ppr:
                        base_score = sum(entity_ppr) / len(entity_ppr)
                        coverage_bonus = 0.05 * (len(entity_ppr) - 1)
                        hop_score = min(1.0, base_score + coverage_bonus)
                    else:
                        hop_score = self._HOP_SCORE_DEFAULT
                else:
                    # Fallback: fixed hop-score table (used when subgraph has no edges).
                    entity_hop_scores = [
                        self._HOP_SCORE.get(entities[eid]["min_hops"], self._HOP_SCORE_DEFAULT)
                        for eid in linked if eid in entities
                    ]
                    if entity_hop_scores:
                        base_score = sum(entity_hop_scores) / len(entity_hop_scores)
                        coverage_bonus = 0.05 * (len(entity_hop_scores) - 1)
                        hop_score = min(1.0, base_score + coverage_bonus)
                    else:
                        hop_score = self._HOP_SCORE_DEFAULT

                min_hop = min(
                    (entities[eid]["min_hops"] for eid in linked if eid in entities),
                    default=2,
                )
                chunks.append({
                    "text": row["text"],
                    "chunk_id": row["chunk_id"],
                    "chunk_element_id": row["chunk_element_id"],
                    "score": hop_score,
                    "min_hop": min_hop,
                    "document": row["document"],
                    "kg_name": row.get("kg_name"),
                    "position": row.get("position"),
                    "source": row.get("source"),
                    "question_id": row.get("question_id"),
                    "passage_index": row.get("passage_index"),
                    "chunk_local_index": row.get("chunk_local_index"),
                    "entities": [],
                    "linked_entity_count": int(row.get("linked_entity_count") or 0),
                    # Retained for post-truncation entity pruning in iterative retrieval
                    "linked_entity_ids": linked,
                })
                documents.add(row["document"])

            # --- Traversal relevance gate ---
            # Drop hop-1+ chunks whose text is not similar enough to the query.
            # This prevents entity neighbours from flooding the context with
            # passages that share an entity name but are topically off-target
            # (e.g. a historical "Vicente García González" pulled in because the
            # graph connected him to "Vicente García" the musician by name).
            # Seed chunks (min_hop == 0) are always kept.
            # Embeddings are fetched in a single batched query to avoid N+1 overhead.
            _min_sim = self._TRAVERSAL_CHUNK_MIN_SIM
            if _min_sim > 0.0 and query_embedding:
                import numpy as np
                qvec = np.array(query_embedding, dtype=float)
                qnorm = np.linalg.norm(qvec)
                # Collect element IDs of hop-1+ chunks that need similarity checks
                traversal_eids = [
                    c["chunk_element_id"]
                    for c in chunks
                    if c.get("min_hop", 1) > 0 and c.get("chunk_element_id")
                ]
                eid_to_sim: Dict[str, float] = {}
                if traversal_eids and qnorm > 0:
                    try:
                        _emb_rows = graph.query(
                            "MATCH (c:Chunk) WHERE elementId(c) IN $eids "
                            "RETURN elementId(c) AS eid, c.embedding AS emb",
                            {"eids": traversal_eids},
                        )
                        for _row in (_emb_rows or []):
                            _emb = _row.get("emb")
                            if not _emb:
                                continue
                            cvec = np.array(_emb, dtype=float)
                            cnorm = np.linalg.norm(cvec)
                            sim = float(np.dot(qvec, cvec) / (qnorm * cnorm)) if cnorm > 0 else 0.0
                            eid_to_sim[_row["eid"]] = sim
                    except Exception as _ge:
                        logging.debug("Traversal relevance gate: batch embedding fetch failed: %s", _ge)
                filtered_chunks = []
                for c in chunks:
                    if c.get("min_hop", 1) == 0:
                        filtered_chunks.append(c)
                        continue
                    eid = c.get("chunk_element_id")
                    sim = eid_to_sim.get(eid)
                    if sim is None or sim >= _min_sim:
                        # Keep if similarity unknown (no embedding stored) or above threshold
                        filtered_chunks.append(c)
                    else:
                        logging.debug(
                            "Entity-first: dropped traversal chunk (passage_index=%s, sim=%.3f < %.3f)",
                            c.get("passage_index"), sim, _min_sim,
                        )
                n_before = len(chunks)
                if filtered_chunks:
                    chunks = filtered_chunks
                logging.info(
                    "Traversal relevance gate: kept %d/%d chunks (min_sim=%.2f)",
                    len(chunks), n_before, _min_sim,
                )

            # --- Per-passage chunk cap ---
            # Limit how many chunks from any single source passage (same
            # questionId + passageIndex) enter the context.  A single passage
            # may produce many entity-linked chunks; taking them all starves
            # other passages that hold complementary evidence.
            if self._MAX_CHUNKS_PER_PASSAGE > 0:
                chunks = self._apply_per_passage_chunk_cap(chunks)

            # Sort by hop score descending so the LLM sees the most directly supported
            # chunks first (mirrors C2RAG's constraint-aligned evidence ordering).
            chunks.sort(key=self._chunk_rank_key)
            chunks = chunks[:max_chunks]

            # Step 5: relationships between all entities
            relationships = []
            if all_entity_ids:
                rel_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_entity_ids,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel in rel_results:
                    rel_key = (
                        rel.get("relationship_element_id")
                        or f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                        f"-negated={bool(rel.get('negated', False))}"
                    )
                    relationships.append({
                        "key": rel_key,
                        "source": rel["source"],
                        "source_element_id": rel["source_element_id"],
                        "target": rel["target"],
                        "target_element_id": rel["target_element_id"],
                        "type": rel["relationship_type"],
                        "element_id": rel["relationship_element_id"],
                        "negated": bool(rel.get("negated", False)),
                        "condition": rel.get("condition"),
                        "quantitative": rel.get("quantitative"),
                        "confidence": rel.get("confidence"),
                        "question_ids": rel.get("question_ids") or [],
                        "passage_keys": rel.get("passage_keys") or [],
                        "provenance_positions": rel.get("provenance_positions") or [],
                    })

            return {
                "query": query,
                "chunks": chunks,
                "entities": entities,
                "relationships": relationships,
                "graph_neighbors": expansion["neighbors"],
                "traversal_paths": expansion["paths"],
                "documents": list(documents),
                "total_score": float(sum(c["score"] for c in chunks)),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "search_method": "entity_first",
                "kg_name": kg_name,
                "seed_entity_count": len(seed_ids),
                "grounding_quality": grounding_q,
                "retrieval_route": "entity_first",
                "route_reason": "success",
                "diagnostics": {
                    "seed_entities": [entities[eid].get("name", eid) for eid in seed_ids[:10]],
                    "ann_match_count": len([e for e in matched if e.get("_source_kind") == "embedding"]),
                    "symbolic_match_count": len([e for e in matched if e.get("_source_kind") == "symbolic"]),
                    "rfge_fired": False,
                    "subgraph_edge_count": len(subgraph_edges),
                    "top_ppr_entities": sorted(
                        ((eid, score) for eid, score in ppr_scores.items()),
                        key=lambda x: -x[1],
                    )[:5] if ppr_scores else [],
                    "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto"),
                },
            }

        except Exception as e:
            logging.warning("Entity-first search failed: %s", e)
            return None

    def _retriever_first_graph_expansion(
        self,
        graph,
        query: str,
        max_chunks: int = 20,
        kg_name: str = None,
        max_hops: int = 2,
        question_id: str = None,
        document_names: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retriever-first graph expansion (RFGE).

        Complements entity-first by starting from dense vector retrieval rather
        than from question entities.  Entity-first excels when the question contains
        explicit named entities that exist in the KG; RFGE excels when the question
        is abstract or paraphrased but semantically-close passages are retrievable.

        Flow:
          1. Vector search → top-k chunks (with their linked entities).
          2. Collect entity IDs from those chunks as graph seeds.
          3. Expand the graph from those seeds (_expand_entities_via_graph).
          4. Fetch subgraph edges → run PPR to score all entities.
          5. Re-score chunks: combined = 0.5 * cosine + 0.5 * ppr_chunk_score.
          6. Fetch additional chunks for graph-neighbor entities not already retrieved.
          7. Return merged context with grounding_quality derived from PPR coverage.
        """
        try:
            # Step 1: vector retrieval — overfetch so we have enough seeds.
            vec_ctx = self._semantic_similarity_search(
                graph,
                query,
                document_names=document_names,
                similarity_threshold=self._RFGE_VECTOR_THRESHOLD,
                max_chunks=min(max_chunks * 3, 60),
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
            )
            if not vec_ctx or not vec_ctx.get("chunks"):
                return None

            # Step 2: collect entity seeds from returned chunks.
            vec_entity_ids: List[str] = []
            cosine_by_chunk: Dict[str, float] = {}
            _MAX_ENTITIES_PER_CHUNK = 4  # cap seeds per chunk to avoid generic-entity flooding
            for chunk in vec_ctx["chunks"]:
                cosine_by_chunk[chunk["chunk_id"]] = float(chunk.get("score", 0.0))
                chunk_ents = chunk.get("entities", [])
                added = 0
                for ent in chunk_ents:
                    if added >= _MAX_ENTITIES_PER_CHUNK:
                        break
                    eid = ent.get("id")
                    if not eid:
                        continue
                    eid_lower = eid.lower().strip()
                    if len(eid_lower) <= 2 or eid_lower in self._ENTITY_MATCH_STOPWORDS:
                        continue
                    if eid not in vec_entity_ids:
                        vec_entity_ids.append(eid)
                        added += 1

            if not vec_entity_ids:
                # No entities tagged on any retrieved chunk — RFGE can't expand.
                return None

            # Step 3: graph expansion from passage-derived seeds.
            expansion = self._expand_entities_via_graph(
                graph,
                vec_entity_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
            )

            # Build the combined entity pool: passage seeds + graph neighbors.
            all_entities: Dict[str, Any] = {}
            for eid in vec_entity_ids:
                all_entities[eid] = {"id": eid, "min_hops": 0, "source": "retriever_seed"}
            for nid, ninfo in expansion["neighbors"].items():
                all_entities[nid] = {**ninfo, "min_hops": ninfo.get("min_hops", 1)}

            # Step 4: PPR scoring over the full entity pool.
            all_eid_list = list(all_entities.keys())
            subgraph_edges = self._fetch_subgraph_edges(
                graph,
                all_eid_list,
                kg_name=kg_name,
                question_id=question_id,
                document_names=document_names,
            )
            ppr_scores: Dict[str, float] = {}
            max_ppr = 0.0
            if subgraph_edges:
                ppr_scores = self._ppr_entity_scores(vec_entity_ids, all_eid_list, subgraph_edges)
                max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0

            # Step 5: re-score existing chunks with combined cosine + PPR signal.
            chunks = []
            for chunk in vec_ctx["chunks"]:
                cid = chunk["chunk_id"]
                cosine = cosine_by_chunk.get(cid, float(chunk.get("score", 0.0)))
                linked = [e.get("id") for e in chunk.get("entities", []) if e.get("id")]
                if ppr_scores and max_ppr > 0:
                    ppr_vals = [ppr_scores.get(eid, 0.0) / max_ppr for eid in linked if eid in ppr_scores]
                    if ppr_vals:
                        ppr_chunk = sum(ppr_vals) / len(ppr_vals) + 0.05 * (len(ppr_vals) - 1)
                        combined = 0.5 * cosine + 0.5 * min(1.0, ppr_chunk)
                    else:
                        combined = 0.5 * cosine  # no PPR signal for this chunk
                else:
                    combined = cosine
                chunks.append({**chunk, "score": combined, "linked_entity_ids": linked})

            # Step 6: fetch chunks for graph-neighbor entities not yet retrieved.
            neighbor_eids = [
                eid for eid in expansion["neighbors"]
                if ppr_scores.get(eid, 0.0) >= self._RFGE_MIN_PPR_SCORE
            ]
            if neighbor_eids:
                existing_cids = {c["chunk_id"] for c in chunks}
                extra_params: Dict[str, Any] = {
                    "entity_ids": neighbor_eids[:20],
                    "limit": max(4, max_chunks // 3),
                }
                scope = ""
                if kg_name:
                    scope += " AND d.kgName = $kg_name"
                    extra_params["kg_name"] = kg_name
                if document_names:
                    scope += " AND d.fileName IN $document_names"
                    extra_params["document_names"] = document_names
                if question_id:
                    scope += " AND c.questionId = $question_id"
                    extra_params["question_id"] = question_id
                extra_rows = graph.query(
                    f"""
                    MATCH (e:__Entity__)<-[:HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
                    WHERE e.id IN $entity_ids {scope}
                    RETURN DISTINCT c.text AS text, c.id AS chunk_id,
                           elementId(c) AS chunk_element_id,
                           d.fileName AS document, d.kgName AS kg_name,
                           c.questionId AS question_id,
                           c.position AS position,
                           c.passageIndex AS passage_index,
                           c.chunkLocalIndex AS chunk_local_index,
                           e.id AS seed_entity_id
                    LIMIT $limit
                    """,
                    extra_params,
                ) or []
                for row in extra_rows:
                    cid = row["chunk_id"]
                    if not cid or cid in existing_cids:
                        continue
                    eid = row.get("seed_entity_id", "")
                    ppr_val = ppr_scores.get(eid, 0.0) / max_ppr if max_ppr > 0 else 0.0
                    chunks.append({
                        "text": row["text"],
                        "chunk_id": cid,
                        "chunk_element_id": row["chunk_element_id"],
                        "score": 0.3 * ppr_val,  # graph-only: lower base score
                        "min_hop": 1,
                        "document": row["document"],
                        "kg_name": row.get("kg_name"),
                        "question_id": row.get("question_id"),
                        "position": row.get("position"),
                        "passage_index": row.get("passage_index"),
                        "chunk_local_index": row.get("chunk_local_index"),
                        "entities": [],
                        "linked_entity_ids": [eid] if eid else [],
                        "linked_entity_count": 1 if eid else 0,
                    })
                    existing_cids.add(cid)

            chunks.sort(key=lambda c: -float(c.get("score", 0.0)))
            chunks = chunks[:max_chunks]

            # Grounding quality: fraction of retrieved entities with PPR > threshold.
            supported = sum(1 for eid in vec_entity_ids if ppr_scores.get(eid, 0.0) >= self._RFGE_MIN_PPR_SCORE)
            grounding_q = supported / len(vec_entity_ids) if vec_entity_ids else 0.0

            relationships = []
            if all_eid_list:
                rel_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_eid_list[:40],
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel in rel_results:
                    relationships.append({
                        "key": (
                            rel.get("relationship_element_id")
                            or f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                            f"-negated={bool(rel.get('negated', False))}"
                        ),
                        "source": rel["source"],
                        "source_element_id": rel["source_element_id"],
                        "target": rel["target"],
                        "target_element_id": rel["target_element_id"],
                        "type": rel["relationship_type"],
                        "element_id": rel["relationship_element_id"],
                        "negated": bool(rel.get("negated", False)),
                        "condition": rel.get("condition"),
                        "quantitative": rel.get("quantitative"),
                        "confidence": rel.get("confidence"),
                        "question_ids": rel.get("question_ids") or [],
                        "passage_keys": rel.get("passage_keys") or [],
                        "provenance_positions": rel.get("provenance_positions") or [],
                    })

            logging.info(
                "RFGE: %d seeds, %d total entities, %d edges, %d chunks, grounding=%.2f",
                len(vec_entity_ids), len(all_eid_list), len(subgraph_edges),
                len(chunks), grounding_q,
            )
            return {
                "query": query,
                "chunks": chunks,
                "entities": all_entities,
                "relationships": relationships,
                "graph_neighbors": expansion["neighbors"],
                "traversal_paths": expansion["paths"],
                "documents": list({c["document"] for c in chunks if c.get("document")}),
                "total_score": sum(c["score"] for c in chunks),
                "entity_count": len(all_entities),
                "relationship_count": len(relationships),
                "search_method": "retriever_first_graph_expansion",
                "kg_name": kg_name,
                "seed_entity_count": len(vec_entity_ids),
                "grounding_quality": grounding_q,
                "retrieval_route": "rfge",
                "route_reason": "entity_first_failed",
                "diagnostics": {
                    "seed_entities": vec_entity_ids[:10],
                    "ann_match_count": 0,
                    "symbolic_match_count": 0,
                    "rfge_fired": True,
                    "subgraph_edge_count": len(subgraph_edges),
                    "top_ppr_entities": sorted(
                        ((eid, score) for eid, score in ppr_scores.items()),
                        key=lambda x: -x[1],
                    )[:5] if ppr_scores else [],
                    "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto"),
                },
            }

        except Exception as exc:
            logging.warning("Retriever-first graph expansion failed: %s", exc)
            return None
