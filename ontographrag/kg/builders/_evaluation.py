import re
from typing import Dict, Any, List, Optional
from collections import Counter

class EvaluationMixin:
    """Knowledge-graph quality evaluation and reporting.

    Mixin for :class:`OntologyGuidedKGCreator`; method bodies are
    unchanged from the original monolithic implementation.
    """

    @staticmethod
    def _kg_eval_normalize_text(value: Any) -> str:
        """Normalize strings for lightweight KG evaluation comparisons."""
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip()).lower()

    @classmethod
    def _kg_eval_entity_lookup(
        cls,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Return id -> canonical entity metadata for evaluation helpers."""
        lookup: Dict[str, Dict[str, str]] = {}
        for node in nodes or []:
            props = node.get("properties") or {}
            entity_id = str(node.get("id") or props.get("id") or "").strip()
            if not entity_id:
                continue
            name = str(
                props.get("name")
                or node.get("name")
                or entity_id
            ).strip()
            entity_type = str(
                node.get("label")
                or node.get("type")
                or props.get("type")
                or props.get("ontology_class")
                or "Unknown"
            ).strip()
            lookup[entity_id] = {
                "id": entity_id,
                "name": name,
                "type": entity_type or "Unknown",
                "name_key": cls._kg_eval_normalize_text(name or entity_id),
                "type_key": cls._kg_eval_normalize_text(entity_type or "Unknown"),
            }
        return lookup

    @classmethod
    def _kg_eval_entity_keys(
        cls,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Return canonical entity key -> display metadata."""
        records: Dict[str, Dict[str, str]] = {}
        for info in cls._kg_eval_entity_lookup(nodes).values():
            key = f"{info['name_key']}||{info['type_key']}"
            records[key] = {
                "id": info["id"],
                "name": info["name"],
                "type": info["type"],
            }
        return records

    @classmethod
    def _kg_eval_relationship_records(
        cls,
        relationships: List[Dict[str, Any]],
        entity_lookup: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Return canonical relationship records for evaluation comparisons."""
        records: List[Dict[str, Any]] = []
        for rel in relationships or []:
            props = rel.get("properties") or {}
            rel_type = str(rel.get("type") or props.get("type") or "").strip()
            if not rel_type:
                continue

            source_ref = rel.get("source") or rel.get("from") or props.get("source")
            target_ref = rel.get("target") or rel.get("to") or props.get("target")
            source_ref = str(source_ref or "").strip()
            target_ref = str(target_ref or "").strip()

            source_name = str(
                props.get("source_name")
                or entity_lookup.get(source_ref, {}).get("name")
                or source_ref
            ).strip()
            target_name = str(
                props.get("target_name")
                or entity_lookup.get(target_ref, {}).get("name")
                or target_ref
            ).strip()

            negated = bool(
                rel.get("negated", props.get("negated", False))
            )
            condition = str(
                rel.get("condition")
                or props.get("condition")
                or ""
            ).strip()
            quantitative = str(
                rel.get("quantitative")
                or props.get("quantitative")
                or ""
            ).strip()

            base_key = "||".join(
                (
                    cls._kg_eval_normalize_text(source_name),
                    rel_type.strip().upper(),
                    cls._kg_eval_normalize_text(target_name),
                )
            )
            qualified_key = "||".join(
                (
                    base_key,
                    "1" if negated else "0",
                    cls._kg_eval_normalize_text(condition),
                    cls._kg_eval_normalize_text(quantitative),
                )
            )
            records.append(
                {
                    "base_key": base_key,
                    "qualified_key": qualified_key,
                    "source_name": source_name,
                    "target_name": target_name,
                    "type": rel_type,
                    "negated": negated,
                    "condition": condition,
                    "quantitative": quantitative,
                }
            )
        return records

    @staticmethod
    def _kg_eval_pr_report(
        predicted_keys: set,
        gold_keys: set,
    ) -> Dict[str, Any]:
        """Compute precision / recall / F1 for canonical key sets."""
        tp = len(predicted_keys & gold_keys)
        fp = len(predicted_keys - gold_keys)
        fn = len(gold_keys - predicted_keys)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall / max(1e-9, precision + recall))
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "gold_count": len(gold_keys),
            "predicted_count": len(predicted_keys),
        }

    def evaluate_knowledge_graph(
        self,
        kg: Dict[str, Any],
        *,
        reference: Optional[Any] = None,
        reference_triples: Optional[List[Dict[str, Any]]] = None,
        print_report: bool = True,
    ) -> Dict[str, Any]:
        """Compute quality metrics for a built KG and optionally compare to a gold set.

        Operates entirely on the in-memory KG dict returned by
        ``generate_knowledge_graph`` / ``generate_knowledge_graph_from_passages``.
        No Neo4j connection required.

        Parameters
        ----------
        kg:
            KG dict as returned by the generate methods.
        reference:
            Optional reference KG dict or list of gold triples. When a KG dict is
            provided, entity precision/recall is computed from its nodes and
            relationship precision/recall is computed from its relationships.
        reference_triples:
            Backward-compatible alias for a list of gold triples. Each entry must
            provide ``source``, ``type``, and ``target`` keys. Relationship
            comparison resolves entity ids back to names when node metadata exists,
            so reference triples can use either names or canonical ids.
        print_report:
            When True, prints a human-readable summary to stdout.

        Returns
        -------
        Dict with sections: ``summary``, ``entity_metrics``, ``relationship_metrics``,
        ``grounding_metrics``, ``pipeline_metrics``, and optionally
        ``entity_precision_recall`` / ``relationship_precision_recall``.
        """
        nodes = kg.get("nodes") or []
        relationships = kg.get("relationships") or []
        chunks = kg.get("chunks") or []
        meta = kg.get("metadata") or {}
        entity_lookup = self._kg_eval_entity_lookup(nodes)

        # ── Entity metrics ─────────────────────────────────────────────────
        type_counts: Dict[str, int] = Counter(
            str(n.get("label") or n.get("type") or "Unknown") for n in nodes
        )
        anchor_grounded = sum(
            1 for n in nodes if (n.get("properties") or {}).get("anchor_spans")
        )
        umls_linked = sum(
            1 for n in nodes if (n.get("properties") or {}).get("umls_cui")
        )
        has_synonyms = sum(
            1 for n in nodes
            if len((n.get("properties") or {}).get("all_names") or []) > 1
        )
        n_entities = len(nodes)

        # degree distribution — entities appear as source/target in relationships
        degree: Dict[str, int] = Counter()
        for rel in relationships:
            src = rel.get("source") or rel.get("from")
            tgt = rel.get("target") or rel.get("to")
            if src:
                degree[str(src)] += 1
            if tgt:
                degree[str(tgt)] += 1
        degrees = list(degree.values()) if degree else [0]
        sorted_degrees = sorted(degrees, reverse=True)
        top10_degree_share = (
            sum(sorted_degrees[:10]) / max(1, sum(sorted_degrees))
        ) if sorted_degrees else 0.0
        isolated_entities = n_entities - len(degree)
        degree_buckets = {"0": 0, "1": 0, "2-4": 0, "5-9": 0, "10+": 0}
        for node in nodes:
            entity_id = str(node.get("id") or "").strip()
            node_degree = int(degree.get(entity_id, 0))
            if node_degree <= 0:
                degree_buckets["0"] += 1
            elif node_degree == 1:
                degree_buckets["1"] += 1
            elif node_degree <= 4:
                degree_buckets["2-4"] += 1
            elif node_degree <= 9:
                degree_buckets["5-9"] += 1
            else:
                degree_buckets["10+"] += 1
        top_hub_entities = []
        for entity_id, node_degree in degree.most_common(10):
            info = entity_lookup.get(entity_id, {})
            top_hub_entities.append(
                {
                    "id": entity_id,
                    "name": info.get("name") or entity_id,
                    "type": info.get("type") or "Unknown",
                    "degree": int(node_degree),
                }
            )

        # ── Relationship metrics ────────────────────────────────────────────
        n_rels = len(relationships)
        rel_type_counts: Dict[str, int] = Counter(
            str(r.get("type") or "UNKNOWN") for r in relationships
        )
        negated_count = sum(1 for r in relationships if r.get("negated"))
        contradiction_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("contradiction_detected")
               or r.get("contradiction_detected")
        )
        conditioned_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("condition")
        )
        quantified_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("quantitative")
        )

        # evidence scope distribution
        scope_counts: Dict[str, int] = Counter()
        confidence_values: List[float] = []
        for r in relationships:
            props = r.get("properties") or {}
            scope = props.get("evidence_scope") or "unknown"
            scope_counts[scope] += 1
            raw_conf = props.get("confidence")
            if raw_conf not in (None, "", "null", "NULL"):
                confidence_values.append(self._confidence_score(raw_conf))

        mean_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else None
        )
        high_conf_rels = sum(1 for c in confidence_values if c >= 0.7)
        low_conf_rels = sum(1 for c in confidence_values if c < 0.4)

        restoration_full = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("restoration_status") == "full"
        )
        restoration_partial = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("restoration_status") == "partial"
        )
        restoration_failed = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("restoration_status") == "failed"
        )

        # ── Grounding metrics ───────────────────────────────────────────────
        n_chunks = len(chunks)
        entities_per_chunk = n_entities / max(1, n_chunks)
        rels_per_chunk = n_rels / max(1, n_chunks)
        anchor_rel_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("anchor_text")
        )
        anchor_grounding_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("anchor_grounding")
        )
        section_names: set = {
            str(c.get("section_name") or "")
            for c in chunks
            if c.get("section_name")
        }

        # ── Pipeline health metrics ─────────────────────────────────────────
        dropped_entities = meta.get("schema_enforcement_dropped_entities", 0)
        dropped_rels_schema = meta.get("schema_enforcement_dropped_relationships", 0)
        dropped_rels_unmapped = meta.get("harmonization_relationships_dropped_unmapped", 0)
        deduped_rels = meta.get("harmonization_relationships_deduped", 0)
        contradiction_groups = meta.get("harmonization_relationship_contradiction_groups", 0)
        stored_relationships = meta.get("stored_relationships")
        relationship_store_failures = meta.get("relationship_store_failures", 0)
        relationships_skipped_low_confidence = meta.get("relationships_skipped_low_confidence", 0)
        relationships_skipped_schema_mismatch = meta.get("relationships_skipped_schema_mismatch", 0)
        relationships_reverified_kept = meta.get("relationships_reverified_kept", 0)
        relationships_reverified_rejected = meta.get("relationships_reverified_rejected", 0)
        relationship_store_ratio = meta.get("relationship_store_ratio")

        raw_entities_approx = n_entities + dropped_entities
        entity_pass_rate = n_entities / max(1, raw_entities_approx)
        raw_rels_approx = n_rels + dropped_rels_schema + dropped_rels_unmapped + deduped_rels
        rel_pass_rate = n_rels / max(1, raw_rels_approx)

        # ── Precision / recall vs. gold ─────────────────────────────────────
        entity_precision_recall: Optional[Dict[str, Any]] = None
        relationship_precision_recall: Optional[Dict[str, Any]] = None
        qualified_relationship_precision_recall: Optional[Dict[str, Any]] = None
        reference_nodes: List[Dict[str, Any]] = []
        reference_relationships: List[Dict[str, Any]] = []
        if isinstance(reference, dict):
            reference_nodes = list(reference.get("nodes") or reference.get("entities") or [])
            reference_relationships = list(reference.get("relationships") or reference.get("triples") or [])
        elif isinstance(reference, list):
            reference_relationships = list(reference)
        if reference_triples:
            reference_relationships = list(reference_relationships) + list(reference_triples)

        if reference_nodes:
            gold_entity_keys = set(self._kg_eval_entity_keys(reference_nodes))
            pred_entity_keys = set(self._kg_eval_entity_keys(nodes))
            entity_precision_recall = self._kg_eval_pr_report(
                pred_entity_keys,
                gold_entity_keys,
            )
            entity_precision_recall["gold_entities"] = entity_precision_recall.pop("gold_count")
            entity_precision_recall["predicted_entities"] = entity_precision_recall.pop("predicted_count")

        if reference_relationships:
            reference_lookup = dict(entity_lookup)
            reference_lookup.update(self._kg_eval_entity_lookup(reference_nodes))
            pred_relationship_records = self._kg_eval_relationship_records(
                relationships,
                entity_lookup,
            )
            gold_relationship_records = self._kg_eval_relationship_records(
                reference_relationships,
                reference_lookup,
            )
            pred_keys = {record["base_key"] for record in pred_relationship_records}
            gold_keys = {record["base_key"] for record in gold_relationship_records}
            relationship_precision_recall = self._kg_eval_pr_report(
                pred_keys,
                gold_keys,
            )
            relationship_precision_recall["gold_triples"] = relationship_precision_recall.pop("gold_count")
            relationship_precision_recall["predicted_triples"] = relationship_precision_recall.pop("predicted_count")

            has_qualified_reference = any(
                record["negated"] or record["condition"] or record["quantitative"]
                for record in (pred_relationship_records + gold_relationship_records)
            )
            if has_qualified_reference:
                pred_qualified_keys = {
                    record["qualified_key"] for record in pred_relationship_records
                }
                gold_qualified_keys = {
                    record["qualified_key"] for record in gold_relationship_records
                }
                qualified_relationship_precision_recall = self._kg_eval_pr_report(
                    pred_qualified_keys,
                    gold_qualified_keys,
                )
                qualified_relationship_precision_recall["gold_triples"] = qualified_relationship_precision_recall.pop("gold_count")
                qualified_relationship_precision_recall["predicted_triples"] = qualified_relationship_precision_recall.pop("predicted_count")

        # ── Assemble report ─────────────────────────────────────────────────
        report: Dict[str, Any] = {
            "summary": {
                "entities": n_entities,
                "relationships": n_rels,
                "chunks": n_chunks,
                "entity_types": len(type_counts),
                "relationship_types": len(rel_type_counts),
                "sections_detected": sorted(section_names),
                "extraction_method": meta.get("extraction_method", "unknown"),
                "kg_name": meta.get("kg_name"),
                "created_at": meta.get("created_at"),
            },
            "entity_metrics": {
                "type_distribution": dict(type_counts.most_common()),
                "anchor_grounded": anchor_grounded,
                "anchor_grounded_pct": round(100 * anchor_grounded / max(1, n_entities), 1),
                "umls_linked": umls_linked,
                "umls_linked_pct": round(100 * umls_linked / max(1, n_entities), 1),
                "with_synonyms": has_synonyms,
                "isolated_no_edges": isolated_entities,
                "mean_degree": round(sum(degrees) / max(1, len(degrees)), 2),
                "max_degree": max(degrees),
                "top10_entity_degree_share_pct": round(100 * top10_degree_share, 1),
                "degree_distribution": degree_buckets,
                "hub_entities_topk": top_hub_entities,
            },
            "relationship_metrics": {
                "type_distribution": dict(rel_type_counts.most_common(20)),
                "negated": negated_count,
                "negated_pct": round(100 * negated_count / max(1, n_rels), 1),
                "contradiction_flagged": contradiction_count,
                "contradiction_groups": contradiction_groups,
                "contradiction_rate_pct": round(100 * contradiction_count / max(1, n_rels), 1),
                "conditioned": conditioned_count,
                "conditioned_pct": round(100 * conditioned_count / max(1, n_rels), 1),
                "quantified": quantified_count,
                "quantified_pct": round(100 * quantified_count / max(1, n_rels), 1),
                "evidence_scope_distribution": dict(scope_counts),
                "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
                "high_confidence_pct": round(100 * high_conf_rels / max(1, len(confidence_values)), 1),
                "low_confidence_pct": round(100 * low_conf_rels / max(1, len(confidence_values)), 1),
                "restoration_full": restoration_full,
                "restoration_partial": restoration_partial,
                "restoration_failed": restoration_failed,
                "anchor_relation_phrase_pct": round(100 * anchor_rel_count / max(1, n_rels), 1),
                "anchor_grounding_pct": round(100 * anchor_grounding_count / max(1, n_rels), 1),
            },
            "grounding_metrics": {
                "entities_per_chunk": round(entities_per_chunk, 2),
                "relationships_per_chunk": round(rels_per_chunk, 2),
                "chunks_with_section_tags": sum(1 for c in chunks if c.get("section_name")),
            },
            "pipeline_metrics": {
                "schema_enforcement_dropped_entities": dropped_entities,
                "schema_enforcement_dropped_relationships": dropped_rels_schema,
                "harmonization_dropped_unmapped": dropped_rels_unmapped,
                "harmonization_deduped_relationships": deduped_rels,
                "entity_schema_pass_rate_pct": round(100 * entity_pass_rate, 1),
                "relationship_pass_rate_pct": round(100 * rel_pass_rate, 1),
                "stored_relationships": stored_relationships,
                "relationship_store_failures": relationship_store_failures,
                "relationships_skipped_low_confidence": relationships_skipped_low_confidence,
                "relationships_skipped_schema_mismatch": relationships_skipped_schema_mismatch,
                "relationships_reverified_kept": relationships_reverified_kept,
                "relationships_reverified_rejected": relationships_reverified_rejected,
                "relationship_store_ratio": round(float(relationship_store_ratio), 4)
                if isinstance(relationship_store_ratio, (int, float))
                else relationship_store_ratio,
            },
        }
        if entity_precision_recall is not None:
            report["entity_precision_recall"] = entity_precision_recall
        if relationship_precision_recall is not None:
            report["relationship_precision_recall"] = relationship_precision_recall
            # Backward-compatible alias for earlier callers that only expected
            # relationship-level precision / recall.
            report["precision_recall"] = relationship_precision_recall
        if qualified_relationship_precision_recall is not None:
            report["qualified_relationship_precision_recall"] = qualified_relationship_precision_recall

        if print_report:
            self._print_kg_evaluation_report(report)

        return report

    @staticmethod
    def _print_kg_evaluation_report(report: Dict[str, Any]) -> None:
        """Print a human-readable KG evaluation report to stdout."""
        sep = "─" * 60
        s = report["summary"]
        em = report["entity_metrics"]
        rm = report["relationship_metrics"]
        gm = report["grounding_metrics"]
        pm = report["pipeline_metrics"]

        lines = [
            sep,
            "KG EVALUATION REPORT",
            sep,
            f"KG name          : {s.get('kg_name') or '(unnamed)'}",
            f"Extraction method: {s['extraction_method']}",
            f"Created at       : {s.get('created_at') or 'N/A'}",
            "",
            "── SUMMARY ──────────────────────────────────────────────────",
            f"  Entities        : {s['entities']}  ({s['entity_types']} types)",
            f"  Relationships   : {s['relationships']}  ({s['relationship_types']} types)",
            f"  Chunks          : {s['chunks']}",
            f"  Sections        : {', '.join(s['sections_detected']) or 'none detected'}",
            "",
            "── ENTITY QUALITY ───────────────────────────────────────────",
            f"  Anchor-grounded : {em['anchor_grounded']}  ({em['anchor_grounded_pct']}%)",
            f"  UMLS-linked     : {em['umls_linked']}  ({em['umls_linked_pct']}%)",
            f"  With synonyms   : {em['with_synonyms']}",
            f"  Isolated (no edges): {em['isolated_no_edges']}",
            f"  Mean degree     : {em['mean_degree']}  |  Max: {em['max_degree']}",
            f"  Top-10 hub share: {em['top10_entity_degree_share_pct']}% of all edges",
            f"  Degree buckets  : {em['degree_distribution']}",
            f"  Type distribution (top 10):",
        ]
        for etype, cnt in list(em["type_distribution"].items())[:10]:
            lines.append(f"    {etype:<30} {cnt}")
        if em.get("hub_entities_topk"):
            lines.append("  Top hub entities:")
            for hub in em["hub_entities_topk"][:5]:
                lines.append(
                    f"    {hub['name']:<25} {hub['degree']}  ({hub['type']})"
                )

        lines += [
            "",
            "── RELATIONSHIP QUALITY ─────────────────────────────────────",
            f"  Negated         : {rm['negated']}  ({rm['negated_pct']}%)",
            f"  Contradictions  : {rm['contradiction_flagged']} edges in {rm['contradiction_groups']} groups ({rm['contradiction_rate_pct']}%)",
            f"  Conditioned     : {rm['conditioned']}  ({rm['conditioned_pct']}%)",
            f"  Quantified      : {rm['quantified']}  ({rm['quantified_pct']}%)",
            f"  Mean confidence : {rm['mean_confidence']}  |  High-conf: {rm['high_confidence_pct']}%  Low-conf: {rm['low_confidence_pct']}%",
            f"  Anchor relation phrase: {rm['anchor_relation_phrase_pct']}%",
            f"  Anchor grounding      : {rm['anchor_grounding_pct']}%",
            f"  Restoration (full/partial/failed): {rm['restoration_full']} / {rm['restoration_partial']} / {rm['restoration_failed']}",
            f"  Evidence scope:",
        ]
        for scope, cnt in sorted(rm["evidence_scope_distribution"].items(), key=lambda x: -x[1]):
            pct = round(100 * cnt / max(1, s["relationships"]), 1)
            lines.append(f"    {scope:<25} {cnt}  ({pct}%)")

        lines += [
            "",
            "── GROUNDING ────────────────────────────────────────────────",
            f"  Entities/chunk  : {gm['entities_per_chunk']}",
            f"  Relations/chunk : {gm['relationships_per_chunk']}",
            f"  Section-tagged chunks: {gm['chunks_with_section_tags']} / {s['chunks']}",
            "",
            "── PIPELINE HEALTH ──────────────────────────────────────────",
            f"  Entity schema pass rate : {pm['entity_schema_pass_rate_pct']}%",
            f"  Relation pass rate      : {pm['relationship_pass_rate_pct']}%",
            f"  Dropped entities (schema): {pm['schema_enforcement_dropped_entities']}",
            f"  Dropped rels (schema)    : {pm['schema_enforcement_dropped_relationships']}",
            f"  Dropped rels (unmapped)  : {pm['harmonization_dropped_unmapped']}",
            f"  Deduped relationships    : {pm['harmonization_deduped_relationships']}",
            f"  Stored relationships     : {pm['stored_relationships']}",
            f"  Store failures           : {pm['relationship_store_failures']}",
            f"  Skipped low-confidence   : {pm['relationships_skipped_low_confidence']}",
            f"  Skipped schema-mismatch  : {pm['relationships_skipped_schema_mismatch']}",
            f"  Reverified kept/rejected : {pm['relationships_reverified_kept']} / {pm['relationships_reverified_rejected']}",
            f"  Store ratio              : {pm['relationship_store_ratio']}",
        ]

        if "entity_precision_recall" in report:
            pr = report["entity_precision_recall"]
            lines += [
                "",
                "── ENTITY PRECISION / RECALL vs. GOLD ──────────────────────",
                f"  Gold entities   : {pr['gold_entities']}",
                f"  Predicted       : {pr['predicted_entities']}",
                f"  True positives  : {pr['true_positives']}",
                f"  False positives : {pr['false_positives']}",
                f"  False negatives : {pr['false_negatives']}",
                f"  Precision       : {pr['precision']}",
                f"  Recall          : {pr['recall']}",
                f"  F1              : {pr['f1']}",
            ]

        if "precision_recall" in report:
            pr = report["precision_recall"]
            lines += [
                "",
                "── PRECISION / RECALL vs. GOLD ──────────────────────────────",
                f"  Gold triples    : {pr['gold_triples']}",
                f"  Predicted       : {pr['predicted_triples']}",
                f"  True positives  : {pr['true_positives']}",
                f"  False positives : {pr['false_positives']}",
                f"  False negatives : {pr['false_negatives']}",
                f"  Precision       : {pr['precision']}",
                f"  Recall          : {pr['recall']}",
                f"  F1              : {pr['f1']}",
            ]
        if "qualified_relationship_precision_recall" in report:
            pr = report["qualified_relationship_precision_recall"]
            lines += [
                "",
                "── QUALIFIED RELATIONSHIP P/R vs. GOLD ─────────────────────",
                f"  Gold triples    : {pr['gold_triples']}",
                f"  Predicted       : {pr['predicted_triples']}",
                f"  True positives  : {pr['true_positives']}",
                f"  False positives : {pr['false_positives']}",
                f"  False negatives : {pr['false_negatives']}",
                f"  Precision       : {pr['precision']}",
                f"  Recall          : {pr['recall']}",
                f"  F1              : {pr['f1']}",
            ]

        lines.append(sep)
        print("\n".join(lines))
