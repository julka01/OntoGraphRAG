"""
Regression tests for retrieval hardening helpers.

Pure unit tests — no live Neo4j calls required.
"""

import os
import sys

import pytest

pytest.importorskip("langchain_neo4j", reason="langchain_neo4j not installed — skipping retrieval helper tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem


class TestEnhancedGroundingHelpers:
    system = EnhancedRAGSystem.__new__(EnhancedRAGSystem)

    def test_content_query_tokens_drop_common_question_words(self):
        tokens = self.system._content_query_tokens(
            "What company owns the manufacturer of Learjet 60?"
        )
        assert tokens == ["owns", "manufacturer", "learjet"]

    def test_matched_query_tokens_do_not_inflate_with_many_entities(self):
        query = "What company owns the manufacturer of Learjet 60?"
        matched = self.system._count_matched_query_tokens(
            query,
            [
                "Learjet 60",
                "Learjet",
                "Bombardier Aerospace",
                "Bombardier Inc.",
                "Learjet 60",
            ],
        )
        grounding = self.system._grounding_quality(query, matched)

        assert matched == 1
        assert grounding == pytest.approx(1 / 3)

    def test_grounding_quality_caps_at_one(self):
        query = "Which province borders the province containing Lago District?"
        matched = self.system._count_matched_query_tokens(
            query,
            ["Lago District", "Province", "Border Province", "Containing Province"],
        )
        grounding = self.system._grounding_quality(query, matched)

        assert 0.0 <= grounding <= 1.0

    def test_symbolic_entity_lookup_query_splits_alias_use_across_with_clauses(self):
        query = self.system._build_symbolic_entity_lookup_query("")

        assert "toLower(coalesce(e.name, '')) AS entity_name" in query
        assert "entity_name," in query
        assert "[tok IN $query_tokens WHERE entity_name CONTAINS tok] AS matched_tokens" in query
        assert "coalesce(e.all_names, [])" in query
