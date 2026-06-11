import ast
import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

from ontographrag.schemas.models import (
    DataBinding,
    EntityType,
    OntologySchema,
    PropertyType,
    RelationshipAttribute,
    RelationshipType,
)


def _load_validate_ontology_schema():
    """Load validate_ontology_schema from app.py without importing the whole app."""
    app_path = Path(__file__).resolve().parents[1] / "ontographrag" / "api" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(app_path))
    fn_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_ontology_schema"
    )
    fn_source = ast.get_source_segment(source, fn_node)
    namespace = {
        "json": json,
        "os": __import__("os"),
        "logger": logging.getLogger("test"),
    }
    exec(fn_source, namespace)
    return namespace["validate_ontology_schema"]


def _load_builder_methods():
    """Load selected OntologyGuidedKGCreator methods without importing langchain."""
    builder_path = (
        Path(__file__).resolve().parents[1]
        / "ontographrag"
        / "kg"
        / "builders"
        / "_ontology.py"
    )
    source = builder_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(builder_path))
    class_node = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "OntologySchemaMixin"
    )
    wanted = {
        node.name: ast.get_source_segment(source, node)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_prop_type", "_load_ontology_owl", "_build_schema_card"}
    }
    namespace = {
        "ET": ET,
        "hashlib": __import__("hashlib"),
        "json": json,
        "os": __import__("os"),
        "datetime": __import__("datetime").datetime,
        "defaultdict": defaultdict,
        "OntologySchema": OntologySchema,
        "OntEntityType": EntityType,
        "OntRelType": RelationshipType,
        "DataBinding": DataBinding,
        "RelationshipAttribute": RelationshipAttribute,
        "PropertyType": PropertyType,
        "Optional": __import__("typing").Optional,
        "List": __import__("typing").List,
        "Dict": __import__("typing").Dict,
    }
    exec(wanted["_prop_type"], namespace)
    exec(wanted["_load_ontology_owl"], namespace)
    exec(wanted["_build_schema_card"], namespace)
    return (
        namespace["_prop_type"],
        namespace["_load_ontology_owl"],
        namespace["_build_schema_card"],
    )


def test_compatible_relationships_ranks_full_then_partial_then_unconstrained():
    schema = OntologySchema(
        entity_types=[
            EntityType(id="Drug", label="Drug"),
            EntityType(id="Disease", label="Disease"),
            EntityType(id="Patient", label="Patient"),
        ],
        relationship_types=[
            RelationshipType(id="generic", label="generic"),
            RelationshipType(id="treats_any", label="treatsAny", domain="Drug"),
            RelationshipType(id="treats", label="treats", domain="Drug", range="Disease"),
        ],
    )

    rel_ids = [r.id for r in schema.compatible_relationships("Drug", "Disease")]

    assert rel_ids == ["treats", "treats_any", "generic"]


def test_prop_type_preserves_datetime_decimal_and_double():
    prop_type, _, _ = _load_builder_methods()

    assert prop_type("datetime") == PropertyType.DATETIME
    assert prop_type("decimal") == PropertyType.DECIMAL
    assert prop_type("double") == PropertyType.DOUBLE


def test_load_ontology_owl_parses_datatype_properties_and_rel_attributes(tmp_path):
    owl = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
    xmlns:ont="http://example.org/ontology/test/">
  <owl:Class rdf:about="http://example.org/ontology/test/Patient">
    <rdfs:label>Patient</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/ontology/test/Encounter">
    <rdfs:label>Encounter</rdfs:label>
  </owl:Class>
  <owl:ObjectProperty rdf:about="http://example.org/ontology/test/hasEncounter">
    <rdfs:label>hasEncounter</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/ontology/test/Patient"/>
    <rdfs:range rdf:resource="http://example.org/ontology/test/Encounter"/>
    <ont:cardinality>one_to_many</ont:cardinality>
  </owl:ObjectProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/ontology/test/patient_id">
    <rdfs:label>patientId</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/ontology/test/Patient"/>
    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    <ont:isIdentifier>true</ont:isIdentifier>
  </owl:DatatypeProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/ontology/test/status">
    <rdfs:label>status</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/ontology/test/Encounter"/>
    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    <ont:propertyType>enum</ont:propertyType>
    <ont:enumValues>scheduled,completed</ont:enumValues>
  </owl:DatatypeProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/ontology/test/hasEncounter_note">
    <rdfs:label>note</rdfs:label>
    <ont:relationshipAttributeOf>hasEncounter</ont:relationshipAttributeOf>
    <ont:attributeType>string</ont:attributeType>
  </owl:DatatypeProperty>
</rdf:RDF>
"""
    owl_path = tmp_path / "test.owl"
    owl_path.write_text(owl, encoding="utf-8")

    prop_type, load_ontology_owl, _ = _load_builder_methods()

    class _StubCreator:
        pass

    creator = _StubCreator()
    creator._prop_type = prop_type
    schema = load_ontology_owl(creator, str(owl_path))

    patient = next(et for et in schema.entity_types if et.id == "Patient")
    encounter = next(et for et in schema.entity_types if et.id == "Encounter")
    rel = next(rt for rt in schema.relationship_types if rt.id == "hasEncounter")

    assert any(p.name == "patientId" and p.identifier for p in patient.properties)
    assert any(p.name == "status" and p.type == PropertyType.ENUM for p in encounter.properties)
    assert rel.cardinality == "one_to_many"
    assert any(a.name == "note" for a in rel.attributes)


def test_build_schema_card_hash_changes_when_property_signature_changes():
    _, _, build_schema_card = _load_builder_methods()

    base_entity_types = [
        EntityType(
            id="Patient",
            label="Patient",
            properties=[
                DataBinding(
                    name="patientId",
                    type=PropertyType.STRING,
                    identifier=True,
                )
            ],
        ),
        EntityType(id="Encounter", label="Encounter"),
    ]
    rel_types = [
        RelationshipType(
            id="hasEncounter",
            label="hasEncounter",
            domain="Patient",
            range="Encounter",
        )
    ]

    schema_a = OntologySchema(
        entity_types=base_entity_types,
        relationship_types=rel_types,
        source_format="json",
        source_path="ontology.json",
    )
    schema_b = OntologySchema(
        entity_types=[
            EntityType(
                id="Patient",
                label="Patient",
                properties=[
                    DataBinding(
                        name="patientId",
                        type=PropertyType.INTEGER,
                        identifier=True,
                    )
                ],
            ),
            EntityType(id="Encounter", label="Encounter"),
        ],
        relationship_types=rel_types,
        source_format="json",
        source_path="ontology.json",
    )

    class _StubCreator:
        pass

    creator = _StubCreator()
    creator.ontology_path = None
    creator.ontology_classes = [
        {"id": "Patient", "label": "Patient"},
        {"id": "Encounter", "label": "Encounter"},
    ]
    creator.ontology_relationships = [
        {
            "id": "hasEncounter",
            "label": "hasEncounter",
            "domain": "Patient",
            "range": "Encounter",
        }
    ]

    creator._ontology_schema = schema_a
    card_a = build_schema_card(creator)

    creator._ontology_schema = schema_b
    card_b = build_schema_card(creator)

    assert card_a["schemaHash"] != card_b["schemaHash"]


def test_validate_ontology_schema_rejects_missing_identifier_property():
    validate_ontology_schema = _load_validate_ontology_schema()
    ontology = {
        "classes": [
            {
                "id": "Patient",
                "properties": [
                    {"name": "name", "type": "string"},
                ],
            }
        ],
        "relationships": [],
    }

    errors = validate_ontology_schema(json.dumps(ontology).encode("utf-8"), "ontology.json")

    assert any("missing identifier property" in err for err in errors)


def test_validate_ontology_schema_rejects_invalid_cardinality():
    validate_ontology_schema = _load_validate_ontology_schema()
    ontology = {
        "classes": [
            {
                "id": "Patient",
                "properties": [
                    {"name": "patientId", "type": "string", "identifier": True},
                ],
            },
            {
                "id": "Encounter",
                "properties": [
                    {"name": "encounterId", "type": "string", "identifier": True},
                ],
            },
        ],
        "relationships": [
            {
                "id": "hasEncounter",
                "from": "Patient",
                "to": "Encounter",
                "cardinality": "manyish",
            }
        ],
    }

    errors = validate_ontology_schema(json.dumps(ontology).encode("utf-8"), "ontology.json")

    assert any("invalid cardinality" in err for err in errors)
