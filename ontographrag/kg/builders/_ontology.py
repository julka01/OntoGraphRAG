import json
import re
import hashlib
import difflib
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import os
import logging
from ontographrag.schemas.models import (
    OntologySchema,
    EntityType as OntEntityType,
    RelationshipType as OntRelType,
    DataBinding,
    RelationshipAttribute,
    PropertyType,
)

from ontographrag.kg.builders._creator_shared import (
    _is_valid_entity_name,
)

class OntologySchemaMixin:
    """Ontology loading, schema validation, and type normalisation.

    Mixin for :class:`OntologyGuidedKGCreator`; method bodies are
    unchanged from the original monolithic implementation.
    """

    @staticmethod
    def _prop_type(raw: str) -> PropertyType:
        """Map a raw type string from JSON to a PropertyType enum value."""
        _map = {
            "string": PropertyType.STRING, "str": PropertyType.STRING,
            "integer": PropertyType.INTEGER, "int": PropertyType.INTEGER,
            "decimal": PropertyType.DECIMAL, "numeric": PropertyType.DECIMAL,
            "double": PropertyType.DOUBLE,
            "float": PropertyType.FLOAT, "number": PropertyType.FLOAT,
            "boolean": PropertyType.BOOLEAN, "bool": PropertyType.BOOLEAN,
            "date": PropertyType.DATE, "datetime": PropertyType.DATETIME,
            "enum": PropertyType.ENUM,
            "id": PropertyType.ID, "identifier": PropertyType.ID,
        }
        return _map.get((raw or "string").strip().lower(), PropertyType.STRING)

    def _load_ontology(self, ontology_path: str):
        """Load ontology from OWL/RDF (XML) or Ontology Playground-style JSON.

        Both paths normalise into:
          self._ontology_schema     — OntologySchema (full typed model)
          self.ontology_classes     — List[dict]  (legacy flat list)
          self.ontology_relationships — List[dict]  (legacy flat list)
        """
        ext = os.path.splitext(ontology_path)[1].lower()
        is_json = ext == '.json'
        if not is_json and ext not in ('.owl', '.rdf', '.ttl', '.xml'):
            # Peek at first byte to detect JSON
            try:
                with open(ontology_path, 'r', encoding='utf-8') as _f:
                    _peek = _f.read(3).lstrip()
                is_json = _peek.startswith('{') or _peek.startswith('[')
            except OSError:
                pass

        try:
            if is_json:
                self._ontology_schema = self._load_ontology_json(ontology_path)
            else:
                self._ontology_schema = self._load_ontology_owl(ontology_path)
        except Exception as e:
            logging.error("Error loading ontology: %s", e)
            raise

        # Populate legacy flat lists for backwards compatibility
        self.ontology_classes = [
            {'id': et.id, 'uri': et.uri or '', 'label': et.label,
             'description': et.description or ''}
            for et in self._ontology_schema.entity_types
        ]
        self.ontology_relationships = [
            {'id': rt.id, 'uri': rt.uri or '', 'label': rt.label,
             'description': rt.description or '',
             'domain': rt.domain or '', 'range': rt.range or '',
             'cardinality': rt.cardinality or ''}
            for rt in self._ontology_schema.relationship_types
        ]
        logging.info(
            "Loaded ontology (%s): %d entity types, %d relationship types",
            self._ontology_schema.source_format,
            len(self.ontology_classes), len(self.ontology_relationships),
        )

    def _load_ontology_json(self, ontology_path: str) -> OntologySchema:
        """Parse an Ontology Playground-style JSON file.

        Accepts layout A: {"classes": [...], "relationships": [...]}
        and layout B:     {"entity_types": [...], "relationship_types": [...]}
        """
        with open(ontology_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        raw_classes = raw.get('classes') or raw.get('entity_types') or []
        raw_rels = raw.get('relationships') or raw.get('relationship_types') or []

        entity_types: List[OntEntityType] = []
        for cls in raw_classes:
            if not isinstance(cls, dict):
                continue
            eid = cls.get('id') or cls.get('name') or ''
            if not eid:
                continue
            props = []
            for p in cls.get('properties') or []:
                pname = (p.get('name') or p.get('id') or '') if isinstance(p, dict) else ''
                if not pname:
                    continue
                props.append(DataBinding(
                    name=pname,
                    type=self._prop_type(p.get('type', 'string')),
                    description=p.get('description') or None,
                    identifier=bool(p.get('identifier', False)),
                    required=bool(p.get('required', False)),
                    enum_values=list(p.get('enum_values') or p.get('values') or []),
                    unit=p.get('unit') or None,
                ))
            entity_types.append(OntEntityType(
                id=eid,
                label=cls.get('label') or eid.replace('_', ' ').title(),
                description=cls.get('description') or None,
                uri=cls.get('uri') or None,
                properties=props,
            ))

        relationship_types: List[OntRelType] = []
        for rel in raw_rels:
            if not isinstance(rel, dict):
                continue
            rid = rel.get('id') or rel.get('name') or rel.get('type') or ''
            if not rid:
                continue
            attrs = []
            for a in rel.get('attributes') or rel.get('properties') or []:
                aname = (a.get('name') or a.get('id') or '') if isinstance(a, dict) else ''
                if not aname:
                    continue
                attrs.append(RelationshipAttribute(
                    name=aname,
                    type=self._prop_type(a.get('type', 'string')),
                    description=a.get('description') or None,
                    unit=a.get('unit') or None,
                ))
            relationship_types.append(OntRelType(
                id=rid,
                label=rel.get('label') or rid.replace('_', ' ').title(),
                description=rel.get('description') or None,
                uri=rel.get('uri') or None,
                domain=rel.get('from') or rel.get('domain') or None,
                range=rel.get('to') or rel.get('range') or None,
                cardinality=rel.get('cardinality') or None,
                attributes=attrs,
            ))

        return OntologySchema(
            entity_types=entity_types, relationship_types=relationship_types,
            source_format='json', source_path=ontology_path,
        )

    def _load_ontology_owl(self, ontology_path: str) -> OntologySchema:
        """Parse an OWL/RDF XML ontology into OntologySchema."""
        tree = ET.parse(ontology_path)
        root = tree.getroot()

        ns = {
            'owl':  'http://www.w3.org/2002/07/owl#',
            'rdf':  'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        }
        _rdf_about  = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about'
        _rdfs_label = '{http://www.w3.org/2000/01/rdf-schema#}label'
        _rdfs_cmt   = '{http://www.w3.org/2000/01/rdf-schema#}comment'
        _rdfs_dom   = '{http://www.w3.org/2000/01/rdf-schema#}domain'
        _rdfs_rng   = '{http://www.w3.org/2000/01/rdf-schema#}range'
        _rdf_rsrc   = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'

        def _local(uri: str) -> str:
            return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]

        def _res_local(elem):
            if elem is None:
                return None
            r = elem.get(_rdf_rsrc, '')
            return _local(r) if r else None

        def _child_text_by_local_name(parent, local_name: str) -> Optional[str]:
            if parent is None:
                return None
            for child in list(parent):
                tag = child.tag
                if isinstance(tag, str):
                    child_local = tag.split('}')[-1] if '}' in tag else tag.split(':')[-1]
                    if child_local == local_name and child.text:
                        return child.text.strip()
            return None

        def _bool_text(value: Optional[str]) -> bool:
            return str(value or "").strip().lower() in {"true", "1", "yes"}

        def _xsd_to_prop_type(range_uri: Optional[str], explicit_type: Optional[str]) -> PropertyType:
            if explicit_type:
                return self._prop_type(explicit_type)
            local = _local(range_uri) if range_uri else ""
            return {
                "string": PropertyType.STRING,
                "integer": PropertyType.INTEGER,
                "int": PropertyType.INTEGER,
                "long": PropertyType.INTEGER,
                "decimal": PropertyType.DECIMAL,
                "float": PropertyType.FLOAT,
                "double": PropertyType.DOUBLE,
                "date": PropertyType.DATE,
                "dateTime": PropertyType.DATETIME,
                "boolean": PropertyType.BOOLEAN,
            }.get(local, PropertyType.STRING)

        entity_types: List[OntEntityType] = []
        for cls_elem in root.findall('.//owl:Class', ns):
            uri = cls_elem.get(_rdf_about, '')
            if not uri:
                continue
            local = _local(uri)
            if not local:
                continue
            lbl_el = cls_elem.find(_rdfs_label)
            cmt_el = cls_elem.find(_rdfs_cmt)
            entity_types.append(OntEntityType(
                id=local, uri=uri,
                label=(lbl_el.text.strip() if lbl_el is not None and lbl_el.text else local.replace('_', ' ').title()),
                description=(cmt_el.text.strip() if cmt_el is not None and cmt_el.text else None),
            ))

        entity_by_id = {et.id: et for et in entity_types}

        relationship_attribute_map: Dict[str, List[RelationshipAttribute]] = defaultdict(list)

        for dt_elem in root.findall('.//owl:DatatypeProperty', ns):
            uri = dt_elem.get(_rdf_about, '')
            if not uri:
                continue

            local = _local(uri)
            label = _child_text_by_local_name(dt_elem, 'label') or local
            description = _child_text_by_local_name(dt_elem, 'comment')
            domain = _res_local(dt_elem.find('.//' + _rdfs_dom))
            range_uri = dt_elem.find('.//' + _rdfs_rng)
            range_local = _res_local(range_uri)
            explicit_type = _child_text_by_local_name(dt_elem, 'propertyType') or _child_text_by_local_name(dt_elem, 'attributeType')
            prop_type = _xsd_to_prop_type(range_local, explicit_type)
            enum_values_text = _child_text_by_local_name(dt_elem, 'enumValues')
            enum_values = [v.strip() for v in (enum_values_text or '').split(',') if v.strip()]
            unit = _child_text_by_local_name(dt_elem, 'unit')
            identifier = _bool_text(_child_text_by_local_name(dt_elem, 'isIdentifier'))
            relationship_attr_of = _child_text_by_local_name(dt_elem, 'relationshipAttributeOf')

            if relationship_attr_of:
                relationship_attribute_map[relationship_attr_of].append(
                    RelationshipAttribute(
                        name=label,
                        type=prop_type,
                        description=description,
                        unit=unit,
                    )
                )
                continue

            if not domain or domain not in entity_by_id:
                continue

            entity_by_id[domain].properties.append(
                DataBinding(
                    name=label,
                    type=prop_type,
                    description=description,
                    identifier=identifier or prop_type == PropertyType.ID,
                    required=False,
                    enum_values=enum_values,
                    unit=unit,
                )
            )

        relationship_types: List[OntRelType] = []
        for prop_elem in root.findall('.//owl:ObjectProperty', ns):
            uri = prop_elem.get(_rdf_about, '')
            if not uri:
                continue
            local = _local(uri)
            if not local:
                continue
            lbl_el = prop_elem.find(_rdfs_label)
            cmt_el = prop_elem.find(_rdfs_cmt)
            dom_el = prop_elem.find('.//' + _rdfs_dom)
            rng_el = prop_elem.find('.//' + _rdfs_rng)
            relationship_types.append(OntRelType(
                id=local, uri=uri,
                label=(lbl_el.text.strip() if lbl_el is not None and lbl_el.text else local.replace('_', ' ').title()),
                description=(cmt_el.text.strip() if cmt_el is not None and cmt_el.text else None),
                domain=_res_local(dom_el),
                range=_res_local(rng_el),
                cardinality=_child_text_by_local_name(prop_elem, 'cardinality'),
                attributes=relationship_attribute_map.get(local, []),
            ))

        return OntologySchema(
            entity_types=entity_types, relationship_types=relationship_types,
            source_format='owl', source_path=ontology_path,
        )

    def _validate_ontology_structure(self):
        """
        Validate and clean ontology class and relationship structures to prevent "string indices must be integers" errors
        """
        # Clean ontology classes
        valid_classes = []
        for cls in self.ontology_classes:
            if isinstance(cls, dict) and 'id' in cls and 'label' in cls:
                valid_classes.append(cls)
            else:
                logging.warning(f"Removing invalid ontology class entry: {cls}")

        self.ontology_classes = valid_classes
        logging.info(f"Validated ontology classes: {len(self.ontology_classes)} valid entries")

        # Clean ontology relationships
        valid_relationships = []
        for rel in self.ontology_relationships:
            if isinstance(rel, dict) and 'id' in rel and 'label' in rel:
                valid_relationships.append(rel)
            else:
                logging.warning(f"Removing invalid ontology relationship entry: {rel}")

        self.ontology_relationships = valid_relationships
        logging.info(f"Validated ontology relationships: {len(self.ontology_relationships)} valid entries")

    @staticmethod
    def _normalize_ontology_identifier(value: Optional[str]) -> str:
        """Normalize ontology ids/labels for exact and fuzzy matching."""
        if not isinstance(value, str):
            return ""
        normalized = value.strip().lower()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def _schema_generic_entity_type(self) -> Optional[str]:
        """Return a generic ontology entity class when the schema defines one."""
        schema = self._ontology_schema
        if schema and schema.entity_types:
            preferred = {"concept", "entity", "thing", "unknown", "other"}
            for et in schema.entity_types:
                if (
                    self._normalize_ontology_identifier(et.id) in preferred
                    or self._normalize_ontology_identifier(et.label) in preferred
                ):
                    return et.id
        return None

    def _match_ontology_entity_type(
        self,
        raw_type: Optional[str],
        *,
        allow_fuzzy: bool = True,
        min_score: float = 0.80,
    ) -> Optional[str]:
        """Map a raw entity type string onto a known ontology class id."""
        normalized_raw = self._normalize_ontology_identifier(raw_type)
        if not normalized_raw:
            return None

        schema = self._ontology_schema
        if schema and schema.entity_types:
            best_match = None
            best_score = 0.0
            for et in schema.entity_types:
                candidates = [
                    self._normalize_ontology_identifier(et.id),
                    self._normalize_ontology_identifier(et.label),
                ]
                if normalized_raw in candidates:
                    return et.id
                if allow_fuzzy:
                    score = max(
                        difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                        for candidate in candidates
                        if candidate
                    )
                    if score > best_score:
                        best_score = score
                        best_match = et
            if allow_fuzzy and best_match and best_score >= min_score:
                return best_match.id

        best_match = None
        best_score = 0.0
        for cls in self.ontology_classes:
            cls_id = self._normalize_ontology_identifier(cls.get("id"))
            cls_label = self._normalize_ontology_identifier(cls.get("label"))
            if normalized_raw in {cls_id, cls_label}:
                return cls.get("id")
            if allow_fuzzy:
                score = max(
                    difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                    for candidate in (cls_id, cls_label)
                    if candidate
                )
                if score > best_score:
                    best_score = score
                    best_match = cls
        if allow_fuzzy and best_match and best_score >= min_score:
            return best_match.get("id")
        return None

    def _coerce_entity_type_with_ontology(
        self,
        raw_type: Optional[str],
        entity_text: Optional[str] = None,
    ) -> Optional[str]:
        """Coerce extracted entity types onto the active ontology when present."""
        has_ontology = bool(self._ontology_schema and self._ontology_schema.entity_types) or bool(self.ontology_classes)
        if not has_ontology:
            if raw_type:
                return str(raw_type)
            if entity_text:
                return self._classify_entity_with_ontology(entity_text)
            return None

        matched = self._match_ontology_entity_type(raw_type)
        if matched:
            return matched

        if entity_text:
            matched = self._match_ontology_entity_type(entity_text, allow_fuzzy=False)
            if matched:
                return matched
            classified = self._classify_entity_with_ontology(entity_text)
            matched = self._match_ontology_entity_type(classified, allow_fuzzy=False) or self._match_ontology_entity_type(classified)
            if matched:
                return matched

        generic_type = self._schema_generic_entity_type()
        if generic_type:
            if raw_type and self._normalize_ontology_identifier(raw_type) != self._normalize_ontology_identifier(generic_type):
                logging.info(
                    "Coercing off-schema entity type '%s' for '%s' to generic ontology type '%s'",
                    raw_type,
                    entity_text or "",
                    generic_type,
                )
            return generic_type

        logging.warning(
            "Dropping entity '%s' with off-schema type '%s' (no compatible ontology class found)",
            entity_text or "",
            raw_type or "",
        )
        return None

    def _schema_generic_relationship_type(self) -> Optional[str]:
        """Return a schema-defined generic relationship type when available."""
        preferred = {"related_to", "associated_with", "connects_to", "linked_to"}
        schema = self._ontology_schema
        if schema and schema.relationship_types:
            for rt in schema.relationship_types:
                if (
                    self._normalize_ontology_identifier(rt.id) in preferred
                    or self._normalize_ontology_identifier(rt.label) in preferred
                ):
                    return rt.id.replace(" ", "_").replace("-", "_").upper()
        for rel in self.ontology_relationships:
            rel_id = rel.get("id", "")
            rel_label = rel.get("label", "")
            if (
                self._normalize_ontology_identifier(rel_id) in preferred
                or self._normalize_ontology_identifier(rel_label) in preferred
            ):
                return rel_id.replace(" ", "_").replace("-", "_").upper()
        return None

    def _build_schema_card(self) -> dict:
        """Build a versioned snapshot of the ontology for this KG build.

        Stored on the Document node so future queries can detect ontology drift.
        Includes full property signatures, domain/range, cardinalities, and
        attribute schemas when a typed OntologySchema is available.
        """
        classes = [c.get('id', '') for c in self.ontology_classes if isinstance(c, dict)]
        rels    = [r.get('id', '') for r in self.ontology_relationships if isinstance(r, dict)]

        ontology_file_hash = None
        if self.ontology_path and os.path.exists(self.ontology_path):
            try:
                with open(self.ontology_path, 'rb') as f:
                    ontology_file_hash = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                pass

        card: dict = {
            "ontologyFileHash": ontology_file_hash or "",
            "ontologyPath":     os.path.basename(self.ontology_path) if self.ontology_path else "",
            "sourceFormat":     (self._ontology_schema.source_format if self._ontology_schema else "unknown"),
            "classes":          sorted(classes),
            "relationships":    sorted(rels),
            "classCount":       len(classes),
            "relationshipCount": len(rels),
            "builtAt":          datetime.now().isoformat(),
        }

        # Enrich with typed property signatures and domain/range when available
        schema = self._ontology_schema
        if schema:
            card["entityTypes"] = [
                {
                    "id": et.id,
                    "label": et.label,
                    "description": et.description,
                    "properties": [
                        {
                            "name": p.name, "type": p.type.value,
                            "identifier": p.identifier, "required": p.required,
                            "enum_values": p.enum_values, "unit": p.unit,
                        }
                        for p in et.properties
                    ],
                }
                for et in schema.entity_types
            ]
            card["relationshipTypes"] = [
                {
                    "id": rt.id, "label": rt.label, "description": rt.description,
                    "domain": rt.domain, "range": rt.range, "cardinality": rt.cardinality,
                    "attributes": [
                        {"name": a.name, "type": a.type.value, "unit": a.unit}
                        for a in rt.attributes
                    ],
                }
                for rt in schema.relationship_types
            ]

        fingerprint_payload = {
            "sourceFormat": card.get("sourceFormat", "unknown"),
            "classes": card.get("classes", []),
            "relationships": card.get("relationships", []),
            "entityTypes": card.get("entityTypes", []),
            "relationshipTypes": card.get("relationshipTypes", []),
        }
        fingerprint_str = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
        )
        schema_hash = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
        card["schemaVersion"] = schema_hash[:16]
        card["schemaHash"] = schema_hash

        return card

    @staticmethod
    def _example_surface_form_for_type(type_label: str) -> str:
        label = str(type_label or "").lower()
        if "drug" in label or "medication" in label or "treatment" in label:
            return "Metformin"
        if "disease" in label or "condition" in label or "disorder" in label or "syndrome" in label:
            return "type 2 diabetes"
        if "gene" in label or "variant" in label or "mutation" in label:
            return "BRCA1"
        if "protein" in label or "enzyme" in label or "receptor" in label or "biomarker" in label:
            return "TP53 protein"
        if "person" in label or "patient" in label:
            return "Marie Curie"
        if "organization" in label or "hospital" in label or "institute" in label or "university" in label:
            return "World Health Organization"
        if "location" in label or "city" in label or "country" in label:
            return "Paris"
        if "film" in label or "work" in label or "book" in label or "album" in label:
            return "Inception"
        return type_label.replace("_", " ") or "Example entity"

    def _build_ontology_few_shot_examples(self) -> str:
        """Generate a few compact schema-aware examples for the extraction prompt."""
        if getattr(self, "few_shot_example_count", 0) <= 0:
            return ""

        examples: List[str] = []
        schema = self._ontology_schema
        if schema and schema.relationship_types and schema.entity_types:
            for rt in schema.relationship_types[: max(1, self.few_shot_example_count)]:
                source_type = next((et for et in schema.entity_types if et.id == rt.domain), None)
                target_type = next((et for et in schema.entity_types if et.id == rt.range), None)
                if not source_type or not target_type:
                    continue
                source_name = self._example_surface_form_for_type(source_type.label)
                target_name = self._example_surface_form_for_type(target_type.label)
                examples.append(
                    (
                        f"Example {len(examples) + 1}\n"
                        f"TEXT: \"{source_name} {rt.label.lower().replace('_', ' ')} {target_name}.\"\n"
                        "JSON:\n"
                        "{\n"
                        '  "relationships": [\n'
                        f'    {{"source": "{source_name}", "target": "{target_name}", "type": "{rt.id}", "negated": false, "properties": {{"description": "{source_name} {rt.label.lower().replace("_", " ")} {target_name}", "condition": null, "quantitative": null, "confidence": "demonstrated"}}}}\n'
                        "  ],\n"
                        '  "entities": [\n'
                        f'    {{"id": "{source_name}", "type": "{source_type.id}", "properties": {{"name": "{source_name}", "description": "{source_type.label}: {source_name}"}}}},\n'
                        f'    {{"id": "{target_name}", "type": "{target_type.id}", "properties": {{"name": "{target_name}", "description": "{target_type.label}: {target_name}"}}}}\n'
                        "  ]\n"
                        "}"
                    )
                )
                if len(examples) >= self.few_shot_example_count:
                    break
        elif self.ontology_classes:
            for cls in self.ontology_classes[: self.few_shot_example_count]:
                entity_name = self._example_surface_form_for_type(cls.get("label", cls.get("id", "Entity")))
                examples.append(
                    (
                        f"Example {len(examples) + 1}\n"
                        f'TEXT: "{entity_name} is mentioned in the document."\n'
                        "JSON:\n"
                        "{\n"
                        '  "relationships": [],\n'
                        '  "entities": [\n'
                        f'    {{"id": "{entity_name}", "type": "{cls.get("id", "Concept")}", "properties": {{"name": "{entity_name}", "description": "{cls.get("label", cls.get("id", "Concept"))}: {entity_name}"}}}}\n'
                        "  ]\n"
                        "}"
                    )
                )

        return "\n\n".join(examples)

    def _normalize_anchor_inventory(
        self,
        raw_inventory: Dict[str, Any],
        *,
        chunk_text: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize discovered anchors and attach exact local spans."""
        inventory = {
            "entity_anchors": [],
            "relation_anchors": [],
            "attribute_anchors": [],
        }
        if not isinstance(raw_inventory, dict):
            return inventory

        seen_entities = set()
        for anchor in raw_inventory.get("entity_anchors", []) or []:
            text = None
            anchor_type = None
            if isinstance(anchor, str):
                text = anchor
            elif isinstance(anchor, dict):
                text = anchor.get("text") or anchor.get("id")
                anchor_type = anchor.get("type")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if not _is_valid_entity_name(text):
                continue
            spans = self._find_exact_text_spans(chunk_text, [text])
            if not spans:
                continue
            exact_text = spans[0]["text"]
            key = self._normalize_entity_text(exact_text)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            coerced_type = self._coerce_entity_type_with_ontology(anchor_type, exact_text)
            if not coerced_type:
                continue
            inventory["entity_anchors"].append(
                {
                    "text": exact_text,
                    "type": coerced_type,
                    "anchor_spans": spans,
                }
            )

        seen_relations = set()
        for anchor in raw_inventory.get("relation_anchors", []) or []:
            if isinstance(anchor, str):
                text = anchor
                type_hint = anchor
            elif isinstance(anchor, dict):
                text = anchor.get("text")
                type_hint = anchor.get("type_hint") or anchor.get("type") or text
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            spans = self._find_exact_text_spans(chunk_text, [text.strip()])
            if not spans:
                continue
            exact_text = spans[0]["text"]
            key = exact_text.lower()
            if key in seen_relations:
                continue
            seen_relations.add(key)
            inventory["relation_anchors"].append(
                {
                    "text": exact_text,
                    "type_hint": str(type_hint or exact_text).strip(),
                    "anchor_spans": spans,
                }
            )

        seen_attributes = set()
        for anchor in raw_inventory.get("attribute_anchors", []) or []:
            if isinstance(anchor, str):
                text = anchor
                role = "other"
            elif isinstance(anchor, dict):
                text = anchor.get("text")
                role = str(anchor.get("role") or "other").strip().lower()
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            spans = self._find_exact_text_spans(chunk_text, [text.strip()])
            if not spans:
                continue
            exact_text = spans[0]["text"]
            key = (exact_text.lower(), role)
            if key in seen_attributes:
                continue
            seen_attributes.add(key)
            inventory["attribute_anchors"].append(
                {
                    "text": exact_text,
                    "role": role if role in {"condition", "quantitative", "other"} else "other",
                    "anchor_spans": spans,
                }
            )

        return inventory

    def _classify_entity_with_ontology(self, entity_text: str) -> str:
        """
        Classify entity using ontology guidance.

        Strategy (in priority order):
        1. Embedding-based cosine similarity against pre-computed ontology class label embeddings
           (threshold ≥ 0.50 required to accept the match).
        2. Exact substring match against ontology class id/label (legacy fallback).
        3. Keyword heuristics (final fallback).
        """
        entity_lower = entity_text.lower()
        schema = self._ontology_schema

        # Strategy 1: embedding similarity (preferred — robust to abbreviations / paraphrases)
        if self._ontology_class_embeddings and self.embedding_function:
            try:
                entity_emb = self.embedding_function.embed_query(entity_text)
                best_class, best_score = None, 0.0
                for cls_id, cls_emb in self._ontology_class_embeddings:
                    # Cosine similarity (both vectors are L2-normalised by the embedding model)
                    score = float(
                        sum(a * b for a, b in zip(entity_emb, cls_emb))
                        / (
                            (sum(a * a for a in entity_emb) ** 0.5 + 1e-9)
                            * (sum(b * b for b in cls_emb) ** 0.5 + 1e-9)
                        )
                    )
                    if score > best_score:
                        best_score, best_class = score, cls_id
                if best_class and best_score >= 0.50:
                    return best_class
            except Exception as e:
                logging.debug(f"Embedding classification failed for '{entity_text}': {e}")

        # Strategy 2: exact substring match against ontology class labels
        for cls in self.ontology_classes:
            if cls['id'].lower() in entity_lower or cls['label'].lower() in entity_lower:
                return cls['id']

        # Strategy 3: keyword heuristics
        fallback = 'Concept'
        if any(word in entity_lower for word in ['disease', 'cancer', 'tumor', 'syndrome', 'disorder', 'carcinoma', 'malignancy']):
            fallback = 'Disease'
        elif any(word in entity_lower for word in ['drug', 'medication', 'treatment', 'therapy', 'surgery', 'chemotherapy', 'radiotherapy']):
            fallback = 'Treatment'
        elif any(word in entity_lower for word in ['patient', 'person', 'individual', 'male', 'female']):
            fallback = 'Patient'
        elif any(word in entity_lower for word in ['doctor', 'physician', 'surgeon', 'specialist', 'oncologist', 'urologist']):
            fallback = 'Physician'
        elif any(word in entity_lower for word in ['hospital', 'clinic', 'center', 'institute', 'department']):
            fallback = 'Hospital'
        elif any(word in entity_lower for word in ['symptom', 'sign', 'manifestation', 'pain', 'fever']):
            fallback = 'Symptom'
        elif any(word in entity_lower for word in ['gene', 'mutation', 'protein', 'biomarker', 'receptor', 'marker']):
            fallback = 'Biomarker'
        elif any(word in entity_lower for word in ['score', 'grade', 'stage', 'classification', 'risk']):
            fallback = 'ClinicalFinding'

        if schema and schema.entity_types:
            matched = self._match_ontology_entity_type(fallback, allow_fuzzy=False)
            if matched:
                return matched
            generic = self._schema_generic_entity_type()
            if generic:
                return generic

        return fallback

    def _classify_relationship_with_ontology(self, source: str, target: str) -> str:
        """
        Classify relationship using ontology guidance
        """
        # ------------------------------------------------------------------
        # Schema-constrained relationship classification (step 5).
        # When OntologySchema is available, find relationship types whose
        # domain/range are compatible with the given entity types, then rank
        # by lexical similarity to the entity names as a tiebreaker.
        # Falls back to hardcoded heuristics only when no schema is loaded.
        # ------------------------------------------------------------------
        schema = self._ontology_schema
        if schema and schema.relationship_types:
            source_type = self._classify_entity_with_ontology(source)
            target_type = self._classify_entity_with_ontology(target)
            candidates = schema.compatible_relationships(source_type, target_type)
            if candidates:
                # Rank by lexical similarity of the relationship label to
                # the concatenated source+target text as a weak domain signal
                combined = (source + " " + target).lower()
                best = max(
                    candidates,
                    key=lambda rt: difflib.SequenceMatcher(
                        None, rt.label.lower(), combined
                    ).ratio(),
                )
                return best.id.replace(' ', '_').replace('-', '_').upper()

        # Heuristic fallback (no schema or no compatible candidates found)
        source_lower = source.lower()
        target_lower = target.lower()
        if any(w in source_lower for w in ['treatment', 'therapy', 'drug']) or \
           any(w in target_lower for w in ['treatment', 'therapy', 'drug']):
            return 'TREATS'
        if any(w in source_lower for w in ['disease', 'cancer']) and \
           any(w in target_lower for w in ['symptom', 'sign']):
            return 'HAS_SYMPTOM'
        if any(w in source_lower for w in ['physician', 'doctor']) and \
           any(w in target_lower for w in ['patient', 'person']):
            return 'DIAGNOSES'
        return 'RELATED_TO'

    def _canonicalize_relationship_type(
        self,
        raw_type: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> Optional[str]:
        """Map a raw LLM-generated relationship type to the closest ontology relationship.

        Steps:
        1. Exact label/id match (case/space-insensitive) filtered to schema-compatible
           candidates when source_type and target_type are provided.
        2. Fuzzy match (SequenceMatcher ≥ 0.72) with schema-compatibility boost
           (+0.10) for fully-compatible domain/range matches.
        3. If no schema is active, sanitize the raw type if it passes regex; else
           fall back to ASSOCIATED_WITH.
        4. If a schema is active and no safe match exists, return a schema-defined
           generic relationship type when available; otherwise return None so the
           caller can skip the off-schema edge.
        """
        schema = self._ontology_schema
        if not raw_type:
            return self._schema_generic_relationship_type() if schema else 'ASSOCIATED_WITH'

        normalized_raw = raw_type.lower().replace(' ', '_').replace('-', '_')

        candidates = (
            schema.compatible_relationships(source_type, target_type)
            if schema and (source_type or target_type)
            else (schema.relationship_types if schema else [])
        )
        # Merge with legacy flat list for non-schema path
        ont_rels = self.ontology_relationships

        if schema and candidates:
            # Step 1: exact match within schema-compatible candidates first
            for rt in candidates:
                cand = rt.label.lower().replace(' ', '_')
                if cand == normalized_raw or rt.id.lower() == normalized_raw:
                    logging.debug("Exact schema-compatible rel match: '%s' → '%s'", raw_type, rt.id)
                    return rt.id.replace(' ', '_').replace('-', '_').upper()
            # Step 1b: exact match in full schema (less preferred)
            for rt in schema.relationship_types:
                cand = rt.label.lower().replace(' ', '_')
                if cand == normalized_raw or rt.id.lower() == normalized_raw:
                    return rt.id.replace(' ', '_').replace('-', '_').upper()

            # Step 2: semantic label match using ontology relationship embeddings.
            semantic_match = self._semantic_relationship_match(
                raw_type,
                [rt.id for rt in candidates] or [rt.id for rt in schema.relationship_types],
            )
            if semantic_match:
                matched_id, matched_score = semantic_match
                logging.debug(
                    "Semantic schema rel: '%s' → '%s' (score=%.2f)", raw_type, matched_id, matched_score
                )
                return matched_id.replace(' ', '_').replace('-', '_').upper()

            # Step 3: fuzzy match with schema-compatibility boost
            compat_ids = {rt.id for rt in candidates}
            best_match, best_score = None, 0.0
            for rt in schema.relationship_types:
                cand_label = rt.label.lower().replace(' ', '_')
                score = max(
                    difflib.SequenceMatcher(None, normalized_raw, cand_label).ratio(),
                    difflib.SequenceMatcher(None, normalized_raw, rt.id.lower()).ratio(),
                )
                # Boost schema-compatible candidates
                if rt.id in compat_ids:
                    score += 0.10
                if score > best_score:
                    best_score, best_match = score, rt
            if best_match and best_score >= 0.72:
                logging.debug(
                    "Fuzzy schema rel: '%s' → '%s' (score=%.2f)", raw_type, best_match.id, best_score
                )
                return best_match.id.replace(' ', '_').replace('-', '_').upper()

        elif ont_rels:
            # Legacy flat-list path (no OntologySchema)
            for ont_rel in ont_rels:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                if candidate == normalized_raw or ont_rel['id'].lower() == normalized_raw:
                    return ont_rel['id'].replace(' ', '_').replace('-', '_').upper()
            semantic_match = self._semantic_relationship_match(
                raw_type,
                [rel.get("id", "") for rel in ont_rels if rel.get("id")],
            )
            if semantic_match:
                matched_id, matched_score = semantic_match
                logging.debug(
                    "Semantic rel (legacy): '%s' → '%s' (score=%.2f)", raw_type, matched_id, matched_score
                )
                return matched_id.replace(' ', '_').replace('-', '_').upper()
            best_match, best_score = None, 0.0
            for ont_rel in ont_rels:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                score = difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                id_score = difflib.SequenceMatcher(None, normalized_raw, ont_rel['id'].lower()).ratio()
                best_s = max(score, id_score)
                if best_s > best_score:
                    best_score, best_match = best_s, ont_rel
            if best_match and best_score >= 0.72:
                logging.debug(
                    "Fuzzy rel (legacy): '%s' → '%s' (score=%.2f)", raw_type, best_match['id'], best_score
                )
                return best_match['id'].replace(' ', '_').replace('-', '_').upper()

        # Step 3: sanitize raw type
        sanitized = raw_type.strip().replace(' ', '_').replace('-', '_').upper()
        if schema:
            generic_rel = self._schema_generic_relationship_type()
            if generic_rel:
                logging.info(
                    "Coercing off-schema relationship type '%s' to generic ontology type '%s'",
                    raw_type,
                    generic_rel,
                )
                return generic_rel
            logging.warning(
                "Dropping relationship with off-schema type '%s' between %s and %s",
                raw_type,
                source_type or "?",
                target_type or "?",
            )
            return None
        if len(sanitized) > 50 or not re.match(r'^[A-Z][A-Z0-9_]*$', sanitized):
            logging.debug("Rel type '%s' failed sanitization; using ASSOCIATED_WITH", raw_type)
            return 'ASSOCIATED_WITH'
        return sanitized

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for duplicate detection.

        Applies generic, domain-agnostic normalization:
        - lowercase + collapse whitespace
        - strip leading articles / conjunctions
        - remove punctuation that doesn't affect identity
        - collapse runs of underscores/spaces

        Domain-specific aliases are intentionally NOT hardcoded here.
        The LLM is expected to produce consistent names; UUID5-based
        deduplication in _harmonize_entities handles surface variants.
        """
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Strip leading articles and conjunctions
        normalized = re.sub(r'^(the |an |a |and |or )', '', normalized)

        # Remove punctuation that doesn't carry semantic weight
        normalized = re.sub(r'[,()\[\];:.]', '', normalized)

        # Collapse hyphens/slashes to underscores for stable keys
        normalized = re.sub(r'[\-/]', '_', normalized)

        # Final cleanup
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        return normalized

    @staticmethod
    def _entity_type_specificity(entity_type: Any) -> int:
        """Score whether an entity type is more specific than generic fallback labels."""
        generic_types = {"concept", "entity", "unknown", "other", ""}
        normalized = str(entity_type or "").strip().lower()
        return 0 if normalized in generic_types else 1

    def _coerce_harmonized_entities_to_schema(
        self,
        harmonized_entities: List[Dict[str, Any]],
        harmonized_relationships: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply ontology type enforcement after harmonization as a final backstop."""
        has_ontology = bool(self._ontology_schema and self._ontology_schema.entity_types) or bool(self.ontology_classes)
        if not has_ontology:
            return harmonized_entities, harmonized_relationships

        filtered_entities: List[Dict[str, Any]] = []
        valid_entity_ids = set()
        dropped_entities = 0
        for entity in harmonized_entities:
            coerced_type = self._coerce_entity_type_with_ontology(
                entity.get("type"),
                entity.get("id"),
            )
            if not coerced_type:
                dropped_entities += 1
                continue
            entity_copy = dict(entity)
            entity_copy["type"] = coerced_type
            properties = dict(entity_copy.get("properties") or {})
            properties["type"] = coerced_type
            entity_copy["properties"] = properties
            filtered_entities.append(entity_copy)
            valid_entity_ids.add(entity_copy.get("uuid"))

        filtered_relationships = [
            rel for rel in harmonized_relationships
            if rel.get("source") in valid_entity_ids and rel.get("target") in valid_entity_ids
        ]
        dropped_relationships = len(harmonized_relationships) - len(filtered_relationships)
        if dropped_entities or dropped_relationships:
            logging.info(
                "Ontology enforcement dropped %d harmonized entities and %d relationships",
                dropped_entities,
                dropped_relationships,
            )
        self._last_schema_enforcement_stats = {
            "dropped_entities": dropped_entities,
            "dropped_relationships": dropped_relationships,
            "kept_entities": len(filtered_entities),
            "kept_relationships": len(filtered_relationships),
        }
        return filtered_entities, filtered_relationships
