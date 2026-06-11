import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import logging

from ontographrag.kg.builders.ontology_guided_kg_creator import OntologyGuidedKGCreator

# Import CSV processors for bulk operations
try:
    from ontographrag.kg.csv_processor import MedicalReportCSVProcessor, DocumentCSVProcessor
except ImportError:
    logging.warning("CSV processors not available - bulk CSV operations disabled")
    MedicalReportCSVProcessor = None
    DocumentCSVProcessor = None


class UnifiedOntologyGuidedKGCreator(OntologyGuidedKGCreator):
    """
    Unified Knowledge Graph Creator combining ontology-guided LLM extraction with patient-specific parsing.
    Supports both general biomedical documents and patient reports with maximum detail depth.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        embedding_model: str = None,
        ontology_path: Optional[str] = None,
        max_chunks: int = None,  # For testing - limit chunks processed per report
        enable_coreference_resolution: bool = False,
        enable_heuristic_coreference_resolution: bool = True,
        retrieval_chunk_size: Optional[int] = None,
        retrieval_chunk_overlap: Optional[int] = None,
        strict_ontology: bool = True,
        self_consistency_n: int = 1,
        few_shot_example_count: int = 2,
        min_triple_confidence: float = 0.15,
        relationship_type_similarity_threshold: float = 0.62,
        enable_low_confidence_triple_reverification: bool = False,
        low_confidence_reverify_threshold: float = 0.4,
        enable_umls_linking: bool = False,
        umls_spacy_model: Optional[str] = None,
        enable_anchor_constrained_extraction: bool = True,
        enable_self_reflection: bool = True,
        enable_anchor_coverage_supplement: bool = True,
        enable_cross_passage_relation_recovery: bool = True,
        enable_soft_entity_linking: bool = False,
        soft_entity_similarity_threshold: float = 0.88,
        enable_fragmentation_repair: bool = False,
        fragmentation_bridge_similarity_threshold: float = 0.92,
        max_fragmentation_bridges: int = 8,
        enable_graph_summaries: bool = False,
        enable_claim_extraction: bool = False,
        max_summary_entities: int = 6,
        max_summary_relationships: int = 6,
    ):
        # Resolve env-var defaults here so no hardcoded URIs appear in source code
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
        embedding_model = embedding_model or os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

        # Pass ontology_path to parent so it loads ontology in the correct dict format
        # (strings-only loading via _load_biomedical_ontology would break the LLM prompt builder)
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            embedding_model=embedding_model,
            ontology_path=ontology_path,
            enable_coreference_resolution=enable_coreference_resolution,
            enable_heuristic_coreference_resolution=enable_heuristic_coreference_resolution,
            retrieval_chunk_size=retrieval_chunk_size,
            retrieval_chunk_overlap=retrieval_chunk_overlap,
            strict_ontology=strict_ontology,
            self_consistency_n=self_consistency_n,
            few_shot_example_count=few_shot_example_count,
            min_triple_confidence=min_triple_confidence,
            relationship_type_similarity_threshold=relationship_type_similarity_threshold,
            enable_low_confidence_triple_reverification=enable_low_confidence_triple_reverification,
            low_confidence_reverify_threshold=low_confidence_reverify_threshold,
            enable_umls_linking=enable_umls_linking,
            umls_spacy_model=umls_spacy_model,
            enable_anchor_constrained_extraction=enable_anchor_constrained_extraction,
            enable_self_reflection=enable_self_reflection,
            enable_anchor_coverage_supplement=enable_anchor_coverage_supplement,
            enable_cross_passage_relation_recovery=enable_cross_passage_relation_recovery,
            enable_soft_entity_linking=enable_soft_entity_linking,
            soft_entity_similarity_threshold=soft_entity_similarity_threshold,
            enable_fragmentation_repair=enable_fragmentation_repair,
            fragmentation_bridge_similarity_threshold=fragmentation_bridge_similarity_threshold,
            max_fragmentation_bridges=max_fragmentation_bridges,
            enable_graph_summaries=enable_graph_summaries,
            enable_claim_extraction=enable_claim_extraction,
            max_summary_entities=max_summary_entities,
            max_summary_relationships=max_summary_relationships,
        )

        # Set max_chunks for testing
        self._test_max_chunks = max_chunks

    def _extract_patient_demographics(self, text: str) -> Dict[str, Any]:
        """Extract patient demographics from text"""
        patient_info = {}

        name_match = re.search(r'Name:\s*([^\n\r]+)', text, re.IGNORECASE)
        if name_match:
            patient_info['name'] = name_match.group(1).strip()

        unit_match = re.search(r'Unit No:\s*([^\n\r]+)', text, re.IGNORECASE)
        if unit_match:
            patient_info['unit_no'] = unit_match.group(1).strip()

        dob_match = re.search(r'Date of Birth:\s*([^\n\r]+)', text, re.IGNORECASE)
        if dob_match:
            patient_info['date_of_birth'] = dob_match.group(1).strip()

        admission_match = re.search(r'Admission Date:\s*([^\n\r]+)', text, re.IGNORECASE)
        if admission_match:
            patient_info['admission_date'] = admission_match.group(1).strip()

        discharge_match = re.search(r'Discharge Date:\s*([^\n\r]+)', text, re.IGNORECASE)
        if discharge_match:
            patient_info['discharge_date'] = discharge_match.group(1).strip()

        sex_match = re.search(r'Sex:\s*([^\n\r]+)', text, re.IGNORECASE)
        if sex_match:
            patient_info['sex'] = sex_match.group(1).strip()

        service_match = re.search(r'Service:\s*([^\n\r]+)', text, re.IGNORECASE)
        if service_match:
            patient_info['service'] = service_match.group(1).strip()

        attending_match = re.search(r'Attending:\s*([^\n\r]+)', text, re.IGNORECASE)
        if attending_match:
            patient_info['attending'] = attending_match.group(1).strip()

        return patient_info

    def _extract_medical_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical conditions from various sections"""
        conditions = []

        pmh_section = self._extract_section(text, "Past Medical History")
        if pmh_section:
            condition_matches = re.findall(r'\d+\.\s*([^.\n]+)', pmh_section, re.MULTILINE)
            for match in condition_matches:
                condition = match.strip()
                if condition and len(condition) > 3 and not condition.startswith(('No ', 'no ')):
                    conditions.append({
                        'condition': condition,
                        'category': 'past_medical_history',
                        'status': 'chronic'
                    })

            pmh_conditions = ['cirrhosis', 'HIV', 'IVDU', 'COPD', 'bipolar', 'PTSD', 'cancer', 'diabetes', 'hypertension']
            for condition in pmh_conditions:
                if condition.lower() in pmh_section.lower():
                    conditions.append({
                        'condition': condition,
                        'category': 'past_medical_history',
                        'status': 'chronic'
                    })

        hopi_section = self._extract_section(text, "History of Present Illness")
        if hopi_section:
            condition_matches = re.findall(r'(?:c/b|complicated by|due to|secondary to)\s+([^.,;]+)', hopi_section, re.IGNORECASE)
            for match in condition_matches:
                match = match.strip()
                if len(match) > 2:
                    conditions.append({
                        'condition': match,
                        'category': 'present_illness',
                        'status': 'acute'
                    })

            if 'ascites' in hopi_section.lower():
                conditions.append({
                    'condition': 'ascites',
                    'category': 'present_illness',
                    'status': 'acute'
                })

        chief_complaint_section = self._extract_section(text, "Chief Complaint")
        if chief_complaint_section:
            complaint_text = chief_complaint_section.strip()
            if complaint_text.startswith('-'):
                complaint_text = complaint_text[1:].strip()

            if complaint_text and len(complaint_text) > 3:
                if 'abdominal' in complaint_text.lower() and 'distension' in complaint_text.lower():
                    conditions.append({
                        'condition': 'abdominal distension',
                        'category': 'chief_complaint',
                        'status': 'presenting'
                    })
                else:
                    conditions.append({
                        'condition': complaint_text,
                        'category': 'chief_complaint',
                        'status': 'presenting'
                    })

        discharge_dx_section = self._extract_section(text, "Discharge Diagnosis")
        if discharge_dx_section:
            if 'ascites' in discharge_dx_section.lower():
                conditions.append({
                    'condition': 'ascites from portal hypertension',
                    'category': 'discharge_diagnosis',
                    'status': 'final'
                })

        seen = set()
        unique_conditions = []
        for condition in conditions:
            key = condition['condition'].lower().strip()
            if key not in seen:
                seen.add(key)
                unique_conditions.append(condition)

        return unique_conditions

    def _extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """Extract medications from admission and discharge sections"""
        medications = []

        admission_meds_section = self._extract_section(text, "Medications on Admission")
        if admission_meds_section:
            med_lines = [line.strip('- ').strip() for line in admission_meds_section.split('\n') if line.strip() and line[0].isdigit()]
            for med in med_lines:
                if med and len(med) > 2:
                    medications.append({'medication': med, 'timing': 'admission', 'status': 'chronic'})

        discharge_meds_section = self._extract_section(text, "Discharge Medications")
        if discharge_meds_section:
            med_lines = [line.strip('- ').strip() for line in discharge_meds_section.split('\n') if line.strip() and line[0].isdigit()]
            for med in med_lines:
                if med and len(med) > 2:
                    medications.append({'medication': med, 'timing': 'discharge', 'status': 'prescribed'})

        hospital_course = self._extract_section(text, "Brief Hospital Course")
        if hospital_course:
            med_changes = re.findall(r'(?:Furosemide|Lasix|Spironolactone|Aldactone|metoprolol|atorvastatin|simvastatin|omeprazole|lisinopril|enalapril)\s+\d+(?:\s*mg)?', hospital_course, re.IGNORECASE)
            for med_change in med_changes:
                medications.append({'medication': med_change.strip(), 'timing': 'hospital_course', 'status': 'adjusted'})

        return medications

    def _extract_social_history(self, text: str) -> Dict[str, Any]:
        """Extract social history information"""
        social_info = {}

        social_section = self._extract_section(text, "Social History")
        if social_section:
            if 'smoking' in social_section.lower() or 'smoker' in social_section.lower():
                if 'quit' in social_section.lower() or 'former' in social_section.lower():
                    social_info['smoking_status'] = 'former_smoker'
                elif 'current' in social_section.lower() or 'active' in social_section.lower():
                    social_info['smoking_status'] = 'current_smoker'
                else:
                    social_info['smoking_status'] = 'unknown'

            if 'alcohol' in social_section.lower():
                social_info['alcohol_use'] = 'none' if ('none' in social_section.lower() or 'quit' in social_section.lower()) else 'present'

            if 'drug' in social_section.lower():
                social_info['drug_use'] = 'none' if ('none' in social_section.lower() or 'quit' in social_section.lower()) else 'history'

        return social_info

    def _extract_lab_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract laboratory values and results"""
        lab_values = []
        lab_patterns = [
            r'(\w+)\s*[:-]\s*(\d+(?:\.\d+)?)\*?',
            r'(\w+)\((\w+)\)\s*[:-]\s*(\d+(?:\.\d+)?)\*?',
        ]

        pertinent_results_section = self._extract_section(text, "Pertinent Results")
        if pertinent_results_section:
            for pattern in lab_patterns:
                matches = re.findall(pattern, pertinent_results_section)
                for match in matches:
                    if len(match) == 2:
                        lab_name, value = match
                        lab_values.append({'test': lab_name.strip(), 'value': value, 'abnormal': '*' in str(match)})
                    elif len(match) == 3:
                        full_name, short_name, value = match
                        lab_values.append({'test': f"{full_name} ({short_name})", 'value': value, 'abnormal': '*' in str(match)})

        return lab_values

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the patient report"""
        section_match = re.search(rf'{section_name}[:\s]*', text, re.IGNORECASE)
        if not section_match:
            return ""

        start_pos = section_match.end()
        next_section_patterns = [
            r'\n(?:Admission|Discharge|Physical|History|Brief|Medications|Pertinent|Social|Family)\s+(?:Medical\s+)?History[:\(\s]',
            r'\n[A-Z][A-Z\s]+:\s*'
        ]

        text_remaining = text[start_pos:]
        min_end_pos = len(text)
        for pattern in next_section_patterns:
            next_match = re.search(pattern, text_remaining)
            if next_match:
                min_end_pos = min(min_end_pos, next_match.start())

        return text_remaining[:min_end_pos].strip()

    def bulk_process_medical_reports(self, csv_path: str, start_row: int = 0, batch_size: int = 50, llm=None) -> Dict[str, Any]:
        """Process multiple medical reports in bulk from a pipe-delimited CSV file."""
        logging.info(f"Starting bulk processing of medical reports from {csv_path}")

        csv_processor = MedicalReportCSVProcessor(delimiter='|')
        validation = csv_processor.validate_csv_format(csv_path)
        if not validation['is_valid']:
            raise ValueError(f"CSV validation failed: {validation.get('validation_errors', validation.get('error'))}")

        all_knowledge_graphs = []
        total_processed = 0

        while True:
            try:
                batch_data = csv_processor.load_reports_bulk(csv_path, start_row=start_row + total_processed, max_rows=batch_size)
                reports = batch_data['reports']
                if not reports:
                    break

                batch_kgs = []
                for report in reports:
                    try:
                        report_text = report['sections'].get('full_report_text', '')
                        if not report_text:
                            sections_text = [
                                f"{k.replace('_', ' ').title()}:\n{v}\n\n"
                                for k, v in report['sections'].items()
                                if k != 'full_report_text'
                            ]
                            report_text = '\n'.join(sections_text)

                        if report_text:
                            kg = self.generate_patient_knowledge_graph(
                                report_text, llm=llm,
                                file_name=f"report_{report['row_index']}",
                                max_chunks=getattr(self, '_test_max_chunks', None),
                            )
                            if kg:
                                batch_kgs.append(kg)
                    except Exception as e:
                        logging.error(f"Failed to process report at row {report['row_index']}: {e}")

                all_knowledge_graphs.extend(batch_kgs)
                total_processed += len(reports)
                logging.info(f"Processed batch: {len(reports)} reports, generated {len(batch_kgs)} knowledge graphs")

                if len(reports) < batch_size:
                    break
            except Exception as e:
                logging.error(f"Error processing batch starting at row {start_row + total_processed}: {e}")
                break

        return {
            'knowledge_graph': self._merge_knowledge_graphs(all_knowledge_graphs),
            'metadata': {
                'total_reports_processed': total_processed,
                'total_knowledge_graphs': len(all_knowledge_graphs),
                'csv_validation': validation,
                'bulk_processing_info': {'start_row': start_row, 'batch_size': batch_size,
                                         'total_batches': (total_processed + batch_size - 1) // batch_size},
            },
        }

    def bulk_process_documents(
        self,
        csv_path: str,
        text_column: str,
        id_column: str = None,
        start_row: int = 0,
        batch_size: int = 50,
        llm=None,
        max_chunks: int = None,
        kg_name: str = None,
    ) -> Dict[str, Any]:
        """Process documents from any CSV file and build a knowledge graph."""
        if DocumentCSVProcessor is None:
            raise RuntimeError("DocumentCSVProcessor not available — install csv_processor module")

        logging.info("Starting bulk document processing from %s (text_column=%s)", csv_path, text_column)

        processor = DocumentCSVProcessor(text_column=text_column, id_column=id_column)
        validation = processor.validate_csv_format(csv_path)
        if not validation['is_valid']:
            raise ValueError(f"CSV validation failed: {validation.get('validation_errors')}")

        all_knowledge_graphs = []
        total_processed = 0

        while True:
            try:
                batch_data = processor.load_documents_bulk(csv_path, start_row=start_row + total_processed, max_rows=batch_size)
                documents = batch_data['documents']
                if not documents:
                    break

                batch_kgs = []
                for doc in documents:
                    try:
                        kg = self.generate_knowledge_graph(
                            doc['text'], llm=llm, file_name=doc['doc_id'],
                            max_chunks=max_chunks, kg_name=kg_name,
                            doc_metadata=doc.get('metadata'),
                        )
                        if kg:
                            batch_kgs.append(kg)
                    except Exception as e:
                        logging.error("Failed to process document '%s' (row %d): %s", doc['doc_id'], doc['row_index'], e)

                all_knowledge_graphs.extend(batch_kgs)
                total_processed += len(documents)
                logging.info("Processed batch: %d documents, generated %d knowledge graphs", len(documents), len(batch_kgs))

                if len(documents) < batch_size:
                    break
            except Exception as e:
                logging.error("Error processing batch starting at row %d: %s", start_row + total_processed, e)
                break

        return {
            'knowledge_graph': self._merge_knowledge_graphs(all_knowledge_graphs),
            'metadata': {
                'total_documents_processed': total_processed,
                'total_knowledge_graphs': len(all_knowledge_graphs),
                'csv_validation': validation,
                'bulk_processing_info': {
                    'start_row': start_row, 'batch_size': batch_size,
                    'text_column': text_column, 'id_column': id_column, 'kg_name': kg_name,
                    'total_batches': (total_processed + batch_size - 1) // batch_size if total_processed else 0,
                },
            },
        }

    def _merge_knowledge_graphs(self, knowledge_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple knowledge graphs into a single unified graph."""
        all_nodes, all_relationships, all_chunks = [], [], []
        for kg in knowledge_graphs:
            all_nodes.extend(kg.get('nodes', []))
            all_relationships.extend(kg.get('relationships', []))
            all_chunks.extend(kg.get('chunks', []))

        harmonized_nodes, id_map = self._harmonize_entities(all_nodes, return_id_map=True)
        harmonized_rels = self._harmonize_relationships(all_relationships, id_map)

        return {
            "nodes": harmonized_nodes,
            "relationships": harmonized_rels,
            "chunks": all_chunks,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "kg_type": "bulk_medical_reports",
                "extraction_method": "patient_specific_hybrid",
                "total_entities": len(harmonized_nodes),
                "total_relationships": len(harmonized_rels),
                "source_knowledge_graphs": len(knowledge_graphs),
            },
        }

    def _extract_entities_and_relationships_for_patients(self, text: str) -> Dict[str, Any]:
        """Extract patient-specific entities and relationships using hybrid approach."""
        entities = []
        relationships = []

        patient_info = self._extract_patient_demographics(text)
        if patient_info:
            patient_id = f"Patient_{hashlib.md5(str(patient_info).encode()).hexdigest()[:8]}"
            entities.append({
                "id": patient_id, "type": "Patient",
                "properties": {
                    "name": patient_info.get('name', 'Unknown'), "type": "Patient",
                    "description": f"Patient: {', '.join(f'{k}: {v}' for k, v in patient_info.items())}",
                    **patient_info
                }
            })

        conditions = self._extract_medical_conditions(text)
        for condition in conditions:
            cid = f"Condition_{hashlib.md5(condition['condition'].encode()).hexdigest()[:8]}"
            entities.append({
                "id": cid, "type": "MedicalCondition",
                "properties": {
                    "name": condition['condition'], "type": "MedicalCondition",
                    "description": f"{condition['condition']} ({condition['status']})",
                    "category": condition['category'], "status": condition['status'],
                }
            })
            if patient_info and condition['category'] != 'presenting':
                relationships.append({
                    "source": patient_id, "target": cid, "type": "HAS_CONDITION",
                    "properties": {"description": f"Patient has {condition['condition']}",
                                   "category": condition['category'], "status": condition['status']},
                })

        for medication in self._extract_medications(text):
            mid = f"Medication_{hashlib.md5(medication['medication'].encode()).hexdigest()[:8]}"
            entities.append({
                "id": mid, "type": "Medication",
                "properties": {
                    "name": medication['medication'], "type": "Medication",
                    "description": f"Medication: {medication['medication']} ({medication['timing']})",
                    "timing": medication['timing'], "status": medication['status'],
                }
            })
            if patient_info:
                relationships.append({
                    "source": patient_id, "target": mid, "type": "TAKES_MEDICATION",
                    "properties": {"description": f"Patient takes {medication['medication']}",
                                   "timing": medication['timing'], "status": medication['status']},
                })

        for lab in self._extract_lab_values(text):
            lid = f"Lab_{hashlib.md5(lab['test'].encode()).hexdigest()[:8]}"
            entities.append({
                "id": lid, "type": "LabTest",
                "properties": {
                    "name": lab['test'], "type": "LabTest",
                    "description": f"Lab: {lab['test']} = {lab['value']}{'*' if lab['abnormal'] else ''}",
                    "value": lab['value'], "abnormal": lab['abnormal'],
                }
            })
            if patient_info:
                relationships.append({
                    "source": patient_id, "target": lid, "type": "HAS_LAB_RESULT",
                    "properties": {"description": f"Patient has lab result: {lab['test']} = {lab['value']}",
                                   "value": lab['value'], "abnormal": lab['abnormal']},
                })

        for key, value in self._extract_social_history(text).items():
            sid = f"Social_{key}_{hashlib.md5(str(value).encode()).hexdigest()[:8]}"
            entities.append({
                "id": sid, "type": "SocialFactor",
                "properties": {"name": key, "type": "SocialFactor",
                               "description": f"Social history: {key} = {value}",
                               "factor": key, "value": value}
            })
            if patient_info:
                relationships.append({
                    "source": patient_id, "target": sid, "type": "HAS_SOCIAL_HISTORY",
                    "properties": {"description": f"Patient has social factor: {key} = {value}",
                                   "factor": key, "value": value},
                })

        return {"entities": entities, "relationships": relationships}

    def generate_patient_knowledge_graph(self, text: str, llm=None, file_name: str = None, max_chunks: int = None) -> Dict[str, Any]:
        """Generate a knowledge graph from patient report text with patient-specific extraction."""
        logging.info("Generating patient-specific knowledge graph")
        if llm is None:
            logging.warning(
                "No LLM provided to generate_patient_knowledge_graph — "
                "falling back to regex-only extraction. Pass an LLM for better coverage."
            )

        chunks = self._chunk_text(text)
        if max_chunks is not None and max_chunks > 0:
            chunks = chunks[:max_chunks]

        all_entities, all_relationships = [], []
        for chunk in chunks:
            try:
                result = self._extract_entities_and_relationships_for_patients(chunk['text'])
                all_entities.extend(result.get("entities", []))
                all_relationships.extend(result.get("relationships", []))

                if llm is not None:
                    llm_result = self._extract_entities_and_relationships_with_llm(chunk['text'], llm)
                    all_entities.extend(llm_result.get("entities", []))
                    all_relationships.extend(llm_result.get("relationships", []))
            except Exception as e:
                logging.error(f"Error processing chunk {chunk['chunk_id']}: {e}")

        harmonized_entities, id_map = self._harmonize_entities(all_entities, return_id_map=True)
        harmonized_relationships = self._harmonize_relationships(all_relationships, id_map)

        return {
            "nodes": harmonized_entities,
            "relationships": harmonized_relationships,
            "chunks": chunks,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "file_name": file_name,
                "kg_type": "patient_report",
                "extraction_method": "patient_specific_hybrid",
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
            },
        }
