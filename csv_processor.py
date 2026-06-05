#!/usr/bin/env python3
"""
CSV Processors for Knowledge Graph bulk ingestion.

- DocumentCSVProcessor: generic, works with any CSV. The ontology defines the
  domain; this class only cares about which column holds the document text.
- MedicalReportCSVProcessor: specialisation for pipe-delimited medical reports
  with structured section columns.
"""
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class DocumentCSVProcessor:
    """
    Generic CSV processor for bulk KG ingestion.

    Domain knowledge lives in the ontology, not here. This class only needs to
    know which column contains the document text and (optionally) which column
    to use as a stable document ID.
    """

    def __init__(self, text_column: str, id_column: str = None, delimiter: str = ','):
        """
        Args:
            text_column: Name of the column that contains the document text.
            id_column:   Optional column to use as doc_id. When None, row index
                         is used ("row_0", "row_1", …).
            delimiter:   CSV field delimiter.
        """
        self.text_column = text_column
        self.id_column = id_column
        self.delimiter = delimiter

    def validate_csv_format(self, file_path: str) -> Dict[str, Any]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path, sep=self.delimiter, nrows=5, encoding='utf-8')

        errors = []
        if self.text_column not in df.columns:
            errors.append(f"text_column '{self.text_column}' not found in CSV. "
                          f"Available columns: {list(df.columns)}")
        if self.id_column and self.id_column not in df.columns:
            errors.append(f"id_column '{self.id_column}' not found in CSV. "
                          f"Available columns: {list(df.columns)}")

        return {
            'is_valid': len(errors) == 0,
            'delimiter': self.delimiter,
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'validation_errors': errors,
        }

    def load_documents_bulk(
        self,
        file_path: str,
        start_row: int = 0,
        max_rows: int = 50,
    ) -> Dict[str, Any]:
        """
        Load a batch of documents from CSV.

        Returns:
            {
                'documents': [
                    {
                        'text':      str,   # content of text_column
                        'doc_id':    str,   # id_column value or "row_{row_index}"
                        'row_index': int,   # absolute row number in the file
                        'metadata':  dict,  # all other columns as-is
                    },
                    ...
                ],
                'metadata': {
                    'total_documents': int,
                    'start_row': int,
                    'max_rows': int,
                    'columns': list,
                },
            }
        """
        df = pd.read_csv(
            file_path,
            sep=self.delimiter,
            skiprows=range(1, start_row + 1) if start_row > 0 else None,
            nrows=max_rows,
            encoding='utf-8',
        )

        documents = []
        for local_idx, row in df.iterrows():
            row_index = start_row + local_idx
            text = str(row.get(self.text_column, '') or '')
            if not text.strip():
                continue

            doc_id = (
                str(row[self.id_column])
                if self.id_column and self.id_column in row
                else f"row_{row_index}"
            )

            metadata = {
                col: row[col]
                for col in df.columns
                if col != self.text_column and (self.id_column is None or col != self.id_column)
            }

            documents.append({
                'text': text,
                'doc_id': doc_id,
                'row_index': row_index,
                'metadata': metadata,
            })

        return {
            'documents': documents,
            'metadata': {
                'total_documents': len(documents),
                'start_row': start_row,
                'max_rows': max_rows,
                'columns': list(df.columns),
            },
        }

class MedicalReportCSVProcessor:
    """
    Processor for pipe-delimited CSV files containing medical reports
    with enhanced validation and bulk processing capabilities
    """

    # Expected field names for medical report sections
    EXPECTED_FIELDS = [
        'patient_id',
        'full_name',
        'date_of_birth',
        'admission_date',
        'discharge_date',
        'sex',
        'service',
        'attending',
        'unit_no',
        'chief_complaint',
        'history_present_illness_hopi',
        'past_medical_history_pmh',
        'medications_admission',
        'medications_discharge',
        'brief_hospital_course',
        'pertinent_results',
        'social_history',
        'family_history',
        'discharge_diagnosis',
        'discharge_instructions',
        'follow_up_instructions',
        'full_report_text'  # Last column containing complete report
    ]

    def __init__(self, delimiter: str = '|'):
        """
        Initialize CSV processor

        Args:
            delimiter: CSV field delimiter (default: '|' for pipe-delimited)
        """
        self.delimiter = delimiter
        self.validated_schema = None

    def validate_csv_format(self, file_path: str) -> Dict[str, Any]:
        """
        Validate CSV file format and structure

        Args:
            file_path: Path to CSV file

        Returns:
            Validation results with schema info
        """
        try:
            # First check: file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            # Test read with pipe delimiter
            df_test = pd.read_csv(
                file_path,
                sep=self.delimiter,
                nrows=5,  # Only read first few rows for validation
                encoding='utf-8'
            )

            if df_test.empty:
                raise ValueError("CSV file is empty")

            # Check for expected field structure
            validation_results = {
                'is_valid': True,
                'delimiter': self.delimiter,
                'num_columns': len(df_test.columns),
                'num_rows': len(df_test),
                'column_names': list(df_test.columns),
                'field_mapping': {},
                'validation_errors': []
            }

            # Validate column count (should have all expected sections + full text)
            if len(df_test.columns) < 25:  # Minimum expected medical report sections
                validation_results['validation_errors'].append(
                    f"Insufficient columns. Expected at least {len(self.EXPECTED_FIELDS)}, got {len(df_test.columns)}"
                )

            # Check if last column likely contains full report text (longer text)
            if len(df_test.columns) > 0:
                last_col_samples = df_test.iloc[:, -1].astype(str)
                avg_length = last_col_samples.str.len().mean()
                if avg_length < 100:  # Probably not full report text
                    validation_results['validation_errors'].append(
                        "Last column appears too short for full report text"
                    )

            # Check for common medical report fields
            med_fields = ['history', 'hospital', 'medication', 'diagnosis', 'complaint']
            found_med_fields = [
                col for col in df_test.columns
                if any(field in col.lower() for field in med_fields)
            ]

            if len(found_med_fields) < 3:
                validation_results['validation_errors'].append(
                    "Few medical report fields detected. May not be a proper medical report CSV."
                )

            # Auto-map expected fields to actual columns
            for expected in self.EXPECTED_FIELDS:
                # Find best match (case-insensitive partial match)
                best_match = None
                best_score = 0

                for actual in df_test.columns:
                    # Calculate match score
                    actual_lower = actual.lower().replace('_', '').replace(' ', '')
                    expected_lower = expected.lower().replace('_', '').replace(' ', '')

                    if expected_lower in actual_lower:
                        score = len(expected_lower) / len(actual_lower)
                        if score > best_score:
                            best_score = score
                            best_match = actual

                if best_match:
                    validation_results['field_mapping'][expected] = best_match

            # Mark as invalid if critical errors
            if len(validation_results['validation_errors']) > 2:
                validation_results['is_valid'] = False

            self.validated_schema = validation_results
            return validation_results

        except Exception as e:
            logging.error(f"CSV validation failed: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'delimiter': self.delimiter
            }

    def load_reports_bulk(self, file_path: str, start_row: int = 0, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Load and process medical reports in bulk

        Args:
            file_path: Path to CSV file
            start_row: Starting row number (0-based)
            max_rows: Maximum number of rows to load

        Returns:
            Dictionary containing processed reports and metadata
        """
        try:
            if not self.validated_schema:
                validation = self.validate_csv_format(file_path)
                if not validation['is_valid']:
                    raise ValueError(f"CSV validation failed: {validation.get('validation_errors', [])}")

            # Load CSV with optimized settings
            df = pd.read_csv(
                file_path,
                sep=self.delimiter,
                skiprows=range(1, start_row + 1) if start_row > 0 else None,
                nrows=max_rows,
                encoding='utf-8',
                low_memory=False,
                dtype=str  # Load as strings to preserve formatting
            ).fillna('')

            # Apply field mapping to expected fields
            mapped_df = df.copy()
            if self.validated_schema.get('field_mapping'):
                for expected, actual in self.validated_schema['field_mapping'].items():
                    if actual in df.columns:
                        mapped_df[expected] = df[actual]

            # Process each row into structured report format
            processed_reports = []
            for idx, row in mapped_df.iterrows():
                report = self._process_single_report(row)
                report['row_index'] = idx + start_row
                processed_reports.append(report)

            return {
                'reports': processed_reports,
                'metadata': {
                    'total_reports': len(processed_reports),
                    'start_row': start_row,
                    'max_rows': max_rows,
                    'columns_processed': list(mapped_df.columns),
                    'validation_schema': self.validated_schema
                },
                'schema_info': self.validated_schema
            }

        except Exception as e:
            logging.error(f"Bulk processing failed: {e}")
            raise

    def _process_single_report(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single report row into structured format

        Args:
            row: Pandas Series containing report data

        Returns:
            Structured report dictionary
        """
        report = {
            'sections': {},
            'metadata': {},
            'field_names': {}
        }

        # Extract demographic information
        report['metadata'] = {
            'patient_id': row.get('patient_id', ''),
            'full_name': row.get('full_name', ''),
            'date_of_birth': row.get('date_of_birth', ''),
            'admission_date': row.get('admission_date', ''),
            'discharge_date': row.get('discharge_date', ''),
            'sex': row.get('sex', ''),
            'service': row.get('service', ''),
            'attending': row.get('attending', ''),
            'unit_no': row.get('unit_no', '')
        }

        # Extract and validate sectioned data
        sections_mapping = {
            'chief_complaint': row.get('chief_complaint', ''),
            'history_present_illness_hopi': row.get('history_present_illness_hopi', ''),
            'past_medical_history_pmh': row.get('past_medical_history_pmh', ''),
            'medications_admission': row.get('medications_admission', ''),
            'medications_discharge': row.get('medications_discharge', ''),
            'brief_hospital_course': row.get('brief_hospital_course', ''),
            'pertinent_results': row.get('pertinent_results', ''),
            'social_history': row.get('social_history', ''),
            'family_history': row.get('family_history', ''),
            'discharge_diagnosis': row.get('discharge_diagnosis', ''),
            'discharge_instructions': row.get('discharge_instructions', ''),
            'follow_up_instructions': row.get('follow_up_instructions', ''),
            'full_report_text': row.get('full_report_text', '')
        }

        # Store field names for reference
        report['field_names'] = {k: k for k, v in sections_mapping.items() if v}

        # Clean and validate sections
        for section_name, content in sections_mapping.items():
            if content and str(content).strip():
                # Basic validation - check if content is not just whitespace
                cleaned_content = str(content).strip()
                if len(cleaned_content) > 10:  # Minimum meaningful content length
                    report['sections'][section_name] = cleaned_content
                else:
                    logging.warning(f"Section '{section_name}' has very short content, skipping")

        return report

    def create_csv_template(self, template_path: str, num_sample_rows: int = 3):
        """
        Create a CSV template file with expected medical report field names

        Args:
            template_path: Path where to save the template
            num_sample_rows: Number of sample rows to include
        """
        # Create template DataFrame with expected structure
        template_data = []
        sample_data = self._generate_sample_data()

        for i in range(num_sample_rows):
            row_data = {}
            for field in self.EXPECTED_FIELDS:
                if field in sample_data:
                    row_data[field] = sample_data[field][i % len(sample_data[field])]
                else:
                    row_data[field] = f"Sample {field.replace('_', ' ').title()}"
            template_data.append(row_data)

        df_template = pd.DataFrame(template_data)

        # Write to CSV with pipe delimiter
        df_template.to_csv(
            template_path,
            sep=self.delimiter,
            index=False,
            encoding='utf-8'
        )

        logging.info(f"CSV template created at: {template_path}")
        logging.info(f"Delimiter: {self.delimiter}")
        logging.info(f"Columns: {len(self.EXPECTED_FIELDS)}")

    def _generate_sample_data(self) -> Dict[str, List[str]]:
        """Generate sample medical report data"""
        return {
            'patient_id': ['PAT001', 'PAT002', 'PAT003', 'PAT004', 'PAT005'],
            'full_name': ['John Doe', 'Jane Smith', 'Robert Johnson', 'Mary Williams', 'David Wilson'],
            'date_of_birth': ['1980-01-15', '1975-06-20', '1990-03-10', '1965-11-30', '1982-08-12'],
            'admission_date': ['2024-01-01', '2024-01-15', '2024-02-01', '2024-01-20', '2024-02-10'],
            'discharge_date': ['2024-01-10', '2024-01-25', '2024-02-15', '2024-02-05', '2024-02-18'],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'service': ['Cardiology', 'Internal Medicine', 'Surgery', 'Neurology', 'Orthopedics'],
            'attending': ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis'],
            'unit_no': ['ICU-001', 'WARD-005', 'CCU-002', 'NEURO-003', 'ORTH-102'],
            'chief_complaint': [
                'Acute chest pain',
                'Shortness of breath',
                'Abdominal pain',
                'Severe headache',
                'Hip fracture'
            ],
            'history_present_illness_hopi': [
                'Patient reports severe chest pain starting 2 hours ago, radiating to left arm...',
                'Progressive dyspnea over past week, associated with cough...',
                'Sudden onset abdominal pain after eating, nausea and vomiting...',
                'Thunderclap headache reaching maximum intensity immediately...',
                'Patient slipped and fell, sustaining hip fracture...'
            ],
            'past_medical_history_pmh': [
                'Hypertension, Diabetes Mellitus Type 2, Coronary artery disease',
                'Asthma, GERD, Anxiety',
                'No significant past medical history',
                'Migraine headaches, Hypertension',
                'Osteoporosis, Vitamin D deficiency'
            ],
            'medications_admission': [
                'Lisinopril 10mg daily, Metformin 500mg BID, Aspirin 81mg daily',
                'Albuterol inhaler PRN, Omeprazole 20mg daily, Lorazepam 0.5mg PRN',
                'No home medications',
                'Ibuprofen 600mg PRN, Lisinopril 5mg daily',
                'Calcium 500mg daily, Vitamin D 1000 IU daily'
            ],
            'medications_discharge': [
                'Aspirin 81mg daily, Clopidogrel 75mg daily, Atorvastatin 40mg daily',
                'Metoprolol 25mg BID, Furosemide 20mg daily, Lisinopril 10mg daily',
                'Oxycodone 5mg q8hrs PRN pain, Cephalexin 500mg q6hrs',
                'Sumatriptan 100mg PRN, Propranolol 20mg BID',
                'Warfarin 5mg daily, Calcium supplements'
            ],
            'brief_hospital_course': [
                'Patient admitted with chest pain, started on anticoagulation protocol...',
                'Managed with bronchodilators, oxygen, gradually improved over 5 days...',
                'Initial workup showed no acute pathology, observation period...',
                'Started on triptans for headache control, Neurology consultation...',
                'Underwent hip replacement surgery, good postoperative course...'
            ],
            'pertinent_results': [
                'Troponin negative x2, EKG showed ST elevations, Echo: EF 35%',
                'CXR clear, ABG showed hypoxemia, PFTs: obstructive pattern',
                'CT abdomen negative for pathology, Labs unremarkable',
                'CT head negative for bleed, LP: opening pressure normal',
                'X-ray confirmed hip fracture, pre-op labs normal'
            ],
            'social_history': [
                'Former smoker, quit 5 years ago (20 pack-years), occasional alcohol',
                'Never smoker, moderate alcohol use (2 drinks/day), works in office',
                'Current smoker (1 pack/day), drinks heavily on weekends',
                'Never smoker, no alcohol use, teacher by occupation',
                'Former smoker, quit 10 years ago, minimal alcohol'
            ],
            'family_history': [
                'Father with CAD at age 60, Mother with diabetes, Brother with hypertension',
                'No family history of lung disease, Father with GERD',
                'No significant family history',
                'Brother with migraine headaches',
                'Mother with osteoporosis, Father normal'
            ],
            'discharge_diagnosis': [
                'Acute myocardial infarction, Heart failure with reduced EF',
                'Acute exacerbation of COPD, Respiratory failure',
                'Acute gastroenteritis, Dehydration',
                'Migraine headache, Cluster headache',
                'Hip fracture, Osteoarthritis'
            ],
            'discharge_instructions': [
                'Take medications as prescribed, follow low-sodium diet, no smoking',
                'Use inhaler as needed, pulmonary rehabilitation, stop smoking',
                'Avoid heavy meals, oral rehydration, BRAT diet',
                'Take triptans at first sign of headache, avoid triggers',
                'Weight bearing as tolerated, physical therapy as scheduled'
            ],
            'follow_up_instructions': [
                'Cardiology clinic in 3 days, Primary care in 1 week',
                'Pulmonology in 2 weeks, Primary care in 1 week',
                'Return if symptoms worsen, Primary care in 3 days',
                'Neurology in 1 week, PCP for medication refill',
                'Orthopedics in 2 weeks, Physical therapy Monday'
            ],
            'full_report_text': [
                'COMPETE PATIENT REPORT: John Doe, 44-year-old male admitted for acute chest pain... (full medical report follows)',
                'COMPETE PATIENT REPORT: Jane Smith, 48-year-old female with shortness of breath... (full medical report follows)',
                'COMPETE PATIENT REPORT: Robert Johnson, 34-year-old male admitted for abdominal pain... (full medical report follows)',
                'COMPETE PATIENT REPORT: Mary Williams, 58-year-old female with severe headache... (full medical report follows)',
                'COMPETE PATIENT REPORT: David Wilson, 42-year-old male with hip fracture... (full medical report follows)'
            ]
        }
        """Generate sample medical report data"""
        return {
            'patient_id': ['PAT001', 'PAT002', 'PAT003'],
            'full_name': ['John Doe', 'Jane Smith', 'Robert Johnson'],
            'date_of_birth': ['1980-01-15', '1975-06-20', '1990-03-10'],
            'admission_date': ['2024-01-01', '2024-01-15', '2024-02-01'],
            'discharge_date': ['2024-01-10', '2024-01-25', '2024-02-15'],
            'sex': ['M', 'F', 'M'],
            'service': ['Cardiology', 'Internal Medicine', 'Surgery'],
            'attending': ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams'],
            'unit_no': ['ICU-001', 'WARD-005', 'CCU-002'],
            'chief_complaint': [
                'Acute chest pain',
                'Shortness of breath',
                'Abdominal pain'
            ],
            'history_present_illness_hopi': [
                'Patient reports severe chest pain starting 2 hours ago...',
                'Progressive dyspnea over past week...',
                'Sudden onset abdominal pain after eating...'
            ],
            'past_medical_history_pmh': [
                'Hypertension, Diabetes Mellitus Type 2',
                'Asthma, GERD',
                'No significant past medical history'
            ],
            'medications_admission': [
                'Lisinopril 10mg daily, Metformin 500mg BID',
                'Albuterol inhaler PRN, Omeprazole 20mg daily',
                'No home medications'
            ],
            'medications_discharge': [
                'Aspirin 81mg daily, Clopidogrel 75mg daily',
                'Metoprolol 25mg BID, Furosemide 20mg daily',
                'Oxycodone 5mg q8hrs PRN pain'
            ],
            'brief_hospital_course': [
                'Patient admitted with chest pain, started on anticoagulation...',
                'Managed with bronchodilators, oxygen, gradually improved...',
                'Initial workup showed no acute pathology...'
            ],
            'pertinent_results': [
                'Troponin negative x2, EKG showed ST elevations',
                'CXR clear, ABG showed hypoxemia',
                'CT abdomen negative for pathology'
            ],
            'social_history': [
                'Former smoker, quit 5 years ago',
                'Moderate alcohol use, occasional smoking',
                'Never smoker, minimal alcohol'
            ],
            'family_history': [
                'Father with CAD, Mother with diabetes',
                'No family history of kidney disease',
                'Family history of cancer'
            ],
            'discharge_diagnosis': [
                'Acute myocardial infarction',
                'Acute exacerbation of COPD',
                'Acute gastroenteritis'
            ],
            'discharge_instructions': [
                'Take medications as prescribed, follow up in 1 week',
                'Use inhaler as needed, pulmonary rehabilitation',
                'Avoid heavy meals, oral rehydration'
            ],
            'follow_up_instructions': [
                'Cardiology clinic in 3 days',
                'Primary care physician in 2 weeks',
                'Return if symptoms worsen'
            ],
            'full_report_text': [
                'Complete patient report including all sections for KG generation...',
                'All medical information consolidated in final column...',
                'Comprehensive report for analysis and processing...'
            ]
        }


# Convenience function for creating template
def create_medical_csv_template(template_path: str = 'medical_reports_template.csv'):
    """
    Create a pipe-delimited CSV template for medical reports

    Args:
        template_path: Path for the template file
    """
    processor = MedicalReportCSVProcessor()
    processor.create_csv_template(template_path)


if __name__ == "__main__":
    # Example usage: create template and validate
    processor = MedicalReportCSVProcessor()

    # Create template
    processor.create_csv_template('medical_reports_template.csv')
    logging.info("Template created successfully")
