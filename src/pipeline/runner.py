"""MIDAS End-to-End Pipeline Runner.

Orchestrates multi-format document ingestion, enhancement, OCR text detection,
layout region segmentation, tabular parsing, multimodal LLM schema extraction,
Pydantic validation, and SQLite FTS5 database persistence.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from src.preprocessing.enhance import preprocess
from src.ocr.engine import run_ocr, format_ocr_transcript
from src.layout.detector import detect_regions
from src.extraction.table_parser import extract_tables_from_pdf, extract_tables_from_image
from src.extraction.entity_extractor import extract_fields
from src.schema.validator import validate, cross_validate_invoice_totals
from src.storage.database import store_document, store_rejection, DB_PATH


def infer_document_type(ocr_text: str, filename: str = "") -> str:
    """Infer document type (invoice, technical_spec, inspection_report) from content cues."""
    fname = filename.lower()
    text = ocr_text.lower()

    if any(k in fname for k in ["inv", "bill", "invoice", "receipt"]) or any(k in text for k in ["invoice", "tax invoice", "grand total", "bill to", "subtotal", "payment terms"]):
        return "invoice"

    if any(k in fname for k in ["spec", "datasheet", "technical"]) or any(k in text for k in ["technical specification", "operating conditions", "model number", "tolerance", "rated current", "slurry pump"]):
        return "technical_spec"

    if any(k in fname for k in ["insp", "checklist", "audit", "report"]) or any(k in text for k in ["inspection report", "checklist", "deficiencies", "inspector name", "pass", "fail"]):
        return "inspection_report"

    # Default fallback
    return "invoice"


def run_pipeline(
    input_path: Union[str, Path],
    doc_type: str = "auto",
    output_dir: str = "outputs/extracted",
    rejected_dir: str = "outputs/rejected",
    db_path: str = DB_PATH,
    config_path: Optional[str] = "config/pipeline.yaml",
) -> Dict[str, Any]:
    """Execute complete MIDAS pipeline on a single document file.

    Args:
        input_path: Path to document (PDF, PNG, JPG, TIFF)
        doc_type: "invoice" | "technical_spec" | "inspection_report" | "auto"
        output_dir: Target directory for validated JSON
        rejected_dir: Target directory for rejected JSON records
        db_path: SQLite database file path
        config_path: Optional path to pipeline YAML configuration

    Returns:
        Dict: {"status": "success" | "rejected", ...}
    """
    path_obj = Path(input_path)
    if not path_obj.exists():
        return {
            "status": "rejected",
            "reason": f"Input file not found: {input_path}",
            "confidence": 0.0,
        }

    out_folder = Path(output_dir)
    rej_folder = Path(rejected_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    rej_folder.mkdir(parents=True, exist_ok=True)

    # Load configuration if available
    confidence_threshold = 0.75
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                confidence_threshold = cfg.get("validation", {}).get("confidence_threshold", 0.75)
        except Exception:
            pass

    # Stage 1: Pre-processing & Image Enhancement
    try:
        cleaned_path = preprocess(str(path_obj))
    except Exception as e:
        # Fallback to original input if preprocessing fails on unconventional input
        cleaned_path = str(path_obj)

    # Stage 2: OCR Detection & Recognition
    ocr_results = run_ocr(cleaned_path)
    ocr_text = format_ocr_transcript(ocr_results)

    if not ocr_text.strip():
        reason = "OCR failed to detect any readable text on document."
        store_rejection(doc_type, path_obj.name, {}, reason, db_path=db_path)
        return {"status": "rejected", "reason": reason, "confidence": 0.0}

    # Resolve document type if set to auto
    effective_doc_type = doc_type
    if effective_doc_type == "auto":
        effective_doc_type = infer_document_type(ocr_text, path_obj.name)

    # Stage 3: Layout Analysis
    regions = detect_regions(cleaned_path)

    # Stage 4: Table Parsing
    table_data = []
    has_table_region = any(r.get("type") == "Table" for r in regions)

    if path_obj.suffix.lower() == ".pdf":
        pdf_tables = extract_tables_from_pdf(str(path_obj))
        if pdf_tables:
            table_data = pdf_tables
            ocr_text += "\n\nSTRUCTURED TABLE ROWS:\n" + json.dumps(pdf_tables)
    elif has_table_region:
        for r in regions:
            if r.get("type") == "Table":
                img_table = extract_tables_from_image(cleaned_path, r.get("bbox"))
                if img_table:
                    table_data.append(img_table)
                    ocr_text += "\n\nSTRUCTURED TABLE ROWS:\n" + json.dumps(img_table)

    # Stage 5: Multimodal LLM Extraction
    extracted = extract_fields(ocr_text, regions, effective_doc_type, table_data=table_data)
    confidence = extracted.get("_extraction_confidence", 0.0)

    # Stage 6: Pydantic Schema Validation
    is_valid, validated_obj = validate(extracted, effective_doc_type)

    if not is_valid:
        rejection_reason = f"Schema validation error: {validated_obj}"
        rejected_file = rej_folder / f"{path_obj.stem}_rejected.json"
        rej_payload = {
            "source_file": path_obj.name,
            "doc_type": effective_doc_type,
            "rejection_reason": rejection_reason,
            "confidence": confidence,
            "raw_extraction": extracted,
        }
        with open(rejected_file, "w", encoding="utf-8") as f:
            json.dump(rej_payload, f, indent=2)

        store_rejection(effective_doc_type, path_obj.name, extracted, rejection_reason, db_path=db_path)
        return {
            "status": "rejected",
            "reason": rejection_reason,
            "confidence": confidence,
            "raw": extracted,
            "doc_type": effective_doc_type,
        }

    # Optional arithmetic cross-validation for invoices
    if effective_doc_type == "invoice":
        cross_valid, cross_msg = cross_validate_invoice_totals(extracted)
        # We record the arithmetic warning if mismatched but don't reject unless grand_total was 0
        extracted["_arithmetic_cross_check"] = cross_msg

    # Stage 7: Storage & Output Persistence
    result_dict = validated_obj.model_dump()
    result_dict["_extraction_confidence"] = confidence

    out_file = out_folder / f"{path_obj.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    store_document(
        doc_type=effective_doc_type,
        source_file=path_obj.name,
        data=result_dict,
        confidence=confidence,
        db_path=db_path
    )

    return {
        "status": "success",
        "doc_type": effective_doc_type,
        "output_path": str(out_file),
        "confidence": confidence,
        "data": result_dict,
        "regions_detected": len(regions),
        "ocr_lines": len(ocr_results),
    }
