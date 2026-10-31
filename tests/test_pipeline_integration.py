"""Integration tests for complete MIDAS document analysis pipeline."""

import pytest
import os
from pathlib import Path
from src.pipeline.runner import run_pipeline


def test_pipeline_invoice_integration():
    """Assert pipeline runs end-to-end on synthetic invoice."""
    sample_path = Path("data/generated/invoices/INV-0047.pdf")
    if not sample_path.exists():
        pytest.skip("Test sample not found")

    result = run_pipeline(str(sample_path), doc_type="invoice")
    assert result["status"] == "success"
    assert result["confidence"] >= 0.75
    assert os.path.exists(result["output_path"])
    assert result["data"]["vendor_name"] is not None


def test_pipeline_spec_integration():
    """Assert pipeline runs end-to-end on synthetic technical specification."""
    sample_path = Path("data/generated/specs/SPEC-0003.pdf")
    if not sample_path.exists():
        pytest.skip("Test sample not found")

    result = run_pipeline(str(sample_path), doc_type="technical_spec")
    assert result["status"] == "success"
    assert result["confidence"] >= 0.75
    assert len(result["data"]["parameters"]) >= 1


def test_pipeline_missing_file_rejection():
    """Assert non-existent file path produces rejection status without unhandled crash."""
    result = run_pipeline("data/non_existent_file.pdf", doc_type="invoice")
    assert result["status"] == "rejected"
    assert "not found" in result["reason"].lower()
