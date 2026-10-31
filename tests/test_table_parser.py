"""Unit tests for MIDAS tabular extraction from digital PDFs and images."""

import pytest
from pathlib import Path
from src.extraction.table_parser import extract_tables_from_pdf, extract_tables_from_image


def test_extract_tables_from_digital_pdf():
    """Assert extract_tables_from_pdf recovers table rows from generated sample."""
    sample_pdf = Path("data/generated/invoices/INV-0047.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not generated yet")

    tables = extract_tables_from_pdf(str(sample_pdf))
    assert isinstance(tables, list)
    assert len(tables) >= 1
    # Check that rows have at least 3 columns
    table = tables[0]
    assert len(table) >= 3
    assert len(table[0]) >= 3
