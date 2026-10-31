"""Unit tests for MIDAS Pydantic schema validation contracts."""

import pytest
from src.schema.validator import (
    validate,
    cross_validate_invoice_totals,
    Invoice,
    TechnicalSpecification,
    InspectionReport,
)


def test_valid_invoice_schema():
    """Assert valid invoice dictionary passes validation."""
    data = {
        "vendor_name": "TATA METALIKS LTD",
        "invoice_number": "INV-0047",
        "invoice_date": "2026-06-15",
        "line_items": [
            {"description": "Hydraulic Valve", "qty": 2.0, "unit_price": 5000.0, "total": 10000.0}
        ],
        "grand_total": 11800.0,
        "currency": "INR"
    }
    valid, result = validate(data, "invoice")
    assert valid is True
    assert isinstance(result, Invoice)
    assert result.vendor_name == "TATA METALIKS LTD"
    assert result.grand_total == 11800.0


def test_invoice_rejects_negative_or_zero_total():
    """Assert invoice validation fails when grand_total is negative or zero."""
    data = {
        "vendor_name": "TATA METALIKS LTD",
        "invoice_number": "INV-0047",
        "grand_total": -500.0,
    }
    valid, result = validate(data, "invoice")
    assert valid is False
    assert "grand_total must be strictly positive" in str(result)


def test_technical_spec_requires_parameters():
    """Assert technical specification rejects empty parameters list."""
    # Empty parameters should raise validation error (Journal Entry 10)
    data = {
        "equipment_name": "Centrifugal Slurry Pump",
        "model_number": "CSP-450X",
        "parameters": []
    }
    valid, result = validate(data, "technical_spec")
    assert valid is False


def test_inspection_report_validates_status_enum():
    """Assert checklist status must strictly be PASS, FAIL, or NA."""
    data = {
        "site_name": "Hazira Complex",
        "inspector_name": "Rajesh Sharma",
        "inspection_date": "2026-01-15",
        "checklist_items": [
            {"item": "Hydrostatic pressure test", "status": "MAYBE"}
        ]
    }
    valid, result = validate(data, "inspection_report")
    assert valid is False
    assert "status must be PASS, FAIL, or NA" in str(result)


def test_cross_validate_arithmetic_consistency():
    """Assert arithmetic cross-check detects total discrepancies."""
    data = {
        "grand_total": 1000.0,
        "line_items": [
            {"total": 400.0},
            {"total": 500.0}
        ]
    }
    valid, msg = cross_validate_invoice_totals(data, tolerance=0.01)
    assert valid is False
    assert "Arithmetic mismatch" in msg
