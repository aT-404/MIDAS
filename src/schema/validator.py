"""MIDAS Schema Validation Layer.

Enforces strict typed contracts via Pydantic v2 schemas for all document classes.
Performs arithmetic cross-validation (grand total vs line-item sums) and
guards against hallucinated or empty structured parameter sets.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class LineItem(BaseModel):
    """Line item in a commercial invoice."""
    description: str = Field(..., min_length=1, description="Product or service line description")
    qty: float = Field(..., ge=0, description="Quantity")
    unit_price: float = Field(..., ge=0, description="Unit price per item")
    total: float = Field(..., ge=0, description="Computed total price for item")


class Invoice(BaseModel):
    """Supplier or vendor invoice schema with arithmetic validation."""
    vendor_name: str = Field(..., min_length=2, description="Supplier company name")
    vendor_address: Optional[str] = None
    invoice_number: str = Field(..., min_length=1, description="Unique invoice number")
    invoice_date: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    grand_total: float = Field(..., description="Total payable amount")
    currency: str = Field(default="INR", max_length=10)
    payment_terms: Optional[str] = None
    bank_details: Optional[str] = None
    _extraction_confidence: float = 0.0

    @field_validator("grand_total")
    @classmethod
    def total_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"grand_total must be strictly positive (got {v})")
        return round(v, 2)


class SpecParameter(BaseModel):
    """Technical specification parameter row."""
    name: str = Field(..., min_length=1, description="Parameter or rating name")
    value: str = Field(..., min_length=1, description="Rating or threshold value")
    unit: Optional[str] = None
    tolerance: Optional[str] = None


class TechnicalSpecification(BaseModel):
    """Equipment engineering data sheet schema."""
    equipment_name: str = Field(..., min_length=2, description="Machinery or equipment designation")
    model_number: str = Field(..., min_length=1, description="Model, part, or serial number")
    manufacturer: Optional[str] = None
    parameters: List[SpecParameter] = Field(..., min_length=1, description="Must contain at least 1 parameter")
    operating_conditions: Optional[Dict[str, Any]] = None
    certification_standards: List[str] = Field(default_factory=list)
    revision_date: Optional[str] = None
    _extraction_confidence: float = 0.0

    @field_validator("parameters")
    @classmethod
    def parameters_must_not_be_empty(cls, v: List[SpecParameter]) -> List[SpecParameter]:
        """A valid technical specification must contain at least one extracted parameter."""
        if not v or len(v) == 0:
            raise ValueError("Technical specification must contain at least 1 valid parameter row")
        return v


class ChecklistItem(BaseModel):
    """Quality and safety checklist inspection item."""
    item: str = Field(..., min_length=2, description="Inspection point description")
    status: str = Field(..., description="PASS / FAIL / NA")
    remarks: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in {"PASS", "FAIL", "NA"}:
            raise ValueError(f"Checklist status must be PASS, FAIL, or NA (got '{v}')")
        return upper


class InspectionReport(BaseModel):
    """Field inspection and quality audit report schema."""
    site_name: str = Field(..., min_length=2, description="Inspection site or facility")
    inspector_name: str = Field(..., min_length=2, description="Certified inspector name")
    inspection_date: str = Field(..., min_length=4, description="Date of inspection")
    checklist_items: List[ChecklistItem] = Field(..., min_length=1)
    deficiencies_found: Optional[str] = None
    corrective_actions: Optional[str] = None
    next_inspection_date: Optional[str] = None
    _extraction_confidence: float = 0.0

    @field_validator("checklist_items")
    @classmethod
    def checklist_must_not_be_empty(cls, v: List[ChecklistItem]) -> List[ChecklistItem]:
        if not v or len(v) == 0:
            raise ValueError("Inspection report must contain at least 1 checklist evaluation item")
        return v


SCHEMAS: Dict[str, Any] = {
    "invoice": Invoice,
    "technical_spec": TechnicalSpecification,
    "inspection_report": InspectionReport,
}


def cross_validate_invoice_totals(data: Dict[str, Any], tolerance: float = 0.05) -> Tuple[bool, str]:
    """Verify arithmetic consistency of invoice line items against grand total.

    grand_total should match subtotal + tax_amount (or sum(line_items) * (1 + tax))
    within the specified tolerance.
    """
    grand_total = data.get("grand_total")
    line_items = data.get("line_items", [])

    if not grand_total or not line_items:
        return True, "No line items to cross-validate"

    items_sum = sum(float(item.get("total", 0.0)) for item in line_items)
    if items_sum <= 0:
        return True, "Line item total is zero"

    # Direct match or match with tax
    diff_direct = abs(grand_total - items_sum)
    if diff_direct <= max(1.0, items_sum * tolerance):
        return True, f"Grand total matches line items sum ({items_sum:.2f})"

    # Check with tax_amount
    tax_amount = data.get("tax_amount", 0.0) or 0.0
    diff_tax = abs(grand_total - (items_sum + tax_amount))
    if diff_tax <= max(1.0, items_sum * tolerance):
        return True, f"Grand total matches line items sum + tax ({items_sum + tax_amount:.2f})"

    return False, f"Arithmetic mismatch: grand_total ({grand_total}) != line items sum ({items_sum:.2f})"


def validate(data: Dict[str, Any], doc_type: str) -> Tuple[bool, Union[BaseModel, str]]:
    """Validate raw extracted dictionary against typed Pydantic schema.

    Returns:
        (True, validated_model) if valid, or (False, error_message_string) if rejected.
    """
    schema_cls = SCHEMAS.get(doc_type)
    if not schema_cls:
        return False, f"Unknown document type '{doc_type}'. Supported: {list(SCHEMAS.keys())}"

    # Strip internal keys
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}

    try:
        model_instance = schema_cls(**clean_data)
        # Preserve confidence score
        if "_extraction_confidence" in data:
            model_instance._extraction_confidence = data["_extraction_confidence"]
        return True, model_instance
    except Exception as e:
        return False, str(e)
