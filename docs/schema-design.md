# MIDAS Schema Design Guide

MIDAS uses strict Pydantic v2 data models to enforce schema compliance before persistence.

---

## 1. Invoice (`src/schema/validator.py`)

Commercial invoice schema designed for industrial procurement.

```python
class LineItem(BaseModel):
    description: str
    qty: float
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor_name: str
    vendor_address: Optional[str] = None
    invoice_number: str
    invoice_date: Optional[str] = None
    line_items: List[LineItem] = []
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    grand_total: float
    currency: str = "INR"
    payment_terms: Optional[str] = None
    bank_details: Optional[str] = None
```

### Constraints & Validators
- `grand_total`: Must be strictly positive ($> 0.00$).
- `cross_validate_invoice_totals`: Verifies whether $\text{grand\_total} \approx \text{subtotal} + \text{tax\_amount}$ within a 5% tolerance margin.

---

## 2. Technical Specification (`src/schema/validator.py`)

Engineering specification schema for machinery, valves, pumps, and electrical systems.

```python
class SpecParameter(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    tolerance: Optional[str] = None

class TechnicalSpecification(BaseModel):
    equipment_name: str
    model_number: str
    manufacturer: Optional[str] = None
    parameters: List[SpecParameter]
    operating_conditions: Optional[Dict[str, Any]] = None
    certification_standards: List[str] = []
    revision_date: Optional[str] = None
```

### Constraints & Validators
- `parameters`: Enforces `min_length=1`. A specification with 0 parameters is invalid and routed to the rejection queue.

---

## 3. Inspection Report (`src/schema/validator.py`)

Quality assurance and site inspection checklist schema.

```python
class ChecklistItem(BaseModel):
    item: str
    status: str  # PASS / FAIL / NA
    remarks: Optional[str] = None

class InspectionReport(BaseModel):
    site_name: str
    inspector_name: str
    inspection_date: str
    checklist_items: List[ChecklistItem]
    deficiencies_found: Optional[str] = None
    corrective_actions: Optional[str] = None
    next_inspection_date: Optional[str] = None
```

### Constraints & Validators
- `status`: Must strictly belong to the set `{"PASS", "FAIL", "NA"}`.
- `checklist_items`: Enforces `min_length=1`.
