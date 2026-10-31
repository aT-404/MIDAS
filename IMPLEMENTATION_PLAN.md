# MIDAS — Detailed Implementation Plan

This is the full, block-by-block build plan for the Multimodal Intelligent Document Analysis System. Each block lists exact files, configuration, code where meaningful, and a "Done when" test. No block starts until the previous one passes its test. Use this alongside `PROJECT_JOURNAL.md` for the narrative context behind each decision.

Dataset used: A three-class synthetic document set — 30 invoices, 20 technical specifications, and 15 inspection reports — generated via `data/scripts/generate_docs.py` using Faker + ReportLab. Supplemented by 20 real-world document samples from the SROIE benchmark and 10 from the CORD dataset (both publicly available for research).

---

## Block 0 — Environment Setup

**Goal:** A clean Python environment with every dependency installed, the repo skeleton in place, and a valid Gemini API key verified end-to-end before any OCR code is written.

Steps:

1. Create a Google AI Studio account at `aistudio.google.com`. Generate an API key for the Gemini API — the free tier includes Gemini 1.5 Flash at 15 requests per minute, which is enough for batch testing. Note the key.

2. Repo skeleton:

```
MIDAS/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── IMPLEMENTATION_PLAN.md
├── PROJECT_JOURNAL.md
├── config/
│   ├── schemas/        ← JSON schemas for each document type
│   └── pipeline.yaml   ← confidence thresholds and model settings
├── data/
│   ├── samples/        ← real and synthetic test documents
│   │   ├── invoices/
│   │   ├── specs/
│   │   └── inspection/
│   └── generated/      ← output of the generation script
├── src/
│   ├── preprocessing/
│   ├── ocr/
│   ├── layout/
│   ├── extraction/
│   ├── schema/
│   ├── storage/
│   ├── pipeline/
│   └── main.py
├── dashboard/
├── outputs/extracted/
├── docs/
└── tests/
```

3. `.env.example`:

```
GEMINI_API_KEY=your_api_key_here
EXTRACTION_MODEL=gemini-1.5-flash
OCR_LANG=en
CONFIDENCE_THRESHOLD=0.75
DB_PATH=outputs/midas.db
```

4. `requirements.txt`:

```
paddlepaddle
paddleocr
layoutparser[layoutmodels,tesseract]
camelot-py[cv]
google-generativeai
pydantic>=2.0
opencv-python-headless
Pillow
PyMuPDF
streamlit
faker
reportlab
pytest
python-dotenv
shapely
```

5. `config/pipeline.yaml`:

```yaml
ocr:
  lang: en
  use_angle_cls: true
  det_db_thresh: 0.3

layout:
  model: lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config
  threshold: 0.5

llm:
  model: gemini-1.5-flash
  temperature: 0.0
  max_tokens: 2048

validation:
  confidence_threshold: 0.75
  require_all_required_fields: true

storage:
  db_path: outputs/midas.db
  enable_fts: true
```

**Done when:** `python -c "import paddleocr; import google.generativeai as genai; genai.configure(api_key='your_key'); print('OK')"` runs without errors. A simple `genai.GenerativeModel('gemini-1.5-flash').generate_content('test').text` returns a non-empty string.

---

## Block 1 — Dataset Collection & Synthetic Document Generation

**Goal:** Build the document corpus that every downstream test will run against. Three document types, three difficulty levels each (clean digital, light scan, degraded scan/photo).

### 1a — Synthetic document generation

`data/scripts/generate_docs.py` uses Faker + ReportLab to produce realistic documents:

**Invoice schema (fields to include):**
- `vendor_name`, `vendor_address`, `invoice_number`, `invoice_date`
- `line_items: [{description, qty, unit_price, total}]`
- `subtotal`, `tax_rate`, `tax_amount`, `grand_total`
- `payment_terms`, `bank_details`

**TechnicalSpecification schema:**
- `equipment_name`, `model_number`, `manufacturer`
- `parameters: [{name, value, unit, tolerance}]`
- `operating_conditions: {temp_min, temp_max, voltage, current}`
- `certification_standards: [string]`
- `revision_date`

**InspectionReport schema:**
- `site_name`, `inspector_name`, `inspection_date`
- `checklist_items: [{item, status (PASS/FAIL/NA), remarks}]`
- `deficiencies_found`, `corrective_actions`, `next_inspection_date`

```python
# data/scripts/generate_docs.py — outline
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph

fake = Faker("en_IN")

def generate_invoice(output_path: str) -> dict:
    """Generate a synthetic invoice PDF and return the ground-truth dict."""
    ...

def generate_technical_spec(output_path: str) -> dict:
    ...

def generate_inspection_report(output_path: str) -> dict:
    ...

if __name__ == "__main__":
    for i in range(30):
        generate_invoice(f"data/generated/invoices/INV-{i:04d}.pdf")
    for i in range(20):
        generate_technical_spec(f"data/generated/specs/SPEC-{i:04d}.pdf")
    for i in range(15):
        generate_inspection_report(f"data/generated/inspection/INSP-{i:04d}.pdf")
```

The script saves both the PDF and a matching `*_ground_truth.json` file. These ground-truth files are what the test suite will compare extracted output against.

### 1b — Degradation simulation

For scanned/photographed realism, `data/scripts/degrade.py` adds:
- Random rotation: ±5–12°
- Gaussian noise: sigma 5–20
- Brightness variation: ±30%
- Simulated fold shadow: a dark diagonal stripe across one quadrant

This ensures the pre-processing pipeline is tested against realistic degraded inputs, not just clean digital PDFs.

### 1c — SROIE and CORD datasets

Download from their respective repositories:
- SROIE (Scanned Receipts OCR and Information Extraction): `github.com/zzzDavid/ICDAR-2019-SROIE`
- CORD (Consolidated Receipt Dataset): `github.com/clovaai/cord`

Place 10 SROIE samples under `data/samples/invoices/sroie/` and 10 CORD samples under `data/samples/invoices/cord/`. These are real-world scanned receipts that stress-test the preprocessing stage.

**Done when:** `python data/scripts/generate_docs.py` runs without errors and produces 65 PDF files across the three directories, each paired with a `_ground_truth.json`. Manual spot-check of 5 PDFs confirms readable content.

---

## Block 2 — MVP Slice (Single Invoice, End-to-End)

**Goal:** One invoice, fully extracted and written to JSON via the complete pipeline, before any of the advanced blocks are built. This is the most important block — if the LLM extraction doesn't produce usable output, there is no pipeline.

`src/ocr/engine.py` — minimal version:

```python
from paddleocr import PaddleOCR
from pathlib import Path

_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

def run_ocr(image_path: str) -> list[dict]:
    """Returns a list of {text, confidence, bbox} dicts."""
    result = _ocr.ocr(image_path, cls=True)
    extracted = []
    for line in result[0]:
        bbox, (text, confidence) = line
        extracted.append({"text": text, "confidence": float(confidence), "bbox": bbox})
    return extracted
```

`src/extraction/entity_extractor.py` — minimal version:

```python
import google.generativeai as genai
import json
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel(os.getenv("EXTRACTION_MODEL", "gemini-1.5-flash"))

INVOICE_PROMPT = """
You are a document extraction engine. Below is the OCR text from an invoice document.
Extract the following fields and return ONLY valid JSON. If a field is missing, use null.

Fields to extract:
- vendor_name (string)
- invoice_number (string)
- invoice_date (string, ISO 8601 if possible)
- line_items (array of {description, qty, unit_price, total})
- grand_total (number)
- currency (string, 3-letter code)

OCR TEXT:
{ocr_text}

Return only JSON, no explanation.
"""

def extract_invoice_fields(ocr_text: str) -> dict:
    prompt = INVOICE_PROMPT.format(ocr_text=ocr_text)
    response = _model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)
```

`src/main.py` — temporary MVP wiring:

```python
from src.ocr.engine import run_ocr
from src.extraction.entity_extractor import extract_invoice_fields
import json

def process_invoice(image_path: str, output_path: str):
    ocr_results = run_ocr(image_path)
    ocr_text = " ".join(r["text"] for r in ocr_results)
    extracted = extract_invoice_fields(ocr_text)
    with open(output_path, "w") as f:
        json.dump(extracted, f, indent=2)
    print(f"Extracted to {output_path}")

if __name__ == "__main__":
    process_invoice(
        "data/samples/invoices/INV-0047.jpg",
        "outputs/extracted/INV-0047.json"
    )
```

**Done when:** `python src/main.py` runs on one sample invoice and produces a JSON file with at least `vendor_name`, `invoice_number`, and `grand_total` correctly populated. Check by opening the output JSON — the values should match what is visible in the document.

---

## Block 3 — Pre-processing Pipeline

**Goal:** Images that are skewed, noisy, or low-contrast should pass through pre-processing and come out as clean binary images that PaddleOCR can read reliably. The pipeline must be robust enough that the "degraded" test documents from Block 1b produce OCR accuracy within 10% of the clean-scan results.

`src/preprocessing/enhance.py`:

```python
import cv2
import numpy as np
from PIL import Image

def deskew(image: np.ndarray) -> np.ndarray:
    """Correct image rotation using Hough line transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return image
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 20:
            angles.append(angle)
    if not angles:
        return image
    median_angle = float(np.median(angles))
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

def preprocess(image_path: str) -> str:
    """Full preprocessing pipeline. Returns path to the cleaned image."""
    image = cv2.imread(image_path)
    image = deskew(image)
    image = denoise(image)
    binary = binarize(image)
    out_path = image_path.replace("data/", "data/preprocessed/")
    cv2.imwrite(out_path, binary)
    return out_path
```

**Done when:** Running `preprocess()` on the 5 most degraded samples from Block 1b and then OCR-ing the output produces text that is readable by eye — no garbled single characters where whole words should appear. Specifically, the supplier/vendor name field should parse correctly in all 5 cases.

---

## Block 4 — Layout Analysis & Table Extraction

**Goal:** Know what type each region is before sending text to the LLM. A table cell value sent as part of flowing prose will confuse the extraction prompt. Layout-aware separation fixes this.

`src/layout/detector.py`:

```python
import layoutparser as lp
import cv2

_model = lp.Detectron2LayoutModel(
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)

def detect_regions(image_path: str) -> list[dict]:
    """Returns a list of {type, bbox, text_blocks} for each detected layout region."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    layout = _model.detect(image_rgb)
    regions = []
    for block in layout:
        regions.append({
            "type": block.type,
            "bbox": [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2],
            "score": float(block.score),
        })
    return regions
```

`src/extraction/table_parser.py`:

Two modes, selected based on input type:

**Digital PDFs** — use Camelot:
```python
import camelot

def extract_tables_from_pdf(pdf_path: str) -> list[list[list[str]]]:
    tables = camelot.read_pdf(pdf_path, flavor="lattice", pages="all")
    return [t.df.values.tolist() for t in tables]
```

**Scanned images** — use OpenCV contour grid detection:
```python
def extract_tables_from_image(image_path: str, table_bbox: list) -> list[list[str]]:
    """Crops the table region, finds cell contours, and reads text per cell via PaddleOCR."""
    ...
```

The image-based extractor crops the table bounding box from the layout detector output, finds horizontal and vertical lines via morphological operations, intersects them to get cell boundaries, and OCRs each cell independently. This handles tables with no visible grid lines by detecting whitespace gaps instead.

**Done when:** Running both extractors on one clean digital PDF invoice and one scanned invoice returns a 2D list where each row is a line item. Spot-checking 3 rows manually confirms values match the source document.

---

## Block 5 — Multimodal LLM Extraction Layer (Full Version)

**Goal:** The MVP slice from Block 2 used a single prompt for invoices. This block builds the full extraction layer: separate, well-engineered prompts for each document type, fallback handling, and per-field confidence scoring.

`src/extraction/entity_extractor.py` — extended:

```python
PROMPTS = {
    "invoice": INVOICE_PROMPT,
    "technical_spec": TECHNICAL_SPEC_PROMPT,
    "inspection_report": INSPECTION_REPORT_PROMPT,
}

TECHNICAL_SPEC_PROMPT = """
You are a document extraction engine reading an industrial technical specification.
Extract these fields and return ONLY valid JSON. Use null for missing fields.

Fields:
- equipment_name (string)
- model_number (string)
- manufacturer (string)
- parameters (array of {name, value, unit, tolerance})
- operating_conditions ({temp_min_c, temp_max_c, input_voltage_v, rated_current_a})
- certification_standards (array of strings, e.g. ["ISO 9001", "IEC 61000"])
- revision_date (string)

OCR TEXT:
{ocr_text}

DETECTED REGIONS (layout):
{region_summary}

Return only JSON.
"""

def extract_fields(ocr_text: str, regions: list[dict], doc_type: str) -> dict:
    prompt = PROMPTS[doc_type].format(
        ocr_text=ocr_text,
        region_summary="\n".join(f"- {r['type']}: {r['score']:.2f}" for r in regions),
    )
    try:
        response = _model.generate_content(prompt)
        raw = _clean_response(response.text)
        result = json.loads(raw)
        result["_extraction_confidence"] = _estimate_confidence(result, doc_type)
        return result
    except (json.JSONDecodeError, Exception) as e:
        return {"_error": str(e), "_extraction_confidence": 0.0}

def _estimate_confidence(result: dict, doc_type: str) -> float:
    """Count non-null required fields and return fill-rate as confidence."""
    required_fields = {
        "invoice": ["vendor_name", "invoice_number", "invoice_date", "grand_total"],
        "technical_spec": ["equipment_name", "model_number", "parameters"],
        "inspection_report": ["site_name", "inspector_name", "inspection_date", "checklist_items"],
    }
    required = required_fields.get(doc_type, [])
    if not required:
        return 1.0
    filled = sum(1 for f in required if result.get(f) not in (None, [], ""))
    return filled / len(required)
```

**Done when:** Running the extractor on 5 invoices, 5 specs, and 5 inspection reports returns structured JSON for all 15. The `_extraction_confidence` score is ≥ 0.75 for at least 12 of the 15. The 3 that fail (if any) should be the most degraded documents, not the clean ones.

---

## Block 6 — Pydantic Schema Validation

**Goal:** Every extracted record is validated against a typed schema before it touches the database. Invalid records are written to `outputs/rejected/` with the reason attached — never silently dropped.

`src/schema/validator.py`:

```python
from pydantic import BaseModel, field_validator
from typing import Optional
import json

class LineItem(BaseModel):
    description: str
    qty: float
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: Optional[str] = None
    line_items: list[LineItem] = []
    grand_total: float
    currency: str = "INR"
    _extraction_confidence: float = 0.0

    @field_validator("grand_total")
    @classmethod
    def total_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("grand_total must be positive")
        return v

class SpecParameter(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    tolerance: Optional[str] = None

class TechnicalSpecification(BaseModel):
    equipment_name: str
    model_number: str
    manufacturer: Optional[str] = None
    parameters: list[SpecParameter]
    revision_date: Optional[str] = None

class ChecklistItem(BaseModel):
    item: str
    status: str  # PASS / FAIL / NA
    remarks: Optional[str] = None

class InspectionReport(BaseModel):
    site_name: str
    inspector_name: str
    inspection_date: str
    checklist_items: list[ChecklistItem]
    deficiencies_found: Optional[str] = None
    next_inspection_date: Optional[str] = None

SCHEMAS = {
    "invoice": Invoice,
    "technical_spec": TechnicalSpecification,
    "inspection_report": InspectionReport,
}

def validate(data: dict, doc_type: str) -> tuple[bool, object | str]:
    schema = SCHEMAS.get(doc_type)
    if not schema:
        return False, f"Unknown doc_type: {doc_type}"
    try:
        validated = schema(**data)
        return True, validated
    except Exception as e:
        return False, str(e)
```

**Done when:** Feeding a known-good extraction through `validate()` returns `(True, model_instance)`. Feeding a deliberately broken extraction (missing required `vendor_name`) returns `(False, "error message")`. Both cases tested manually before wiring into the pipeline.

---

## Block 7 — SQLite Storage & Full-Text Search

**Goal:** All validated extractions land in a local SQLite database with FTS5 enabled. The dashboard's search bar queries this store.

`src/storage/database.py`:

```python
import sqlite3
import json
from pathlib import Path

DB_PATH = "outputs/midas.db"

def init_db(db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            source_file TEXT NOT NULL,
            extracted_json TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
        USING fts5(doc_type, extracted_json, content='documents', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, doc_type, extracted_json)
            VALUES (new.id, new.doc_type, new.extracted_json);
        END;
    """)
    con.commit()
    con.close()

def store_document(doc_type: str, source_file: str, data: dict, confidence: float, db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO documents (doc_type, source_file, extracted_json, confidence) VALUES (?, ?, ?, ?)",
        (doc_type, source_file, json.dumps(data), confidence),
    )
    con.commit()
    con.close()

def search(query: str, doc_type: str = None, db_path: str = DB_PATH) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    if doc_type:
        rows = con.execute(
            "SELECT d.* FROM documents d JOIN documents_fts f ON d.id = f.rowid WHERE documents_fts MATCH ? AND d.doc_type = ?",
            (query, doc_type),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT d.* FROM documents d JOIN documents_fts f ON d.id = f.rowid WHERE documents_fts MATCH ?",
            (query,),
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]
```

**Done when:** Running `init_db()` once creates the database. Storing 5 extracted invoices then running `search("TATA METALIKS")` returns the matching record. `search("4625")` on a spec database returns no results (confirming FTS is not cross-contaminating document types when filtered).

---

## Block 8 — Pipeline Runner & Full Integration

**Goal:** Wire Blocks 3–7 into a single `src/pipeline/runner.py` that accepts any document and document type, runs the full stack, and writes the output to both JSON and SQLite.

`src/pipeline/runner.py`:

```python
from src.preprocessing.enhance import preprocess
from src.ocr.engine import run_ocr
from src.layout.detector import detect_regions
from src.extraction.table_parser import extract_tables_from_pdf, extract_tables_from_image
from src.extraction.entity_extractor import extract_fields
from src.schema.validator import validate
from src.storage.database import store_document
import json, os
from pathlib import Path

def run(input_path: str, doc_type: str, output_dir: str = "outputs/extracted/") -> dict:
    # Step 1: Pre-process
    cleaned_path = preprocess(input_path)

    # Step 2: OCR
    ocr_results = run_ocr(cleaned_path)
    ocr_text = " ".join(r["text"] for r in ocr_results)

    # Step 3: Layout
    regions = detect_regions(cleaned_path)

    # Step 4: Table extraction (adds structured table text to ocr_text if table region found)
    has_table = any(r["type"] == "Table" for r in regions)
    if has_table and input_path.endswith(".pdf"):
        tables = extract_tables_from_pdf(input_path)
        ocr_text += "\n\nTABLE DATA:\n" + json.dumps(tables)

    # Step 5: LLM extraction
    extracted = extract_fields(ocr_text, regions, doc_type)
    confidence = extracted.pop("_extraction_confidence", 0.0)

    # Step 6: Validate
    valid, result = validate(extracted, doc_type)
    if not valid:
        rejected_path = Path("outputs/rejected") / Path(input_path).stem + "_rejected.json"
        with open(rejected_path, "w") as f:
            json.dump({"source": input_path, "error": result, "raw": extracted}, f, indent=2)
        return {"status": "rejected", "reason": result}

    # Step 7: Store
    result_dict = result.model_dump()
    source_name = Path(input_path).name
    out_path = Path(output_dir) / (Path(input_path).stem + ".json")
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    store_document(doc_type, source_name, result_dict, confidence)

    return {"status": "success", "output": str(out_path), "confidence": confidence}
```

**Done when:** Running `run("data/samples/invoices/INV-0047.jpg", "invoice")` on 5 different invoices produces 5 JSON files and 5 database records. Running it twice on the same file does not create a duplicate database entry (add a unique constraint check on `source_file` to enforce idempotency). Running it on a deliberately corrupt image returns `{"status": "rejected", ...}` instead of crashing.

---

## Block 9 — Dashboard

**Goal:** A Streamlit app that lets anyone upload a document, see extracted fields with confidence scores, and search the stored extraction history.

`dashboard/app.py` — layout:

- **Sidebar:** Document type selector (Invoice / Technical Spec / Inspection Report), file uploader (drag-and-drop, accepts PDF and image), "Extract Document" button.
- **Main — Upload Tab:**
  - After extraction: table of `{Field, Value, Confidence}` rows. Low-confidence fields highlighted in orange.
  - "Download JSON" button.
  - Rejected documents show the validation error clearly.
- **Main — Search Tab:**
  - Text input for full-text search query.
  - Optional doc-type filter.
  - Results shown as expandable JSON cards with source file name and extraction timestamp.
- **Main — Stats Tab:**
  - Total documents extracted (by type).
  - Average confidence per document type.
  - Rejection rate.

Build this last — the pipeline is the deliverable, the dashboard is the demonstration layer.

---

## Block 10 — Tests, Docs, and Polish

**Tests** (`tests/`):

- `test_preprocessing.py` — assert that `deskew()` on a 10°-rotated image returns an image within ±1° of vertical.
- `test_schema_validation.py` — feed known-good and known-bad dicts into `validate()`. Assert `Invoice` rejects negative `grand_total`. Assert `TechnicalSpecification` requires `equipment_name`.
- `test_table_parser.py` — feed a known digital PDF into `extract_tables_from_pdf()` and assert the returned 2D list has the expected row count.
- `test_pipeline_integration.py` — run the full `runner.run()` on 3 clean synthetic documents (one per type). Assert all 3 return `status: success` and the output JSON is non-empty.

**Docs** (`docs/`):

- `architecture.md` — the flowchart from the README, plus a one-paragraph explanation of each stage and why it exists in that order.
- `extraction-methodology.md` — how PaddleOCR was selected over Tesseract, how the LLM prompt is structured, how confidence is estimated, and what happens when extraction fails.
- `schema-design.md` — each Pydantic schema, field descriptions, and the reasoning behind required vs optional fields.
- `setup.md` — Google AI Studio API key setup, PaddleOCR installation notes (it downloads model weights on first run — needs internet), LayoutParser Detectron2 installation, running the synthetic document generator.

**README:** Problem statement → architecture diagram → INV-0047 walkthrough → quick start → documentation links → disclaimer.

**GitHub Actions** (`.github/workflows/tests.yml`) — run `pytest` on Python 3.11 for every push and pull request. Add this after all tests pass locally.

**Done when:** `pytest` returns 100% pass rate. The Streamlit dashboard runs cleanly, shows the INV-0047 extraction, and the FTS search returns the correct record for a vendor name query.

---

## Dataset Coverage Summary

| Document | Type | Difficulty | Expected Extraction Confidence |
|---|---|---|---|
| `INV-0047.jpg` | Invoice (photographed) | Medium (angled, shadow) | ≥ 0.80 |
| `SPEC-0003.pdf` | Technical Spec (digital) | Easy (clean) | ≥ 0.90 |
| `INSP-0011.jpg` | Inspection Report (scanned) | Hard (low contrast) | ≥ 0.70 |
| SROIE samples | Invoice (real scanned receipts) | Medium–Hard | ≥ 0.75 average |
| CORD samples | Invoice (complex layout) | Hard | ≥ 0.65 average |

The CORD samples are deliberately included to show the limits of the system — complex overlapping layout zones are an honest failure mode, and the rejection pipeline ensures they are surfaced rather than silently producing wrong data.
