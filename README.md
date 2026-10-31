# MIDAS — Multimodal Intelligent Document Analysis System

A production-grade document intelligence pipeline that ingests scanned PDFs, photographed documents, and digital files — engineering specifications, invoices, inspection checklists — and extracts structured, validated JSON output using OCR, layout analysis, and a multimodal generative AI extraction layer.

Built around a real operational bottleneck: in large enterprises, critical information (vendor terms, equipment parameters, safety inspection results) lives in paper and unstructured files that should be in a database. MIDAS bridges that gap with an end-to-end automated pipeline.

---

## Key Features

- **Multi-Format Ingestion:** Accepts scanned PDFs, phone-photographed documents, digital PDFs, and raster images (JPG, PNG, TIFF). Pre-processes for skew, noise, shadow, and low contrast before OCR is called.
- **PaddleOCR v4 Engine:** Performs text detection and recognition with support for rotated text blocks, multi-column layouts, and embedded tables — outperforms Tesseract on complex document structures including handwritten annotations.
- **Layout-Aware Region Detection:** LayoutParser classifies each region (title, paragraph, table, figure, header/footer) before extraction, preventing text from table cells from bleeding into prose blocks.
- **Multimodal LLM Extraction Layer:** A structured prompt sent to Google Gemini Flash reads the OCR transcript and region map together, returning typed key-value pairs — handles abbreviations, field aliases, and contextual inference that rule-based parsers cannot.
- **Pydantic Schema Validation:** Every extracted document is validated against a typed schema (Invoice, TechnicalSpecification, InspectionReport). Low-confidence or missing required fields are flagged, not silently dropped.
- **SQLite Storage with Full-Text Search:** Extracted records are stored in a local SQLite database with FTS5 indexing — query across every extracted invoice line-item or specification parameter without opening a single JSON file.
- **Interactive Streamlit Dashboard:** Drag-and-drop document upload, real-time extraction trigger, field-level confidence display, and JSON export. Supports batch review workflows.

---

## Architecture Diagram

```
flowchart TD
    A["Raw Input (PDF / Scanned Image / Photo)"] --> B["Pre-processing Engine (src/preprocessing/enhance.py)"]
    B --> C["PaddleOCR v4 Engine (src/ocr/engine.py)"]
    C --> D["Layout Region Classifier (src/layout/detector.py)"]
    D --> E["Table Parser (src/extraction/table_parser.py)"]
    D --> F["Entity Extractor — LLM Layer (src/extraction/entity_extractor.py)"]
    E --> G["Pydantic Schema Validator (src/schema/validator.py)"]
    F --> G
    G --> H["SQLite + FTS5 Storage (src/storage/database.py)"]
    G --> I["JSON Output Files (outputs/extracted/)"]
    H --> J["Streamlit Dashboard (dashboard/app.py)"]
    I --> J
```

---

## Extraction Walkthrough: `INV-0047` (Supplier Invoice, Partially Scanned)

Input: A photographed A4 invoice taken at an 8° angle under uneven fluorescent lighting. Supplier name printed in a condensed typeface that Tesseract 4 consistently misreads.

1. **Pre-processing:** Deskew corrects the 8° tilt. Adaptive thresholding removes the lighting shadow from the top-left corner. Output: a clean 300 DPI binary image.
2. **OCR:** PaddleOCR detects 34 text regions. Supplier name correctly reads `TATA METALIKS LTD` — Tesseract produced `TATA MET/4LlKS LlD` on the same input.
3. **Layout Detection:** 6 region types classified: Title (1), Header blocks (2), Table (1), Footer blocks (2).
4. **Table Parser:** Extracts the 6-row line-item table with columns `Item`, `Qty`, `Unit Price`, `Total`. Grid reconstructed from contour spacing — the original scan has no visible grid lines.
5. **LLM Extraction:** Gemini Flash receives the OCR transcript and region labels. Returns `{"vendor": "TATA METALIKS LTD", "invoice_number": "INV-0047", "date": "2025-11-14", "line_items": [...], "total_amount": 148750.00, "currency": "INR"}`.
6. **Validation:** Pydantic `Invoice` schema passes. All required fields present. `total_amount` matches the sum of line-item totals within 0.01 tolerance (arithmetic cross-check).
7. **Storage:** Written to SQLite `invoices` table, FTS5 index updated, and `outputs/extracted/INV-0047.json` written to disk.

---

## Quick Start & Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
EXTRACTION_MODEL=gemini-1.5-flash
OCR_LANG=en
CONFIDENCE_THRESHOLD=0.75
```

### 3. Launch Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

In the sidebar you can:
- **Upload a document** via drag-and-drop, select its type (invoice / spec / inspection), and click **"Extract"**.
- **Browse extracted records** with full-text search across all stored documents.
- **Download JSON** for any extracted record with one click.

### 4. Run Pipeline via CLI (Alternative)

```bash
python src/main.py --input data/samples/invoices/INV-0047.jpg --doc-type invoice
```

### 5. Run Test Suite

```bash
pytest
```

---

## Repository Structure

```
MIDAS/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── PROJECT_JOURNAL.md
├── config/
│   ├── schemas/
│   │   ├── invoice.json
│   │   ├── technical_spec.json
│   │   └── inspection_report.json
│   └── pipeline.yaml
├── data/
│   ├── samples/
│   │   ├── invoices/
│   │   ├── specs/
│   │   └── inspection/
│   └── generated/
├── src/
│   ├── preprocessing/
│   │   └── enhance.py
│   ├── ocr/
│   │   └── engine.py
│   ├── layout/
│   │   └── detector.py
│   ├── extraction/
│   │   ├── table_parser.py
│   │   └── entity_extractor.py
│   ├── schema/
│   │   └── validator.py
│   ├── storage/
│   │   └── database.py
│   ├── pipeline/
│   │   └── runner.py
│   └── main.py
├── dashboard/
│   └── app.py
├── outputs/
│   └── extracted/
├── docs/
│   ├── architecture.md
│   ├── extraction-methodology.md
│   ├── schema-design.md
│   └── setup.md
├── tests/
│   ├── test_preprocessing.py
│   ├── test_schema_validation.py
│   ├── test_table_parser.py
│   └── test_pipeline_integration.py
└── .github/
    └── workflows/
        └── tests.yml
```

---

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Extraction Methodology](docs/extraction-methodology.md)
- [Schema Design](docs/schema-design.md)
- [Setup Guide](docs/setup.md)

---

## Disclaimer

The LLM extraction layer uses Google Gemini Flash with a structured output prompt. Extraction accuracy depends on OCR quality and document type. The confidence thresholds in `config/pipeline.yaml` are project-specific heuristics tuned on the sample dataset — they are not vendor-certified accuracy guarantees and should be recalibrated for production document types.
