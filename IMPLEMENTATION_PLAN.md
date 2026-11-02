# MIDAS — Implementation Plan

**Multimodal Intelligent Document Analysis System**  
End-to-end OCR, layout-aware extraction, and structured data validation for scanned and photographed industrial documents.

---

## How to read this plan

Each block represents a self-contained build phase. Blocks were implemented roughly in order; later blocks depend on earlier ones. Each block records:

- **Goal** — what it adds to the system
- **Files** — exact paths created or changed
- **Key implementation** — the core design decision with a representative code excerpt
- **Done when** — the concrete check that confirmed it worked

---

## Block 0 — Repo skeleton, config, and environment

**Goal:** Establish project layout, dependency manifest, and local environment config before writing any module code.

**Files:**
```
requirements.txt
config/pipeline.yaml
config/schemas/invoice.json
config/schemas/technical_spec.json
config/schemas/inspection_report.json
.env.example
.gitignore
```

**Key implementation:**

`config/pipeline.yaml` centralises all tuneable parameters so changing a threshold doesn't require touching module code:

```yaml
ocr:
  engine: "rapidocr"
  lang: "en"
  fallback: "paddleocr"

layout:
  model: "contour_heuristic"

llm:
  model_name: "gemini-1.5-flash"
  temperature: 0.1
  max_tokens: 2048

validation:
  confidence_threshold: 0.75
  invoice_arithmetic_tolerance: 0.05

storage:
  db_path: "outputs/midas.db"
  fts_enabled: true
```

**Done when:** `python -c "import yaml; yaml.safe_load(open('config/pipeline.yaml'))"` completes without errors and all dependency packages install cleanly from `requirements.txt`.

---

## Block 1 — Synthetic document corpus generation

**Goal:** Build a deterministic ground-truth corpus of 65 industrial documents (30 invoices, 20 specifications, 15 inspection reports) to drive all downstream testing without requiring real enterprise data.

**Files:**
```
data/scripts/generate_docs.py
data/generated/invoices/INV-0001.pdf … INV-0030.pdf
data/generated/specs/SPEC-0001.pdf … SPEC-0020.pdf
data/generated/inspection/INSP-0001.pdf … INSP-0015.pdf
data/generated/invoices/INV-0001_ground_truth.json … (and so on)
```

**Key implementation:**

Vendors and part names drawn from real industrial supply chains (L&T, TATA METALIKS, BHARAT FORGE, etc.) to ensure OCR vocabulary is representative of actual deployment targets:

```python
VENDORS = [
    "TATA METALIKS LTD", "L&T HEAVY ENGINEERING", "BHARAT FORGE LTD",
    "KIRLOSKAR BROTHERS LTD", "THERMAX INDIA LTD",
]

PART_NAMES = [
    "Hydraulic Control Valve", "Slurry Pump Casing", "Industrial Ball Bearing",
    "Centrifugal Pump Impeller", "High-Pressure Hose Assembly",
]

def generate_invoice() -> dict:
    vendor = random.choice(VENDORS)
    items = [
        {
            "description": random.choice(PART_NAMES),
            "qty": random.randint(1, 20),
            "unit_price": round(random.uniform(500, 15000), 2),
        }
        for _ in range(random.randint(3, 8))
    ]
    for item in items:
        item["total"] = round(item["qty"] * item["unit_price"], 2)
    subtotal = sum(i["total"] for i in items)
    tax_rate = random.choice([0.05, 0.09, 0.12, 0.18])
    tax_amount = round(subtotal * tax_rate, 2)
    grand_total = round(subtotal + tax_amount, 2)
    return { "vendor_name": vendor, "line_items": items,
             "subtotal": subtotal, "tax_rate": tax_rate,
             "tax_amount": tax_amount, "grand_total": grand_total }
```

Each generated PDF has a paired `_ground_truth.json` to enable precision/recall evaluation.

**Done when:** `python data/scripts/generate_docs.py` produces 65 PDFs and 65 matching ground truth JSON files in `data/generated/`.

---

## Block 2 — Degradation simulator and test samples

**Goal:** Transform clean generated PDFs into realistic smartphone captures and poor-quality scans for preprocessing stress tests.

**Files:**
```
data/scripts/degrade.py
data/samples/invoices/INV-0047.jpg
data/samples/specs/SPEC-0003.pdf
data/samples/inspection/INSP-0011.jpg
```

**Key implementation:**

Three degradation stages applied sequentially to simulate a real-world smartphone photo: skew rotation, diagonal shadow gradient, and Gaussian sensor noise:

```python
def add_shadow(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    # Shadow darkest at top-left corner, falls off exponentially
    shadow_mask = 1.0 - (intensity * np.exp(-dist * 1.8))
    shadow_mask = np.clip(shadow_mask, 0.2, 1.0)
    degraded = image.astype(np.float32)
    for c in range(3):
        degraded[:, :, c] *= shadow_mask
    return np.clip(degraded, 0, 255).astype(np.uint8)

def degrade_document(input_pdf_or_img, output_path,
                     rotation_angle=8.0, add_shadow_effect=True,
                     noise_sigma=10.0, dpi=200):
    # rasterize PDF → rotate → shadow → noise → lens blur
    ...
    degraded = add_rotation(img, angle=rotation_angle)
    if add_shadow_effect:
        degraded = add_shadow(degraded, intensity=0.55)
    if noise_sigma > 0:
        degraded = add_noise(degraded, sigma=noise_sigma)
    degraded = cv2.GaussianBlur(degraded, (3, 3), 0.5)
    cv2.imwrite(out_str, degraded)
```

INV-0047 receives 8° skew + shadow + noise (smartphone photo scenario). INSP-0011 receives -4° skew + heavy noise only (poor flatbed scan scenario). SPEC-0003 is kept as a clean digital PDF for the baseline comparison.

**Done when:** `python data/scripts/degrade.py` produces three files in `data/samples/` with visually distinct degradation patterns.

---

## Block 3 — Document preprocessing pipeline

**Goal:** Implement the image enhancement pipeline that normalizes scanned and photographed documents before OCR: deskewing, shadow removal, denoising, and binarization.

**Files:**
```
src/preprocessing/enhance.py
outputs/preprocessed/   (created at runtime)
```

**Key implementation:**

Deskew uses the probabilistic Hough line transform with a length filter (`min_line_len_ratio`) to exclude checkbox tick marks and short table dividers that otherwise bias the angle estimator:

```python
def deskew(image, max_skew_angle=15.0, min_line_len_ratio=0.25):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Require line segments longer than 25% of image width
    min_line_length = int(w * min_line_len_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=min_line_length, maxLineGap=20)

    angles = []
    for line in (lines or []):
        x1, y1, x2, y2 = line.flatten()
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_skew_angle:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.2:   # skip micro-rotations to avoid blur
        return image

    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```

Shadow removal uses morphological dilation to estimate the background illumination surface and subtract it from the document:

```python
def remove_shadow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    background = cv2.medianBlur(background, 21)
    diff = 255 - cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
```

**Done when:** Passing `data/samples/invoices/INV-0047.jpg` through `preprocess()` writes a visually clean, upright, binarized PNG to `outputs/preprocessed/` and returns the output path.

---

## Block 4 — OCR engine (singleton, dual-backend)

**Goal:** Implement a singleton OCR wrapper that tries RapidOCR (PP-OCRv4 via ONNX) first and falls back to PaddleOCR, producing a uniform output schema regardless of which backend runs.

**Files:**
```
src/ocr/engine.py
```

**Key implementation:**

The singleton pattern ensures the heavy model weights load once per process. Backend selection is transparent to all callers:

```python
class OCREngine:
    _instance = None
    _engine = None
    _engine_type = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._engine_type = "rapidocr"
            return
        except Exception:
            pass
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._engine_type = "paddleocr"
        except Exception:
            self._engine_type = "none"
```

Results are sorted into reading order (top-to-bottom, left-to-right) with a 15-pixel y-band to group text on the same visual line:

```python
results.sort(key=lambda item: (item["center"][1] // 15, item["center"][0]))
```

**Done when:** `OCREngine().process("data/samples/invoices/INV-0047.jpg")` returns a non-empty list of dicts with `text`, `confidence`, `bbox`, and `center` keys. Calling `OCREngine()` a second time in the same process returns the same instance (singleton confirmed by `is` identity check).

---

## Block 5 — Layout detection and region classification

**Goal:** Identify structural regions in the document image (title, header, table, text body, footer) so the extraction layer receives contextual region labels rather than a flat text dump.

**Files:**
```
src/layout/detector.py
```

**Key implementation:**

The detector tries LayoutParser's Detectron2 model first; if the library isn't installed, it falls back to a contour-based heuristic that detects table boxes using morphological horizontal and vertical line kernels:

```python
def _detect_table_boxes(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    h_lines  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.add(h_lines, v_lines)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    total_area = h * w
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > 0.4 * w and bh > 0.08 * h and (bw * bh) > 0.04 * total_area:
            boxes.append({"x": x, "y": y, "w": bw, "h": bh})
    return boxes
```

Text regions outside table boxes are classified by vertical position: top 12% → Title, top 22% → Header, bottom 12% → Footer, tall blocks → Text, otherwise → List.

**Done when:** `LayoutDetector().detect(image)` on a generated invoice returns at least one region with `type == "Table"` detected in the line items section.

---

## Block 6 — Table extraction (PDF-native and image-based)

**Goal:** Extract tabular data as structured row/column arrays — using PyMuPDF's native `find_tables()` for vector PDFs and an OCR + row-clustering approach for rasterized images.

**Files:**
```
src/extraction/table_parser.py
```

**Key implementation:**

PyMuPDF's `find_tables()` operates directly on PDF vector data without rasterizing, preserving numeric precision:

```python
def extract_tables_from_pdf(pdf_path, page_index=0):
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    try:
        tabs = page.find_tables()
        if tabs.tables:
            table = tabs.tables[0]
            return [row for row in table.extract() if any(c for c in row)]
    except Exception:
        pass
    # Fallback: rasterize and use image-based extraction
    mat = fitz.Matrix(200 / 72.0, 200 / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
    return extract_tables_from_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

For image-based extraction, OCR results are grouped into rows by proximity of their y-centres using a running-average threshold:

```python
y_thresh = max(14, int(h * 0.025))
rows = []
current_row = []
last_y = None
for item in sorted_items:
    cy = item["center"][1]
    if last_y is None or abs(cy - last_y) <= y_thresh:
        current_row.append(item)
        last_y = cy if last_y is None else (last_y + cy) / 2
    else:
        rows.append(sorted(current_row, key=lambda x: x["center"][0]))
        current_row = [item]
        last_y = cy
```

**Done when:** `extract_tables_from_pdf("data/generated/invoices/INV-0047.pdf")` returns a list-of-lists where each inner list represents one invoice line item row with description, quantity, unit price, and total populated.

---

## Block 7 — LLM extraction layer and deterministic fallback

**Goal:** Use Gemini Flash to convert OCR transcripts and layout region summaries into typed JSON matching each document schema, with a regex-based heuristic extractor as the offline fallback.

**Files:**
```
src/extraction/entity_extractor.py
```

**Key implementation:**

Each document type has its own prompt embedding a schema contract, ensuring the LLM response is structurally constrained before JSON parsing:

```python
INVOICE_PROMPT = """
You are a document extraction engine. Extract fields from the OCR transcript below.
Return ONLY a valid JSON object matching this exact schema:
{{
  "vendor_name": "string",
  "invoice_number": "string (e.g. INV-0047, Bill No 123)",
  "invoice_date": "string (ISO format preferred)",
  "line_items": [{{"description": "str", "qty": number, "unit_price": number, "total": number}}],
  "subtotal": number_or_null,
  "tax_rate": number_or_null,
  "tax_amount": number_or_null,
  "grand_total": number,
  "currency": "string (default INR)"
}}
OCR transcript:
{ocr_text}
Region summary: {region_summary}
"""
```

Dual SDK support handles the Google GenAI SDK version split without breaking on either:

```python
def _call_gemini_api(prompt, model_name="gemini-1.5-flash"):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        import google.genai as genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name, contents=prompt)
        return _clean_json_response(response.text)
    except Exception:
        pass
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return _clean_json_response(response.text)
    except Exception:
        return None
```

The heuristic fallback uses document-type-specific regex patterns when the API is unavailable or returns malformed JSON:

```python
def _extract_invoice_heuristically(ocr_text):
    inv_pattern = re.compile(
        r'(?:INV|Invoice|Bill)\s*[-#:]?\s*(\d{4,8})', re.IGNORECASE)
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',            # ISO
        r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',      # DD/MM/YYYY
        r'\b(\d{1,2}[-\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-\s]\d{2,4})\b'
    ]
    total_pattern = re.compile(
        r'(?:GRAND\s+TOTAL|Total\s+Amount)[^\d]*(\d[\d,]*\.?\d*)',
        re.IGNORECASE)
    ...
```

**Done when:** With `GEMINI_API_KEY` set, `extract_fields(ocr_text, regions, "invoice", ...)` returns a JSON dict with `vendor_name`, `invoice_number`, and `grand_total` populated. Without the key, the same call returns a populated dict via the heuristic path, with `_extraction_confidence` attached to the result.

---

## Block 8 — Pydantic v2 schema validation and arithmetic cross-checking

**Goal:** Enforce typed schema contracts on all extracted data via Pydantic v2 models and add invoice arithmetic cross-validation as a second integrity layer.

**Files:**
```
src/schema/validator.py
tests/test_schema_validation.py
```

**Key implementation:**

Pydantic v2 field validators enforce domain rules that JSON schema alone cannot express:

```python
class Invoice(BaseModel):
    vendor_name: str = Field(..., min_length=2)
    invoice_number: str
    grand_total: float
    line_items: List[LineItem] = []
    currency: str = "INR"

    @field_validator("grand_total")
    @classmethod
    def grand_total_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("grand_total must be strictly positive")
        return v

class TechnicalSpecification(BaseModel):
    equipment_name: str
    model_number: str
    parameters: List[SpecParameter] = Field(..., min_length=1)

    @model_validator(mode="after")
    def parameters_must_not_be_empty(self):
        if not self.parameters:
            raise ValueError("parameters list cannot be empty")
        return self
```

Arithmetic cross-validation checks that `grand_total ≈ subtotal + tax_amount` within a configurable tolerance (default 5%), providing a second gate against OCR digit misreads on numerical fields:

```python
def cross_validate_invoice_totals(data, tolerance=0.05):
    grand = data.get("grand_total", 0)
    line_sum = sum(i.get("total", 0) for i in data.get("line_items", []))
    tax = data.get("tax_amount", 0) or 0

    if line_sum > 0:
        check_val = line_sum + tax if tax else line_sum
        if abs(grand - check_val) / max(grand, 1e-9) > tolerance:
            return False, f"Arithmetic mismatch: grand_total={grand}, computed={check_val:.2f}"
    return True, "OK"
```

**Done when:** `pytest tests/test_schema_validation.py` passes all 5 tests: valid invoice accepted, negative grand_total rejected, empty parameters rejected, invalid checklist status rejected, and arithmetic mismatch detected.

---

## Block 9 — SQLite + FTS5 storage layer

**Goal:** Persist validated extractions to SQLite with an FTS5 full-text search index maintained by triggers, enabling instant substring and prefix queries across all stored documents.

**Files:**
```
src/storage/database.py
outputs/midas.db   (created at runtime)
```

**Key implementation:**

FTS5 is maintained by three triggers rather than application-side sync calls, ensuring the index stays consistent even if bulk operations bypass the Python layer:

```python
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
USING fts5(doc_type, source_file, extracted_json,
           content='documents', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, doc_type, source_file, extracted_json)
    VALUES (new.id, new.doc_type, new.source_file, new.extracted_json);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, doc_type, source_file, extracted_json)
    VALUES ('delete', old.id, old.doc_type, old.source_file, old.extracted_json);
    INSERT INTO documents_fts(rowid, doc_type, source_file, extracted_json)
    VALUES (new.id, new.doc_type, new.source_file, new.extracted_json);
END;
```

Idempotent upsert on `source_file` prevents duplicate records when the same document is reprocessed:

```python
cur.execute("""
    INSERT INTO documents (doc_type, source_file, extracted_json, confidence)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(source_file) DO UPDATE SET
        extracted_json = excluded.extracted_json,
        confidence     = excluded.confidence,
        created_at     = datetime('now')
""", (doc_type, source_file, json_str, confidence))
```

Search uses phrase-quoted FTS5 queries for multi-word terms with a LIKE fallback for unusual query strings that fail the FTS expression parser:

```python
safe_query = f'"{clean_query}"*' if " " in clean_query else f"{clean_query}*"
try:
    rows = con.execute(
        "SELECT d.* FROM documents d JOIN documents_fts f ON d.id = f.rowid "
        "WHERE documents_fts MATCH ? ORDER BY d.id DESC", (safe_query,)
    ).fetchall()
except sqlite3.OperationalError:
    rows = con.execute(
        "SELECT * FROM documents WHERE extracted_json LIKE ? ORDER BY id DESC",
        (f"%{clean_query}%",)
    ).fetchall()
```

**Done when:** Storing an invoice record and then calling `search("TATA METALIKS")` returns that record. Reprocessing the same source file does not create a duplicate row (row count stays at 1 before and after the second insert).

---

## Block 10 — Pipeline orchestration and CLI

**Goal:** Wire all modules into a single callable `run_pipeline()` function that accepts any supported document file and returns a structured result dict, then expose it via an argparse CLI.

**Files:**
```
src/pipeline/runner.py
src/main.py
```

**Key implementation:**

`run_pipeline()` runs seven stages in sequence with fallback handling at each stage so a failure in preprocessing or layout detection doesn't halt the entire run:

```python
def run_pipeline(input_path, doc_type="auto", output_dir="outputs/extracted",
                 rejected_dir="outputs/rejected", db_path=DB_PATH, config_path=None):

    # Stage 1: Preprocess (fallback to original on exception)
    try:
        preprocessed_path = preprocess(input_path, out_dir="outputs/preprocessed")
    except Exception:
        preprocessed_path = str(input_path)

    # Stage 2: OCR
    ocr_results = run_ocr(preprocessed_path)
    ocr_text    = format_ocr_transcript(ocr_results)

    # Stage 3: Document type inference
    if doc_type == "auto":
        doc_type = infer_document_type(ocr_text, Path(input_path).stem)

    # Stage 4: Layout detection
    image = cv2.imread(preprocessed_path)
    regions = LayoutDetector().detect(image) if image is not None else []

    # Stage 5: Table extraction
    has_table = any(r.get("type") == "Table" for r in regions)
    table_data = []
    if Path(input_path).suffix.lower() == ".pdf":
        table_data = extract_tables_from_pdf(input_path)
    elif has_table:
        table_data = extract_tables_from_image(image)

    # Stage 6: LLM extraction + schema validation
    extracted = extract_fields(ocr_text, regions, doc_type, table_data=table_data)
    confidence = extracted.pop("_extraction_confidence", 0.0)
    valid, result = validate(extracted, doc_type)

    if not valid:
        store_rejection(doc_type, str(input_path), extracted, str(result))
        ...
        return {"status": "rejected", "reason": str(result), ...}

    result_dict = result.model_dump()

    # Stage 7: Arithmetic cross-check (invoices) + storage
    if doc_type == "invoice":
        arith_ok, arith_msg = cross_validate_invoice_totals(result_dict)
        result_dict["_arithmetic_valid"] = arith_ok
        result_dict["_arithmetic_note"]  = arith_msg

    result_dict["_extraction_confidence"] = confidence
    store_document(doc_type, str(input_path), result_dict, confidence)
    ...
    return {"status": "ok", "doc_type": doc_type, "confidence": confidence,
            "data": result_dict, ...}
```

Document type inference uses keyword matching on both the filename stem and OCR text content:

```python
def infer_document_type(ocr_text, filename=""):
    fname = filename.lower()
    if any(k in fname for k in ["inv", "invoice", "bill"]):
        return "invoice"
    if any(k in fname for k in ["spec", "technical"]):
        return "technical_spec"
    if any(k in fname for k in ["insp", "inspection", "report"]):
        return "inspection_report"
    text = ocr_text.lower()
    if any(k in text for k in ["grand total", "invoice no", "bill to"]):
        return "invoice"
    if any(k in text for k in ["operating pressure", "model number", "rated capacity"]):
        return "technical_spec"
    if any(k in text for k in ["pass", "fail", "inspection date", "checklist"]):
        return "inspection_report"
    return "invoice"
```

**Done when:** `python src/main.py --input data/samples/invoices/INV-0047.jpg` completes without errors, writes a JSON file to `outputs/extracted/`, and prints a summary. `python src/main.py --stats` shows a counts table grouped by document type.

---

## Block 11 — Streamlit dashboard

**Goal:** Build an interactive three-tab UI for document upload/processing, FTS search, and pipeline metrics.

**Files:**
```
dashboard/app.py
```

**Key implementation:**

Custom CSS injects confidence-level colour classes that the DataFrame styling layer applies to each processed record:

```python
st.markdown("""
<style>
.conf-high { color: #22c55e; font-weight: bold; }
.conf-med  { color: #f59e0b; font-weight: bold; }
.conf-low  { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
```

The document extraction tab shows raw input alongside extracted fields side-by-side, with invoice arithmetic results surfaced as an explicit pass/fail indicator:

```python
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Input Document")
    if uploaded_path.endswith(".pdf"):
        st.markdown(f"**PDF:** `{os.path.basename(uploaded_path)}`")
    else:
        st.image(uploaded_path, use_column_width=True)

with col2:
    st.subheader("Extracted Fields")
    conf = result.get("confidence", 0)
    label = "High" if conf >= 0.8 else "Medium" if conf >= 0.6 else "Low"
    st.markdown(f'<span class="conf-{label.lower()}">Confidence: {conf:.0%} ({label})</span>',
                unsafe_allow_html=True)
    if result.get("doc_type") == "invoice":
        arith = result["data"].get("_arithmetic_valid", True)
        st.success("Arithmetic check passed") if arith else st.error("Arithmetic mismatch detected")
```

**Done when:** `streamlit run dashboard/app.py` launches without errors. Selecting the INV-0047 sample from the sidebar dropdown, clicking Run, and switching to the Metrics tab all operate without exceptions.

---

## Block 12 — Automated tests and CI workflow

**Goal:** Codify the system's correctness contracts in pytest, and run them automatically on every push via GitHub Actions.

**Files:**
```
tests/test_preprocessing.py
tests/test_schema_validation.py
tests/test_table_parser.py
tests/test_pipeline_integration.py
.github/workflows/tests.yml
```

**Key implementation:**

Schema validation tests act as living documentation of the system's rejection criteria:

```python
def test_invoice_rejects_negative_or_zero_total():
    data = {"vendor_name": "TATA METALIKS LTD",
            "invoice_number": "INV-0047", "grand_total": -500.0}
    valid, result = validate(data, "invoice")
    assert valid is False
    assert "grand_total must be strictly positive" in str(result)

def test_technical_spec_requires_parameters():
    data = {"equipment_name": "Centrifugal Slurry Pump",
            "model_number": "CSP-450X", "parameters": []}
    valid, result = validate(data, "technical_spec")
    assert valid is False
```

CI runs on Python 3.11 and 3.12, generates the synthetic corpus before pytest so integration tests have documents to process:

```yaml
- name: Generate test corpus
  run: |
    python data/scripts/generate_docs.py
    python data/scripts/degrade.py

- name: Run pytest
  run: pytest --tb=short -v
```

**Done when:** `pytest` from the repo root exits with 0 failures across all four test files. The GitHub Actions badge in the README shows passing on both Python versions.

---

*Built with PaddleOCR v4 / RapidOCR (PP-OCRv4 ONNX), Google Gemini Flash, Pydantic v2, PyMuPDF, OpenCV, SQLite FTS5, and Streamlit.*
