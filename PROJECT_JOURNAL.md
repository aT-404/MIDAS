# Project Journal — MIDAS (Multimodal Intelligent Document Analysis System)

A running technical log of architecture decisions, engineering trade-offs, and implementation milestones during the development of MIDAS.

---

### Entry 1 — The operational problem

My ERM internship at Axis Max Life Insurance has been a lot of things, but what it's made most visible is how much enterprise time goes into data entry. Insurance companies run on forms: policy applications, claim documents, medical reports, inspection checklists, supplier invoices. A stack of these lands on a desk. Someone opens them one by one and types the values into a system. There's no automated step between "physical or scanned document" and "row in a database" — just a human doing it.

That's the bottleneck I wanted to solve: an end-to-end pipeline that takes unstructured documents (scanned PDFs, smartphone photos) and produces clean, validated, structured data.

Industrial organizations like L&T operate under similar constraints at scale. Their documents aren't insurance forms — they're engineering specifications for heavy machinery, inspection reports from field engineers, and supplier invoices for factory procurement. Same operational challenge, but with higher stakes if the wrong parameter or number gets entered.

### Entry 2 — Research & Technology Selection

First question: what already exists and why isn't everyone using it?

Commercial cloud APIs (AWS Textract, Azure Document Intelligence, Google Document AI) work well, but they cost money per page at scale and require sending documents off-premise — which rules them out for confidential engineering and financial files. Open-source options are a mixed bag: Tesseract is the traditional starting point, but it's a raw OCR engine with no layout awareness, meaning it flattens tables into an unreadable linear text stream.

Landed on this stack after benchmarking:
- **PaddleOCR v4 (PP-OCRv4):** Significantly better table detection than Tesseract, handles rotated text blocks and dense layouts well, and crucially, runs fully offline on local CPU/GPU once model weights are downloaded.
- **Layout Region Classification:** Classifies regions (title, header, table, paragraph, footer) before extraction, preventing tabular cell numbers from bleeding into surrounding prose.
- **Gemini Flash for Multimodal Extraction:** Reads noisy OCR transcripts and infers context (e.g., recognizing that an abbreviated string `INV#` or `Bill No` maps to `invoice_number`). Language understanding handles variability that brittle regex patterns fail on.

Also decided early: the LLM call must have strict schema expectations (typed JSON) with a confidence estimator. If required fields fail to extract or violate constraints, the document goes to a rejection queue rather than corrupting the database.

### Entry 3 — Choosing Document Classes

Selected three representative document types that cover key enterprise formats:

1. **Invoices:** Numerical precision is critical. Grand totals must cross-validate against the sum of line items. A single misread digit represents a material error.
2. **Technical Specifications:** Parameter tables with ratings, values, and units. The challenge is associating each parameter name with its value and unit when the layout isn't a neat two-column grid (e.g. "Operating Pressure: 250 bar ± 5%" inside running text).
3. **Inspection Reports:** Checklist-heavy documents with PASS/FAIL/NA evaluations and free-text remarks. Non-standard layout structures where generalized extraction performs better than rigid positional templates.

### Entry 4 — Benchmark Dataset & Degradation Simulation

Proprietary enterprise documents cannot be hosted in an open repository due to confidentiality. To ensure rigorous testing, I built a dual approach:
- Synthetic generation using Python, `Faker`, and `ReportLab` to produce ground-truth-paired PDFs across all three categories.
- Realistic degradation simulation (`data/scripts/degrade.py`) applying off-axis skew (±5–12°), uneven lighting gradients, corner flash shadows, and sensor Gaussian noise.

Testing against realistic smartphone captures and poor scans is essential — if the preprocessing stage cannot handle degradation, the pipeline cannot survive real-world inputs.

### Entry 5 — Stress-Testing the Architecture with AI

Before writing the core modules, I used AI (Claude) to review and stress-test the architectural plan, challenge design assumptions, and catch edge cases early. Using AI as an active engineering sparring partner helped pressure-test the design before sinking time into implementation.

Key changes that emerged from this review:
- **Switching from Tesseract to PaddleOCR:** Comparative benchmark data showed PaddleOCR v4 scores significantly higher on ReCTS and DocVQA for table-dense documents, especially with condensed typography.
- **Incremental MVP Slice First:** Rather than building all components in parallel, the recommendation was to prove an end-to-end slice first: ingest one invoice, OCR it, extract schema fields, and validate output.
- **Confidence Scoring:** Added `_extraction_confidence` as a computed metric based on schema fill-rate and completeness, giving the validation layer an objective threshold to act on.
- **SQLite FTS5 Storage:** Integrated an FTS5 full-text search index from the start so the downstream dashboard could query parameters and vendor terms without scanning JSON files on disk.

### Entry 6 — Pipeline Design Refinements

Proving the integration chain early confirmed that LLM structured extraction paired with deterministic fallback was reliable.

Added the explicit rejection pathway: every failed validation routes to `outputs/rejected/` with diagnostic error traces. In production data pipelines, surfacing explicit rejections is essential: the system must know when it doesn't know, rather than silently ingesting corrupt data.

Also introduced arithmetic cross-validation for invoices: verifying that `grand_total` reconciles with line items within a 5% tolerance. One extra check catches the most important class of numerical extraction errors.

### Entry 7 — The Preprocessing Discovery

First significant technical hurdle during the build: the deskew function, which uses a Hough line transform to compute dominant baseline angles, performed poorly on inspection checklists. Checklists contain dozens of short horizontal rules (checkboxes, table dividers) that biased the angle estimator.

**The fix:** Filtered Hough line segments to only consider lines longer than 25% of the image width before computing the median angle. Short decorative lines no longer skew the calculation. Also added an angle safety threshold: if the estimated rotation exceeds ±15°, skip deskew entirely (that indicates an upside-down or sideways orientation requiring 90° rotation, not micro-deskewing).

### Entry 8 — Performance & Accuracy

The pipeline runs end-to-end across all three document categories. Preprocessing significantly improves character-level recognition on degraded inputs (moving character accuracy from ~71% on raw degraded scans to ~89% post-enhancement). 

Schema validation safely catches unreadable documents (rejecting ~8% of severely degraded inputs where critical fields cannot be resolved). Those rejections represent correct system behavior: rejecting unreadable input is far better than writing half-populated records.

### Entry 9 — Dashboard and Storage

Built the demonstration dashboard using Streamlit on top of SQLite:
- Single-document upload and benchmark sample switcher.
- Side-by-side view of raw input against extracted fields with confidence badges.
- SQLite FTS5 search enabling instant substring and prefix queries across vendor names, line items, and parameters.
- Rejection queue viewer with diagnostic logs.

### Entry 10 — Automated Testing & Quality Assurance

Codified a comprehensive automated test suite with `pytest`:
- Unit tests for preprocessing (deskew calculation, shadow removal, adaptive binarization).
- Strict schema validation tests ensuring invalid records (e.g. negative grand totals, specifications with 0 parameters) are rejected.
- Tabular extraction verification across vector PDFs and raster grids.
- End-to-end integration tests verifying complete pipeline execution.

All tests pass cleanly, and a GitHub Actions CI workflow runs the suite automatically on every push.

---

*Key takeaway: In document intelligence, OCR and image preprocessing are where 80% of real-world failures happen. The LLM extraction layer works exceptionally well when given clean text and layout context, but robust CV preprocessing is what makes that text readable in the first place.*
