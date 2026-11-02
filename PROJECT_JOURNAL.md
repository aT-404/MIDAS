# Project Journal — MIDAS (Multimodal Intelligent Document Analysis System)

A running log of decisions, dead ends, and the things I figured out the hard way while building this.

---

### Entry 1 — Where this came from

My internship at Axis Max Life Insurance involves a lot of ERM work — risk registers, compliance reviews, reporting — and what I noticed early on is how much time the team spends on manual data entry from documents. Someone photographs a form. Someone else opens it on their screen and types the values into a spreadsheet. That's the whole workflow. There's no automated step between "scanned document" and "structured data." Just a person doing it one field at a time.

It's not unique to insurance. The same bottleneck exists at industrial companies like L&T — supplier invoices for factory procurement, inspection reports filed by field engineers, technical specifications for heavy machinery. Different documents, same manual entry problem, potentially higher stakes if a number is misread.

That observation turned into this project: a pipeline that takes unstructured scanned or photographed documents and returns validated structured records, with no manual step in between.

### Entry 2 — Choosing the stack

First real decision: what OCR engine to use, and whether to use a cloud API or run everything locally.

Cloud APIs (AWS Textract, Azure Document Intelligence, Google Document AI) are genuinely good — I tested Textract on a few sample invoices and it handled them well. But they cost money per page and require sending documents off-site, which rules them out for confidential engineering specs or financial records. That constraint pushed me toward self-hosted options.

Tesseract is the obvious starting point for open-source OCR, but it has a fundamental limitation for this use case: it's a raw text extractor with no layout understanding. On a structured invoice or spec table, it flattens everything into a linear text stream and loses the column relationships entirely. You get numbers, but not which number belongs to which field.

PaddleOCR v4 (PP-OCRv4) is a different category. It handles rotated text, dense table layouts, and mixed font sizes much better than Tesseract, and it runs fully offline after model download. RapidOCR wraps the same PP-OCRv4 models as ONNX runtimes, which has lower startup overhead. I set up the engine to try RapidOCR first and fall back to the full PaddleOCR if it isn't installed.

For the extraction step — turning OCR text into typed fields — regex patterns alone don't hold up. An invoice from one vendor writes "Invoice No:" while another writes "Bill #" and another just has the number after a thin horizontal rule with no label at all. A language model handles that variability gracefully. Gemini Flash is free-tier accessible and fast enough to stay interactive, so that became the extraction layer. The critical design decision was building a deterministic heuristic fallback that runs when the API is unavailable, so the pipeline works completely offline.

### Entry 3 — Three document types

Selected three categories that cover the main enterprise document structures I was working with:

**Invoices.** The interesting constraint here is arithmetic. A misread digit on a line item total isn't just a field error — it needs to fail an independent check. So I added an arithmetic cross-validation step: `grand_total` must reconcile against the sum of line items plus tax within 5% tolerance, verified separately from the schema validation.

**Technical specifications.** These are parameter tables: operating pressure, rated capacity, shaft diameter, tolerance ranges. The challenge is that the layout isn't always a neat two-column grid. Sometimes the parameter name and value are on separate lines in free text. Sometimes units and tolerances are inline ("250 bar ± 5%"). The LLM handles this context much better than positional templates.

**Inspection reports.** Checklist-heavy with PASS/FAIL/NA evaluations and free-text remarks per item. Probably the most structurally varied of the three, since field inspection forms come from many sources and don't follow one template. The schema enforces that `status` must be one of the three valid values and that the checklist cannot be empty.

### Entry 4 — Building the test corpus

I can't use real enterprise documents in an open repository. The pragmatic solution is synthetic generation: Python scripts using `Faker` and `ReportLab` to generate ground-truth-paired PDFs with realistic content.

The vendor and part names matter here. If the test corpus uses made-up names ("Company A", "Widget XYZ"), the OCR vocabulary doesn't reflect what the system will see in practice. So I seeded the generation with actual industrial supply chain names — TATA METALIKS, L&T Heavy Engineering, BHARAT FORGE, Kirloskar Brothers — and component names like hydraulic control valves, slurry pump casings, centrifugal impellers. The ground truth JSON files paired with each PDF make precision/recall evaluation straightforward.

The degradation simulator applies three effects sequentially: off-axis rotation (±5–12°), a diagonal lighting gradient to simulate corner shadows from a phone camera, and Gaussian sensor noise. Then a slight Gaussian blur for lens softness. The result looks like a document photographed on a desk at an angle, which is what real-world input looks like. Testing only on clean PDFs would tell you very little about real performance.

### Entry 5 — Stress-testing the plan before building

Before writing the core modules, I used Claude to review the architectural plan and push back on it. Not to generate code — to question the design decisions and find the edge cases I hadn't thought through.

The most useful thing that came out of it: the recommendation to prove an end-to-end slice first before building all components in parallel. Process one invoice completely — ingest, OCR, extract fields, validate, store — before building the spec and inspection paths. That way you find the integration issues early, when they're cheap to fix, rather than discovering them when you're wiring everything together at the end.

Two other things that changed from that review: confidence scoring as a first-class metric (I'd been treating extraction quality as a pass/fail rather than a gradient), and the FTS5 design decision — building full-text search into the storage layer from the start rather than adding it later, which avoids a painful migration.

Going back through my notes, I think the most valuable part of that exercise was being forced to defend the choices I'd made. When I had to explain why I chose adaptive Gaussian thresholding over Otsu's method for binarization, I realised I had a vague preference but not a concrete reason. Tested both on degraded samples and confirmed adaptive threshold handles uneven lighting gradients better, which is exactly the failure case the shadow removal step doesn't fully solve.

### Entry 6 — The rejection queue

One of the early architectural decisions that turned out to be important: every pipeline failure routes somewhere explicit. Documents that fail schema validation go to `outputs/rejected/` with a diagnostic JSON containing the raw extraction and the specific validation error. Documents where the LLM returned unparseable output go there too.

This matters because silent failures are worse than loud ones. A pipeline that quietly writes a half-populated record to the database is harder to debug than one that says "this document was rejected, here's why." In a production data pipeline, knowing the rejection rate and the failure reasons is the first signal that something has changed upstream — a new document format, a degradation the preprocessing stage isn't handling, a model response format shift.

The rejection rate on severely degraded inputs is around 8%, which I consider correct behavior. The preprocessing stage brings most documents to an extractable state, but some are too far gone. Rejecting them explicitly is right.

### Entry 7 — The checklist problem with deskew

The most specific bug I hit during the build: the deskew function was performing badly on inspection report checklists and giving wrong rotation corrections.

The deskew algorithm uses the Hough line transform to find dominant line angles in the document image and rotate to correct skew. The median angle across all detected lines is used for the correction. On a text-heavy invoice or spec sheet, the dominant lines are text baselines and table borders — long horizontal runs that accurately represent the document's orientation.

Inspection checklists are different. They contain dozens of short horizontal rules: the tick boxes, the short dividers between checklist rows, the column separators in a small table. These short lines are numerous enough to shift the median away from the actual text baselines, and the deskew function was making corrections based on the wrong angle.

The fix: filter Hough line segments to only include lines longer than 25% of the image width before computing the median. Short lines — anything that could be a checkbox or a decorative rule — are excluded from the angle calculation. I also added a safety threshold: if the estimated rotation exceeds ±15°, skip deskew entirely, because an angle that large indicates a 90° orientation problem that micro-deskewing won't solve.

After the fix, the same checklist samples that were coming out skewed or over-corrected were being handled correctly. The preprocessing tests (`test_deskew_correction`) verify this with a synthetic canvas of angled lines.

### Entry 8 — End-to-end performance

After the preprocessing fix the pipeline ran cleanly across all three document categories.

The improvement from preprocessing is significant on degraded inputs. On raw degraded scans without any enhancement, character-level accuracy on the test corpus was around 71%. After the full preprocessing chain (deskew → shadow removal → denoising → binarization), that comes up to around 89%. The preprocessing stage is doing the heavy lifting — the OCR engine works well when it gets clean input.

Confidence scoring works as intended. High-quality digital PDFs consistently score above 0.8. Degraded scanned documents with partial field extraction land in the 0.6–0.75 range. Severely degraded inputs where critical fields can't be found score below 0.6 and get routed to the rejection queue.

### Entry 9 — Dashboard and storage

The Streamlit dashboard serves two purposes: demo interface and operational monitor.

The extraction tab shows the raw document alongside the extracted fields in a two-column layout. For invoices, the arithmetic cross-check result is displayed explicitly — a green "passed" or a red "mismatch detected" — rather than just a confidence number. Confidence is displayed with colour-coded labels (High / Medium / Low) per result.

The FTS5 search tab exposes the SQLite full-text index directly. Searching "TATA METALIKS" returns all invoices where that vendor name appears in the extracted JSON, with prefix matching so partial terms work too. The LIKE fallback handles edge cases where FTS5 rejects the query string.

The metrics tab shows aggregate stats: total documents processed, rejection rate, average confidence by document type. These are the operational numbers you'd monitor in a real deployment.

### Entry 10 — Tests and CI

Automated tests cover four areas:

**Preprocessing:** deskew on a synthetic rotated canvas, shadow removal on a gradient image, binarization on random noise input. These test behavior, not exact output, since the transformations involve some numerical variation.

**Schema validation:** five tests that document the rejection criteria. Negative grand total must fail. Empty parameters list must fail. Invalid checklist status must fail. Arithmetic mismatch must be detected. These tests are living documentation — if someone changes the validation logic, one of these will break first.

**Table extraction:** verifies that `extract_tables_from_pdf` on a generated invoice returns populated row data. Skips if the corpus hasn't been generated yet.

**Integration:** full pipeline run on a generated invoice and spec. Missing file input must raise an error. These tests catch regressions in the module wiring without requiring the actual LLM API (the heuristic fallback runs offline).

GitHub Actions runs the suite on Python 3.11 and 3.12, generates the test corpus before pytest, and reports failures on push.

---

*The thing I'd tell myself at the start: the preprocessing stage is where the real work is. Clean OCR text makes everything downstream easy. Noisy OCR text makes everything downstream unreliable, no matter how good the extraction logic is. Get the image clean first.*
