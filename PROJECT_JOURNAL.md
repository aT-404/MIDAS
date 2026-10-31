# Project Journal — MIDAS (Multimodal Intelligent Document Analysis System)

A running log of how this project happened and why. Partly for myself, partly because a portfolio project with a paper trail looks substantially better than one that appears fully formed from nowhere.

---

### Entry 1 — The idea

My ERM internship at Axis Max Life Insurance has been a lot of things, but what it's made most visible is how much enterprise time goes into data entry. Insurance companies run on forms: policy applications, claim documents, medical reports, inspection checklists, supplier invoices. A stack of these lands on a desk. Someone opens them one by one and types the values into a system. There's no automated step between "physical or scanned document" and "row in a database" — just a human doing it.

That's the bottleneck I want to solve. Not because I've written a thesis on document AI, but because I've watched the manual version happen every day for weeks.

L&T operates at an industrial scale. Their documents aren't insurance forms — they're engineering specifications for heavy machinery, inspection reports from field engineers, supplier invoices for procurement at the factory level. Same problem, higher stakes if the wrong number gets entered. This is what I want to build: a pipeline that takes any document in, and outputs clean structured data.

### Entry 2 — Research

First question: what already exists and why isn't everyone using it?

Commercial options (AWS Textract, Azure Document Intelligence, Google Document AI) work very well, but they cost money per page at scale and require sending documents to a cloud service — which rules them out for anything confidential. Open-source options are a mix: Tesseract is the obvious starting point but it's a raw OCR engine with no layout awareness, which means it melts tables into an unreadable linear text stream.

Landed on this stack after reading:
- PaddleOCR v4 technical report — significantly better table detection than Tesseract, handles rotated text and dense layouts well, and importantly, it runs fully offline once the model weights are downloaded.
- LayoutParser paper (Shen et al., 2021) — gives you region classification on top of OCR so you know which text came from a table cell vs a paragraph heading.
- The Gemini Flash model for the extraction layer — I want something that can read a messy OCR transcript and infer "this abbreviated string `INV#` means invoice number", which is a language understanding task, not a pattern match task. A rule-based extractor would need a regex for every possible abbreviation variation. A small LLM handles it in the prompt.

Also decided early: the LLM call should have a structured output expectation (return JSON with specific fields) and a confidence estimator on top of it. If half the required fields come back null, the document goes to a rejected queue, not into the database. Garbage in, garbage out is a design choice, not just a saying.

### Entry 3 — Choosing the document types

Three types covers the realistic range without making the schema design unmanageable:

**Invoice** — numerical precision matters. The grand total must cross-validate against the sum of line items. A single misread digit is a meaningful error. This is the hardest type to get exactly right.

**Technical Specification** — parameter tables with values and units. The extraction challenge is not reading the numbers (PaddleOCR handles those well) but associating each parameter name with its value and unit when the layout isn't a clean two-column table. A spec for a hydraulic pump might have "Operating Pressure: 250 bar ± 5%" as running text, not a table row. The LLM extraction layer handles this where a regex wouldn't.

**Inspection Report** — checklist-heavy. Mostly PASS/FAIL/NA status per item, with free-text remarks. The challenge here is that real inspection forms are rarely standardized — the items vary by site and equipment type. The LLM extraction generalizes better than a fixed schema parser.

### Entry 4 — Dataset design

No real enterprise will hand over their documents for a student project. So I build my own.

Two approaches: synthetic generation (Python + Faker + ReportLab, which produces clean PDFs that I then deliberately degrade with noise and skew), and real-world public benchmarks. SROIE and CORD are both scanned receipt/invoice datasets used in the information extraction research community — they're real scans with real OCR challenges, not textbook examples.

The degradation simulation is deliberate and documented: I want the pipeline tested against images that actually look like a photo taken on a phone in bad lighting, not a perfect flatbed scan. If the preprocessing stage can't handle that, the pipeline isn't useful in practice.

### Entry 5 — Seeking a second opinion

Before committing to the full build, I walked through the plan with Claude — same reason as the ASIRS project: I'd rather catch design errors before they cost me a full rebuild, and "I built this with zero external input" is a weaker story than "I built this and knew when to get a second opinion."

Issues that came up:

- My original plan used Tesseract as the OCR engine. After discussing the table handling problem, I switched to PaddleOCR — specifically, the document showing that PaddleOCR v4 scores significantly higher on ReCTS and DocVQA benchmarks for table-heavy documents changed my mind.
- I was planning to build all three document types in parallel before proving any of them worked end-to-end. Same backwards-logic problem as the ASIRS plan. The fix was the same: MVP first. One invoice, fully extracted and written to JSON, before anything else.
- I had no confidence estimation — the pipeline would either extract and accept, or crash. Adding `_extraction_confidence` as a computed field (fill-rate of required fields) gives the validation layer something to act on.
- The FTS5 search was not in the original plan at all. Added it because the dashboard needs a way to query stored records without loading every JSON file into memory.

### Entry 6 — Corrections

Rewrote the build order: Block 2 (MVP, one invoice end-to-end) happens before Block 3 (preprocessing) and Block 4 (layout analysis). This feels counterintuitive — preprocessing is step 1 in the pipeline, why build it second? Because the MVP proves the integration chain works. If the LLM refuses to return valid JSON, or the Gemini API is flaky, I find out immediately. I don't discover it at step 7 after building a polished preprocessing module.

Also added the rejection pathway explicitly. Every failed validation writes to `outputs/rejected/` with the error reason. This is important for the portfolio demo: showing a rejection is more impressive than showing only successes, because it proves the system knows when it doesn't know.

The arithmetic cross-validation on invoices (grand_total must match the sum of line_items within 0.01 tolerance) was also added here. One extra check, but it catches the most important class of extraction errors in financial documents.

### Entry 7 — The preprocessing discovery

First real surprise during the build: my deskew function, which uses a Hough line transform to find the dominant line angle, performs terribly on inspection checklists. Checklists have lots of short horizontal rules (checkbox borders, dividers) that confuse the angle estimator into thinking the page is more rotated than it is.

Fixed by filtering the Hough lines to only consider lines longer than 30% of the image width before computing the median angle. Short decorative lines no longer dominate the calculation. Also added a sanity check: if the estimated rotation is more than ±15°, skip deskew entirely (that's more likely a sideways document than a slightly tilted scan — different problem, different fix).

### Entry 8 — Where things stand

Pipeline now works end-to-end for all three document types on clean inputs. Preprocessing handles the degraded test documents well — average OCR accuracy (character-level) improves from 71% on raw degraded inputs to 89% after preprocessing. The remaining 11% error rate is mostly in dense handwritten annotations in the inspection report samples, which PaddleOCR handles poorly by design (it's not a handwriting recognition engine).

Schema validation rejects approximately 8% of the synthetic test extractions — mostly missing fields in the most degraded inspection reports. Those rejections are correct behavior: the document is genuinely unreadable, and a rejection is more honest than a half-populated record.

### Entry 9 — Block 7 and 8 — Dashboard and storage

Dashboard is built last, which is the right order. The pipeline is the deliverable; the dashboard is the demonstration layer.

One thing I did add during the dashboard build that wasn't in the original plan: a confidence-based color scheme on the extracted fields table. Required fields with confidence below the threshold are shown in orange. Optional fields with confidence below threshold are shown in grey. This makes it immediately visible to a reviewer where the extraction is uncertain, which is more useful than a single aggregate confidence number.

The FTS5 search works well for invoice vendor name lookups. An edge case: Gemini sometimes normalizes "Tata Metaliks Ltd." to "TATA METALIKS LIMITED" in the extraction. Both strings FTS-match on "TATA METALIKS" — the search is case-insensitive and does prefix matching by default — so this doesn't cause missed results.

### Entry 10 — Testing pass

Wrote the test suite last (after Block 9 — deliberately), because the "Done when" tests at the end of each block were already the informal test criteria. The formal pytest tests are mostly codifying what I'd been checking by hand.

One genuine bug caught by the schema test: `TechnicalSpecification.parameters` accepted an empty list without complaint. A spec with zero parameters is technically invalid but Pydantic allowed it. Added a `@field_validator` that raises if the list is empty. Two extracted specs that had previously "passed" validation were reclassified as rejected — correctly, because both came from documents where the parameter table OCR completely failed.

Six tests total, all passing. GitHub Actions workflow runs them on every push. The pipeline is ready for portfolio sharing.

---

*The most honest thing I can say about this project: OCR is a harder problem than it looks when the inputs are real documents rather than textbook scans. The preprocessing stage, which I originally planned as "maybe 50 lines," ended up as the most iterated part of the codebase. The LLM extraction layer was easier than I expected — a well-structured prompt does most of the work. The hard part is giving it clean enough text to work with.*
