# MIDAS Extraction Methodology

## 1. OCR Engine Selection: Why PaddleOCR v4 over Tesseract?

During initial prototyping, Tesseract 4/5 was evaluated alongside PaddleOCR v4 on real-world industrial and commercial documents:

| Capability | Tesseract 4/5 | PaddleOCR v4 (PP-OCRv4) | MIDAS Impact |
|---|---|---|---|
| **Rotated Text Detection** | Fails on angles > 3° without manual OSD | Built-in Angle Classifier handles arbitrary tilt | Ingests mobile phone captures reliably |
| **Dense Table Reading** | Concatenates columns into single garbled lines | DBNet preserves bounding boxes per cell | Line items retain spatial integrity |
| **Condensed Industrial Fonts** | Produces character errors (e.g. `MET/4LlKS`) | High precision on specialized typefaces | Vendor names and part numbers match exactly |
| **Inference Performance** | CPU bound, slow on high DPI images | Optimized ONNX Runtime execution | Sub-second transcription per page |

## 2. Layout-Aware Prompt Engineering

Passing raw linear OCR text into a language model often results in hallucinations, because table rows across adjacent columns get interwoven into single strings. MIDAS solves this by passing both the transcript and a structured layout map:

```
DETECTED REGIONS (layout):
- Title at y=[50..120] (conf: 0.95)
- Header at y=[130..220] (conf: 0.90)
- Table at y=[240..650] (conf: 0.92)
- Footer at y=[700..780] (conf: 0.88)
```

The LLM uses this spatial grounding to associate vendor metadata with the header block, and line items strictly within the table boundaries.

## 3. Confidence Estimation Formula

Extraction confidence is not a subjective metric; it is explicitly computed by measuring the fill-rate of schema-critical fields:

$$\text{Confidence}_{\text{base}} = \frac{\sum_{f \in \text{RequiredFields}} \mathbf{1}_{[\text{field\_present}(f)]}}{|\text{RequiredFields}|}$$

- **Invoice Required Fields:** `[vendor_name, invoice_number, invoice_date, grand_total, line_items]`
- **Spec Required Fields:** `[equipment_name, model_number, parameters]`
- **Inspection Required Fields:** `[site_name, inspector_name, inspection_date, checklist_items]`

A bonus $+0.05$ is awarded when rich structured arrays are populated (e.g. $\ge 2$ line items or $\ge 3$ parameters), capping at $1.00$.

## 4. The Rejection Pathway (No Silent Failures)

An enterprise document pipeline must never silently enter garbage into a database. When a document is illegible or missing vital contract fields:
1. Pydantic validation raises a typed error.
2. The document is flagged with status `rejected`.
3. The raw candidate payload and specific error reason are saved in `outputs/rejected/<stem>_rejected.json`.
4. A diagnostic record is inserted into the SQLite `rejections` table for auditing in the Streamlit dashboard.
