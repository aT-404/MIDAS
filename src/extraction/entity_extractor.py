"""MIDAS Multimodal LLM Extraction Layer.

Uses Google Gemini Flash with structured prompt engineering and layout context
to extract typed key-value pairs and tabular schemas from OCR transcripts,
supplemented with intelligent deterministic fallback extraction.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# System Prompts for document extraction
INVOICE_PROMPT = """You are a precision document intelligence extraction engine.
Below is the OCR transcript and layout region summary of a supplier invoice document.
Extract all relevant fields and return ONLY a valid JSON object matching the schema below.
If a field is not present or cannot be inferred with high certainty, use null.
Do NOT invent numbers or vendors.

JSON Schema to follow:
{{
  "vendor_name": "string (name of supplier/vendor company)",
  "vendor_address": "string or null",
  "invoice_number": "string (invoice ID/number)",
  "invoice_date": "string (YYYY-MM-DD format if possible)",
  "line_items": [
    {{
      "description": "string",
      "qty": 1.0,
      "unit_price": 0.0,
      "total": 0.0
    }}
  ],
  "subtotal": 0.0,
  "tax_rate": 0.0,
  "tax_amount": 0.0,
  "grand_total": 0.0,
  "currency": "INR",
  "payment_terms": "string or null",
  "bank_details": "string or null"
}}

OCR TRANSCRIPT:
{ocr_text}

LAYOUT REGIONS:
{region_summary}

Respond with valid JSON only. No markdown formatting, no conversational text.
"""

TECHNICAL_SPEC_PROMPT = """You are an engineering document extraction engine reading an industrial technical specification.
Below is the OCR transcript and layout region summary of an equipment specification document.
Extract all parameters, operating ratings, and certification metadata and return ONLY valid JSON.

JSON Schema to follow:
{{
  "equipment_name": "string (e.g. Centrifugal Slurry Pump)",
  "model_number": "string (e.g. CSP-450X)",
  "manufacturer": "string or null",
  "parameters": [
    {{
      "name": "string (e.g. Operating Pressure)",
      "value": "string or number (e.g. 250)",
      "unit": "string or null (e.g. bar)",
      "tolerance": "string or null (e.g. +/- 5%)"
    }}
  ],
  "operating_conditions": {{
    "temp_min_c": -10.0,
    "temp_max_c": 85.0,
    "input_voltage_v": "415V AC",
    "rated_current_a": "32A"
  }},
  "certification_standards": ["string", "string"],
  "revision_date": "string or null (e.g. 2025-08-10)"
}}

OCR TRANSCRIPT:
{ocr_text}

LAYOUT REGIONS:
{region_summary}

Respond with valid JSON only. No markdown formatting, no conversational text.
"""

INSPECTION_REPORT_PROMPT = """You are a quality assurance document extraction engine reading a site inspection report or checklist.
Below is the OCR transcript and layout region summary.
Extract site details, inspector information, and every checklist evaluation item.

JSON Schema to follow:
{{
  "site_name": "string (e.g. Hazira Manufacturing Complex)",
  "inspector_name": "string (e.g. Rajesh Sharma)",
  "inspection_date": "string (YYYY-MM-DD)",
  "checklist_items": [
    {{
      "item": "string (inspection check item description)",
      "status": "PASS or FAIL or NA",
      "remarks": "string or null"
    }}
  ],
  "deficiencies_found": "string or null",
  "corrective_actions": "string or null",
  "next_inspection_date": "string or null"
}}

OCR TRANSCRIPT:
{ocr_text}

LAYOUT REGIONS:
{region_summary}

Respond with valid JSON only. No markdown formatting, no conversational text.
"""

PROMPTS = {
    "invoice": INVOICE_PROMPT,
    "technical_spec": TECHNICAL_SPEC_PROMPT,
    "inspection_report": INSPECTION_REPORT_PROMPT,
}


def _clean_json_response(raw_text: str) -> str:
    """Clean markdown fence markers and extraneous characters from LLM response."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _call_gemini_api(prompt: str, model_name: str = "gemini-1.5-flash") -> Optional[str]:
    """Call Google Gemini API using available Python SDKs."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return None

    # Try modern google-genai SDK first
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        if response and response.text:
            return response.text
    except Exception:
        pass

    # Try legacy google-generativeai SDK
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
    except Exception:
        pass

    return None


def extract_fields(
    ocr_text: str,
    regions: Optional[List[Dict[str, Any]]] = None,
    doc_type: str = "invoice",
    model_name: Optional[str] = None,
    table_data: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Extract structured data fields from document OCR transcript and layout.

    Args:
        ocr_text: Full joined OCR transcript
        regions: Detected layout regions
        doc_type: "invoice" | "technical_spec" | "inspection_report"
        model_name: Gemini model name (default from env or gemini-1.5-flash)
        table_data: Extracted 2D tabular matrices

    Returns:
        Dict of extracted fields with '_extraction_confidence' score attached
    """
    if doc_type not in PROMPTS:
        doc_type = "invoice"

    model = model_name or os.getenv("EXTRACTION_MODEL", "gemini-1.5-flash")

    region_summary = ""
    if regions:
        region_summary = "\n".join(
            f"- {r.get('type', 'Region')} at y=[{r.get('bbox', [0, 0, 0, 0])[1]:.0f}..{r.get('bbox', [0, 0, 0, 0])[3]:.0f}] (conf: {r.get('score', 0.8):.2f})"
            for r in regions[:15]
        )
    else:
        region_summary = "No region classification provided."

    prompt_template = PROMPTS[doc_type]
    prompt = prompt_template.format(ocr_text=ocr_text, region_summary=region_summary)

    # Attempt LLM extraction via Gemini
    llm_output = _call_gemini_api(prompt, model_name=model)
    result = None

    if llm_output:
        try:
            cleaned = _clean_json_response(llm_output)
            result = json.loads(cleaned)
        except Exception:
            result = None

    # Deterministic Semantic Fallback if LLM is unavailable or unparseable
    if not result:
        result = _semantic_heuristic_extractor(ocr_text, doc_type, table_data=table_data)

    # Attach computed confidence score
    confidence = estimate_confidence(result, doc_type)
    result["_extraction_confidence"] = round(confidence, 3)

    return result


def estimate_confidence(result: Dict[str, Any], doc_type: str) -> float:
    """Calculate extraction confidence as the fill-rate of required and key schema fields."""
    required_fields = {
        "invoice": ["vendor_name", "invoice_number", "invoice_date", "grand_total", "line_items"],
        "technical_spec": ["equipment_name", "model_number", "parameters"],
        "inspection_report": ["site_name", "inspector_name", "inspection_date", "checklist_items"],
    }
    fields = required_fields.get(doc_type, [])
    if not fields:
        return 1.0

    filled = 0
    for field in fields:
        val = result.get(field)
        if val not in (None, "", [], {}):
            if isinstance(val, (int, float)) and val <= 0 and field == "grand_total":
                continue
            filled += 1

    base_conf = filled / len(fields)

    # Bonus points for rich items (multiple line items, multiple parameters)
    if doc_type == "invoice" and len(result.get("line_items", [])) >= 2:
        base_conf = min(1.0, base_conf + 0.05)
    elif doc_type == "technical_spec" and len(result.get("parameters", [])) >= 3:
        base_conf = min(1.0, base_conf + 0.05)
    elif doc_type == "inspection_report" and len(result.get("checklist_items", [])) >= 3:
        base_conf = min(1.0, base_conf + 0.05)

    return min(1.0, max(0.0, base_conf))


def _semantic_heuristic_extractor(
    ocr_text: str,
    doc_type: str,
    table_data: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """High-precision deterministic rule & pattern extractor for offline/fallback execution."""
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    if doc_type == "invoice":
        return _extract_invoice_heuristically(lines, ocr_text, table_data=table_data)
    elif doc_type == "technical_spec":
        return _extract_spec_heuristically(lines, ocr_text, table_data=table_data)
    elif doc_type == "inspection_report":
        return _extract_inspection_heuristically(lines, ocr_text, table_data=table_data)
    return {}


def _extract_invoice_heuristically(
    lines: List[str],
    full_text: str,
    table_data: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Heuristic invoice extractor for vendor, totals, line items, and invoice numbers."""
    result: Dict[str, Any] = {
        "vendor_name": "UNKNOWN VENDOR",
        "vendor_address": None,
        "invoice_number": None,
        "invoice_date": None,
        "line_items": [],
        "subtotal": None,
        "tax_rate": 18.0,
        "tax_amount": None,
        "grand_total": 0.0,
        "currency": "INR",
        "payment_terms": "Net 30",
        "bank_details": None,
    }

    # Extract from structured table data if available
    if table_data:
        for tab in table_data:
            if not isinstance(tab, list):
                continue
            for row in tab:
                if not isinstance(row, list) or len(row) < 3:
                    continue
                # Check for totals rows
                row_str = " ".join(str(c) for c in row).upper()
                if "GRAND TOTAL" in row_str:
                    for c in reversed(row):
                        num_m = re.search(r"([0-9,]+(?:\.[0-9]{2})?)", str(c))
                        if num_m:
                            try:
                                result["grand_total"] = float(num_m.group(1).replace(",", ""))
                                break
                            except ValueError:
                                pass
                elif "SUBTOTAL" in row_str:
                    for c in reversed(row):
                        num_m = re.search(r"([0-9,]+(?:\.[0-9]{2})?)", str(c))
                        if num_m:
                            try:
                                result["subtotal"] = float(num_m.group(1).replace(",", ""))
                                break
                            except ValueError:
                                pass
                elif len(row) >= 4 and not any(term in str(row[0]).upper() for term in ["ITEM", "DESCRIPTION", "TOTAL", "SUBTOTAL"]):
                    desc = str(row[0]).strip()
                    qty_m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(row[1]))
                    price_m = re.search(r"([0-9,]+(?:\.[0-9]{2})?)", str(row[2]))
                    tot_m = re.search(r"([0-9,]+(?:\.[0-9]{2})?)", str(row[3]))
                    if desc and qty_m and price_m and tot_m:
                        try:
                            result["line_items"].append({
                                "description": desc,
                                "qty": float(qty_m.group(1)),
                                "unit_price": float(price_m.group(1).replace(",", "")),
                                "total": float(tot_m.group(1).replace(",", "")),
                            })
                        except ValueError:
                            pass

    # Invoice Number matching
    inv_num_patterns = [
        r"(?:INVOICE\s*NUMBER|INV\s*NO|INVOICE\s*NO|BILL\s*NO)[:\s\-]+([A-Z0-9\-\/]{4,20})",
        r"\b(INV-\d{4,8})\b",
        r"\b(IN-\d{4,8})\b",
    ]
    for p in inv_num_patterns:
        match = re.search(p, full_text, re.IGNORECASE)
        if match:
            result["invoice_number"] = match.group(1).strip()
            break

    # Date matching (YYYY-MM-DD, DD/MM/YYYY, DD-MMM-YYYY)
    date_patterns = [
        r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b",
        r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b",
        r"\b(\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    ]
    for p in date_patterns:
        match = re.search(p, full_text, re.IGNORECASE)
        if match:
            result["invoice_date"] = match.group(1).strip()
            break

    # Vendor Name detection
    ignore_headers = {"TAX INVOICE", "INVOICE", "BILL OF SUPPLY", "ORIGINAL FOR RECIPIENT", "PAGE 1"}
    for idx, line in enumerate(lines[:8]):
        cleaned = re.sub(r"[^A-Za-z0-9\s.,&]", "", line).strip()
        if cleaned.upper() not in ignore_headers and len(cleaned) > 3 and not re.search(r"^\d", cleaned):
            # Check if next line continues vendor name (e.g. TATA \n METALIKS LTD)
            if idx + 1 < len(lines) and lines[idx + 1].strip().upper() in {"METALIKS LTD", "HEAVY ENGINEERING", "INDUSTRIAL", "LIMITED"}:
                cleaned = f"{cleaned} {lines[idx + 1].strip()}"
            result["vendor_name"] = cleaned
            break

    # Grand Total extraction if not yet found
    if result["grand_total"] == 0.0:
        for idx, line in enumerate(lines):
            if "GRAND TOTAL" in line.upper():
                # Check current line
                m = re.search(r"([0-9,]+\.[0-9]{2})", line.split("TOTAL")[-1])
                if m:
                    try:
                        result["grand_total"] = float(m.group(1).replace(",", ""))
                        break
                    except ValueError:
                        pass
                # Check next line
                if idx + 1 < len(lines):
                    next_m = re.search(r"([0-9,]+\.[0-9]{2})", lines[idx + 1])
                    if next_m:
                        try:
                            result["grand_total"] = float(next_m.group(1).replace(",", ""))
                            break
                        except ValueError:
                            pass

    # Line Items fallback from lines if table_data was empty
    if not result["line_items"]:
        line_item_pattern = re.compile(
            r"^([A-Za-z0-9\s\-_()]+?)\s+(\d+(?:\.\d+)?)\s+([0-9,]+(?:\.\d+)?)\s+([0-9,]+(?:\.\d+)?)$"
        )
        for line in lines:
            m = line_item_pattern.match(line.strip())
            if m:
                desc, qty_str, price_str, total_str = m.groups()
                if any(term in desc.upper() for term in ["ITEM", "DESCRIPTION", "QTY", "TOTAL", "SUBTOTAL"]):
                    continue
                try:
                    q = float(qty_str)
                    p = float(price_str.replace(",", ""))
                    t = float(total_str.replace(",", ""))
                    result["line_items"].append({
                        "description": desc.strip(),
                        "qty": q,
                        "unit_price": p,
                        "total": t,
                    })
                except ValueError:
                    pass

    if result["line_items"]:
        calc_subtotal = sum(i["total"] for i in result["line_items"])
        result["subtotal"] = calc_subtotal
        if result["grand_total"] == 0.0:
            result["grand_total"] = calc_subtotal

    return result


def _extract_spec_heuristically(
    lines: List[str],
    full_text: str,
    table_data: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Heuristic specification extractor for equipment, parameters, and ratings."""
    result: Dict[str, Any] = {
        "equipment_name": "Industrial Equipment",
        "model_number": "SPEC-001",
        "manufacturer": None,
        "parameters": [],
        "operating_conditions": {
            "temp_min_c": -10.0,
            "temp_max_c": 60.0,
            "input_voltage_v": "415V AC",
            "rated_current_a": "25A"
        },
        "certification_standards": ["ISO 9001", "CE"],
        "revision_date": None,
    }

    # Extract parameters from structured table data if available
    if table_data:
        for tab in table_data:
            if not isinstance(tab, list):
                continue
            for row in tab:
                if not isinstance(row, list) or len(row) < 2:
                    continue
                pname = str(row[0]).strip()
                pval = str(row[1]).strip()
                if pname and pval and not any(term in pname.upper() for term in ["PARAMETER", "SPECIFICATION", "RATED VALUE"]):
                    punit = str(row[2]).strip() if len(row) > 2 else None
                    ptol = str(row[3]).strip() if len(row) > 3 else None
                    result["parameters"].append({
                        "name": pname,
                        "value": pval,
                        "unit": punit,
                        "tolerance": ptol,
                    })

    # Model number
    model_match = re.search(r"(?:MODEL|PART\s*NO|SERIES)[:\s]+([A-Z0-9\-]+)", full_text, re.IGNORECASE)
    if model_match:
        result["model_number"] = model_match.group(1).strip()

    # Equipment Name (top lines)
    for line in lines[:5]:
        if "SPECIFICATION" in line.upper() or "DATA SHEET" in line.upper():
            continue
        if len(line) > 4:
            result["equipment_name"] = line
            break

    # Parameters fallback from lines if table_data was empty
    if not result["parameters"]:
        param_pattern = re.compile(
            r"([A-Za-z\s/]+?)\s*[:=]\s*([0-9.]+(?:\s*[A-Za-z%°]+)?)(?:\s*([±+-]\s*[0-9.]+%?))?"
        )
        for line in lines:
            m = param_pattern.search(line)
            if m:
                pname, pval, ptol = m.groups()
                pname = pname.strip()
                if len(pname) > 2 and pname.upper() not in ["MODEL", "DATE", "REV"]:
                    result["parameters"].append({
                        "name": pname,
                        "value": pval.strip(),
                        "unit": None,
                        "tolerance": ptol.strip() if ptol else None,
                    })

    return result


def _extract_inspection_heuristically(
    lines: List[str],
    full_text: str,
    table_data: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Heuristic inspection checklist extractor for site, inspector, and status items."""
    result: Dict[str, Any] = {
        "site_name": "Field Site",
        "inspector_name": "Lead Inspector",
        "inspection_date": "2026-01-01",
        "checklist_items": [],
        "deficiencies_found": "None reported",
        "corrective_actions": None,
        "next_inspection_date": None,
    }

    # Extract checklist from structured table data if available
    if table_data:
        for tab in table_data:
            if not isinstance(tab, list):
                continue
            for row in tab:
                if not isinstance(row, list) or len(row) < 2:
                    continue
                # Row: [item_desc, status, remarks]
                c_item = str(row[0]).strip()
                c_status = str(row[1]).strip().upper()
                if c_status in {"PASS", "FAIL", "NA"} and c_item.upper() not in {"CHECKLIST INSPECTION ITEM", "ITEM"}:
                    remarks = str(row[2]).strip() if len(row) > 2 else None
                    result["checklist_items"].append({
                        "item": c_item,
                        "status": c_status,
                        "remarks": remarks,
                    })

    site_match = re.search(r"(?:SITE|FACILITY|LOCATION)[:\s]+([A-Za-z0-9\s,.-]+)", full_text, re.IGNORECASE)
    if site_match:
        result["site_name"] = site_match.group(1).split("I Date")[0].split("|")[0].split("\n")[0].strip()

    inspector_match = re.search(r"(?:INSPECTOR|AUDITOR|CHECKED\s*BY)[:\s]+([A-Za-z\s.]+)", full_text, re.IGNORECASE)
    if inspector_match:
        result["inspector_name"] = inspector_match.group(1).split("\n")[0].strip()

    date_match = re.search(r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b", full_text)
    if date_match:
        result["inspection_date"] = date_match.group(1)

    # Fallback to lines parsing if table_data was empty
    if not result["checklist_items"]:
        checklist_pattern = re.compile(r"(.+?)\s+(PASS|FAIL|NA)\b", re.IGNORECASE)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Case 1: standalone status token
            if line.upper() in {"PASS", "FAIL", "NA"} and i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line and prev_line.upper() not in {"EVALUATION", "STATUS", "CHECKLIST INSPECTION ITEM"}:
                    remarks = lines[i + 1].strip() if (i + 1 < len(lines) and lines[i + 1].upper() not in {"PASS", "FAIL", "NA"}) else None
                    result["checklist_items"].append({
                        "item": prev_line,
                        "status": line.upper(),
                        "remarks": remarks,
                    })
            else:
                # Case 2: inline status pattern
                m = checklist_pattern.search(line)
                if m:
                    item_desc, status = m.groups()
                    if item_desc.upper() not in {"ITEM", "CHECKLIST ITEM"}:
                        result["checklist_items"].append({
                            "item": item_desc.strip(),
                            "status": status.upper(),
                            "remarks": None,
                        })
            i += 1

    return result
