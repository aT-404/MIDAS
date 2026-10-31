"""MIDAS Synthetic Document Generator.

Generates realistic PDFs with matching ground truth JSON for:
- Commercial Invoices (30 files)
- Industrial Technical Specifications (20 files)
- Quality & Safety Inspection Reports (15 files)
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Dict
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.units import inch

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

OUT_DIR = Path("data/generated")


def generate_invoice(output_path: str, invoice_id: Optional[str] = None, vendor: Optional[str] = None) -> dict:
    """Generate a realistic supplier invoice PDF and save ground truth."""
    inv_num = invoice_id or f"INV-{random.randint(1000, 9999)}"
    vendor_name = vendor or random.choice([
        "TATA METALIKS LTD",
        "LARSEN & TOUBRO HEAVY ENGINEERING",
        "BHARAT FORGE INDUSTRIAL",
        "JINDAL STEEL & POWER LTD",
        "THERMAX PROCESS SYSTEMS",
        "KIRLOSKAR PNEUMATIC CO",
        "CROMPTON GREAVES POWER",
        "HINDALCO INDUSTRIES",
    ])
    vendor_addr = f"{fake.building_number()}, {fake.street_name()}, {fake.city()} - {fake.postcode()}"
    inv_date = fake.date_between(start_date="-1y", end_date="today").isoformat()

    # Line items
    sample_items = [
        ("High-Pressure Hydraulic Valve 25mm", 12500.0),
        ("Submersible Impeller Assembly 316L", 38400.0),
        ("Flanged Carbon Steel Pipe Fitting 4-inch", 4200.0),
        ("Rotary Shaft Oil Seal Viton", 850.0),
        ("Cast Iron Slurry Casing Liner", 45000.0),
        ("Centrifugal Bearing Housing Unit", 18500.0),
        ("Heavy Duty Gasket Kit Neoprene", 1200.0),
        ("Pneumatic Actuator Double Acting", 27800.0),
    ]

    num_items = random.randint(3, 6)
    chosen = random.sample(sample_items, num_items)
    line_items = []
    subtotal = 0.0

    for desc, unit_p in chosen:
        qty = float(random.randint(1, 10))
        tot = round(qty * unit_p, 2)
        subtotal += tot
        line_items.append({
            "description": desc,
            "qty": qty,
            "unit_price": unit_p,
            "total": tot,
        })

    tax_rate = 18.0
    tax_amount = round(subtotal * (tax_rate / 100.0), 2)
    grand_total = round(subtotal + tax_amount, 2)

    ground_truth = {
        "vendor_name": vendor_name,
        "vendor_address": vendor_addr,
        "invoice_number": inv_num,
        "invoice_date": inv_date,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "grand_total": grand_total,
        "currency": "INR",
        "payment_terms": "Net 30 Days",
        "bank_details": f"State Bank of India, A/C: {random.randint(10000000000, 99999999999)}, IFSC: SBIN0004521"
    }

    # Build PDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    elements = []

    # Header
    title_style = ParagraphStyle(
        "InvTitle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1e3a8a")
    )
    elements.append(Paragraph(f"<b>{vendor_name}</b>", title_style))
    elements.append(Paragraph(f"<font size=9 color='#4b5563'>{vendor_addr}</font>", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Meta table
    meta_data = [
        [Paragraph(f"<b>INVOICE NUMBER:</b> {inv_num}", styles["Normal"]),
         Paragraph(f"<b>DATE:</b> {inv_date}", styles["Normal"])],
        [Paragraph(f"<b>PAYMENT TERMS:</b> Net 30 Days", styles["Normal"]),
         Paragraph(f"<b>CURRENCY:</b> INR", styles["Normal"])],
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * inch, 3.5 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor("#d1d5db")),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 16))

    # Line item table
    table_rows = [["Item Description", "Qty", "Unit Price (INR)", "Total (INR)"]]
    for item in line_items:
        table_rows.append([
            item["description"],
            f"{item['qty']:.0f}",
            f"{item['unit_price']:,.2f}",
            f"{item['total']:,.2f}",
        ])

    table_rows.append(["", "", "SUBTOTAL:", f"{subtotal:,.2f}"])
    table_rows.append(["", "", f"GST ({tax_rate:.0f}%):", f"{tax_amount:,.2f}"])
    table_rows.append(["", "", "GRAND TOTAL:", f"{grand_total:,.2f}"])

    items_table = Table(table_rows, colWidths=[3.5 * inch, 0.8 * inch, 1.4 * inch, 1.5 * inch])
    items_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -4), 0.5, colors.HexColor("#9ca3af")),
        ("BACKGROUND", (0, 1), (-1, -4), colors.HexColor("#ffffff")),
        ("FONTNAME", (2, -3), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE", (2, -1), (-1, -1), 1.5, colors.HexColor("#1e3a8a")),
        ("BACKGROUND", (2, -1), (-1, -1), colors.HexColor("#e0e7ff")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 20))

    # Bank details footer
    elements.append(Paragraph(f"<b>Bank Details:</b> {ground_truth['bank_details']}", styles["Normal"]))

    doc.build(elements)

    # Save matching ground truth
    gt_path = output_path.replace(".pdf", "_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def generate_technical_spec(output_path: str, spec_id: Optional[str] = None) -> dict:
    """Generate industrial equipment specification sheet."""
    model_num = spec_id or f"CSP-{random.randint(100, 999)}X"
    equip_name = random.choice([
        "Heavy Duty Centrifugal Slurry Pump",
        "Multi-Stage High Pressure Boiler Feed Pump",
        "Double-Suction Split Case Water Pump",
        "Submersible Waste Water Cutter Pump",
        "Axial Flow Chemical Circulation Pump",
    ])
    manufacturer = random.choice([
        "L&T Heavy Engineering Division",
        "Kirloskar Brothers Limited",
        "Sulzer Pumps India",
        "KSB Mil Controls",
    ])
    rev_date = fake.date_between(start_date="-2y", end_date="today").isoformat()

    parameters = [
        {"name": "Max Operating Pressure", "value": f"{random.choice([150, 200, 250, 300])}", "unit": "bar", "tolerance": "± 5%"},
        {"name": "Volumetric Flow Rate", "value": f"{random.choice([450, 600, 850, 1200])}", "unit": "m3/h", "tolerance": "± 3%"},
        {"name": "Shaft Speed Rating", "value": f"{random.choice([1450, 1780, 2900])}", "unit": "RPM", "tolerance": "± 10 RPM"},
        {"name": "Hydraulic Head Rating", "value": f"{random.choice([85, 120, 160])}", "unit": "m", "tolerance": "± 4%"},
        {"name": "Impeller Diameter", "value": f"{random.choice([380, 420, 510])}", "unit": "mm", "tolerance": "± 1 mm"},
        {"name": "Casing Wall Thickness", "value": "24", "unit": "mm", "tolerance": "Min"},
    ]

    operating_conditions = {
        "temp_min_c": -15.0,
        "temp_max_c": 110.0,
        "input_voltage_v": "415V AC 3-Phase",
        "rated_current_a": "55A",
    }
    certifications = ["ISO 9001:2015", "API 610 12th Ed", "CE Certified", "ATEX Zone 1"]

    ground_truth = {
        "equipment_name": equip_name,
        "model_number": model_num,
        "manufacturer": manufacturer,
        "parameters": parameters,
        "operating_conditions": operating_conditions,
        "certification_standards": certifications,
        "revision_date": rev_date,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>ENGINEERING TECHNICAL SPECIFICATION</b>", styles["Heading1"]))
    elements.append(Paragraph(f"<font size=12 color='#1e3a8a'><b>{equip_name}</b></font>", styles["Heading2"]))
    elements.append(Paragraph(f"Manufacturer: {manufacturer} | Model: <b>{model_num}</b> | Revision: {rev_date}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    # Parameters Table
    p_rows = [["Specification Parameter", "Rated Value", "Unit", "Design Tolerance"]]
    for p in parameters:
        p_rows.append([p["name"], p["value"], p["unit"], p["tolerance"]])

    ptable = Table(p_rows, colWidths=[3.0 * inch, 1.4 * inch, 1.2 * inch, 1.6 * inch])
    ptable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(ptable)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("<b>Operating Environmental Conditions:</b>", styles["Heading3"]))
    elements.append(Paragraph(
        f"Temperature Range: {operating_conditions['temp_min_c']}°C to {operating_conditions['temp_max_c']}°C | "
        f"Input Voltage: {operating_conditions['input_voltage_v']} | Current: {operating_conditions['rated_current_a']}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Applicable Standards & Certifications:</b>", styles["Heading3"]))
    elements.append(Paragraph(", ".join(certifications), styles["Normal"]))

    doc.build(elements)

    gt_path = output_path.replace(".pdf", "_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def generate_inspection_report(output_path: str, report_id: Optional[str] = None) -> dict:
    """Generate quality and safety inspection checklist report."""
    insp_id = report_id or f"INSP-{random.randint(1000, 9999)}"
    site_name = random.choice([
        "Hazira Modular Fabrication Yard",
        "Vadodara Heavy Machinery Complex",
        "Powai Plant 1 Machine Shop",
        "Kattupalli Shipyard Drydock",
        "Talegaon Engine Testing Facility",
    ])
    inspector_name = random.choice([
        "Rajesh Sharma (Lead QA/QC)",
        "Anand Varma (Senior Mechanical Auditor)",
        "Priya Sundaram (NDT Level III)",
        "Vikram Sengupta (Safety Officer)",
    ])
    insp_date = fake.date_between(start_date="-1y", end_date="today").isoformat()

    checklist_pool = [
        ("Visual inspection of casing for hairline casting cracks", "PASS", "Zero defects detected under UV dye"),
        ("Dynamic shaft runout balance check (<0.02mm)", "PASS", "Measured at 0.012mm TIR"),
        ("Hydrostatic pressure containment hold (350 bar / 30 min)", "PASS", "Pressure maintained without drop"),
        ("Mechanical seal leak rate within tolerance (<5 ml/h)", "PASS", "No visible dripping observed"),
        ("Foundation anchor bolt torque verification (450 Nm)", "PASS", "Torqued and calibrated"),
        ("Emergency automatic shutdown circuit trip response", "PASS", "Tripped in 180ms"),
        ("Bearing vibration velocity RMS (<2.8 mm/s)", random.choice(["PASS", "FAIL"]), "Exceeded on drive-end journal" if random.random() < 0.2 else "Normal"),
    ]

    selected_items = checklist_pool[:random.randint(4, 7)]
    checklist_items = [{"item": item[0], "status": item[1], "remarks": item[2]} for item in selected_items]

    deficiencies = "Vibration spike on drive end bearing journal exceeding ISO 10816 limits" if any(i["status"] == "FAIL" for i in checklist_items) else "None reported. Equipment within operational envelope."
    corrective = "Re-align coupling hubs and replace drive-end bearing spacer" if "Vibration" in deficiencies else "Proceed with commissioning"

    ground_truth = {
        "site_name": site_name,
        "inspector_name": inspector_name,
        "inspection_date": insp_date,
        "checklist_items": checklist_items,
        "deficiencies_found": deficiencies,
        "corrective_actions": corrective,
        "next_inspection_date": fake.date_between(start_date="today", end_date="+6m").isoformat(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>QUALITY & SAFETY INSPECTION REPORT: {insp_id}</b>", styles["Heading1"]))
    elements.append(Paragraph(f"<b>Site:</b> {site_name} | <b>Date:</b> {insp_date}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Certified Auditor:</b> {inspector_name}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    c_rows = [["Checklist Inspection Item", "Evaluation", "Inspector Remarks"]]
    for item in checklist_items:
        c_rows.append([item["item"], item["status"], item["remarks"] or ""])

    ctable = Table(c_rows, colWidths=[3.8 * inch, 1.2 * inch, 2.2 * inch])
    ctable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c2d12")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(ctable)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph(f"<b>Deficiencies Identified:</b> {deficiencies}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Corrective Actions:</b> {corrective}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Next Scheduled Inspection:</b> {ground_truth['next_inspection_date']}", styles["Normal"]))

    doc.build(elements)

    gt_path = output_path.replace(".pdf", "_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def build_full_corpus():
    """Build the complete MIDAS benchmark corpus (30 invoices, 20 specs, 15 inspections)."""
    print("Generating MIDAS synthetic document corpus...")

    # 1. Invoices
    inv_dir = OUT_DIR / "invoices"
    inv_dir.mkdir(parents=True, exist_ok=True)
    # Generate canonical INV-0047
    generate_invoice(str(inv_dir / "INV-0047.pdf"), invoice_id="INV-0047", vendor="TATA METALIKS LTD")
    for i in range(1, 30):
        generate_invoice(str(inv_dir / f"INV-{i:04d}.pdf"))

    # 2. Specs
    spec_dir = OUT_DIR / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    generate_technical_spec(str(spec_dir / "SPEC-0003.pdf"), spec_id="SPEC-0003")
    for i in range(1, 20):
        generate_technical_spec(str(spec_dir / f"SPEC-{i:04d}.pdf"))

    # 3. Inspection reports
    insp_dir = OUT_DIR / "inspection"
    insp_dir.mkdir(parents=True, exist_ok=True)
    generate_inspection_report(str(insp_dir / "INSP-0011.pdf"), report_id="INSP-0011")
    for i in range(1, 15):
        generate_inspection_report(str(insp_dir / f"INSP-{i:04d}.pdf"))

    print(f"Corpus generation complete! 30 Invoices, 20 Specs, 15 Inspection reports saved in {OUT_DIR}.")


if __name__ == "__main__":
    build_full_corpus()
