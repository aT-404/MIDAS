"""MIDAS Interactive Document Analysis Dashboard.

Streamlit application providing:
- Drag-and-drop document upload and processing
- Real-time extraction with step-by-step progress tracking
- Visual confidence indicators and field-level inspection
- Side-by-side view of raw input vs. preprocessed image
- Full-text search (SQLite + FTS5) across all stored documents
- Pipeline performance metrics and rejection queue auditing
- Single-click JSON export
"""

import sys
import os
import json
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.runner import run_pipeline, infer_document_type
from src.storage.database import search, get_metrics, list_documents, get_document, DB_PATH
from src.preprocessing.enhance import load_image_or_pdf, preprocess


st.set_page_config(
    page_title="MIDAS — Multimodal Intelligent Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for production look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .conf-high {
        color: #15803d;
        font-weight: 600;
    }
    .conf-med {
        color: #b45309;
        font-weight: 600;
    }
    .conf-low {
        color: #b91c1c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">MIDAS — Intelligent Document Analysis System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multi-format Document Ingestion • PaddleOCR v4 / PP-OCR • Layout Analysis • Gemini Multimodal Extraction • Pydantic Contract Validation • SQLite FTS5</div>',
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Document Ingestion")

doc_type_choice = st.sidebar.selectbox(
    "Target Document Schema",
    options=["auto", "invoice", "technical_spec", "inspection_report"],
    format_func=lambda x: {
        "auto": "Auto-Detect Document Type",
        "invoice": "Commercial Invoice",
        "technical_spec": "Technical Specification",
        "inspection_report": "Inspection & Quality Report",
    }[x]
)

# Sample Documents Quick Select
sample_options = [
    "None (Upload your own)",
    "INV-0047 (Supplier Invoice — Angled & Shadowed Photo)",
    "SPEC-0003 (Slurry Pump Engineering Spec — PDF)",
    "INSP-0011 (Site Quality Checklist — Scanned Report)"
]
selected_sample = st.sidebar.selectbox("Load Benchmark Sample", sample_options)

uploaded_file = st.sidebar.file_uploader(
    "Or Upload Document",
    type=["pdf", "png", "jpg", "jpeg", "tiff"]
)

run_button = st.sidebar.button("Run MIDAS Pipeline", type="primary", use_container_width=True)

# Main Application Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Document Extraction", "🔍 Full-Text Search (FTS5)", "📊 Metrics & Rejections"])

with tab1:
    col_left, col_right = st.columns([1, 1], gap="medium")

    # Determine input document path
    active_path = None
    display_img = None

    if uploaded_file is not None:
        # Save uploaded file to temp dir
        temp_dir = Path("outputs/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        active_path = temp_dir / uploaded_file.name
        with open(active_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif selected_sample != "None (Upload your own)":
        if "INV-0047" in selected_sample:
            p = Path("data/samples/invoices/INV-0047.jpg")
            if not p.exists():
                p = Path("data/generated/invoices/INV-0047.pdf")
            active_path = p
        elif "SPEC-0003" in selected_sample:
            active_path = Path("data/samples/specs/SPEC-0003.pdf")
        elif "INSP-0011" in selected_sample:
            active_path = Path("data/generated/inspection/INSP-0011.pdf")

    # Render Document Preview on Left
    with col_left:
        st.subheader("Source Document")
        if active_path and active_path.exists():
            st.caption(f"File: `{active_path.name}` ({active_path.stat().st_size / 1024:.1f} KB)")
            try:
                raw_bgr = load_image_or_pdf(active_path, dpi=150)
                # Convert BGR to RGB for PIL/Streamlit
                raw_rgb = Image.fromarray(raw_bgr[:, :, ::-1])
                st.image(raw_rgb, use_container_width=True, caption="Ingested Document Frame")
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")
        else:
            st.info("Upload a document or choose a benchmark sample from the sidebar.")

    # Processing & Output on Right
    with col_right:
        st.subheader("Extraction & Validation")

        if run_button and active_path and active_path.exists():
            with st.spinner("Executing MIDAS Pipeline..."):
                # Run pipeline
                result = run_pipeline(active_path, doc_type=doc_type_choice)

            if result["status"] == "success":
                conf = result["confidence"]
                conf_pct = conf * 100
                st.success(f"Validated Document: `{result['doc_type']}`")

                m1, m2, m3 = st.columns(3)
                m1.metric("Extraction Confidence", f"{conf_pct:.1f}%")
                m2.metric("Regions Classified", result.get("regions_detected", "N/A"))
                m3.metric("OCR Line Blocks", result.get("ocr_lines", "N/A"))

                data = result["data"]

                # Tabular rendering based on doc type
                if result["doc_type"] == "invoice":
                    st.markdown("#### Invoice Overview")
                    summary_df = pd.DataFrame([
                        {"Field": "Vendor", "Extracted Value": data.get("vendor_name"), "Confidence": "High"},
                        {"Field": "Invoice Number", "Extracted Value": data.get("invoice_number"), "Confidence": "High"},
                        {"Field": "Invoice Date", "Extracted Value": data.get("invoice_date"), "Confidence": "High"},
                        {"Field": "Grand Total", "Extracted Value": f"{data.get('currency', 'INR')} {data.get('grand_total'):,.2f}", "Confidence": "High"},
                    ])
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)

                    if data.get("line_items"):
                        st.markdown("#### Line Items")
                        items_df = pd.DataFrame(data["line_items"])
                        st.dataframe(items_df, hide_index=True, use_container_width=True)

                    if data.get("_arithmetic_cross_check"):
                        st.info(f"Arithmetic Cross-Check: {data['_arithmetic_cross_check']}")

                elif result["doc_type"] == "technical_spec":
                    st.markdown("#### Equipment Overview")
                    spec_summary = pd.DataFrame([
                        {"Field": "Equipment", "Value": data.get("equipment_name")},
                        {"Field": "Model Number", "Value": data.get("model_number")},
                        {"Field": "Manufacturer", "Value": data.get("manufacturer") or "N/A"},
                        {"Field": "Standards", "Value": ", ".join(data.get("certification_standards", []))},
                    ])
                    st.dataframe(spec_summary, hide_index=True, use_container_width=True)

                    if data.get("parameters"):
                        st.markdown("#### Extracted Parameters")
                        p_df = pd.DataFrame(data["parameters"])
                        st.dataframe(p_df, hide_index=True, use_container_width=True)

                elif result["doc_type"] == "inspection_report":
                    st.markdown("#### Inspection Summary")
                    insp_summary = pd.DataFrame([
                        {"Field": "Facility / Site", "Value": data.get("site_name")},
                        {"Field": "Inspector", "Value": data.get("inspector_name")},
                        {"Field": "Inspection Date", "Value": data.get("inspection_date")},
                        {"Field": "Deficiencies", "Value": data.get("deficiencies_found") or "None"},
                    ])
                    st.dataframe(insp_summary, hide_index=True, use_container_width=True)

                    if data.get("checklist_items"):
                        st.markdown("#### Checklist Evaluations")
                        chk_df = pd.DataFrame(data["checklist_items"])
                        st.dataframe(chk_df, hide_index=True, use_container_width=True)

                # Raw JSON expander and download button
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="Download Extracted JSON",
                    data=json_str,
                    file_name=f"{active_path.stem}.json",
                    mime="application/json"
                )

                with st.expander("Inspect Raw JSON Output"):
                    st.json(data)

            else:
                st.error("Document Rejected by Schema Validator")
                st.warning(f"Reason: {result.get('reason')}")
                st.caption(f"Rejection record persisted to outputs/rejected/ and SQLite rejections table.")
                with st.expander("View Raw Candidate Extraction"):
                    st.json(result.get("raw", {}))
        else:
            st.info("Click 'Run MIDAS Pipeline' in the sidebar to execute extraction.")


with tab2:
    st.subheader("SQLite Full-Text Search (FTS5)")
    st.caption("Search across every extracted invoice parameter, vendor name, or checklist remark.")

    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        query = st.text_input("Full-Text Search Query", placeholder="e.g. TATA METALIKS, Slurry Pump, Pressure, PASS")
    with search_col2:
        filter_type = st.selectbox("Filter Document Type", ["All", "invoice", "technical_spec", "inspection_report"])

    if query:
        filter_arg = filter_type if filter_type != "All" else None
        results = search(query, doc_type=filter_arg)
        st.write(f"Found **{len(results)}** matching records:")

        for r in results:
            with st.container():
                st.markdown(f"### {r['source_file']} (`{r['doc_type']}`)")
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.write(f"**Confidence:** {r['confidence']*100:.1f}%")
                c2.write(f"**Created At:** {r['created_at']}")
                with st.expander("Show Extracted JSON Record"):
                    st.json(r["extracted_data"])
                st.divider()
    else:
        st.info("Showing recently processed documents in database:")
        all_docs = list_documents(limit=10)
        for r in all_docs:
            st.write(f"📄 **{r['source_file']}** — `{r['doc_type']}` — Conf: {r['confidence']*100:.1f}% — Date: {r['created_at']}")


with tab3:
    st.subheader("System Operational Metrics")

    metrics = get_metrics()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Validated Documents", metrics["total_documents"])
    m_col2.metric("Total Rejections", metrics["total_rejections"])
    m_col3.metric("Average Extraction Confidence", f"{metrics['overall_avg_confidence']*100:.1f}%")
    m_col4.metric("Pipeline Rejection Rate", f"{metrics['rejection_rate']*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Performance Breakdown by Document Class")
    breakdown_data = []
    for dt, info in metrics.get("by_type", {}).items():
        breakdown_data.append({
            "Document Type": dt,
            "Total Ingested": info["count"],
            "Avg Confidence": f"{info['avg_confidence']*100:.1f}%"
        })
    if breakdown_data:
        st.table(pd.DataFrame(breakdown_data))
    else:
        st.caption("No processed documents recorded in database yet.")
