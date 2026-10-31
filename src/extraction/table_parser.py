"""MIDAS Table Parsing Engine.

Extracts structured 2D table matrices from digital vector PDFs (via PyMuPDF lattice/stream)
and scanned raster images (via OpenCV morphological grid detection + OCR).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np
import fitz  # PyMuPDF
from src.ocr.engine import run_ocr


def extract_tables_from_pdf(pdf_path: Union[str, Path]) -> List[List[List[str]]]:
    """Extract tables from digital PDF pages using PyMuPDF native table recognition.

    Returns:
        List of tables (one per table found), where each table is a 2D list of strings:
        [
            [ ["Header1", "Header2", ...], ["Row1Col1", "Row1Col2", ...], ... ]
        ]
    """
    path_str = str(pdf_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"PDF not found: {path_str}")

    doc = fitz.open(path_str)
    all_tables: List[List[List[str]]] = []

    for page_idx, page in enumerate(doc):
        # PyMuPDF find_tables extracts vector table structures with high fidelity
        try:
            tabs = page.find_tables()
            for tab in tabs:
                extracted = tab.extract()
                if extracted and len(extracted) > 1:
                    # Clean None values to empty strings
                    cleaned_table = [
                        [str(cell).strip() if cell is not None else "" for cell in row]
                        for row in extracted
                    ]
                    all_tables.append(cleaned_table)
        except Exception:
            pass

    doc.close()

    # Fallback if no vector tables detected: rasterize and run image table extraction
    if not all_tables:
        from src.preprocessing.enhance import load_image_or_pdf
        img = load_image_or_pdf(pdf_path, dpi=200)
        table_data = extract_tables_from_image(img)
        if table_data:
            all_tables.append(table_data)

    return all_tables


def extract_tables_from_image(
    image_input: Union[str, Path, np.ndarray],
    table_bbox: Optional[List[float]] = None
) -> List[List[str]]:
    """Extract tabular grid and cell contents from a raster image or crop.

    Args:
        image_input: Image array or file path
        table_bbox: Optional [x1, y1, x2, y2] to restrict search to a layout box

    Returns:
        2D list of row strings: [ [cell_0, cell_1, ...], ... ]
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return []
    else:
        img = image_input

    # Crop table region if bbox specified
    if table_bbox is not None and len(table_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in table_bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return []

    # Run OCR on the table image region to get bounding boxes
    ocr_results = run_ocr(img)
    if not ocr_results:
        return []

    # Cluster words into rows based on vertical center coordinate
    # Group words within vertical distance threshold (approx 12-18 pixels)
    row_clusters: List[List[Dict[str, Any]]] = []
    # Sort by vertical center first
    sorted_by_y = sorted(ocr_results, key=lambda item: item["center"][1])

    y_thresh = max(14, int(h * 0.025))

    current_cluster: List[Dict[str, Any]] = []
    current_y = None

    for item in sorted_by_y:
        cy = item["center"][1]
        if current_y is None:
            current_y = cy
            current_cluster.append(item)
        elif abs(cy - current_y) <= y_thresh:
            current_cluster.append(item)
            # Running average of y center
            current_y = (current_y + cy) / 2.0
        else:
            if current_cluster:
                # Sort cluster left-to-right
                current_cluster.sort(key=lambda it: it["center"][0])
                row_clusters.append(current_cluster)
            current_cluster = [item]
            current_y = cy

    if current_cluster:
        current_cluster.sort(key=lambda it: it["center"][0])
        row_clusters.append(current_cluster)

    # Format into 2D row strings
    table_matrix: List[List[str]] = []
    for row in row_clusters:
        row_cells = [item["text"].strip() for item in row if item["text"].strip()]
        if row_cells:
            table_matrix.append(row_cells)

    return table_matrix
