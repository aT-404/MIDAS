"""MIDAS Layout Analysis & Region Classifier.

Detects and segments document regions (Title, Header, Table, Paragraph/Text,
List, Figure, Footer) to maintain layout hierarchy and prevent tabular data
from bleeding into continuous text prompts.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
import cv2
import numpy as np


class LayoutDetector:
    """Document layout region analyzer."""

    def __init__(self, model_name: str = "contour_heuristic"):
        self.model_name = model_name
        self._lp_model = None
        if model_name != "contour_heuristic":
            try:
                import layoutparser as lp
                self._lp_model = lp.Detectron2LayoutModel(
                    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                )
            except Exception:
                self._lp_model = None

    def detect(self, image_input: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect layout blocks and classify their semantic region types.

        Returns:
            List of dicts: [
                {
                    "type": "Title" | "Header" | "Table" | "Text" | "List" | "Footer" | "Figure",
                    "bbox": [x1, y1, x2, y2],
                    "score": float
                },
                ...
            ]
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            image = cv2.imread(path_str)
            if image is None:
                raise ValueError(f"Failed to load image for layout detection: {path_str}")
        else:
            image = image_input

        # If layoutparser is configured and loaded, use it
        if self._lp_model is not None:
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                layout = self._lp_model.detect(image_rgb)
                regions = []
                for block in layout:
                    regions.append({
                        "type": str(block.type),
                        "bbox": [float(block.block.x_1), float(block.block.y_1),
                                 float(block.block.x_2), float(block.block.y_2)],
                        "score": float(block.score),
                    })
                return regions
            except Exception:
                pass

        # Robust Computer Vision & Morphological Region Segmentation
        return self._detect_heuristic(image)

    def _detect_heuristic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """High-precision computer vision layout decomposition using morphological kernels."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Invert: white text/lines on black background
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        regions: List[Dict[str, Any]] = []

        # 1. Detect Tables via horizontal and vertical line intersection
        table_boxes = self._detect_table_boxes(thresh, w, h)
        table_mask = np.zeros((h, w), dtype=np.uint8)

        for tb in table_boxes:
            x1, y1, x2, y2 = tb
            cv2.rectangle(table_mask, (x1, y1), (x2, y2), 255, -1)
            regions.append({
                "type": "Table",
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": 0.92,
            })

        # 2. Detect text clusters outside tables
        # Mask out table areas so table cells don't get detected as separate paragraphs
        non_table_thresh = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(table_mask))

        # Horizontal dilation to connect words into text lines and paragraphs
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.05), 5))
        dilated = cv2.dilate(non_table_thresh, kernel_text, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter noise
            if bw < w * 0.08 or bh < 12:
                continue

            x1, y1, x2, y2 = x, y, x + bw, y + bh

            # Classify based on position and aspect ratio
            # Top 15% of page with substantial width is Title or Header
            if y1 < h * 0.12 and bw > w * 0.3:
                region_type = "Title"
                score = 0.88
            elif y1 < h * 0.22:
                region_type = "Header"
                score = 0.82
            # Bottom 12% is Footer
            elif y2 > h * 0.88:
                region_type = "Footer"
                score = 0.85
            # Multi-line or tall blocks are Paragraph/Text
            elif bh > 40:
                region_type = "Text"
                score = 0.86
            else:
                region_type = "List"
                score = 0.80

            regions.append({
                "type": region_type,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": score,
            })

        # Ensure regions are sorted top-to-bottom
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return regions

    def _detect_table_boxes(self, thresh: np.ndarray, w: int, h: int) -> List[List[int]]:
        """Detect tabular bounding rectangles using morphological line kernels."""
        # Horizontal lines kernel
        h_len = max(20, int(w * 0.08))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        # Vertical lines kernel
        v_len = max(20, int(h * 0.04))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        # Table grid = intersection / sum of lines
        table_grid = cv2.add(h_lines, v_lines)

        # Close gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        table_closed = cv2.morphologyEx(table_grid, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(table_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_boxes = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # A table must have reasonable width and height
            area = bw * bh
            if bw > w * 0.40 and bh > h * 0.08 and area > (w * h * 0.04):
                table_boxes.append([x, y, x + bw, y + bh])

        return table_boxes


def detect_regions(image_input: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
    """Convenience helper to detect document layout regions."""
    detector = LayoutDetector()
    return detector.detect(image_input)
