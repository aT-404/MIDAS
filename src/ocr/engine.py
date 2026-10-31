"""MIDAS OCR Engine.

Executes PaddleOCR v4 / RapidOCR text detection and recognition on preprocessed
document images with angle classification, bounding box geometry, and word/line
confidence scoring.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np


class OCREngine:
    """Singleton wrapper for PaddleOCR v4 / RapidOCR engine."""

    _instance = None
    _engine = None
    _engine_type = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OCREngine, cls).__new__(cls)
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        """Initialize the OCR engine."""
        # Try RapidOCR (PP-OCRv4 ONNX runtime) first
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._engine_type = "rapidocr"
            return
        except Exception as e:
            pass

        # Try PaddleOCR if installed
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._engine_type = "paddleocr"
            return
        except Exception as e:
            pass

        self._engine = None
        self._engine_type = "none"

    def process(self, image_input: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """Run OCR on image or file path and return structured bounding boxes & text.

        Returns:
            List of dicts: [
                {
                    "text": str,
                    "confidence": float (0.0 to 1.0),
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "center": [cx, cy]
                },
                ...
            ]
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found for OCR: {path_str}")
            image = cv2.imread(path_str)
            if image is None:
                from src.preprocessing.enhance import load_image_or_pdf
                image = load_image_or_pdf(path_str)
        else:
            image = image_input

        results: List[Dict[str, Any]] = []

        if self._engine_type == "rapidocr":
            # RapidOCR call: ocr(img) -> (ocr_result, elapse_list)
            # ocr_result: list of [dt_boxes, rec_res, score]
            ocr_res, _ = self._engine(image)
            if ocr_res:
                for item in ocr_res:
                    bbox, text, score = item
                    # Format bbox as standard python floats
                    box_coords = [[float(pt[0]), float(pt[1])] for pt in bbox]
                    cx = float(np.mean([pt[0] for pt in box_coords]))
                    cy = float(np.mean([pt[1] for pt in box_coords]))
                    results.append({
                        "text": str(text).strip(),
                        "confidence": float(score),
                        "bbox": box_coords,
                        "center": [cx, cy],
                    })

        elif self._engine_type == "paddleocr":
            paddle_res = self._engine.ocr(image, cls=True)
            if paddle_res and paddle_res[0]:
                for line in paddle_res[0]:
                    bbox, (text, score) = line
                    box_coords = [[float(pt[0]), float(pt[1])] for pt in bbox]
                    cx = float(np.mean([pt[0] for pt in box_coords]))
                    cy = float(np.mean([pt[1] for pt in box_coords]))
                    results.append({
                        "text": str(text).strip(),
                        "confidence": float(score),
                        "bbox": box_coords,
                        "center": [cx, cy],
                    })

        # Sort reading order: top-to-bottom, then left-to-right
        results.sort(key=lambda item: (item["center"][1] // 15, item["center"][0]))
        return results


def run_ocr(image_input: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
    """Convenience helper to run OCR via the singleton engine."""
    engine = OCREngine()
    return engine.process(image_input)


def format_ocr_transcript(ocr_results: List[Dict[str, Any]]) -> str:
    """Format OCR text lines into structured readable transcript."""
    return "\n".join(r["text"] for r in ocr_results if r["text"])
