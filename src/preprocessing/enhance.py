"""MIDAS Preprocessing & Document Enhancement Engine.

Performs robust image deskewing, shadow removal, adaptive binarization,
and noise reduction to maximize downstream OCR accuracy on photographed,
scanned, or degraded document inputs.
"""

import os
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF


def load_image_or_pdf(input_path: Union[str, Path], page_index: int = 0, dpi: int = 300) -> np.ndarray:
    """Load an image file or render a PDF page into an OpenCV BGR numpy array.

    Args:
        input_path: Path to document (PDF, PNG, JPG, TIFF, etc.)
        page_index: Page to extract if PDF (0-indexed)
        dpi: Target rendering resolution for PDF rasterization

    Returns:
        np.ndarray: BGR image array
    """
    path_str = str(input_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Document file not found: {path_str}")

    ext = Path(path_str).suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(path_str)
        if len(doc) == 0:
            raise ValueError(f"Empty PDF document: {path_str}")
        page_idx = min(page_index, len(doc) - 1)
        page = doc[page_idx]

        # Render at target DPI (standard 72 pt/in)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap to numpy array
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
        # PyMuPDF pixmap is RGB; convert to OpenCV BGR
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        doc.close()
        return img_bgr

    # Standard image loading
    image = cv2.imread(path_str)
    if image is None:
        # Fallback with PIL for stubborn formats
        pil_img = Image.open(path_str).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return image


def deskew(image: np.ndarray, max_skew_angle: float = 15.0, min_line_len_ratio: float = 0.25) -> np.ndarray:
    """Correct document rotation using Hough line transform.

    Filters out short lines (e.g., checkboxes, small dividers) that confuse
    the angle estimator, retaining only dominant text-baseline or table-border
    lines longer than min_line_len_ratio * width.

    Args:
        image: BGR image
        max_skew_angle: Maximum rotation in degrees to correct (skips if larger)
        min_line_len_ratio: Minimum line length as a ratio of image width

    Returns:
        np.ndarray: Deskewed BGR image
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform to get explicit line segment lengths
    min_line_length = int(w * min_line_len_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=min_line_length, maxLineGap=20)

    if lines is None or len(lines) == 0:
        # Fallback to standard HoughLines if no long segments found
        std_lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if std_lines is None:
            return image
        angles = []
        for rho, theta in std_lines[:, 0]:
            angle = (theta * 180.0 / np.pi) - 90.0
            if abs(angle) <= max_skew_angle:
                angles.append(angle)
    else:
        angles = []
        for line in lines:
            pts = line.flatten()
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            # Keep near-horizontal lines within bounds
            if abs(angle) <= max_skew_angle:
                angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))

    # Avoid micro-rotations below 0.2 degrees to prevent interpolation blur
    if abs(median_angle) < 0.2:
        return image

    # Compute rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def remove_shadow(image: np.ndarray) -> np.ndarray:
    """Normalize uneven lighting, flash gradients, and shadows across document page.

    Divides the original image by a morphological background estimate to
    flatten shading and brighten dark corners.
    """
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dilate with large rectangular kernel to capture background illumination
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

    # Gaussian blur the background estimation
    background = cv2.medianBlur(background, 21)

    # Normalize image against estimated illumination
    diff = 255 - cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Convert back to 3-channel if input was 3-channel
    if len(image.shape) == 3:
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return norm


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove salt-and-pepper scan noise and sensor grain."""
    if len(image.shape) == 3:
        # Fast non-local means for color
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=7, hColor=7,
            templateWindowSize=7, searchWindowSize=21
        )
    return cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=7, searchWindowSize=21)


def binarize(image: np.ndarray, block_size: int = 31, c_val: int = 10) -> np.ndarray:
    """Produce clean adaptive Gaussian thresholded binary image for crisp text boundaries."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_val
    )
    return binary


def preprocess(
    input_doc: Union[str, Path, np.ndarray],
    out_dir: Optional[Union[str, Path]] = "outputs/preprocessed",
    return_path: bool = True
) -> Union[str, np.ndarray]:
    """Execute complete MIDAS pre-processing pipeline on document.

    Pipeline stages:
    1. Ingestion / PDF rasterization to 300 DPI
    2. Dominant text-line Deskew
    3. Illumination flattening and shadow removal
    4. Non-local means Denoising
    5. Adaptive Gaussian Binarization & contrast enhancement

    Args:
        input_doc: Path to file or numpy array
        out_dir: Directory where preprocessed image is saved
        return_path: If True, writes image to disk and returns path string;
                     if False, returns (image_array, clean_path)

    Returns:
        Path string or np.ndarray depending on return_path
    """
    if isinstance(input_doc, (str, Path)):
        img = load_image_or_pdf(input_doc)
        doc_stem = Path(input_doc).stem
    else:
        img = input_doc.copy()
        doc_stem = "frame"

    # Step 1: Deskew
    deskewed = deskew(img)

    # Step 2: Shadow removal
    deshadowed = remove_shadow(deskewed)

    # Step 3: Denoise
    denoised = denoise(deshadowed)

    # Step 4: Binarization
    binary = binarize(denoised)

    # Convert binary to 3-channel for standard OCR engines
    clean_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if return_path:
        out_folder = Path(out_dir) if out_dir else Path("outputs/preprocessed")
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / f"{doc_stem}_preprocessed.png"
        cv2.imwrite(str(out_path), clean_bgr)
        return str(out_path)

    return clean_bgr
