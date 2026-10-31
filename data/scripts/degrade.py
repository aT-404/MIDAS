"""MIDAS Degradation Simulator.

Simulates real-world scan and smartphone capture artifacts:
- Off-axis rotation / skew (5° to 12°)
- Uneven lighting gradients & corner shadows
- Sensor Gaussian noise
- Contrast degradation & optical blur
"""

import os
from pathlib import Path
from typing import Union, Tuple
import cv2
import numpy as np
import fitz  # PyMuPDF


def add_rotation(image: np.ndarray, angle: float = 8.0) -> np.ndarray:
    """Rotate image by specified angle with border replication."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def add_shadow(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Simulate a dark diagonal illumination shadow gradient across the top-left quadrant."""
    h, w = image.shape[:2]
    # Create 2D gradient mask
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    # Diagonal distance from top-left (0,0)
    dist = np.sqrt(xx**2 + yy**2)
    # Shadow factor: dark at top-left, normal at bottom-right
    shadow_mask = 1.0 - (intensity * np.exp(-dist * 1.8))
    shadow_mask = np.clip(shadow_mask, 0.2, 1.0)

    degraded = image.astype(np.float32)
    for c in range(3):
        degraded[:, :, c] = degraded[:, :, c] * shadow_mask

    return np.clip(degraded, 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Add zero-mean Gaussian sensor noise."""
    gauss = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degrade_document(
    input_pdf_or_img: Union[str, Path],
    output_path: Union[str, Path],
    rotation_angle: float = 8.0,
    add_shadow_effect: bool = True,
    noise_sigma: float = 10.0,
    dpi: int = 200
) -> str:
    """Degrade a clean document file into a realistic smartphone photo or poor-scan image."""
    in_str = str(input_pdf_or_img)
    out_str = str(output_path)
    os.makedirs(os.path.dirname(out_str), exist_ok=True)

    # Load / rasterize
    if in_str.lower().endswith(".pdf"):
        doc = fitz.open(in_str)
        page = doc[0]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, 3))
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        doc.close()
    else:
        img = cv2.imread(in_str)

    # 1. Rotate
    degraded = add_rotation(img, angle=rotation_angle)

    # 2. Add lighting shadow
    if add_shadow_effect:
        degraded = add_shadow(degraded, intensity=0.55)

    # 3. Add sensor noise
    if noise_sigma > 0:
        degraded = add_noise(degraded, sigma=noise_sigma)

    # 4. Slight lens blur
    degraded = cv2.GaussianBlur(degraded, (3, 3), 0.5)

    cv2.imwrite(out_str, degraded)
    return out_str


def build_samples():
    """Create sample test documents in data/samples/ for the demo walkthroughs."""
    print("Building degraded test samples (INV-0047, SPEC-0003, INSP-0011)...")

    # Degrade canonical INV-0047 into an 8° angled photo with lighting shadow
    inv_pdf = "data/generated/invoices/INV-0047.pdf"
    if os.path.exists(inv_pdf):
        degrade_document(
            inv_pdf,
            "data/samples/invoices/INV-0047.jpg",
            rotation_angle=8.0,
            add_shadow_effect=True,
            noise_sigma=12.0
        )
        print("  -> Created data/samples/invoices/INV-0047.jpg (Walkthrough sample)")

    # Clean digital spec sample
    spec_pdf = "data/generated/specs/SPEC-0003.pdf"
    if os.path.exists(spec_pdf):
        os.makedirs("data/samples/specs", exist_ok=True)
        import shutil
        shutil.copy(spec_pdf, "data/samples/specs/SPEC-0003.pdf")
        print("  -> Created data/samples/specs/SPEC-0003.pdf (Clean digital spec)")

    # Low-contrast scanned inspection report
    insp_pdf = "data/generated/inspection/INSP-0011.pdf"
    if os.path.exists(insp_pdf):
        degrade_document(
            insp_pdf,
            "data/samples/inspection/INSP-0011.jpg",
            rotation_angle=-4.0,
            add_shadow_effect=False,
            noise_sigma=15.0
        )
        print("  -> Created data/samples/inspection/INSP-0011.jpg (Scanned inspection report)")


if __name__ == "__main__":
    build_samples()
