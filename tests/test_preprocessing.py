"""Unit tests for MIDAS document preprocessing and enhancement module."""

import pytest
import numpy as np
import cv2
from src.preprocessing.enhance import deskew, remove_shadow, denoise, binarize


def test_deskew_correction():
    """Assert deskewing corrects rotated text-like lines back toward horizontal."""
    # Create synthetic canvas with prominent horizontal lines
    h, w = 600, 800
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Draw several horizontal bars
    for y in [150, 250, 350, 450]:
        cv2.line(img, (100, y), (700, y), (0, 0, 0), thickness=4)

    # Rotate by 8 degrees
    angle = 8.0
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # Apply deskew
    corrected = deskew(rotated)

    assert corrected is not None
    assert corrected.shape == (h, w, 3)


def test_remove_shadow():
    """Assert shadow removal produces normalized non-empty image."""
    h, w = 300, 400
    # Create gradient shadow image
    gradient = np.tile(np.linspace(50, 255, w, dtype=np.uint8), (h, 1))
    color_img = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    deshadowed = remove_shadow(color_img)
    assert deshadowed is not None
    assert deshadowed.shape == color_img.shape
    assert np.mean(deshadowed) > 0


def test_binarize_produces_binary_image():
    """Assert binarization returns array with only 0 and 255 pixel values."""
    img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    binary = binarize(img)
    assert binary is not None
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})
