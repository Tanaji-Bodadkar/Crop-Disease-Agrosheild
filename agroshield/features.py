"""Image validation, preprocessing, and lightweight visual features."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = 128
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
FEATURE_VERSION = "numpy-hsv16x3-hog4x4x8-v1"


def decode_image(raw: bytes) -> Image.Image:
    """Validate uploaded bytes and return an orientation-correct RGB image."""
    if not raw:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than 10 MB. Please resize it and try again.")
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large. Resize the image to 20 megapixels or less and try again.")
            source.verify()
        with Image.open(BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width < 32 or image.height < 32:
                raise ValueError("Image is too small. Upload a crop image at least 32×32 pixels.")
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This file could not be read as an image. Try a JPG or PNG image.") from exc


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize an RGB image to the fixed model input size."""
    return np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Extract normalized HSV histograms and coarse spatial HOG features."""
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        rgb = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))

    # Convert RGB to OpenCV-like HSV ranges without a compiled vision package.
    rgb_float = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
    maximum = np.max(rgb_float, axis=2)
    minimum = np.min(rgb_float, axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8
    red_max = nonzero & (maximum == r)
    green_max = nonzero & (maximum == g)
    blue_max = nonzero & (maximum == b)
    hue[red_max] = np.mod((g[red_max] - b[red_max]) / delta[red_max], 6.0)
    hue[green_max] = (b[green_max] - r[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (r[blue_max] - g[blue_max]) / delta[blue_max] + 4.0
    hue = hue * 30.0  # degrees / 2, matching the conventional 0..180 range
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-8) * 255.0
    value = maximum * 255.0
    hsv = (hue, saturation, value)

    color_features = []
    for channel, value_range in zip(hsv, ((0, 180), (0, 256), (0, 256))):
        hist, _ = np.histogram(channel, bins=16, range=value_range)
        hist = hist / max(float(hist.sum()), 1.0)
        color_features.extend(hist.tolist())

    gray = np.dot(rgb_float, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    hog_features = []
    cell = IMAGE_SIZE // 4
    for row in range(4):
        for col in range(4):
            y0, y1 = row * cell, (row + 1) * cell
            x0, x1 = col * cell, (col + 1) * cell
            bins = np.floor(angle[y0:y1, x0:x1] / (180.0 / 8)).astype(np.int32) % 8
            hist = np.bincount(bins.ravel(), weights=magnitude[y0:y1, x0:x1].ravel(), minlength=8)
            hist = hist / max(float(np.linalg.norm(hist)), 1e-8)
            hog_features.extend(hist.tolist())

    return np.asarray(color_features + hog_features, dtype=np.float32)
