"""Deterministic synthetic images with known properties."""

import io

import cv2
import numpy as np

from PIL import Image


def texture(height=384, width=576, seed=0):
    """Band-limited random detail in mid-tones, standing in for a sharp photograph."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(height, width, 3)).astype(np.float32)
    detail = cv2.GaussianBlur(noise, (0, 0), 1.2)
    detail = (detail - detail.mean()) / detail.std()
    return np.clip(128 + 40 * detail, 0, 255).astype(np.uint8)


def gradient(height=384, width=576):
    ramp = np.linspace(60, 190, width, dtype=np.float32)
    return np.repeat(np.repeat(ramp[None, :, None], height, axis=0), 3, axis=2).astype(np.uint8)


def blur(rgb, sigma):
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


def add_noise(rgb, sigma, seed=1):
    rng = np.random.default_rng(seed)
    noisy = rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape[:2])[..., None]
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def solid(colour, height=384, width=576):
    return np.full((height, width, 3), colour, dtype=np.uint8)


def encode(rgb, fmt="PNG", **options):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format=fmt, **options)
    return buffer.getvalue()
