"""Pinned Smaug inference, adapted from Abacus.AI's Apache-2.0 wrapper.

Upstream: abacusai/Smaug-AI-Image-Detector, c7c95951e04a87b884a4ad479401b1128c3903bd.
Three 528px crops, 140 vision tokens, average crop logits before sigmoid.
"""

from __future__ import annotations

import math
import time

from pathlib import Path

from PIL import Image, ImageOps

MODEL_ID = "abacusai/Smaug-AI-Image-Detector"
REVISION = "c7c95951e04a87b884a4ad479401b1128c3903bd"
DEFAULT_THRESHOLD = 0.5


def image_crops(image: Image.Image) -> list[Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if min(width, height) > 576:
        scale = 576 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    elif min(width, height) < 528:
        scale = 528 / min(width, height)
        image = image.resize((math.ceil(width * scale), math.ceil(height * scale)), Image.Resampling.BICUBIC)
    width, height = image.size
    cx, cy = (width - 528) // 2, (height - 528) // 2
    if width >= height:
        return [image.crop((x, cy, x + 528, cy + 528)) for x in (0, cx, width - 528)]
    return [image.crop((cx, y, cx + 528, y + 528)) for y in (0, cy, height - 528)]


class Smaug:
    def __init__(self, *, device: str | None = None, cache_dir: str | Path | None = None, offline: bool = False):
        import torch

        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        path = snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=cache_dir,
            local_files_only=offline,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
        )
        self.torch, self.device = torch, device
        self.encoder = SentenceTransformer(
            path,
            device=device,
            config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        )
        processor = self.encoder[0].processor.image_processor
        processor.max_soft_tokens = processor.image_seq_length = 140
        self.settings = {
            "model": MODEL_ID,
            "revision": REVISION,
            "crops": 3,
            "crop_size": 528,
            "vision_tokens": 140,
            "precision": "bfloat16" if device == "cuda" else "float32",
            "device": device,
        }

    def predict(self, path: str | Path) -> dict:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            logits = (
                self.encoder.encode(
                    [{"image": crop} for crop in image_crops(rgb)], batch_size=3, convert_to_tensor=True
                )
                .float()
                .view(1, 3)
            )
            score = float(self.torch.sigmoid(logits.mean(1)).item())
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return {
            **self.settings,
            "probability_ai": score,
            "seconds": time.perf_counter() - started,
            "latency_kind": "measured_from_decoded_RGB",
        }
