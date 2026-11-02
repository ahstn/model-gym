"""Localhost-only EmbeddingGemma2 worker, reached through an authenticated SSH tunnel."""

import base64
import hashlib
import io
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
    TOKENIZERS_PARALLELISM="false",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
)

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(2026110110000)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
MODEL_PATH = Path("/workspace/embeddinggemma2-public")
MODEL = SentenceTransformer(
    str(MODEL_PATH),
    device="cuda",
    config_kwargs={"audio_config": None},
    model_kwargs={"dtype": torch.float32},
)
MODEL.eval()
GPU_LOCK = threading.Lock()
IDENTITY = {
    "model": "google/embeddinggemma-2",
    "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
    "dtype": "float32",
    "device": torch.cuda.get_device_name(0),
    "dimensions": 768,
    "audio_config": None,
    "max_soft_tokens": 280,
    "normalization": "L2",
    "tf32": False,
    "parameters": sum(p.numel() for p in MODEL.parameters()),
    "software": {
        name: version(name)
        for name in ["torch", "torchvision", "transformers", "sentence-transformers", "numpy", "Pillow"]
    },
    "weights_sha256": hashlib.sha256((MODEL_PATH / "model.safetensors").read_bytes()).hexdigest(),
    "server_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


def load_image(item):
    if "source" in item:
        process = subprocess.run(
            ["/workspace/embedding-venv/bin/python", "/workspace/corpus_render_exact.py"],
            input=json.dumps(item),
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
            env={**os.environ, "PYTHONPATH": "/workspace/render-deps"},
        )
        if process.returncode:
            raise ValueError("Remote source reconstruction failed: " + process.stderr[-500:])
        payload = Path(process.stdout.strip()).read_bytes()
    else:
        payload = base64.b64decode(item["data"], validate=True)
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError("Delivered image hash mismatch")
    with Image.open(io.BytesIO(payload)) as image:
        return {"image": image.convert("RGB")}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, IDENTITY)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 20_000_000:
                raise ValueError("Invalid request length")
            request = json.loads(self.rfile.read(length))
            records = request["images"]
            if not 1 <= len(records) <= 8:
                raise ValueError("Batch length out of bounds")
            with ThreadPoolExecutor(4) as pool:
                images = list(pool.map(load_image, records))
            started = time.monotonic()
            with GPU_LOCK, torch.inference_mode():
                output = MODEL.encode(
                    images,
                    batch_size=4,
                    processing_kwargs={"image": {"max_soft_tokens": 280}},
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                output = torch.nn.functional.normalize(output.float(), p=2, dim=-1).cpu().numpy()
            if output.shape != (len(records), 768) or not np.isfinite(output).all():
                raise ValueError("Invalid embedding result")
            self.reply(
                200,
                {
                    "embeddings": output.tolist(),
                    "seconds": time.monotonic() - started,
                    "sha256": [item["sha256"] for item in records],
                },
            )
        except Exception as error:  # noqa: BLE001 -- HTTP boundary must report failures without returning fabricated vectors.
            self.reply(400, {"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    print(json.dumps(IDENTITY), flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
