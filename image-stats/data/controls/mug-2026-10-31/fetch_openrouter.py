"""Generate one mug image through OpenRouter and keep the decoded bytes unchanged.

Writes <name>-<n>.<ext> and <name>.response.json (image data and reasoning signatures elided) to OUT.
Needs OPENROUTER_API_KEY. Each call is billed.

    REF=images/reference-openrouter-product-placement.png
    python fetch_openrouter.py OUT images sourceful/riverflow-v2.6-pro riverflow --reference $REF
    python fetch_openrouter.py OUT chat google/gemini-3-pro-image nano-banana-pro --reference $REF
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import urllib.request

from pathlib import Path

PROMPT = Path(__file__).resolve().parents[4] / "docs" / "mug-provenance-prompt-2026-10-31.txt"
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def _data_url(path: Path) -> str:
    return f"data:{mimetypes.guess_type(path.name)[0]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _post(url: str, body: dict) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def _save(out: Path, name: str, index: int, media_type: str, data: bytes) -> str:
    path = out / f"{name}-{index}.{EXTENSIONS.get(media_type, 'bin')}"
    path.write_bytes(data)
    print(path.name, media_type, len(data))
    return f"<{len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("api", choices=["chat", "images"])
    parser.add_argument("model")
    parser.add_argument("name")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    # fal received the prompt without its trailing newline; send the same bytes here.
    prompt = PROMPT.read_text().rstrip("\n")
    reference = {"type": "image_url", "image_url": {"url": _data_url(args.reference)}} if args.reference else None

    if args.api == "images":
        body = {"model": args.model, "prompt": prompt} | ({"input_references": [reference]} if reference else {})
        response = _post("https://openrouter.ai/api/v1/images", body)
        for i, image in enumerate(response.get("data", [])):
            image["b64_json"] = _save(
                args.out, args.name, i, image.get("media_type"), base64.b64decode(image["b64_json"])
            )
    else:
        content = [{"type": "text", "text": prompt}] + ([reference] if reference else [])
        body = {
            "model": args.model,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": content}],
        }
        response = _post("https://openrouter.ai/api/v1/chat/completions", body)
        message = response["choices"][0]["message"]
        for i, image in enumerate(message.get("images", [])):
            header, encoded = image["image_url"]["url"].split(",", 1)
            image["image_url"]["url"] = _save(
                args.out, args.name, i, header[5:].split(";")[0], base64.b64decode(encoded)
            )
        for detail in message.get("reasoning_details", []):
            if "data" in detail:
                detail["data"] = f"<{len(detail['data'])} chars elided>"
    (args.out / f"{args.name}.response.json").write_text(json.dumps(response, indent=1))


if __name__ == "__main__":
    main()
