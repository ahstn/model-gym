"""Load the current image-labeling configuration and build provider requests."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

MODELS = ("luna", "deepseek", "liquid", "jev")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_profile(directory: Path) -> dict:
    """Load one coherent prompt/route pack without making any provider calls."""
    directory = Path(directory)
    files = {
        "teacher": directory / "prompts" / "teacher.json",
        "decision": directory / "prompts" / "decision.json",
        "routes": directory / "routes.json",
    }
    data = {key: json.loads(path.read_text()) for key, path in files.items()}
    versions = {value.get("version") for value in data.values()}
    if versions != {"photo-labeling-current-v1"}:
        raise ValueError("Current labeling profile versions do not match")
    if set(data["routes"].get("routes", {})) != set(MODELS):
        raise ValueError("Current labeling profile has missing or unexpected routes")
    for model in ("luna", "deepseek"):
        if data["routes"]["routes"][model].get("kind") != "chat":
            raise ValueError("Teacher route must use chat completions")
    if data["teacher"].get("response_format", {}).get("type") != "json_schema":
        raise ValueError("Teacher response must use the evaluated JSON schema")
    if data["decision"].get("exclusion_threshold") != 0.5:
        raise ValueError("Eligibility threshold differs from the evaluated rule")
    for model in ("liquid", "jev"):
        route = data["routes"]["routes"][model]
        if (
            route.get("kind") != "decisions"
            or route.get("requested_model") != data["decision"]["models"][model]["model"]
        ):
            raise ValueError("Decision route and question model must match")
        questions = data["decision"]["models"][model]["questions"]
        if set(questions) != {"technical_quality", "aesthetics", "excluded"}:
            raise ValueError("Decision questions differ from the evaluated contract")
        if questions["excluded"].get("type") != "noul":
            raise ValueError("Eligibility decision must use the yes/no contract")
    data["sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    return data


def request_payload(profile: dict, model: str, image_bytes: bytes) -> dict:
    """Build the evaluated request; image preparation and HTTP dispatch are separate."""
    if model not in MODELS:
        raise ValueError("Unknown current labeling model")
    if not isinstance(image_bytes, bytes) or not image_bytes.startswith(PNG_SIGNATURE):
        raise ValueError("A prepared PNG image is required")
    route = profile["routes"]["routes"][model]
    image = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
            "detail": "high",
        },
    }
    if route["kind"] == "decisions":
        prompt = profile["decision"]
        return {
            **copy.deepcopy(route["settings"]),
            "state": [prompt["common"], image],
            "questions": copy.deepcopy(prompt["models"][model]["questions"]),
        }
    prompt = profile["teacher"]
    return {
        **copy.deepcopy(route["settings"]),
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": [{"type": "text", "text": prompt["user"]}, image]},
        ],
        "response_format": copy.deepcopy(prompt["response_format"]),
    }
