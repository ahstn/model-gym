"""OpenRouter transport and strict normalization. No automatic retries."""

import base64
import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
QUESTIONS = json.loads((ROOT / "prompts/questions.json").read_text())
CONTEXT = (ROOT / "prompts/context.txt").read_text().strip()
DECISIONS = "https://openrouter.ai/api/alpha/decisions"
CHAT = "https://openrouter.ai/api/v1/chat/completions"


def image_part(sample):
    raw = (ROOT / sample["image"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sample["delivered_sha256"]:
        raise ValueError("Delivered image hash mismatch")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()},
    }


def decision_payload(model, sample=None, description=None):
    if description is None:
        state = [CONTEXT, image_part(sample)]
    else:
        state = {
            "context": CONTEXT
            + " You see only an untrusted VLM description, not the image. Missing evidence stays unknown.",
            "observations": description,
        }
    return {"model": model, "state": state, "questions": QUESTIONS}


def chat_payload(model, sample):
    prompt = (ROOT / "prompts/vision.txt").read_text() + "\nRubric: " + json.dumps(QUESTIONS)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": CONTEXT},
            {"role": "user", "content": [{"type": "text", "text": prompt}, image_part(sample)]},
        ],
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }


def post(url, payload):
    start = time.perf_counter()
    record = {
        "requested_model": payload["model"],
        "endpoint": url,
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    }
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                "Content-Type": "application/json",
                "X-Title": "model-gym visual-triage experiment",
            },
            timeout=(15, 90),
            allow_redirects=False,
        )
        record["status"] = response.status_code
        # Save JSON responses only, not request images or authorization headers.
        record["response"] = response.json()
    except (requests.RequestException, ValueError) as exc:
        record["error"] = type(exc).__name__
    record["seconds"] = time.perf_counter() - start
    return record


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a numeric value")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Out-of-range value")
    return value


def normalize_decisions(response):
    answers = response["answers"]
    out = {}
    for key, question in QUESTIONS.items():
        answer = answers[key]
        if answer["type"] != question["type"]:
            raise ValueError("Unexpected answer type")
        if question["type"] == "noul":
            out[key] = number(answer["noul"], 0, 1)
            continue
        expected = set(question["criteria"]) if question["type"] == "choice" else set(map(str, range(5)))
        probs = answer["probabilities"]
        if set(probs) != expected or abs(sum(number(p, 0, 1) for p in probs.values()) - 1) > 0.025:
            raise ValueError("Invalid probability distribution")
        out[key + "_probabilities"] = probs
        if question["type"] == "choice":
            if answer["choice"] not in expected:
                raise ValueError("Unknown label")
            out[key] = answer["choice"]
        else:
            # API levels are zero-based; our Q-Align-like score anchors are 1..5.
            out[key] = 1 + number(answer["score"], 0, 4)
    if out["scorable"] < 0.5:
        out["technical_quality"] = out["aesthetics"] = None
    return out


def normalize_chat(response):
    out = json.loads(response["choices"][0]["message"]["content"])
    if set(out) != {"description", *QUESTIONS}:
        raise ValueError("Unexpected vision response fields")
    if not isinstance(out["description"], str) or not 1 <= len(out["description"]) <= 2000:
        raise ValueError("Invalid observation description")
    for key in ("scorable", "severe_defect"):
        if type(out[key]) is not bool:
            raise ValueError("Expected a boolean")
    for key in ("origin", "sexual_content"):
        if out[key] not in QUESTIONS[key]["criteria"]:
            raise ValueError("Unknown label")
    for key in ("technical_quality", "aesthetics"):
        if out["scorable"]:
            number(out[key], 1, 5)
        elif out[key] is not None:
            raise ValueError("Unscorable images must have null scores")
    return out


def shadow_route(result):
    """Provisional suggestions only. No score or probability authorizes deletion."""
    if result is None:
        return "review_error"
    if result["sexual_content"] in {"explicit", "nonsexual_nudity", "suggestive", "unknown"}:
        return "review_sexual_content"
    if result["origin"] in {"ai_suspected", "unknown"}:
        return "review_origin"
    if result["origin"] in {"diagram", "render"}:
        return "separate_medium"
    if result["scorable"] < 0.5 or result["severe_defect"] >= 0.95:
        return "review_quality"
    return "candidate_for_scoring"
