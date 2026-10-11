"""A bounded provider check: color fixtures and identical dataset images."""

import base64
import io
import json

from PIL import Image

from inference import DECISIONS, ROOT, decision_payload, normalize_decisions, post


def main():
    output = ROOT / "evidence/provider-probes.jsonl"
    if output.exists():
        raise SystemExit("Probe evidence already exists; preserve it before another run.")
    samples = json.loads((ROOT / "evidence/samples.json").read_text())["samples"]
    total = 0
    for provider in ("Cloudflare", "PrimeIntellect"):
        for color in ("red", "blue"):
            buf = io.BytesIO()
            Image.new("RGB", (256, 256), color).save(buf, format="PNG")
            part = {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
            }
            payload = {
                "model": "cloudflare/clef-flash",
                "state": ["Inspect the supplied image.", part],
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "What is the dominant color in the image?",
                        "criteria": {
                            "red": "Red",
                            "blue": "Blue",
                            "green": "Green",
                            "unknown": "Cannot tell",
                        },
                    }
                },
            }
            payload["provider"] = {"only": [provider], "allow_fallbacks": False}
            record = post(DECISIONS, payload)
            record.update(probe="color", expected=color, requested_provider=provider)
            with output.open("a") as out:
                out.write(json.dumps(record) + "\n")
            cost = record.get("response", {}).get("usage", {}).get("cost")
            print(
                provider, color, record.get("status"), record.get("response", {}).get("answers"), flush=True
            )
            if cost is None:
                raise SystemExit("Unknown probe cost; stopped")
            total += cost
        for index in (0, 7, 16):
            payload = decision_payload("cloudflare/clef-flash", samples[index])
            payload["provider"] = {"only": [provider], "allow_fallbacks": False}
            record = post(DECISIONS, payload)
            record.update(probe="same_image", sample=samples[index]["id"], requested_provider=provider)
            try:
                record["normalized"] = normalize_decisions(record.get("response", {}))
            except (ValueError, KeyError, TypeError) as exc:
                record["validation_error"] = str(exc)
            with output.open("a") as out:
                out.write(json.dumps(record) + "\n")
            cost = record.get("response", {}).get("usage", {}).get("cost")
            print(
                provider,
                samples[index]["id"],
                record.get("status"),
                record.get("normalized", {}).get("origin"),
                flush=True,
            )
            if cost is None:
                raise SystemExit("Unknown probe cost; stopped")
            total += cost
    print(f"Probe cost: ${total:.6f}")


if __name__ == "__main__":
    main()
