"""Record provider model capabilities/prices without exposing credentials."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import requests

DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01")


def main():
    records = {}
    for model in ("openai/gpt-6-luna", "liquid/d1.1", "typesafe/jev-1.14"):
        url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        records[url] = response.json()
    url = "https://api.runinfra.ai/v1/models"
    response = requests.get(
        url, headers={"Authorization": "Bearer " + os.environ["RUN_INFRA_API_KEY"]}, timeout=30
    )
    response.raise_for_status()
    body = response.json()
    records[url] = {"data": [row for row in body["data"] if row["id"] == "deepseek-v4-1-flash-1017"]}
    if not records[url]["data"]:
        raise ValueError("Requested DeepSeek model unavailable")
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "api-provider-snapshot.json"
    target.write_text(
        json.dumps({"checked_at": datetime.now(UTC).isoformat(), "sources": records}, indent=2) + "\n"
    )
    summary = {}
    for url, value in records.items():
        if "endpoints" in url:
            summary[url] = [
                {
                    key: endpoint.get(key)
                    for key in ("name", "provider_name", "pricing", "context_length", "supported_parameters")
                }
                for endpoint in value["data"]["endpoints"]
                if endpoint.get("provider_name") in {"OpenAI", "Liquid", "TypeSafe"}
            ]
        else:
            summary[url] = value
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
