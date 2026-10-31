"""Save the previous displayed prompt with the user-approved photographic scope."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "prompt-ablation-2026-10-30"
PROJECT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("scope_previous_wire", PREVIOUS / "wire.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    combined = next(arm for arm in wire.load_arms(PREVIOUS / "arms.json") if arm["id"] == "combined")
    previous = wire.prompt_bundle(PROJECT / "prompts/visual-rating-v1", combined)
    prompt = json.loads(json.dumps(previous))
    policy = (HERE / "photo-scope.txt").read_text().strip()
    replacements = {
        "Judge paintings, graphics, and stylized images within their visible medium. Do not penalize them for failing to look photographic.":
            "Apply the photography eligibility rule before judging quality. For eligible photographs, judge the visible photographic style.",
        "Assess the work within its visible medium and style.":
            "For eligible photographs, assess the work within its visible photographic style.",
    }
    for before, after in replacements.items():
        if prompt["user"].count(before) != 1:
            raise ValueError("Expected exactly one prior rubric statement: " + before)
        prompt["user"] = prompt["user"].replace(before, after)
    prompt["user"] = policy + "\n\n" + prompt["user"]
    before = "Apply the same standards across subject types and visual media."
    if prompt["system"].count(before) != 1:
        raise ValueError("Prior system instruction changed")
    prompt["system"] = prompt["system"].replace(
        before,
        "Apply the supplied photography eligibility rule first, then consistent quality standards across eligible photographic subjects.",
    )
    prompt["version"] = "visual-rating-v1-combined-photo-scope-2026-10-31"
    prompt.pop("sha256")
    prompt["sha256"] = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()
    if prompt["response_format"] != previous["response_format"]:
        raise ValueError("Schema must remain unchanged")
    (HERE / "prompt.json").write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n")
    (HERE / "prompt-provenance.json").write_text(json.dumps({
        "previous_arm": "combined",
        "previous_prompt_sha256": previous["sha256"],
        "current_prompt_sha256": prompt["sha256"],
        "schema_changed": False,
        "quality_additions_changed": False,
        "user_confirmed_floor": "Clearly non-photographic content receives 1.00 for both scores.",
        "replacement_statements": replacements,
        "inputs": {str(path): sha(path) for path in (
            Path(__file__), HERE / "photo-scope.txt", PREVIOUS / "wire.py", PREVIOUS / "arms.json",
            PROJECT / "prompts/visual-rating-v1/system.txt", PROJECT / "prompts/visual-rating-v1/joint.txt",
        )},
    }, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"prompt_sha256": prompt["sha256"], "previous_arm": "combined", "schema_changed": False}))


if __name__ == "__main__":
    main()
