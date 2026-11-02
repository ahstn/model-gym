"""Build the v2 two-score profile from pinned repository rubric sources."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
VERSION = "photo-labeling-v2-2026-11-01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    profile = PACKAGE / "labeling"
    old = HERE.parent / "rehearsal-4000-2026-11-01/api-teacher-prompt.json"
    source = PACKAGE / "prompts/visual-rating-v2"
    teacher = json.loads(old.read_text())
    technical = (source / "technical-quality.txt").read_text().split("\nReturn only a JSON object", 1)[0]
    media = "Judge graphics and paintings within their visible medium rather than penalizing them for not being photographs."
    if technical.count(media) != 1:
        raise ValueError("v2 media sentence changed; review the photography scope adaptation")
    technical = technical.replace(
        media, "Apply the photography eligibility rule before judging technical quality."
    )
    before, remaining = teacher["user"].split("TECHNICAL QUALITY RUBRIC\n\n", 1)
    _, after = remaining.split("AESTHETICS RUBRIC\n\n", 1)
    teacher["user"] = (
        before + "TECHNICAL QUALITY RUBRIC\n\n" + technical + "\n\nAESTHETICS RUBRIC\n\n" + after
    )
    teacher["version"] = VERSION
    teacher.pop("sha256", None)
    teacher["response_format"]["json_schema"]["name"] = "joint_visual_rating_v2_photo_scope"
    system = (source / "system.txt").read_text()
    system = system.replace(
        "Apply the same standards across subject types and visual media.",
        "Apply the photography eligibility rule first, then consistent quality standards across eligible photographic subjects.",
    )
    teacher["system"] = system
    target = profile / "prompts/teacher.json"
    target.write_text(json.dumps(teacher, indent=2, ensure_ascii=False) + "\n")
    for name in ("prompts/decision.json", "routes.json"):
        path = profile / name
        value = json.loads(path.read_text())
        value["version"] = VERSION
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    record = {
        "version": VERSION,
        "technical_rubric": "visual-rating-v2",
        "source_commit": "9a2f7cef8265690019d41b0158a558dcfd8a0d6d",
        "source_hashes": {
            str(p.relative_to(PACKAGE)): sha(p)
            for p in (
                source / "system.txt",
                source / "technical-quality.txt",
                source / "response.schema.json",
                old,
            )
        },
        "profile_hashes": {
            name: sha(profile / name)
            for name in ("prompts/teacher.json", "prompts/decision.json", "routes.json")
        },
        "adaptations": [
            "v2 technical rubric up to its standalone output instructions is embedded into the two-task teacher prompt",
            "explicit photography eligibility supersedes the v2 allowance for graphics and paintings",
            "existing independent aesthetic rubric and scoring clarifications are retained",
            "standalone numeric output instructions are replaced by the evaluated exact two-decimal string joint schema",
            "Liquid and Jev keep the revised model-specific Decisions questions from 9a2f7ce unchanged",
        ],
        "reuse_policy": "Old Luna/DeepSeek prompt hashes differ; all chat-model observations are freshly requested.",
        "stats_sent_to_models": False,
    }
    (profile / "profile-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
