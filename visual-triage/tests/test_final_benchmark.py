import copy
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_benchmark import validate_preflight


@pytest.fixture
def frozen_panel():
    prompt = b'{"context": "A frozen rubric."}\n'
    freeze = {
        "prompt_hashes": {"quality": hashlib.sha256(prompt).hexdigest()},
        "reasoning": {"gemini": "low", "luna": "low"},
        "max_output_tokens": 4096,
    }
    manifest = {"samples": [{"id": f"quality-{i:03d}", "delivered_sha256": f"{i:064x}"} for i in range(60)]}
    return freeze, manifest, copy.deepcopy(manifest), prompt


def test_frozen_prompt_accepts_exact_bytes_and_rejects_even_whitespace_change(frozen_panel):
    freeze, manifest, reference, prompt = frozen_panel
    validate_preflight("quality", freeze, manifest, reference, prompt)
    with pytest.raises(ValueError, match="prompt bytes"):
        validate_preflight("quality", freeze, manifest, reference, prompt + b"\n")


def test_all_sixty_reference_hashes_must_match(frozen_panel):
    freeze, manifest, reference, prompt = frozen_panel
    reference["samples"][-1]["delivered_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 values"):
        validate_preflight("quality", freeze, manifest, reference, prompt)


def test_duplicate_reference_cannot_hide_behind_dictionary_conversion(frozen_panel):
    freeze, manifest, reference, prompt = frozen_panel
    reference["samples"].append(reference["samples"][0])
    with pytest.raises(ValueError, match="exactly 60"):
        validate_preflight("quality", freeze, manifest, reference, prompt)
    reference["samples"].pop()
    reference["samples"][-1] = reference["samples"][0]
    with pytest.raises(ValueError, match="60 unique"):
        validate_preflight("quality", freeze, manifest, reference, prompt)


@pytest.mark.parametrize("effort", [None, "xhigh", "high"])
def test_luna_rejects_missing_or_unsupported_effort(frozen_panel, effort):
    freeze, manifest, reference, prompt = frozen_panel
    freeze["reasoning"]["luna"] = effort
    with pytest.raises(ValueError, match="reasoning effort"):
        validate_preflight("quality", freeze, manifest, reference, prompt)


@pytest.mark.parametrize("cap", [True, 127, 16385, None])
def test_invalid_frozen_output_cap_fails_before_requests(frozen_panel, cap):
    freeze, manifest, reference, prompt = frozen_panel
    freeze["max_output_tokens"] = cap
    with pytest.raises(ValueError, match="output-token cap"):
        validate_preflight("quality", freeze, manifest, reference, prompt)
