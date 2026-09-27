"""Prompt rendering and the option-code readout.

The model reads one prompt and we take the next-token logits at the last prompt position. The prompt ends with
ANSWER_PREFIX ("Answer:") and has NO trailing space. A byte-level BPE tokenizer (MiniCPM5 uses one) attaches the
space to the next word, so "Answer: A" tokenizes as [..., "Answer", ":", "ĠA"]. The answer token is therefore the
space-prefixed code token " A", and the prompt must stop right after ":". If the prompt ended with "Answer: ", the
trailing space would become its own token and the code token after it would be the rare "A" without a space.

Readout only accepts a code when "<prompt> <code>" encodes to exactly the prompt ids plus one new id, and that id is
unique among accepted codes. Inside the option list, lines like "\\nAB. text" may tokenize the code differently (for
example merged with the newline). That does not matter: only the token at the answer position is read.
"""

from __future__ import annotations

import string

from itertools import product
from typing import TYPE_CHECKING

from decision.schema import NOUL_OPTIONS, Decision, Kind

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

MAX_OPTIONS = 255
ANSWER_PREFIX = "Answer:"
SYSTEM_PROMPT = (
    "You are a decision model. Read the state and the question. Then answer with the code of exactly one option."
)
KIND_INSTRUCTIONS: dict[Kind, str] = {
    "choice": "Choose the single best option.",
    "score": "Options are ordered levels from lowest to highest. Choose the level that fits best.",
    "noul": "Answer the question with Yes or No.",
}

_PROBE = Decision(
    id="probe",
    dataset="probe",
    source="probe",
    kind="noul",
    state="The door is open.",
    question="Is the door open?",
    options=NOUL_OPTIONS,
    target=(0.0, 1.0),
    group_id="probe",
)


def _candidate_codes() -> list[str]:
    letters = string.ascii_uppercase
    return list(letters) + ["".join(p) for p in product(letters, repeat=2)]


class Readout:
    """Renders a Decision into a prompt ending right before the option-code token and maps codes to token ids."""

    codes: tuple[str, ...]
    code_token_ids: tuple[int, ...]

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        # Placeholder codes for the probe: only the rendered prefix up to ANSWER_PREFIX is used.
        self.codes = ("A", "B")
        probe = self.render(_PROBE)
        base = self._ids(probe)
        codes: list[str] = []
        ids: list[int] = []
        seen: set[int] = set()
        for code in _candidate_codes():
            if len(codes) == MAX_OPTIONS:
                break
            answer_id = self.answer_id(base, probe, code)
            if answer_id is None or answer_id in seen:
                continue
            codes.append(code)
            ids.append(answer_id)
            seen.add(answer_id)
        if len(codes) < MAX_OPTIONS:
            raise ValueError(f"tokenizer offers only {len(codes)} single-token option codes, need {MAX_OPTIONS}")
        self.codes = tuple(codes)
        self.code_token_ids = tuple(ids)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def answer_id(self, prompt_ids: list[int], prompt: str, code: str) -> int | None:
        """Id of `code` at the answer position after `prompt`, or None if it is not exactly one appended token."""
        full = self._ids(prompt + " " + code)
        if len(full) != len(prompt_ids) + 1 or full[:-1] != prompt_ids:
            return None
        return full[-1]

    def render(self, d: Decision) -> str:
        if len(d.options) > len(self.codes):
            raise ValueError(f"{d.id}: {len(d.options)} options, readout has {len(self.codes)} codes")
        option_lines = "\n".join(f"{code}. {text}" for code, text in zip(self.codes, d.options, strict=False))
        user = (
            f"State:\n{d.state}\n\nQuestion:\n{d.question}\n\nOptions:\n{option_lines}\n\n"
            f"{KIND_INSTRUCTIONS[d.kind]} Reply with the option code only."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return text + ANSWER_PREFIX

    def encode(self, d: Decision, *, max_tokens: int | None = None) -> list[int] | None:
        ids = self._ids(self.render(d))
        if max_tokens is not None and len(ids) > max_tokens:
            return None
        return ids
