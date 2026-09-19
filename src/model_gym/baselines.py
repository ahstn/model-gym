"""Case-sensitive lexical and frozen-encoder linear classifiers.

All labels and output columns use zero-based severity order, L1 through L5.
These models return uncalibrated logits/probabilities; action policy and
calibration belong to the caller. Commands are data, never executed.
"""

from __future__ import annotations

import json
import re

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from transformers import AutoModel, AutoTokenizer

from model_gym.labels import NUM_LABELS
from model_gym.metrics import softmax
from model_gym.modeling import resolve_device

LEXICAL_NAME = "lexical.joblib"
EMBEDDING_HEAD_NAME = "embedding-head.joblib"
ENCODER_SETTINGS_NAME = "encoder-settings.json"
ENCODER_DIRECTORY = "encoder"

# Keep flags, quotes, path punctuation, and shell operators rather than applying
# the default word analyzer's punctuation-stripping/lowercasing.
_WORD_PATTERN = r"[^\s;&|<>()]+|&&|\|\||&>>|&>|>>|<<<|<<-?|<>|>\||[;&|<>()]"
_SYNTAX_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?<![\w/-])-[A-Za-z][\w-]*",
        r"(?<![\w/-])--[A-Za-z][\w-]*",
        r"(?<![\w/-])-[A-Z][\w-]*",
        r"&&",
        r"\|\|",
        r"(?<!\|)\|(?!\|)",
        r"(?<!>)>>(?!>)",
        r"(?<![>])>(?![>|])",
        r"(?<!<)<<(?!<)",
        r"(?<!<)<(?!<)",
        r";",
        r"\$\(",
        r"`",
        r"(?<![\w/])/(?!/)",
        r"(?<![\w.])\./",
        r"(?<![\w.])\.\./",
        r"~/",
        r"\b[A-Za-z]:[\\/]",
        r"(?<!\\)\\\\[^\s\\]+",
    )
)


def _check_commands(commands: Sequence[str]) -> None:
    if isinstance(commands, (str, bytes)) or any(not isinstance(command, str) for command in commands):
        raise TypeError("commands must be a sequence of strings, not a single string")


def _training_labels(labels: Sequence[int], count: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != count or values.dtype.kind not in "iu":
        raise ValueError("labels must be a one-dimensional integer array matching the training rows")
    if not np.array_equal(np.unique(values), np.arange(NUM_LABELS)):
        raise ValueError(f"training requires all {NUM_LABELS} label indices, exactly 0 through {NUM_LABELS - 1}")
    return values


def _check_head(head: LogisticRegression) -> None:
    if not isinstance(head, LogisticRegression) or not np.array_equal(head.classes_, np.arange(NUM_LABELS)):
        raise ValueError("classifier head must contain all five label columns in severity order")


def _fit_head(features: Any, labels: np.ndarray, C: float) -> LogisticRegression:
    # lbfgs uses multinomial loss for five classes, and its default penalty is L2.
    # Omitting multi_class keeps compatibility with sklearn versions removing it.
    return LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=0).fit(features, labels)


def _syntax_features(commands: Sequence[str]) -> np.ndarray:
    """Count literal syntax forms, not parsed semantics or environment state."""
    counts = np.empty((len(commands), len(_SYNTAX_PATTERNS)), dtype=np.float64)
    for row, command in enumerate(commands):
        for column, pattern in enumerate(_SYNTAX_PATTERNS):
            counts[row, column] = sum(1 for _ in pattern.finditer(command))
    return counts


@dataclass
class LexicalClassifier:
    """TF-IDF features with a deterministic, L2-regularized multinomial head."""

    vectorizer: FeatureUnion
    head: LogisticRegression
    mode: str
    structured: bool

    @classmethod
    def fit(
        cls,
        commands: Sequence[str],
        labels: Sequence[int],
        *,
        C: float = 1.0,
        mode: Literal["char", "word", "union"] = "union",
        structured: bool = False,
    ) -> LexicalClassifier:
        """Fit all five classes without modifying command case or punctuation.

        ``structured`` adds counts of lexical flag/operator/path forms. These
        features are not a shell AST and make no claims about filesystem state.
        """
        _check_commands(commands)
        values = _training_labels(labels, len(commands))
        if mode not in {"char", "word", "union"}:
            raise ValueError("mode must be 'char', 'word', or 'union'")
        transforms: list[tuple[str, Any]] = []
        if mode in {"char", "union"}:
            transforms.append(
                (
                    "char",
                    TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, sublinear_tf=True),
                )
            )
        if mode in {"word", "union"}:
            transforms.append(
                (
                    "word",
                    TfidfVectorizer(
                        token_pattern=_WORD_PATTERN,
                        ngram_range=(1, 2),
                        lowercase=False,
                        sublinear_tf=True,
                    ),
                )
            )
        if structured:
            transforms.append(("syntax", FunctionTransformer(_syntax_features, validate=False)))
        vectorizer = FeatureUnion(transforms)
        features = vectorizer.fit_transform(commands)
        return cls(vectorizer, _fit_head(features, values, C), mode, structured)

    def predict_logits(self, commands: Sequence[str]) -> np.ndarray:
        """Return uncalibrated logits shaped ``(len(commands), 5)``."""
        _check_commands(commands)
        if len(commands) == 0:
            return np.empty((0, NUM_LABELS), dtype=np.float64)
        return self.head.decision_function(self.vectorizer.transform(commands))

    def predict_proba(self, commands: Sequence[str]) -> np.ndarray:
        """Return probabilities in L1-through-L5 column order."""
        return softmax(self.predict_logits(commands))

    def save(self, directory: Path | str) -> None:
        """Write ``lexical.joblib``; only load this pickle from trusted sources."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / LEXICAL_NAME, compress=3)

    @classmethod
    def load(cls, directory: Path | str) -> LexicalClassifier:
        """Load a trusted-local bundle; joblib/pickle can execute arbitrary code."""
        classifier = joblib.load(Path(directory) / LEXICAL_NAME)
        if not isinstance(classifier, cls):
            raise TypeError("bundle does not contain a LexicalClassifier")
        _check_head(classifier.head)
        return classifier


class FrozenEncoder:
    """Extract frozen AutoModel features without adding a classification head.

    Both Hub model identifiers and local ``save_pretrained`` directories work.
    Vocabulary and weights are never changed. Inputs are right-padded, with the
    literal ``prefix`` prepended before tokenization; callers supply any model's
    task instruction, including Nomic's documented task prefix.
    """

    def __init__(
        self,
        model_name: Path | str,
        *,
        prefix: str = "",
        pooling: Literal["mean", "cls"] = "mean",
        normalize: bool = True,
        max_seq_length: int = 256,
        device: str | torch.device | None = None,
    ) -> None:
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        if not isinstance(max_seq_length, int) or isinstance(max_seq_length, bool) or max_seq_length < 1:
            raise ValueError("max_seq_length must be a positive integer")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        self.model_name = str(model_name)
        self.prefix = prefix
        self.pooling = pooling
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.device = resolve_device(str(device) if device is not None else None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.requires_grad_(requires_grad=False)
        self.model.eval()

    @property
    def settings(self) -> dict[str, Any]:
        """Serializable constructor settings, including the resolved device."""
        return {
            "model_name": self.model_name,
            "prefix": self.prefix,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_seq_length": self.max_seq_length,
            "device": str(self.device),
        }

    @torch.no_grad()
    def encode(self, commands: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Return float32 features in input order; padding never enters pooling."""
        _check_commands(commands)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        features = np.empty((len(commands), self.model.config.hidden_size), dtype=np.float32)
        self.model.eval()
        for start in range(0, len(commands), batch_size):
            batch = [self.prefix + command for command in commands[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
                max_length=self.max_seq_length,
                padding=True,
            ).to(self.device)
            hidden = self.model(**encoded, return_dict=True).last_hidden_state.float()
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].to(hidden.dtype)
                pooled = torch.bmm(mask.unsqueeze(1), hidden).squeeze(1)
                pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            features[start : start + len(batch)] = pooled.cpu().numpy()
        return features


@dataclass
class EmbeddingClassifier:
    """A linear head learned from cached features, sharing its frozen encoder.

    Fitting candidates neither copies nor saves encoder weights. Select a head
    using cached features first, then save the desired self-contained bundle.
    """

    head: LogisticRegression
    encoder: FrozenEncoder

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[int],
        encoder: FrozenEncoder,
        *,
        C: float = 1.0,
    ) -> EmbeddingClassifier:
        """Fit only the multinomial head over supplied encoder feature rows."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != encoder.model.config.hidden_size:
            raise ValueError("features must be a matrix with the encoder's hidden dimension")
        values = _training_labels(labels, len(features))
        return cls(_fit_head(features, values, C), encoder)

    def predict_logits(self, commands: Sequence[str]) -> np.ndarray:
        """Encode commands and return uncalibrated five-class logits."""
        features = self.encoder.encode(commands)
        if len(features) == 0:
            return np.empty((0, NUM_LABELS), dtype=np.float64)
        return self.head.decision_function(features)

    def predict_proba(self, commands: Sequence[str]) -> np.ndarray:
        """Return probabilities in L1-through-L5 column order."""
        return softmax(self.predict_logits(commands))

    def save(self, directory: Path | str) -> None:
        """Save head/settings and local encoder weights, tokenizer, and config.

        ``embedding-head.joblib`` is a trusted-local-only pickle. Encoder files
        use Hugging Face's ``save_pretrained`` format with safetensors weights.
        No training feature cache or second copy of the encoder is serialized.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        encoder_directory = directory / ENCODER_DIRECTORY
        self.encoder.model.save_pretrained(encoder_directory, safe_serialization=True)
        self.encoder.tokenizer.save_pretrained(encoder_directory)
        joblib.dump(self.head, directory / EMBEDDING_HEAD_NAME, compress=3)
        (directory / ENCODER_SETTINGS_NAME).write_text(
            json.dumps(self.encoder.settings, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        device: str | torch.device | None = None,
    ) -> EmbeddingClassifier:
        """Load a trusted-local bundle, never the original encoder source.

        joblib/pickle can execute arbitrary code. ``device`` selects deployment
        hardware (auto-detected by default), not the machine recorded at save.
        """
        directory = Path(directory)
        head = joblib.load(directory / EMBEDDING_HEAD_NAME)
        _check_head(head)
        settings = json.loads((directory / ENCODER_SETTINGS_NAME).read_text(encoding="utf-8"))
        encoder = FrozenEncoder(
            directory / ENCODER_DIRECTORY,
            prefix=settings["prefix"],
            pooling=settings["pooling"],
            normalize=settings["normalize"],
            max_seq_length=settings["max_seq_length"],
            device=device,
        )
        if head.n_features_in_ != encoder.model.config.hidden_size:
            raise ValueError("saved head and encoder feature dimensions do not match")
        return cls(head, encoder)
