"""Feature baselines preserve syntax and local encoder inference contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from transformers import BertConfig, BertModel, BertTokenizerFast, ModernBertConfig, ModernBertModel

from model_gym.baselines import EmbeddingClassifier, FrozenEncoder, LexicalClassifier
from model_gym.metrics import softmax


@pytest.mark.parametrize("mode,structured", [("char", False), ("word", False), ("union", True)])
def test_lexical_case_and_punctuation_survive_training_and_reload(
    tmp_path: Path, mode: str, *, structured: bool
) -> None:
    commands = ["git clean -f", "git clean -F", "printf x>out", "printf x>>out", "printf x|out"]
    labels = [4, 0, 3, 1, 2]
    classifier = LexicalClassifier.fit(commands, labels, C=100, mode=mode, structured=structured)
    probabilities = classifier.predict_proba(commands)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)
    np.testing.assert_allclose(probabilities, softmax(classifier.predict_logits(commands)))
    classifier.save(tmp_path / "bundle")
    loaded = LexicalClassifier.load(tmp_path / "bundle")
    np.testing.assert_array_equal(loaded.predict_proba(commands), probabilities)


@pytest.fixture(params=["bert", "modernbert"])
def local_encoder(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    directory = tmp_path / "source-encoder"
    directory.mkdir()
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "git",
        "status",
        "cat",
        "curl",
        "echo",
        "rm",
        "x",
        "out",
        ">",
        "|",
        "search_classification",
        ":",
    ]
    (directory / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(
        vocab_file=str(directory / "vocab.txt"),
        do_lower_case=False,
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(directory)
    dimensions = {
        "vocab_size": len(vocabulary),
        "hidden_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
    }
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        if request.param == "bert":
            model = BertModel(BertConfig(**dimensions, hidden_dropout_prob=0.3))
        else:
            model = ModernBertModel(
                ModernBertConfig(
                    **dimensions,
                    bos_token_id=2,
                    cls_token_id=2,
                    eos_token_id=3,
                    sep_token_id=3,
                    local_attention=16,
                    embedding_dropout=0.3,
                    mlp_dropout=0.3,
                    reference_compile=False,
                )
            )
        model.save_pretrained(directory, safe_serialization=True)
    return directory


@pytest.mark.parametrize("pooling,normalize", [("mean", True), ("cls", False)])
def test_frozen_extraction_matches_pooling_and_single_row_batches(
    local_encoder: Path, pooling: str, *, normalize: bool
) -> None:
    commands = ["git status", "cat x", "curl x > out | cat out"]
    encoder = FrozenEncoder(
        local_encoder,
        prefix="search_classification: ",
        pooling=pooling,
        normalize=normalize,
        max_seq_length=16,
        device="cpu",
    )
    tokens = encoder.tokenizer(
        ["search_classification: " + command for command in commands],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16,
    )
    with torch.no_grad():
        hidden = encoder.model(**tokens).last_hidden_state
        if pooling == "mean":
            mask = tokens["attention_mask"].unsqueeze(-1)
            expected = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            expected = hidden[:, 0]
        if normalize:
            expected = torch.nn.functional.normalize(expected, dim=1)
    features = encoder.encode(commands, batch_size=3)
    np.testing.assert_allclose(features, expected.numpy(), atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(features, encoder.encode(commands, batch_size=1), atol=1e-6, rtol=1e-5)


def test_embedding_bundle_reloads_without_original_encoder(local_encoder: Path, tmp_path: Path) -> None:
    commands = ["rm x", "git status", "curl x", "echo x", "cat out"]
    labels = [3, 0, 4, 1, 2]
    encoder = FrozenEncoder(
        local_encoder, prefix="search_classification: ", normalize=False, max_seq_length=16, device="cpu"
    )
    features = encoder.encode(commands)
    classifier = EmbeddingClassifier.fit(features, labels, encoder, C=1000)
    probabilities = classifier.predict_proba(commands)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)
    np.testing.assert_array_equal(encoder.encode(commands), features)
    bundle = tmp_path / "bundle"
    classifier.save(bundle)
    local_encoder.rename(tmp_path / "unavailable-original")
    loaded = EmbeddingClassifier.load(bundle, device="cpu")
    np.testing.assert_allclose(loaded.predict_logits(commands), classifier.predict_logits(commands), atol=1e-6)
    np.testing.assert_allclose(loaded.predict_proba(commands), probabilities, atol=1e-6)


def test_lexical_training_rejects_incomplete_class_support() -> None:
    with pytest.raises(ValueError, match="all 5 label indices"):
        LexicalClassifier.fit(["rm", "git", "cat", "echo"], [0, 1, 2, 3])


def test_embedding_training_rejects_level_numbers_instead_of_indices(local_encoder: Path) -> None:
    encoder = FrozenEncoder(local_encoder, device="cpu")
    with pytest.raises(ValueError, match="label indices"):
        EmbeddingClassifier.fit(encoder.encode(["rm", "git", "cat", "echo", "curl"]), [1, 2, 3, 4, 5], encoder)
