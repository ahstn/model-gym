import torch

from decision.modeling import Batch, code_logits, code_logits_shared_prefix
from decision.prompt import MAX_OPTIONS


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(config).float().eval()


def _reference(model, encoded, num_options, code_ids):
    n, width = len(encoded), max(len(e) for e in encoded)
    input_ids = torch.zeros((n, width), dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    batch = Batch(
        input_ids=input_ids,
        attention_mask=mask,
        last_index=torch.tensor([len(e) - 1 for e in encoded]),
        num_options=torch.tensor(num_options),
        targets=torch.zeros((n, MAX_OPTIONS)),
    )
    return code_logits(model, batch, code_ids)


def _sliding_softcap_model():
    """Gemma 3 text model: sliding-window layers shorter than the prefix, and final logit soft-capping."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=3.0,
    )
    return Gemma3ForCausalLM(config).float().eval()


def _check(prefix_len, model=None):
    model = model or _model()
    gen = torch.Generator().manual_seed(1)
    prefix = torch.randint(1, 300, (prefix_len,), generator=gen).tolist()
    suffix_lens = [3, 20, 7, 12, 5]
    encoded = [prefix + torch.randint(1, 300, (s,), generator=gen).tolist() for s in suffix_lens]
    num_options = [2, 5, 3, MAX_OPTIONS, 4]
    code_ids = torch.randint(1, 300, (MAX_OPTIONS,), generator=gen)
    with torch.inference_mode():
        expected = _reference(model, encoded, num_options, code_ids)
        got = code_logits_shared_prefix(model, encoded, num_options, code_ids, max_batch_tokens=250, device="cpu")
    assert torch.equal(torch.isinf(got), torch.isinf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(got[finite], expected[finite], atol=1e-4, rtol=0)


def test_shared_prefix_matches_independent_rows_across_chunks():
    _check(100)  # 250 // (100 + 20) = 2 rows per chunk -> 3 chunks


def test_short_prefix_falls_back():
    _check(10)


def test_shared_prefix_keeps_sliding_window_and_softcap():
    _check(100, _sliding_softcap_model())  # prefix 100 > window 16: sliding layers hold a cropped cache


def test_code_logits_match_full_lm_logits_with_softcap():
    model = _sliding_softcap_model()
    gen = torch.Generator().manual_seed(2)
    encoded = [torch.randint(1, 300, (n,), generator=gen).tolist() for n in (40, 25)]
    code_ids = torch.randint(1, 300, (MAX_OPTIONS,), generator=gen)
    with torch.inference_mode():
        got = _reference(model, encoded, [MAX_OPTIONS, MAX_OPTIONS], code_ids)
        for i, ids in enumerate(encoded):
            full = model(input_ids=torch.tensor([ids])).logits[0, -1]
            torch.testing.assert_close(got[i], full[code_ids], atol=1e-4, rtol=0)
