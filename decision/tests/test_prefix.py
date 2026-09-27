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


def _check(prefix_len):
    model = _model()
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
