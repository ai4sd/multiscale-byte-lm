import pytest
import torch
import torch.nn as nn

from mblm.model.multi_stage_token_embedding import MultiStageTokenEmbedding, _StageTokenEmbedding


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)


def _build(num_tokens=257, pad_id=256, model_dims=(1024, 1024), seq_lens=(1024, 8)):
    """Helper to construct the reversed module list like the legacy initializer."""
    modules = MultiStageTokenEmbedding.build(
        model_dims=model_dims,
        seq_lens=seq_lens,
        vocab_size=num_tokens,
        pad_token_id=pad_id,
    )
    return modules


def test_returns_reversed_order_modulelist():
    model_dims = (512, 512, 512)
    seq_lens = (32, 8, 4)
    modules = _build(model_dims=model_dims, seq_lens=seq_lens)
    # Expect one local stage + len(model_dims)-1 global stages
    assert isinstance(modules, nn.ModuleList)
    assert len(modules) == len(model_dims)  # local + all globals
    # First element must be local stage module (last in the hierarchy)
    assert isinstance(modules[0], _StageTokenEmbedding)
    assert modules[0].is_local is True
    # The rest are global
    for module in modules[1:]:
        assert isinstance(module, _StageTokenEmbedding)
        assert module.is_local is False


def test_local_stage_ids_vs_embeds_identical():
    num_tokens, pad_id = 260, 0
    model_dims = (128, 128)
    seq_lens = (16, 4)
    modules = _build(num_tokens=num_tokens, pad_id=pad_id, model_dims=model_dims, seq_lens=seq_lens)

    # local stage is index 0 in the reversed list
    local_stage = modules[0]
    batch_size, seq_len = 2, 11
    input_ids = torch.randint(0, num_tokens, (batch_size, seq_len))

    # Path 1: via input_ids
    out_ids = local_stage(input_ids=input_ids)

    # Path 2: via inputs_embeds (same embedding lookup externally)
    embeds = local_stage.embedding(input_ids)
    out_embeds = local_stage(inputs_embeds=embeds)

    assert out_ids.shape == (batch_size, seq_len, model_dims[-1])
    assert torch.allclose(out_ids, out_embeds, atol=0, rtol=0)


def test_padding_row_is_zero_for_local_stage():
    num_tokens, pad_id = 260, 5
    model_dims = (64, 64)
    seq_lens = (8, 2)
    modules = _build(num_tokens=num_tokens, pad_id=pad_id, model_dims=model_dims, seq_lens=seq_lens)

    local_stage = modules[0]
    with torch.no_grad():
        # padding row must be zeros
        row = local_stage.embedding.weight[pad_id]
        assert torch.all(row == 0)


def test_error_if_both_or_none_inputs():
    modules = _build()
    local_stage = modules[0]

    batch_size, seq_len = 2, 7
    ids = torch.randint(0, 257, (batch_size, seq_len))
    embeds = local_stage.embedding(ids)

    with pytest.raises(ValueError):
        _ = local_stage()  # neither

    with pytest.raises(ValueError):
        _ = local_stage(input_ids=ids, inputs_embeds=embeds)  # both


def test_global_stage_ids_vs_embeds_identical_projection():
    """
    For a global stage:
      - ids path: ids -> embedding -> [B, R, d] -> flatten/proj -> [B, D_model]
      - embeds path: [B, R, d] -> flatten/proj -> [B, D_model]
    If embeds come from the SAME embedding lookup, outputs must match exactly.
    """
    num_tokens, pad_id = 257, 256
    # 3 stages: D1(global), D2(global), D3(local=last)
    model_dims = (384, 256, 128)
    seq_lens = (16, 4, 2)  # P2, P3 used in global stages
    modules = _build(num_tokens=num_tokens, pad_id=pad_id, model_dims=model_dims, seq_lens=seq_lens)

    # modules[0] -> local (D3=128)
    # modules[1] -> global for D2=256 with patch_size = P3=2
    # modules[2] -> global for D1=384 with patch_size = P2*P3
    global_stage = modules[1]  # the one right after local

    # R is patch_size for this global stage:
    patch_size = global_stage.patch_size
    batch_size = 3

    # ids path
    ids = torch.randint(0, num_tokens, (batch_size, patch_size))
    out_ids = global_stage(input_ids=ids)

    # embeds path, but ensure they come from the SAME embedding weights
    embeds = global_stage.embedding(ids)  # [B, R, d]
    out_embeds = global_stage(inputs_embeds=embeds)

    assert out_ids.shape == (batch_size, model_dims[-2])  # D2
    assert torch.allclose(out_ids, out_embeds, atol=0, rtol=0)


def test_global_patch_size_accumulation_matches_seq_lens():
    """
    For n stages, reversed globals should have patch sizes:
        stage n-1: prod(seq_lens[n])      (= P_n)
        stage n-2: prod(seq_lens[n-1:n])  (= P_{n-1} * P_n)
        ...
    """
    model_dims = (256, 192, 128, 96)  # 4 stages, last=local
    seq_lens = (8, 4, 2, 1)
    modules = _build(model_dims=model_dims, seq_lens=seq_lens)

    # modules: [local, global_3, global_2, global_1]
    local_stage, global_stage_3, global_stage_2, global_stage_1 = modules

    assert local_stage.is_local
    # Expected patch sizes:
    p2, p3, p4 = seq_lens[1], seq_lens[2], seq_lens[3]
    assert global_stage_3.patch_size == p4
    assert global_stage_2.patch_size == p3 * p4
    assert global_stage_1.patch_size == p2 * p3 * p4


def test_local_shapes_and_types():
    modules = _build()
    local_stage = modules[0]

    batch_size, seq_len = 2, 9
    ids = torch.randint(0, 257, (batch_size, seq_len))
    out = local_stage(input_ids=ids)

    assert out.dtype == local_stage.embedding.weight.dtype
    assert out.shape[-1] == local_stage.local_dim


def test_global_shapes_and_types():
    modules = _build(model_dims=(256, 256, 256), seq_lens=(12, 3, 2))
    # global stage right after local
    global_stage = modules[1]

    batch_size = 4
    patch_size = global_stage.patch_size  # 2
    ids = torch.randint(0, 257, (batch_size, patch_size))
    out = global_stage(input_ids=ids)

    assert out.dtype == global_stage.embedding.weight.dtype
    assert out.shape == (batch_size, 256)


def test_gradient_flows_through_both_paths():
    """
    Ensure we can backprop through ids->embed->proj and embeds->proj paths.
    """
    num_tokens = 300
    modules = _build(num_tokens=num_tokens, model_dims=(128, 128, 128), seq_lens=(10, 5, 2))
    local_stage = modules[0]
    global_stage = modules[1]

    # Local: ids path
    ids_local = torch.randint(0, num_tokens, (2, 7))
    out_local = local_stage(input_ids=ids_local)
    loss_local = out_local.pow(2).mean()
    loss_local.backward(retain_graph=True)
    # At least embedding grads for some rows should exist (non-padding)
    assert local_stage.embedding.weight.grad is not None

    # Global: embeds path
    patch_size = global_stage.patch_size
    ids_global = torch.randint(0, num_tokens, (2, patch_size))
    embeds = global_stage.embedding(ids_global).detach().requires_grad_(True)
    out_global = global_stage(inputs_embeds=embeds)
    loss_global = out_global.abs().mean()
    loss_global.backward()
    # Check that projection (Linear) received gradients
    linear_layer = next(m for m in global_stage._post if isinstance(m, nn.Linear))
    assert linear_layer.weight.grad is not None


def test_inputs_embeds_must_have_correct_shape_for_local_and_global():
    modules = _build(model_dims=(128, 96, 64), seq_lens=(8, 4, 2))
    local_stage = modules[0]
    global_stage = modules[1]  # has patch_size = 2

    # local expects [..., L, d]
    wrong_local_embeds = torch.randn(2, 5, local_stage.local_dim + 1)
    with pytest.raises(RuntimeError):
        _ = local_stage(inputs_embeds=wrong_local_embeds)

    # global expects [..., R, d]
    wrong_global_embeds = torch.randn(2, global_stage.patch_size + 1, global_stage.local_dim)
    with pytest.raises(RuntimeError):
        _ = global_stage(inputs_embeds=wrong_global_embeds)
