from mblm.utils.seed import seed_everything

seed_everything(8)


def test_from_config():
    import torch
    from mblm import (
        MBLM,
        MambaBlockConfig,
        MBLMModelConfig,
        MBLMReturnType,
        TransformerBlockConfig,
    )

    mblm = MBLM(
        MBLMModelConfig(
            num_tokens=257,
            hidden_dims=[1024, 1024],
            seq_lens=[1024, 8],
            num_layers=[5, 5],
            pad_token_id=256,
            train_checkpoint_chunks=None,
            block=[
                MambaBlockConfig(
                    d_state=128,
                    d_conv=4,
                    expand=2,
                    headdim=64,
                    pos_emb_type=None,
                ),
                TransformerBlockConfig(
                    attn_head_dims=64,
                    attn_num_heads=16,
                    attn_use_rot_embs=True,
                    use_flash_attn=True,
                    pos_emb_type="fixed",
                ),
            ],
        )
    )

    x = torch.randint(0, 258, (1, 12)).long()

    # Choose between any of the return types
    logits = mblm.forward(x, return_type=MBLMReturnType.LOGITS)
    loss = mblm.forward(x, return_type=MBLMReturnType.LOSS)
    loss, logits = mblm.forward(x, return_type=MBLMReturnType.LOSS_LOGITS)

    assert logits.shape == (1, 12, 257)
    assert loss.ndim == 0


def test_from_yaml():
    import torch
    import yaml
    from mblm import MBLM, MBLMModelConfig, MBLMReturnType

    yml_model_config = """
    num_tokens: 257
    hidden_dims: [1024, 1024]
    seq_lens: [1024, 8]
    num_layers: [5, 5]
    pad_token_id: 256
    train_checkpoint_chunks: null
    block:
      - d_state: 128
        d_conv: 4
        expand: 2
        headdim: 64
        pos_emb_type: null
      - attn_head_dims: 64
        attn_num_heads: 16
        attn_use_rot_embs: true
        use_flash_attn: true
        pos_emb_type: fixed
    """

    parsed_config = yaml.safe_load(yml_model_config)
    mblm = MBLM(MBLMModelConfig.model_validate(parsed_config))
    x = torch.randint(0, 258, (1, 12)).long()
    mblm.forward(x, return_type=MBLMReturnType.LOSS)
