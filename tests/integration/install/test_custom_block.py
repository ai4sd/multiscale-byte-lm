from mblm.utils.seed import seed_everything

seed_everything(8)


def test_from_config():
    import torch

    from mblm import MBLM, MBLMModelConfig, MBLMReturnType, TransformerBlock
    from mblm.model.block import StageBlock

    # Define any custom model
    class LSTM(torch.nn.Module):
        def __init__(self, lstm: torch.nn.LSTM):
            super().__init__()
            self.lstm = lstm

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # Wrap the LSTM forward to extract the output
            out, _ = self.lstm(input_ids)
            return out

    # Add a block config and inherit from StageBlock
    class LSTMBlock(StageBlock):
        block_type: str = "lstm"

        # Add whatever is needed
        dropout: float

        def to_model(self, model_dim: int, num_layers: int) -> torch.nn.Module:
            return LSTM(
                torch.nn.LSTM(
                    input_size=model_dim,
                    hidden_size=model_dim,
                    batch_first=True,
                    dropout=self.dropout,
                    num_layers=num_layers,
                )
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
                LSTMBlock(
                    dropout=0.1,
                    pos_emb_type=None,
                ),
                TransformerBlock(
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
    mblm.forward(x, return_type=MBLMReturnType.LOSS)


def test_from_yaml():
    import torch
    import yaml

    from mblm import MBLM, MBLMModelConfig, MBLMReturnType
    from mblm.model.block import StageBlock
    from mblm.model.config import block_registry  # Import this!

    # Define any custom model
    class LSTM(torch.nn.Module):
        def __init__(self, lstm: torch.nn.LSTM):
            super().__init__()
            self.lstm = lstm

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # Wrap the LSTM forward to extract the output
            out, _ = self.lstm(input_ids)
            return out

    # Add a block config and inherit from StageBlock
    @block_registry.register()
    class LSTMBlock(StageBlock):
        block_type: str = "lstm"

        # Add whatever is needed
        dropout: float
        my_property: int

        def to_model(self, model_dim: int, num_layers: int) -> torch.nn.Module:
            return LSTM(
                torch.nn.LSTM(
                    input_size=model_dim,
                    hidden_size=model_dim,
                    batch_first=True,
                    dropout=self.dropout,
                    num_layers=num_layers,
                )
            )

    yml_model_config = """
    num_tokens: 257
    hidden_dims: [1024, 1024]
    seq_lens: [1024, 8]
    num_layers: [5, 5]
    pad_token_id: 256
    train_checkpoint_chunks: null
    block:
        - dropout: 0.1
          my_property: 1
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

    # keep hidden in docs
    assert isinstance(mblm.stage_models[0], LSTM)
