import math

import pytest
import torch
import torch.nn.functional
from einops import rearrange

from mblm import MBLM, MBLMModelConfig, MBLMReturnType, TransformerBlock
from mblm.model.config import MBLMEncoderModelConfig
from mblm.model.mblm import MBLMEncoder


def pad_and_reshape_inputs_embeds(
    flat_embs: torch.Tensor, seq_lens: tuple[int, ...]
) -> tuple[torch.Tensor, int, int]:
    """
    Convert (batch_size, seq_len, dim_n) -> (batch_size, p1_prime, p2, ..., pn, dim_n)
    by padding to the nearest multiple of prod(seq_lens[1:]) and reshaping.

    Returns: (nested_embs, p1_prime, padding_len)

    For single-stage (len(seq_lens) == 1), this just returns (batch_size, p1_prime=seq_len, dim_n).
    """
    assert flat_embs.ndim == 3, "Expected flat inputs_embeds of shape (B, L, Dn)"
    batch_size, seq_len, dim_n = flat_embs.shape
    inner = seq_lens[1:]
    multiple_of = math.prod(inner) if len(inner) > 0 else 1
    padding_len = (-seq_len) % multiple_of

    if padding_len:
        # pad along sequence dim (dim=1) on the right with zeros in emb space
        flat_embs = torch.nn.functional.pad(flat_embs, (0, 0, 0, padding_len))

    seq_len_padded = flat_embs.shape[1]
    p1_prime = seq_len_padded // max(multiple_of, 1)

    if len(seq_lens) == 1:
        nested = flat_embs.view(batch_size, p1_prime, dim_n)
    else:
        nested = flat_embs.view(batch_size, p1_prime, *inner, dim_n)

    return nested, p1_prime, padding_len


class TestMBLMInputsEmbeds:
    """Test class."""

    num_tokens = 256 + 1
    pad_token_id = 256
    num_attn_heads = 8
    dim_attn_heads = 64
    ff_mult = 4
    dropout = 0
    use_rot_emb = True
    use_flash_attn = False

    def _build_model(self, model_dims: tuple[int, ...], seq_lens: tuple[int, ...]) -> MBLM:
        return MBLM(
            MBLMModelConfig(
                num_tokens=self.num_tokens,
                hidden_dims=model_dims,
                seq_lens=seq_lens,
                pad_token_id=self.pad_token_id,
                num_layers=(1,) * len(model_dims),
                train_checkpoint_chunks=None,
                block=[
                    TransformerBlock(
                        attn_head_dims=self.dim_attn_heads,
                        attn_num_heads=self.num_attn_heads,
                        attn_dropout=self.dropout,
                        ff_multiplier=self.ff_mult,
                        ff_dropout=self.dropout,
                        pos_emb_type="fixed",
                        attn_use_rot_embs=self.use_rot_emb,
                        use_flash_attn=self.use_flash_attn,
                    )
                ]
                * len(model_dims),
            )
        )

    def test_inputs_embeds_only_allows_hidden_state_single_stage(self):
        """
        Inputs_embeds is supported only with return_type == HIDDEN_STATE.
        Other return types must raise.
        """
        model_dims = (64,)
        seq_lens = (9,)
        mblm = self._build_model(model_dims, seq_lens)
        mblm.eval()

        batch_size, seq_len, dim_n = 2, 7, model_dims[-1]
        flat_embs = torch.randn(batch_size, seq_len, dim_n)

        # For single-stage, flat (B, L, Dn) is already (B, p1_prime, Dn) with p1_prime = L
        nested = flat_embs

        with torch.no_grad():
            _ = mblm.forward(
                input_ids=None,
                inputs_embeds=nested,
                return_type=MBLMReturnType.HIDDEN_STATE,
            )

        with pytest.raises(ValueError):
            _ = mblm.forward(inputs_embeds=nested, return_type=MBLMReturnType.LOGITS)

        with pytest.raises(ValueError):
            _ = mblm.forward(inputs_embeds=nested, return_type=MBLMReturnType.LOSS)

    def test_inputs_embeds_hidden_state_shape_single_stage(self):
        """
        For single-stage, output should be (batch_size, 1 + p1_prime, dim_n) where p1_prime = seq_len.
        """
        model_dims = (96,)
        seq_lens = (11,)
        mblm = self._build_model(model_dims, seq_lens)
        mblm.eval()

        batch_size, seq_len, dim_n = 3, 8, model_dims[-1]
        flat_embs = torch.randn(batch_size, seq_len, dim_n)
        # single-stage: nested == flat along (B, p1_prime, Dn)
        nested = flat_embs  # p1_prime = seq_len

        with torch.no_grad():
            out = mblm.forward(inputs_embeds=nested, return_type=MBLMReturnType.HIDDEN_STATE)

        assert out.shape == torch.Size(
            [batch_size, 1 + seq_len, dim_n]
        ), f"got {out.shape}, expected {(batch_size, 1 + seq_len, dim_n)}"

    def test_encoder_hidden_state_matches_mblm_single_stage(self):
        """
        Encoder should delegate to the same network. Synchronize weights to
        assert numerical equality.
        """
        model_dims = (128,)
        seq_lens = (13,)
        cfg = MBLMModelConfig(
            num_tokens=self.num_tokens,
            hidden_dims=model_dims,
            seq_lens=seq_lens,
            pad_token_id=self.pad_token_id,
            num_layers=(1,),
            train_checkpoint_chunks=None,
            block=[
                TransformerBlock(
                    attn_head_dims=self.dim_attn_heads,
                    attn_num_heads=self.num_attn_heads,
                    attn_dropout=self.dropout,
                    ff_multiplier=self.ff_mult,
                    ff_dropout=self.dropout,
                    pos_emb_type="fixed",
                    attn_use_rot_embs=self.use_rot_emb,
                    use_flash_attn=self.use_flash_attn,
                )
            ],
        )

        mblm = MBLM(cfg)
        encoder = MBLMEncoder(
            MBLMEncoderModelConfig(mask_token_id=self.pad_token_id + 1, mblm_config=cfg)
        )
        encoder.mblm.load_state_dict(mblm.state_dict())  # sync weights

        mblm.eval()
        encoder.eval()

        batch_size, seq_len, dim_n = 2, 5, model_dims[-1]
        flat_embs = torch.randn(batch_size, seq_len, dim_n)
        nested = flat_embs  # single-stage

        with torch.no_grad():
            h_mblm = mblm.forward(inputs_embeds=nested, return_type=MBLMReturnType.HIDDEN_STATE)
            h_encoder = encoder.forward(
                inputs_embeds=nested, return_type=MBLMReturnType.HIDDEN_STATE
            )

        assert h_mblm.shape == h_encoder.shape
        assert torch.allclose(
            h_mblm, h_encoder, atol=1e-5
        ), "Encoder hidden states differ from MBLM after syncing weights"

    def test_inputs_embeds_nested_multistage(self):
        """
        Test inputs_embeds with multistage model
        """
        model_dims = (128, 64)
        seq_lens = (5, 4)  # prod(inner)=4
        mblm = self._build_model(model_dims, seq_lens)
        mblm.eval()

        batch_size, seq_len, dim_n = (
            1,
            7,
            model_dims[-1],
        )  # choose L <= P1 * prod(inner) to keep p1' <= P1
        flat_embs = torch.randn(batch_size, seq_len, dim_n)

        nested, _, _ = pad_and_reshape_inputs_embeds(flat_embs, seq_lens)

        with torch.no_grad():
            output = mblm.forward(inputs_embeds=nested, return_type=MBLMReturnType.HIDDEN_STATE)

        assert output.shape == (
            nested.shape[0],
            nested.shape[1],
            nested.shape[2] + 1,
            model_dims[-1],
        )

    @pytest.mark.parametrize(
        "model_dims, seq_lens",
        [((64,), (9,)), ((128, 64), (5, 4))],
    )
    def test_ids_vs_inputs_embeds_consistency_end2end(self, model_dims, seq_lens):
        """
        End-to-end consistency:
        1) Forward with nested input_ids -> h_ids
        2) Compute local (most local stage) inputs_embeds EXACTLY as the model does:
           local_inputs_embeds = mblm.token_embs_rev[0](nested_ids, None)
           (i.e., call the local embedding on the NESTED ids, not flat)
           -> forward(..., inputs_embeds) -> h_embs
        3) Normalize (drop final-stage start token, flatten, slice to original seq_len) and compare.
        """
        mblm = self._build_model(model_dims, seq_lens)
        mblm.eval()

        batch_size = 2
        p1 = seq_lens[0]
        inner = seq_lens[1:]
        prod_inner = math.prod(inner) if inner else 1

        # Choose a valid global patch count so that p1_prime <= p1 (fixed positional embeddings at global)
        p1_prime = min(2, p1)
        seq_len = p1_prime * prod_inner  # original token count without any start tokens

        # 1) Build nested input_ids and run ids path
        nested_shape = (batch_size, p1_prime, *inner) if inner else (batch_size, p1_prime)
        input_ids_nested = torch.randint(0, self.num_tokens, size=nested_shape, dtype=torch.long)

        with torch.no_grad():
            h_ids = mblm.forward(
                input_ids=input_ids_nested,
                inputs_embeds=None,
                return_type=MBLMReturnType.HIDDEN_STATE,
            )

        # 2) Compute inputs_embeds for ALL stages exactly as forward would:
        with torch.no_grad():
            inputs_embeds_list = []
            ids_buf = input_ids_nested
            embeds_buf = None

            # Process stages in reverse order (local to global), matching the forward loop
            for stage_idx in range(len(seq_lens) - 1, -1, -1):
                token_emb = mblm.token_embs_rev[len(seq_lens) - 1 - stage_idx]
                # Compute embeddings for this stage
                stage_embeds = token_emb(ids_buf, embeds_buf)
                inputs_embeds_list.append(stage_embeds)

                # Rearrange for next (more global) stage, except for the most local
                if stage_idx < len(seq_lens) - 1:
                    if ids_buf is not None:
                        ids_buf = rearrange(ids_buf, "... m n -> ... (m n)")
                    else:
                        embeds_buf = rearrange(embeds_buf, "... m n d -> ... (m n) d")

            # forward on inputs_embeds sequence
            h_embs = mblm.forward(
                input_ids=None,
                inputs_embeds=inputs_embeds_list,
                return_type=MBLMReturnType.HIDDEN_STATE,
            )

        # 3) Normalize both to (B, L, Dn): drop final-stage start token if present,
        #    flatten hierarchical dims, and slice to original seq_len
        def normalize_hidden(hidden: torch.Tensor) -> torch.Tensor:
            if len(seq_lens) == 1:
                # Single-stage: shapes are (B, S, D)
                # If start token included: S == p1_prime + 1 -> drop the first token on that axis
                if hidden.shape[1] == p1_prime + 1:
                    hidden = hidden[:, 1:, :]
                elif hidden.shape[1] != p1_prime:
                    raise AssertionError(
                        f"Unexpected single-stage shape {tuple(hidden.shape)} with p1'={p1_prime}"
                    )
                # Already (B, p1_prime, D)
                out_hidden = hidden
            else:
                # Multi-stage: last seq axis is the final stage
                # If start token included: size == pn + 1 -> drop the first token on that axis
                if hidden.shape[-2] == seq_lens[-1] + 1:
                    hidden = hidden[..., 1:, :]
                elif hidden.shape[-2] != seq_lens[-1]:
                    raise AssertionError(
                        f"Unexpected multi-stage final seq size {hidden.shape[-2]} given seq_lens={seq_lens}"
                    )
                # Flatten all hierarchical sequence dims: (B, p1_prime, p2, ..., pn, D) -> (B, L_pad, D)
                # We can do generic flatten: "b ... d -> b (...) d" since we only keep batch and last feature
                out_hidden = rearrange(hidden, "b ... d -> b (...) d")

            # Slice to the *original* seq_len (avoid any padding mismatch)
            return out_hidden[:, :seq_len, :]

        h_ids_flat = normalize_hidden(h_ids)
        h_embs_flat = normalize_hidden(h_embs)

        assert (
            h_ids_flat.shape == h_embs_flat.shape
        ), f"Shape mismatch after normalization: ids={h_ids_flat.shape}, embeds={h_embs_flat.shape}"
        assert torch.allclose(
            h_ids_flat, h_embs_flat, atol=1e-5
        ), "Hidden states differ between ids and inputs_embeds paths after normalization"
