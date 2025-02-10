__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import logging
import math
from typing import Iterable, Literal, Sequence, cast, overload

import torch
import torch.nn.functional as F  # noqa: N812
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from mblm.model.block import StageBlock
from mblm.model.config import MBLMModelConfig, MBLMReturnType
from mblm.model.utils import RoPE, gumbel_sample, top_k
from mblm.utils.stream import ByteStreamer

"""
Wording:
    - A hierarchy consists of n stages
    - Stage 1 to n-1 corresponds to global models
    - Stage n corresponds to the local model
Inline comment abbreviations:
    -------------------------------------------------------------------
    Global notation
    -------------------------------------------------------------------
    - V        Vocabulary size (256 for bytes)
    - B        Batch size
    - L        The input sequence length
    - P_n      The sequence length/number of patches at any stage n
    - D_n      The model dimension at any stage n
"""


class MBLM(nn.Module):
    """
    Multiscale Byte Language Model is a hierarchical byte-level
    sequence-to-sequence model for multimodal tasks.

    Based on a refined fork of https://github.com/lucidrains/MEGABYTE-pytorch
    (version 0.3.5).
    """

    def __init__(self, cfg: MBLMModelConfig):
        super().__init__()

        assert len(cfg.hidden_dims) == len(cfg.num_layers) == len(cfg.seq_lens)
        if cfg.train_checkpoint_chunks:
            assert len(cfg.hidden_dims) - 1 == len(cfg.train_checkpoint_chunks)

        self.num_stages = len(cfg.hidden_dims)
        self.seq_lens = cfg.seq_lens
        self.pad_token_id = cfg.pad_token_id
        self.checkpoint_chunks: list[int | None] = (
            [None] + cfg.train_checkpoint_chunks
            if cfg.train_checkpoint_chunks
            else [None] * self.num_stages
        )

        self.start_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(model_dim)) for model_dim in cfg.hidden_dims]
        )
        stage_blocks = cfg.stage_blocks()

        self.pos_embs = self._init_positional_embeddings(
            cfg.hidden_dims, cfg.seq_lens, stage_blocks
        )

        self.token_embs_rev = self._init_token_embeddings(
            cfg.hidden_dims, cfg.seq_lens, cfg.num_tokens, cfg.pad_token_id
        )

        self.stage_models, self.to_next_stage_proj = self._init_models_at_stages(
            cfg.hidden_dims, cfg.seq_lens, cfg.num_layers, stage_blocks
        )

        self.to_logits = nn.Linear(cfg.hidden_dims[-1], cfg.num_tokens)

    @classmethod
    def _init_positional_embeddings(
        cls,
        model_dims: Iterable[int],
        seq_lens: Iterable[int],
        blocks: Iterable[StageBlock],
    ) -> nn.ModuleList:
        """
        Create positional embeddings for each patch.
        """
        modules = nn.ModuleList()
        for model_dim, seq_len, block in zip(model_dims, seq_lens, blocks):
            if not block.pos_emb_type:
                modules.append(nn.Identity())
            elif block.pos_emb_type == "fixed":
                modules.append(nn.Embedding(seq_len, model_dim))
            elif block.pos_emb_type == "rope":
                modules.append(
                    RoPE(
                        model_dim,
                        # only used for caching, will not fail if we're feeding
                        # in longer sequences - cache will be rebuilt with the
                        # new sequence length
                        max_seq_len=seq_len,
                    )
                )
        return modules

    @classmethod
    def _init_token_embeddings(
        cls,
        model_dims: Sequence[int],
        seq_lens: Sequence[int],
        vocab_size: int,
        pad_token_id: int,
    ) -> nn.ModuleList:
        """
        Embed the tokens for each stage (in reverse order).
        """
        local_dim = model_dims[-1]
        token_embs_rev = nn.ModuleList(
            [nn.Embedding(vocab_size, local_dim, padding_idx=pad_token_id)]
        )
        patch_size = 1
        for model_dim, seq_len in zip(
            # all except the local model
            reversed(model_dims[:-1]),  # (D_n-1, ..., D_1)
            reversed(seq_lens[1:]),  # (P_2, ..., P_n)
        ):
            # for the global models, fuse the embedding and patch projection
            # step
            patch_size *= seq_len
            token_embs_rev.append(
                nn.Sequential(
                    nn.Embedding(vocab_size, local_dim, padding_idx=pad_token_id),
                    Rearrange("... r d -> ... (r d)"),
                    nn.LayerNorm(patch_size * local_dim),
                    nn.Linear(patch_size * local_dim, model_dim),
                    nn.LayerNorm(model_dim),
                )
            )
        return token_embs_rev

    @classmethod
    def _init_models_at_stages(
        cls,
        model_dims: Sequence[int],
        seq_lens: Sequence[int],
        num_stage_layers: Iterable[int],
        blocks: Iterable[StageBlock],
    ) -> tuple[nn.ModuleList, nn.ModuleList]:
        """
        Initialize the models and next-stage projections for each stage.
        """
        stage_models = nn.ModuleList([])
        to_next_stage_proj = nn.ModuleList([])

        for block, model_dim, num_layers, next_model_dim, next_seq_len in zip(
            blocks,
            model_dims,
            num_stage_layers,
            tuple(model_dims[1:]) + (None,),
            tuple(seq_lens[1:]) + (None,),
        ):
            stage_models.append(block.to_model(model_dim=model_dim, num_layers=num_layers))

            proj: torch.nn.Module = nn.Identity()

            if next_model_dim and next_seq_len:
                proj = nn.Sequential(
                    Rearrange("b ... d -> b (...) d"),
                    nn.Linear(model_dim, next_model_dim * next_seq_len),
                    Rearrange("b m (n d) -> (b m) n d", n=next_seq_len),
                )

            to_next_stage_proj.append(proj)
        return stage_models, to_next_stage_proj

    def forward_empty(self, batch_size: int) -> torch.Tensor:
        """
        Take care of special case where you sample from input of length 0 (start
        token only).
        """

        prev_stage_tokens_repr: torch.Tensor | None = None
        tokens: torch.Tensor = torch.empty(0)

        for stage_start_tokens, stage_model, proj in zip(
            self.start_tokens, self.stage_models, self.to_next_stage_proj
        ):
            tokens = repeat(stage_start_tokens, "d -> b 1 d", b=batch_size)

            if prev_stage_tokens_repr is not None:
                tokens = tokens + prev_stage_tokens_repr[..., : tokens.shape[-2], :]

            tokens = stage_model(tokens)
            prev_stage_tokens_repr = proj(tokens)

        return self.to_logits(tokens)

    @overload
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_type: Literal[MBLMReturnType.LOSS_LOGITS] = ...,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_type: Literal[MBLMReturnType.LOSS, MBLMReturnType.LOGITS] = ...,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_type: MBLMReturnType = MBLMReturnType.LOSS,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        A single forward pass.

        Args:
            input_ids: The token ids as torch.LongTensor in shape (B, L)
            return_type: What to return - the loss, the logits or both.
            loss_mask: An optional masking tensor that enables interpolation between
                self-supervised and supervised learning. It determines which tokens in the
                prediction should contribute to the loss with what weight and should have
                the same shape as `input_ids`. For example, if `loss_mask` is a tensor
                `torch.ones(input_ids)`, we learn to predict `input_ids` (self-supervised).
                In other scenarios, `input_ids` might be a flat tensor with `...question,
                ...answer`. In this case, one can provide a `loss_mask` in which the
                question tokens have a mask of 0 and the answer a mask of 1. The loss
                is then only computed on the answer, resulting in a supervised setting.
                By default, not providing a `loss_mask` is equivalent to a `loss_mask`
                consisting of all 1
        """
        batch_size = input_ids.shape[0]

        assert input_ids.ndim in {2, self.num_stages + 1}

        if input_ids.numel() == 0:
            return self.forward_empty(batch_size)

        if loss_mask is not None:
            assert loss_mask.shape == input_ids.shape

        flattened_dims = input_ids.ndim == 2
        flat_seq_len = input_ids.shape[-1]

        # if the input is given as (B, L), reshape and distribute it among the
        # hierarchy sequence lengths, filling up the inner dimensions
        # first. padding is applied so that the output shape is (B, P_1', P_2,
        # ..., P_n). for the largest possible L == prod(seq_lens), P_1' = P_1.
        # in all other cases, P_1' < P_1, meaning no padding is applied to the
        # global sequence P_1.
        #
        # here's two examples for a model model (P_1, P_2, P_3) = (5, 4, 3):
        #
        # input: (B, L) = (1, 13), output: (B, P_1', P_2, P_3) = (1, 2, 4, 3)
        # input: (B, L) = (1, 60), output: (B, P_1,  P_2, P_3) = (1, 5, 4, 3)
        #
        # in the 2nd example, L = prod(seq_lens) = 5 * 4 * 3 = 60 = P_1 = P_1'
        if flattened_dims:
            # pad/fill up all inner sequence lengths (all except most global)
            local_seq_lens = self.seq_lens[1:]
            multiple_of = math.prod(local_seq_lens)
            # use the complement of modulo - the difference to the next multiple
            # of multiple_of - to infer the right padding length
            padding = -flat_seq_len % multiple_of
            input_ids = F.pad(input_ids, (0, padding), value=self.pad_token_id)
            # reshape and infer the P_1' dimension
            input_ids = input_ids.reshape(batch_size, -1, *local_seq_lens)

        # make sure the above condition holds, i.e., P_1' <= P_1
        _P_1_prime, _P_1 = input_ids.shape[1], self.seq_lens[0]  # noqa: N806
        fixed_global_patch_encoding = isinstance(self.pos_embs[0], nn.Embedding)
        if fixed_global_patch_encoding:
            assert _P_1_prime <= _P_1, (
                f"Because you are using a fixed global patch embedding, "
                f"the input sequence length ({_P_1_prime}) "
                f"must be less than the first tuple element of seq_lens ({_P_1})"
            )

        token_embs_at_stages = [torch.empty(0) for _ in range(self.num_stages)]
        # at this stage, we're working with nested ids - hence, embed the bytes
        # for each stage in reverse order, starting from the local and ending at
        # the most global model. at each stage, add positional embeddings and
        # rerrange the input shape to match the dimension of the previous
        # (local) stage. with three stages:
        #
        # [0]: (B, P_1', D_1)
        # [1]: (B, P_1', P_2, D_2)
        # [2]: (B, P_1', P_2, P_3, D_3)
        for stage_idx, pos_emb, token_emb in zip(
            range(self.num_stages - 1, -1, -1),
            reversed(self.pos_embs),
            self.token_embs_rev,
        ):
            stage_token_embs: torch.Tensor = token_emb(input_ids)
            stage_seq_len = stage_token_embs.shape[-2]

            if isinstance(pos_emb, nn.Embedding):
                positions: torch.Tensor = pos_emb(
                    torch.arange(stage_seq_len, device=input_ids.device),
                )
                stage_token_embs = stage_token_embs + positions
            elif isinstance(pos_emb, RoPE):
                batch, *seq_lens, hidden_dim = stage_token_embs.shape
                # RoPE expects as input [batch, seq_len, num_heads, head_dim] ->
                # pack to artificial batch dim and add an empty head dimension
                stage_token_embs = stage_token_embs.view(-1, seq_lens[-1], 1, hidden_dim)
                stage_token_embs = pos_emb.forward(stage_token_embs)
                # reshape back to input
                stage_token_embs = stage_token_embs.view(batch, *seq_lens, hidden_dim)

            token_embs_at_stages[stage_idx] = stage_token_embs

            # skip rearranging for the most local model
            if stage_idx == self.num_stages - 1:
                continue
            input_ids = rearrange(input_ids, "... m n -> ... (m n)")

        # initials
        prev_stage_tokens_repr: torch.Tensor | None = None
        attended: torch.Tensor = torch.empty(0)

        for stage_idx in range(self.num_stages):
            stage_start_token = self.start_tokens[stage_idx]
            stage_seq_len = self.seq_lens[stage_idx]
            stage_emb_tokens = token_embs_at_stages[stage_idx]
            model = self.stage_models[stage_idx]
            checkpoint_chunk = self.checkpoint_chunks[stage_idx]
            proj = self.to_next_stage_proj[stage_idx]
            # for each stage n, going from global to local, pack the tokens into
            # a new tensor so that the last two dimensions correspond to the
            # sequence length and the dimension of that stage - later, the shape
            # is restored by the unpack operations.

            # the first dimension ("*"), after packing, corresponds to the
            # number of patches in the current stage (which can be considered a
            # batch for the stage because they are processed in parallel). the
            # patch size of model n-1 becomes the batch size of model n, and so
            # on
            stage_tokens, ps = pack([stage_emb_tokens], "* s d")
            patch_batch = stage_tokens.shape[0]  # "*" in the pack operation above

            # reshape the start tokens and prepend them to the stage tokens
            stage_start_token = repeat(stage_start_token, "d -> b 1 d", b=patch_batch)
            stage_tokens = torch.cat((stage_start_token, stage_tokens), dim=-2)
            # sum the previous hierarchy's representation
            if prev_stage_tokens_repr is not None:
                prev_stage_tokens_repr = F.pad(
                    # pad the second last dimensions (stage sequence) by 1 to
                    # the top to skip adding the embedding to the start token
                    # (i.e., addition with 0)
                    prev_stage_tokens_repr,
                    (0, 0, 1, 0),
                    value=0,
                )
                _start_token = stage_tokens[..., 0, :]
                stage_tokens = stage_tokens + prev_stage_tokens_repr
                assert _start_token.equal(stage_tokens[..., 0, :])
            # stage_tokens is now [B, P_n, D_n]. for the first stage, B is the
            # actual batch size whereas for the other stages, the batch size
            # corresponds to the sequence length of the previous hierarchy stage

            # skip checkpointing for the first global stage
            if checkpoint_chunk is None:
                attended = model.forward(stage_tokens)

            else:
                chunks: list[torch.Tensor] = []
                batch_chunks = torch.chunk(stage_tokens, checkpoint_chunk, dim=0)

                for batch_chunk in batch_chunks:
                    # although not explicitly needed, do not checkpoint for
                    # inference - this saves a little bit of time (around 0.1s /
                    # sequence for a sequence over 1M elements)
                    out = (
                        cast(
                            torch.Tensor,
                            checkpoint(
                                model.forward,
                                batch_chunk,
                                use_reentrant=False,
                            ),
                        )
                        if self.training
                        else model.forward(batch_chunk)
                    )

                    chunks.append(out)
                attended = torch.cat(chunks, dim=0)

            # the output and input share are equal -> by unpacking again, we
            # restore the initial shape
            attended = unpack(attended, ps, "* s d")[0]
            # project for next stage in the hierarchy, dropping the last patch:
            # from (..., P_n, D_n) to (..., P_n, P_n+1, D_n+1)
            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        logits = self.to_logits.forward(attended)  # (B, P_1', P_2, ..., 1 + P_n, V)

        logits_out = logits

        if flattened_dims:
            # drop the start tokens and combine inner dimensions into one
            logits_out = rearrange(logits_out[..., 1:, :], "b ... v -> b (...) v")
            # remove the padding
            logits_out = logits_out[:, :flat_seq_len]

        if return_type == MBLMReturnType.LOGITS:
            return logits_out

        # the most local sequences still contain the preprended start tokens
        # (i.e., the logits) - extract this token once and later prepend it to
        # the flattened sequence
        first_start_token_logits_idx = (slice(None), *((0,) * (logits.ndim - 2)), slice(None))
        # extract it
        start_token_logits = logits[first_start_token_logits_idx]  # type: ignore

        # drop ALL start tokens and combine inner dimensions into one...
        logits_rearranged = rearrange(logits[..., 1:, :], "b ... v -> b (...) v")

        # spread across batches
        start_token_logits = rearrange(start_token_logits, "b v -> b 1 v")
        # right-shift by one by appending the start token to the flat sequence
        logits_rearranged = torch.cat((start_token_logits, logits_rearranged), dim=-2)
        # drop the last item to match lengths
        logits_rearranged = logits_rearranged[..., :-1, :]

        # rearrange for loss calculation: the k-dimensional loss expects (B, V,
        # L)
        preds = rearrange(logits_rearranged, "b l v -> b v l")
        targets = rearrange(input_ids, "b ... -> b (...)")

        # same shape as targets with 0 everywhere where the token equals the pad
        # token. this assumes the same pad token is used for intra-batch padding
        # (ensured by the datasets/dataloaders) as well as patch-padding
        # (ensured by bootstrapping MBLM with the right pad token id)
        loss_tensor: torch.Tensor = F.cross_entropy(
            preds,  # (B, V, L)
            targets,  # (B, L)
            ignore_index=self.pad_token_id,
            reduction="none",
        )

        # remove the hierarchy padding (ignored in the loss calculation) and
        # bring to same shape as input_ids
        loss_tensor = loss_tensor[:, :flat_seq_len]

        if loss_mask is not None:
            # potentially apply the loss mask. this does not involve
            # broadcasting as after slicing above, the mask and the loss tensor
            # have the exact same shape again
            loss_tensor *= loss_mask

        # after applying the mask, some elements might be 0 - they should not be
        # accounted for in the loss calculation
        nonzero_idxs = torch.nonzero(loss_tensor, as_tuple=True)
        loss = loss_tensor[nonzero_idxs].mean()
        if torch.isnan(loss):
            # special case when the loss is zero across target elements - should
            # theoretically never happen
            logging.fatal("Edge case detected, loss is nan")
            loss = torch.zeros_like(loss, requires_grad=True)

        if return_type == MBLMReturnType.LOSS:
            return loss
        return loss, logits_out

    @torch.inference_mode()
    def generate(
        self,
        prime: torch.Tensor | None = None,
        stream: ByteStreamer | None = None,
        num_tokens_to_generate: int | None = None,
        end_of_gen_token_id: int = -1,
        temperature: float = 1.0,
        filter_thres: float = 1.0,
        enable_progress: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens.

        Args:
            prime: The prompt as a 1D tensor of type torch.long. Batches are
                not supported for now.
            stream: Instead of generating and then returning the sequene, stream
                the intermediate results to a writeable stream. Supports
                incremental UTF-8 decoding.
            num_tokens_to_generate: If set, ignore the context window and
                generate an exact amount of tokens. If `None`, generate up to
                the maximum possible sequence length.
            temperature: A float in the range (0, 1] that determines the
                temperature. 1 means deterministic, 0.1 very random. Has
                no effect if `filter_thres` is set to 1
            filter_thres: A float in the range [0, 1] to determine the
                threshold for the top k logits. 1 means the token with
                the maximum logit is sampled.
        """
        if prime is not None:
            assert prime.ndim == 1, "Batch mode not supported"
            prime = prime.unsqueeze(0)
        else:
            device = next(self.parameters()).device
            prime = torch.empty((1, 0), dtype=torch.long, device=device)

        total_iters = num_tokens_to_generate or math.prod(self.seq_lens) - prime.shape[-1]

        iter_rng = range(total_iters)
        iterator = tqdm(iter_rng, leave=False) if (stream is None and enable_progress) else iter_rng

        sequence = prime
        for _ in iterator:
            logits = self.forward(sequence, return_type=MBLMReturnType.LOGITS)[:, -1]
            logits = top_k(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
            sequence = torch.cat((sequence, rearrange(sampled, "b -> b 1")), dim=-1)

            newest_token = int(sampled.item())

            if stream is not None:
                stream.write(newest_token)

            if newest_token == end_of_gen_token_id:
                break
        if stream is not None:
            stream.flush()
        return sequence.squeeze()
