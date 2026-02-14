from __future__ import annotations

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

from typing import Optional, Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class _StageTokenEmbedding(nn.Module):
    """
    Per-stage embedding module that supports either input_ids or inputs_embeds.

    For local stage (is_local=True):
        - If input_ids is given: returns embedding(input_ids)  -> [..., L, local_dim]
        - If inputs_embeds is given: returns inputs_embeds as-is (expects [..., L, local_dim])

    For global stage (is_local=False):
        - If input_ids is given: embedding -> [..., R, local_dim] -> flatten/proj -> [..., model_dim]
        - If inputs_embeds is given: expects [..., R, local_dim] -> flatten/proj -> [..., model_dim]

    In all cases, the projection (flatten + LN + Linear + LN) reuses the same weights,
    so switching between input_ids and inputs_embeds produces identical results
    (provided inputs_embeds come from the same embedding lookup).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_token_id: int,
        local_dim: int,
        is_local: bool,
        patch_size: int = 1,  # product of downstream seq_lens
        model_dim: Optional[int] = None,  # required if is_local=False
    ) -> None:
        super().__init__()
        self.is_local = is_local
        self.local_dim = int(local_dim)
        self.patch_size = int(patch_size)
        self.model_dim = None if model_dim is None else int(model_dim)

        # The (shared) token embedding for input_ids path
        self.embedding = nn.Embedding(
            num_embeddings=int(vocab_size),
            embedding_dim=self.local_dim,
            padding_idx=pad_token_id,
        )

        if self.is_local:
            # No projection for local stage
            self._post = nn.Identity()
        else:
            if self.model_dim is None:
                raise ValueError("model_dim must be provided for global stages.")
            flat_dim = self.patch_size * self.local_dim
            self._post = nn.Sequential(  # type: ignore
                Rearrange("... r d -> ... (r d)"),
                nn.LayerNorm(flat_dim),
                nn.Linear(flat_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Exactly one of (input_ids, inputs_embeds) must be provided.

        Shapes:
          - local stage:
              input_ids:     [..., L]         -> [..., L, local_dim]
              inputs_embeds: [..., L, local_dim] (returned unchanged)
          - global stage:
              input_ids:     [..., R]         -> [..., R, local_dim] -> proj -> [..., model_dim]
              inputs_embeds: [..., R, local_dim]             -> proj -> [..., model_dim]
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")

        if input_ids is not None:
            # IDs -> embedding lookup
            x = self.embedding(input_ids)  # local: [..., L, d]; global: [..., R, d]
        else:
            # bypass embedding layer; reuse projection weights
            x = inputs_embeds
            # --- Shape validation for inputs_embeds ---
            if not isinstance(x, torch.Tensor):
                raise RuntimeError("inputs_embeds must be a torch.Tensor.")
            if x.ndim < 2:
                raise RuntimeError(f"inputs_embeds must be at least 2D (got {x.ndim}D).")
            # Local stage expects [..., L, local_dim]
            if self.is_local:
                if x.shape[-1] != self.local_dim:
                    raise RuntimeError(
                        f"Local stage expects inputs_embeds[..., local_dim={self.local_dim}], "
                        f"but got last dim={x.shape[-1]}."
                    )
            else:
                # Global stage expects [..., R, local_dim] with R == self.patch_size
                if x.ndim < 2:
                    raise RuntimeError(
                        "Global stage expects inputs_embeds with at least 2 dims [..., R, d]."
                    )
                if x.shape[-2] != self.patch_size:
                    raise RuntimeError(
                        f"Global stage expects inputs_embeds[..., R={self.patch_size}, d], "
                        f"but got R={x.shape[-2]}."
                    )
                if x.shape[-1] != self.local_dim:
                    raise RuntimeError(
                        f"Global stage expects last dim local_dim={self.local_dim}, "
                        f"but got {x.shape[-1]}."
                    )

        if self.is_local:
            # local stage has no projection
            return x
        else:
            # global stage: flatten(r, d) and project
            return self._post(x)


class MultiStageTokenEmbedding:
    """
    Factory that replicates the behavior of _init_token_embeddings but returns
    per-stage modules that accept both input_ids and inputs_embeds.

    The returned ModuleList is in REVERSE order, identical to your current API.
    """

    @classmethod
    def build(
        cls,
        *,
        model_dims: Sequence[int],
        seq_lens: Sequence[int],
        vocab_size: int,
        pad_token_id: int,
    ) -> nn.ModuleList:
        """
        Replicates the original initializer semantics (reverse order, fused global projection).

        Returns:
            nn.ModuleList in reverse stage order:
                [ local_stage_module, global_n-1, ..., global_1 ]
        """
        model_dims = list(model_dims)
        seq_lens = list(seq_lens)

        local_dim = model_dims[-1]  # local model hidden dim
        token_embs_rev = nn.ModuleList(
            [
                _StageTokenEmbedding(
                    vocab_size=vocab_size,
                    pad_token_id=pad_token_id,
                    local_dim=local_dim,
                    is_local=True,  # last stage is local
                    patch_size=1,
                    model_dim=None,
                )
            ]
        )

        patch_size = 1
        # iterate over global models in reverse, accumulating patch_size
        for model_dim, seq_len in zip(
            reversed(model_dims[:-1]),  # (D_{n-1}, ..., D_1)
            reversed(seq_lens[1:]),  # (P_2, ..., P_n)
        ):
            patch_size *= int(seq_len)
            token_embs_rev.append(
                _StageTokenEmbedding(
                    vocab_size=vocab_size,
                    pad_token_id=pad_token_id,
                    local_dim=local_dim,
                    is_local=False,
                    patch_size=patch_size,
                    model_dim=int(model_dim),
                )
            )

        return token_embs_rev
