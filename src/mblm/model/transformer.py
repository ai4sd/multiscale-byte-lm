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


from typing import Callable, Iterable, cast

import torch
from MEGABYTE_pytorch.megabyte import Attention, FeedForward, RMSNorm, RotaryEmbedding, token_shift
from pydantic import Field

from mblm.model.block import StageBlock


class TransformerBlock(StageBlock):
    """
    General config for creating a Transformer Decocer block inside MBLM.
    """

    attn_head_dims: int
    attn_num_heads: int
    attn_use_rot_embs: bool
    attn_dropout: float = 0.0
    ff_multiplier: int = 2
    ff_dropout: float = 0.0
    use_flash_attn: bool = False

    block_type: str = Field(init=False, default="transformer")

    def to_model(self, model_dim, num_layers):
        return TransformerDecoder(
            model_dim=model_dim,
            num_layers=num_layers,
            attn_head_dim=self.attn_head_dims,
            attn_num_heads=self.attn_num_heads,
            attn_dropout=self.attn_dropout,
            attn_use_rot_embs=self.attn_use_rot_embs,
            ff_dropout=self.ff_dropout,
            ff_mult=self.ff_multiplier,
            use_flash_attn=self.use_flash_attn,
        )


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_layers: int,
        attn_head_dim: int = 64,
        attn_num_heads: int = 8,
        attn_dropout: float = 0.0,
        attn_use_rot_embs: bool = True,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(attn_head_dim) if attn_use_rot_embs else None

        self.layers = torch.nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        Attention(
                            dim=model_dim,
                            dim_head=attn_head_dim,
                            heads=attn_num_heads,
                            dropout=attn_dropout,
                            flash=use_flash_attn,
                        ),
                        FeedForward(dim=model_dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(model_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        n = input_ids.shape[-2]
        rotary_emb: torch.Tensor | None = self.rotary_emb(n) if self.rotary_emb else None

        for attn, ff in cast(Iterable[tuple[Callable, Callable]], self.layers):
            input_ids = attn(token_shift(input_ids), rotary_emb=rotary_emb) + input_ids
            input_ids = ff(token_shift(input_ids)) + input_ids

        return self.norm(input_ids)
