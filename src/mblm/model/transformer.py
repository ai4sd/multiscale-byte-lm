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

import typing
from functools import wraps
from typing import Callable, Iterable, cast

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from MEGABYTE_pytorch.megabyte import (
    Attend,
    Attention,
    FeedForward,
    RMSNorm,
    RotaryEmbedding,
    token_shift,
)
from packaging import version
from pydantic import Field, model_validator
from torch import einsum, nn
from torch.amp import autocast
from torch.nn.attention import SDPBackend

from mblm.model.block import StageBlock


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def exists(val):
    return val is not None


class TransformerBlock(StageBlock):
    """
    General config for creating a Transformer Decoder block inside MBLM.
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

    @model_validator(mode="after")
    def validate_block_type(self):
        if self.block_type != "transformer":
            raise ValueError("This model is a transformer")
        return self


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


class TransformerEncoderBlock(StageBlock):
    """General config for the Transformer encoder block"""

    attn_head_dims: int
    attn_num_heads: int
    attn_use_rot_embs: bool
    attn_dropout: float = 0.0
    ff_multiplier: int = 2
    ff_dropout: float = 0.0
    use_flash_attn: bool = False

    block_type: str = Field(init=False, default="transformerEncoder")

    def to_model(self, model_dim, num_layers):
        return TransformerEncoder(
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

    @model_validator(mode="after")
    def validate_block_type(self):
        if self.block_type != "transformerEncoder":
            raise ValueError("This model is a transformerEncoder")
        return self


@typing.no_type_check
class TransformerEncoder(torch.nn.Module):
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
                        AttentionEncoder(
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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        n = input_ids.shape[-2]
        rotary_emb: torch.Tensor | None = self.rotary_emb(n) if self.rotary_emb else None

        for attn, ff in cast(Iterable[tuple[Callable, Callable]], self.layers):
            input_ids = (
                attn(input_ids, rotary_emb=rotary_emb, attention_mask=attention_mask) + input_ids
            )  # Skip-connection
            input_ids = ff(input_ids) + input_ids

        return self.norm(input_ids)


# Code Adapted from lucidrain/megabyte


@typing.no_type_check
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@typing.no_type_check
@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


@typing.no_type_check
class AttentionEncoder(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, flash=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = AttendWithMask(causal=False, flash=flash, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, attention_mask: torch.Tensor = None, rotary_emb=None):  # type: ignore
        h, device = self.heads, x.device  # noqa: F841

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))  # type: ignore
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))  # type: ignore

        if rotary_emb is not None:
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask=attention_mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


@typing.no_type_check
class AttendWithMask(Attend):
    """
    Computation of the attention mechanism with support for custom masks
    """

    def __init__(self, causal=False, dropout=0.0, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # default cpu attention configs
        self.attn_cfg = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
        ]

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.attn_cfg = [SDPBackend.FLASH_ATTENTION]
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.attn_cfg = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    @typing.no_type_check
    def flash_attn(self, q, k, v, mask=None, attn_bias=None):  # noqa ARG002 type: ignore
        _, heads, q_len, _, k_len, _, device = *q.shape, k.shape[-2], q.is_cuda, q.device  # type: ignore

        # single headed key / values
        if k.ndim == 3:
            k = rearrange(k, "b n d -> b 1 n d")

        if v.ndim == 3:
            v = rearrange(v, "b n d -> b 1 n d")

        is_causal = self.causal

        if mask is not None:
            if mask.ndim != 4:
                # Expand (b, j) -> (b, 1, 1, j)
                mask = rearrange(mask, "b j -> b 1 1 j")
                mask = mask.expand(-1, heads, q_len, -1)

            if self.causal:
                # PyTorch SDPA requires True for "attend" and False for "attention_mask out".
                # We build a causal attention_mask and logically AND it with the provided attention_mask.
                causal_mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).tril()
                mask = mask & causal_mask
                # Disable SDPA's built-in causal handling since we merged it manually
                is_causal = False

        with torch.nn.attention.sdpa_kernel(self.attn_cfg):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        return out

    def forward(self, q, k, v, mask=None):
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        # similarity
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # FIX 2: Apply the user-provided attention_mask in the non-flash fallback
        if mask is not None:
            if mask.ndim == 2:
                mask = rearrange(mask, "b j -> b 1 1 j")
            # ~attention_mask implies False means "attention_mask out", True means "keep"
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal attention_mask
        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            # get_mask returns True for the upper triangle (future tokens)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
