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

from itertools import repeat
from typing import Sequence

from pydantic import BaseModel

from mblm.model.mamba import MambaBlockConfig
from mblm.model.transformer import TransformerBlockConfig


class MBLMModelConfig(BaseModel):
    """
    General config for creating a MBLM model. For all iterables,
    the order corresponds to global to most local stage from left to right.

    Params:
        num_tokens: The vocabulary size
        pad_token_id: Id of the padding token
        hidden_dims: The model's hidden dimensions at each stage
        num_layers: The number of layers at each stage
        seq_lens: The sequence length at each stage
        block: Either a Transformer or Mamba block configuration
    """

    num_tokens: int
    pad_token_id: int
    hidden_dims: Sequence[int]
    num_layers: Sequence[int]
    seq_lens: Sequence[int]
    train_checkpoint_chunks: list[int] | None
    block: (
        TransformerBlockConfig | MambaBlockConfig | list[TransformerBlockConfig | MambaBlockConfig]
    )

    def blocks(self) -> list[TransformerBlockConfig | MambaBlockConfig]:
        if isinstance(self.block, list):
            assert len(self.block) == len(
                self.hidden_dims
            ), "If blocks are given as list, lengths must match"
            return self.block
        return list(repeat(self.block, len(self.hidden_dims)))
