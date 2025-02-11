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

from enum import Enum, auto
from itertools import repeat
from typing import Any, Sequence

from pydantic import (
    BaseModel,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from mblm.model.block import StageBlock, StageBlockRegistry
from mblm.model.mamba import MambaBlock
from mblm.model.transformer import TransformerBlock

block_registry = StageBlockRegistry()
block_registry.register()(TransformerBlock)
block_registry.register()(MambaBlock)


class MBLMReturnType(str, Enum):
    LOGITS = auto()
    LOSS = auto()
    LOSS_LOGITS = auto()


class MBLMModelConfig(BaseModel):
    """
    General config for creating a MBLM model. For all iterables,
    the order corresponds to global to local stage from left to right.

    Params:
        num_tokens: The vocabulary size
        pad_token_id: Id of the padding token
        hidden_dims: The model's hidden dimensions at each stage
        num_layers: The number of layers at each stage
        seq_lens: The sequence length at each stage
        block: A Transformer or Mamba block
    """

    num_tokens: int
    pad_token_id: int
    hidden_dims: Sequence[int]
    num_layers: Sequence[int]
    seq_lens: Sequence[int]
    train_checkpoint_chunks: list[int] | None

    block: StageBlock | Sequence[StageBlock]

    @field_validator("block", mode="before")
    @classmethod
    def parse_block(cls, value: Any) -> Any:
        """
        Try to parse the block to any registered stage block
        """
        given_as_list = isinstance(value, list)
        value = value if given_as_list else [value]
        parsed = []
        for block_data in value:
            if isinstance(block_data, StageBlock):
                parsed.append(block_data)
                continue
            parsed_block = block_registry.try_parse(
                block_data, lambda model, data: model.model_validate(data)
            )
            parsed.append(parsed_block)
        return parsed if given_as_list else parsed[0]

    @model_validator(mode="after")
    def validate_stages(self):
        is_equal_num_stages = len(self.hidden_dims) == len(self.num_layers) == len(self.seq_lens)
        if not is_equal_num_stages:
            raise ValueError("hidden_dims, num_layers and seq_lens are not of the same length")

        if isinstance(self.block, list) and len(self.block) != len(self.hidden_dims):
            raise ValueError(
                "When given as a list of blocks, must have the appropriate number of stages"
            )
        return self

    @field_serializer("block")
    def serialize_block(
        self, block: StageBlock | Sequence[StageBlock], info: SerializationInfo
    ) -> dict | list[dict]:
        if not isinstance(block, Sequence):
            return block.model_dump(mode=info.mode)
        return [b.model_dump(mode=info.mode) for b in block]

    def stage_blocks(self) -> Sequence[StageBlock]:
        if isinstance(self.block, Sequence):
            return self.block
        return list(repeat(self.block, len(self.hidden_dims)))
