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


from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, TypeVar

import torch
from pydantic import BaseModel, Field


class StageBlock(ABC, BaseModel):
    """
    Configuration for a single model block at an MBLM stage.
    """

    block_type: str = Field(
        description="A name for the block for easy identification, which can be any string name"
    )

    pos_emb_type: Literal["fixed", "rope"] | None = Field(
        default=None,
        description="The type of positional embedding to add to tokens of the stage block",
    )

    @abstractmethod
    def to_model(
        self,
        model_dim: int,
        num_layers: int,
    ) -> torch.nn.Module: ...

    """
    An abstract method that creates a `torch.nn.Module` from the stage block
    configuration.
    """


TStageBlock = TypeVar("TStageBlock", bound=StageBlock)


class StageBlockRegistry(set[type[StageBlock]]):
    """
    Stage block registry that allows (custom) stage blocks to be registered for
    usage with MBLM. Blocks only need to be registered when they require parsing
    from a YAML config file. When a MBLM configuration is read from a YAML
    config file, the registry will try to find a matching stage block
    implementation.
    """

    def register(self) -> Callable[[type[TStageBlock]], type[TStageBlock]]:
        """
        Decorator to register a stage block class so it can be validated from a YAML
        configuration file.

        Usage:
            @block_registry.register()
            class MyBlock(StageBlock):
                pass
        """

        def decorator(stage_block_klass: type[TStageBlock]) -> type[TStageBlock]:
            self.add(stage_block_klass)
            return stage_block_klass

        return decorator

    def try_parse(
        self,
        data: Any,
        parse_func: Callable[[type[StageBlock], Any], StageBlock],
    ):
        """
        Try to parse configuration data to a registered stage block class.
        """
        for klass in self:
            try:
                return parse_func(klass, data)
            except Exception:
                pass

        raise ValueError(f"Coult not parse data to any of {self}")
