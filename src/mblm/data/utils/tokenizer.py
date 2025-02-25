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


from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass
class TokenizerOptions:
    """
    Options for to be used for the Tokenizer class.

    Args:
        pad_token_id: ID of the padding token
        eom_token_id: End-of-modality token ID
            (can be considered a start-of-sentence-token in language modeling)
        som_text_token_id: Start-of-text-modality token ID
        som_image_token_id: Start-of-image-modality token ID
    """

    pad_token_id: int
    eom_token_id: int | None
    som_text_token_id: int | None
    som_image_token_id: int | None
    dtype = torch.long


@dataclass
class Tokenizer:
    """
    A preprocessor to add special tokens to 1D tensors.


    **Example**::

        pipeline = Tokenizer(
            TokenizerOptions(
                pad_token_id=10,
                eom_token_id=11,
                som_image_token_id=12,
                som_text_token_id=13,
            )
        ).pipeline
        input_t = torch.arange(0, 10)  # length 10
        input_t = pipeline(input_t).with_eom().with_som_text().pad_right_to(15).ok()

        assert input_t.size(0) == 15
        assert input_t[0].item() == 13  # som token
        assert input_t[11].item() == 11  # eom token
        assert input_t[12:].equal(torch.tensor([10, 10, 10]))  # padding token
    """

    options: TokenizerOptions

    def pipeline(self, tensor: torch.Tensor) -> TokenizerPipeline:
        """
        Start a piepline to process the 1D input tensor.
        """
        return TokenizerPipeline(tensor=tensor, options=self.options)


@dataclass
class TokenizerPipeline:
    tensor: torch.Tensor
    options: TokenizerOptions

    def __post_init__(self):
        if (dim := self.tensor.ndim) > 1:
            raise ValueError(f"Can only process 1D tensors, input is {dim}D")

    def _tensor_from_val(self, fill_value: int | None) -> torch.Tensor:
        if fill_value is None:
            return torch.empty((0), dtype=self.options.dtype)
        else:
            return torch.full((1,), fill_value=fill_value, dtype=self.options.dtype)

    def with_eom(self) -> TokenizerPipeline:
        filled = self._tensor_from_val(self.options.eom_token_id)
        self.tensor = torch.cat([self.tensor, filled])
        return self

    def with_som_text(self) -> TokenizerPipeline:
        filled = self._tensor_from_val(self.options.som_text_token_id)
        self.tensor = torch.cat([filled, self.tensor])
        return self

    def with_som_image(self) -> TokenizerPipeline:
        filled = self._tensor_from_val(self.options.som_image_token_id)
        self.tensor = torch.cat([filled, self.tensor])
        return self

    def pad_right_to(self, to_length: int, strict: bool = True) -> TokenizerPipeline:
        current_length = self.tensor.size(0)
        pad_right = to_length - current_length
        if strict and pad_right < 0:
            raise ValueError(
                f"Tensor at dim 0 (length {current_length}) larger than "
                f"desired padded size {to_length}"
            )
        self.tensor = F.pad(self.tensor.long(), (0, pad_right), value=self.options.pad_token_id)
        return self

    def to_long_tensor(self) -> torch.Tensor:
        return self.tensor.long()
