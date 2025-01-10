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


from pydantic import Field

from mblm.model.block import StageBlock
from mblm.model.mamba_shim import Mamba1, Mamba1Config, Mamba2Mixer


class MambaBlock(StageBlock):
    """
    General config for creating a Mamba block inside MBLM.
    Uses roughly 3 * expand * d_model^2 parameters.

    Parameters in brackets [x] denote the notation used in Mambabyte

    Parameters:
        n_layers: Number of layers [n]
        d_model: Residual stream dimension [d]
        expand: Linear layers expansion factor (always 2) [e]
        d_state: SSM state dimension [n_state]
        d_conv: Convolutional kernel size (always 4) [k]
        headdim: Head dimension
        dropout: The dropout in the Mamba block
    Note:
        `headdim` is a Mamba2 parameter only. There is no `dt_rank` [r]
        low-rank projection dimension in Mamba2
    """

    d_state: int
    d_conv: int
    expand: int
    headdim: int  # Mamba2 only
    dropout: float = 0.0  # Mamba2 only. Default for backwards compatibility

    block_type: str = Field(init=False, default="mamba1" if Mamba2Mixer is None else "mamba2")

    def to_model(self, model_dim, num_layers):
        if Mamba2Mixer is not None:
            return Mamba2Mixer(
                d_model=model_dim,
                n_layers=num_layers,
                d_state=self.d_state,
                d_conv=self.d_conv,
                headdim=self.headdim,
                expand=self.expand,
                dropout=self.dropout,
            )
        if Mamba1 is not None and Mamba1Config is not None:
            return Mamba1(
                Mamba1Config(
                    d_model=model_dim,
                    n_layers=num_layers,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand_factor=self.expand,
                )
            )
        raise RuntimeError(
            "Failed to import any Mamba version - this should never happen",
        )
