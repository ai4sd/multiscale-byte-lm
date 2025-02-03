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

"""
Module shims for Mamba. Because the official Mamba package relies on a Linux
environment and a few special dependencies that require CUDA, this module
provides a shim to enable development across platforms.

1. If the official mamba_ssm package is available, we import the new Mamba 2
   package [https://arxiv.org/pdf/2405.21060]
2. If not - which is the case for non-Linux platforms OR when mamba_ssm has not
   been installed - we fall back to a pure-PyTorch community implementation of
   Mamba 1 [https://arxiv.org/pdf/2312.00752]

"""

import os
import warnings
from functools import partial
from typing import cast

import torch

Mamba1 = None
Mamba1Config = None
Mamba2Mixer = None  # type: ignore[no-redef]


try:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # mamba-ssm and its triton dependency issue a handful of FutureWarnings,
        # that's none of our business
        from mamba_ssm.models.mixer_seq_simple import (
            _init_weights,  # type: ignore[import-not-found]
            create_block,  # type: ignore[import-not-found]
        )
        from mamba_ssm.ops.triton.layer_norm import (
            RMSNorm,  # type: ignore[import-not-found]
            layer_norm_fn,  # type: ignore[import-not-found]
        )
        from mamba_ssm.utils.generation import (
            InferenceParams,  # type: ignore[import-not-found]
        )

    class Mamba2Mixer(torch.nn.Module):  # type: ignore[no-redef]
        """
        Simplified and typed version of
        mamba_ssm.models.mixer_seq_simple.MixerModel without the embeddings, our
        wrapper model takes care of these.

        Notation used in Mamba paper:
            d_model: Model dimension [D]
            d_state: SSM state dimension/state size/state expansion factor  [N]
            d_conv: Local convolution width
            expand: Block expansion factor [E]

            Sequence length [L]
        """

        def __init__(
            self,
            d_model: int,
            n_layers: int,
            d_state: int,
            d_conv: int,
            headdim: int,
            expand: int,
            intermediate_factor: int = 0,
            norm_epsilon: float = 1e-5,
            dropout: float = 0.0,
            residual_in_fp32: bool = False,
            fused_add_norm: bool = True,
        ):
            super().__init__()
            self.residual_in_fp32 = residual_in_fp32
            self.fused_add_norm = fused_add_norm
            self.dropout = dropout

            # raw params that are passed to the Mamba 1/2 block
            ssm_cfg = {
                # required according to create_block fn
                "layer": "Mamba2",
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "headdim": headdim,
            }

            self.layers = torch.nn.ModuleList(
                [
                    create_block(
                        d_model,
                        d_intermediate=intermediate_factor * d_model,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=True,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                    )
                    for i in range(n_layers)
                ]
            )
            self.norm_f = RMSNorm(d_model, eps=norm_epsilon, dropout_p=dropout)
            self.apply(
                partial(
                    _init_weights,
                    n_layer=n_layers,
                    initializer_range=0.02,  # Now only used for embedding layer.
                    rescale_prenorm_residual=True,
                    n_residuals_per_layer=1 if intermediate_factor == 0 else 2,  # 2 if we have MLP
                )
            )

        def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
            return {
                i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
                for i, layer in enumerate(self.layers)
            }

        def forward(
            self,
            input_ids: torch.Tensor,
            inference_params: InferenceParams | None = None,
            **mixer_kwargs,
        ):
            hidden_states: torch.Tensor = input_ids
            residual: torch.Tensor | None = None
            for layer in self.layers:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, **mixer_kwargs
                )
            if not self.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = cast(
                    torch.Tensor,
                    layer_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        dropout_p=self.dropout,
                        residual_in_fp32=self.residual_in_fp32,
                        is_rms_norm=isinstance(self.norm_f, RMSNorm),
                    ),
                )
            return hidden_states

except ImportError as err:
    import sys

    from mambapy.mamba import Mamba as Mamba1  # type: ignore[no-redef]
    from mambapy.mamba import MambaConfig as Mamba1Config  # type: ignore[no-redef]

    reason_failed = "Platform is not Linux"
    if sys.platform.startswith("linux"):
        reason_failed = err.msg

    skip_warning = "PYTEST_CURRENT_TEST" in os.environ

    if not skip_warning:
        warnings.warn(
            f"Failed to import Mamba2, falling back to Mamba1 (PyTorch version). Reason: {reason_failed}",
        )


__all__ = ["Mamba1", "Mamba1Config", "Mamba2Mixer"]
