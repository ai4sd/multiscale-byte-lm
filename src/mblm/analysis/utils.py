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

import re
from pathlib import Path
from typing import TypeAlias

import polars as pl
from tabulate import tabulate

from mblm import MBLM, MBLMModelConfig
from mblm.data.utils import Bytes
from mblm.trainer.mblm import TrainOutputConfig
from mblm.utils.io import load_model_state, load_yml

ExpCollection: TypeAlias = list[tuple[str, str]]


def extract_prompt_with_offset(text: str, txt_offset: int, prompt_len_bytes: int) -> str:
    # because the text offset is computed in UTF-8 and the prompt/context length
    # in bytes, convert between the two for slicing
    prompt_as_tensor = Bytes.str_to_tensor(text[txt_offset:])
    return Bytes.tensor_to_str(prompt_as_tensor[:prompt_len_bytes])


LINE_BREAK_TABS_RE = re.compile(r"[\r\n]+")


def strip_line_breaks(text: str) -> str:
    return LINE_BREAK_TABS_RE.sub(" ", text)


def dataframe_to_md_table(df: pl.DataFrame, output_file: str | Path) -> None:
    as_str = tabulate(df.to_dict(), headers=df.columns, tablefmt="github")
    with Path(output_file).open("w") as f:
        f.write(as_str)
    return None


def load_model(
    model_id: str,
    model_dir: Path,
    device: str,
) -> tuple[MBLM, TrainOutputConfig]:
    config_file = model_dir / (model_id + ".yml")
    state_file = model_dir / (model_id + ".pth")

    config = load_yml(config_file, TrainOutputConfig)
    model = MBLM(
        MBLMModelConfig(
            num_tokens=config.params.num_tokens,
            hidden_dims=tuple(config.params.hidden_dims),
            seq_lens=tuple(config.params.seq_lens),
            pad_token_id=config.params.pad_token_id,
            num_layers=tuple(config.params.num_layers),
            train_checkpoint_chunks=config.params.train_checkpoint_chunks,
            block=config.params.block,
        )
    ).to(device)

    model, _ = load_model_state(
        state_file,
        model,
        map_location=device,
        # we've renamed this module
        map_rename_modules=(("pos_embs", "patch_pos_embs"),),
    )
    return model, config
