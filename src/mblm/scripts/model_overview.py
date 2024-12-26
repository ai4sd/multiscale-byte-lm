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
from itertools import chain
from pathlib import Path
from typing import Sequence

from tabulate import tabulate

from mblm import MBLM
from mblm.scripts.train_mblm import TrainEntryConfig
from mblm.utils.io import load_yml
from mblm.utils.misc import count_params


def resolve_configs(
    file_or_dirs: Sequence[Path], re_filter: str | None = None
) -> list[tuple[Path, TrainEntryConfig]]:
    if len(file_or_dirs) == 1 and (config_path := file_or_dirs[0]).is_file():
        return [(config_path, load_yml(config_path, parse_to=TrainEntryConfig))]
    yaml_files = chain.from_iterable(p.rglob("*.yaml") for p in file_or_dirs if p.is_dir())
    yml_files = chain.from_iterable(p.rglob("*.yml") for p in file_or_dirs if p.is_dir())

    pattern = re.compile(re_filter or ".*")
    config_files = filter(lambda p: pattern.match(p.name), chain(yaml_files, yml_files))
    return [(path, load_yml(path, parse_to=TrainEntryConfig)) for path in config_files]


def print_model_sizes(
    file_or_dirs: list[Path], re_filter: str | None = None, count_model_params: bool = False
):
    table_data: list[list] = []
    header = ["Config", "Inp. seq len", "Seq. lens", "# Layers"]
    if count_model_params:
        header += ["Params"]

    configs = sorted(
        resolve_configs(file_or_dirs, re_filter),
        key=lambda pc: (pc[1].params.input_seq_len, pc[1].params.seq_lens),
    )
    for path, conf in configs:
        if conf.io.dataset_id != "pg19":
            continue
        inp_len = conf.params.input_seq_len
        seg_lens = conf.params.seq_lens
        layers = str(conf.params.num_layers)
        row = [path.name, inp_len, seg_lens, layers]
        if count_model_params:
            model = MBLM(conf.params)
            params = f"{count_params(model)[0] / 1e6:.0f}m"
            row += [params]

        table_data.append(row)

    print(tabulate(table_data, headers=header, tablefmt="simple_grid"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_or_dir",
        type=Path,
        nargs="+",
        help="Path to a folder with experiments specified as YAML files or path to a single YAML file",
    )
    parser.add_argument(
        "-p",
        dest="count_model_params",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether or not to count model parameters",
    )
    parser.add_argument(
        "-f",
        dest="regex_filter",
        type=str,
        help="Regex filter expression on yaml files",
    )

    args = parser.parse_args()
    print_model_sizes(args.file_or_dir, args.regex_filter, args.count_model_params)
