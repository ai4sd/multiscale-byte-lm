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

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, NamedTuple, TypeAlias, TypeVar

import torch
import yaml
from filelock import FileLock
from pydantic import BaseModel
from torch import load as torch_load
from torch import save as torch_save
from torch.nn import Module as TorchModule
from torch.serialization import MAP_LOCATION


def _to_path(path: str | Path) -> Path:
    return Path(path) if isinstance(path, str) else path


# Although type parameter lists exist, they're only available in Python >=3.12.
# Use TypeVar and ParamSpec objects instead, which are backwards compatible with
# 3.10. See https://docs.python.org/3/library/typing.html#typing.ParamSpec
_T = TypeVar("_T")
_TBaseModel = TypeVar("_TBaseModel", bound=BaseModel)


def load_yml(
    path: str | Path, parse_to: type[_TBaseModel], try_yaml_suffixes: bool = False
) -> _TBaseModel:
    """
    Load any configuration from a (nested) yaml file.
    Args:
        path (str | Path): The path to the yaml file
        parse_to: An instance of a Pydantic model
        check_yaml_suffixes (bool=False): If `True`, try
            loading the file with either a `yaml` or
            `yml` extension
    Returns:
        A populated instance of the Pydantic model
    """
    path = _to_path(path)
    yaml_suffixes = {".yaml", ".yml"}
    assert path.suffix in yaml_suffixes, f"{path} is not a yaml file"
    try:
        with Path.open(path, "r") as file:
            yaml_data = yaml.safe_load(file)
            return parse_to.model_validate(yaml_data)
    except FileNotFoundError as exception:
        if try_yaml_suffixes:
            yaml_suffixes.remove(path.suffix)
            new_suffix = next(iter(yaml_suffixes))
            return load_yml(path.with_suffix(new_suffix), parse_to, try_yaml_suffixes=False)
        raise exception


def dump_yml(path: str | Path, data: BaseModel) -> Path:
    path = _to_path(path).with_suffix(".yaml")
    with Path.open(path, "w") as file:
        yaml.safe_dump(data.model_dump(), file)
    return path


_TNamedTuple = TypeVar("_TNamedTuple", bound=NamedTuple)

# dont pollute with package internal logs
logging.getLogger("filelock").setLevel(logging.WARNING)


class CSVWriter(Generic[_TNamedTuple]):
    def __init__(
        self,
        output_dir: str | Path,
        file_name: str,
        noop: bool = False,
    ) -> None:
        self.noop = noop
        if noop:
            return
        output_dir = _to_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._file = (output_dir / file_name).with_suffix(".csv")

        with self._file.open("w", encoding="utf-8"):
            # clear existing file contents
            pass
        self._lock = FileLock(str(self._file) + ".lock")

    def _file_is_empty(self) -> bool:
        """Check if the file is empty."""
        return self._file.stat().st_size == 0

    def write_row(self, row: _TNamedTuple):
        """
        Write a row.

        Args:
            row: Named tuple to write.
        """
        if self.noop:
            return

        with self._lock:
            file_empty = self._file_is_empty()
            with self._file.open("a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if file_empty:
                    # automatically write a header once
                    writer.writerow(row._fields)
                writer.writerow(row)


class NDJSONWriter(Generic[_TBaseModel]):
    def __init__(self, path: Path | str):
        self.path = _to_path(path)
        self.stream_positions: list[int] = []

    def write_line(self, obj: _TBaseModel) -> None:
        with self.path.open("a", encoding="utf8") as f:
            self.stream_positions.append(f.tell())
            f.write(obj.model_dump_json() + "\n")

    def remove_last_line(self) -> None:
        if len(self.stream_positions) == 0:
            return None
        last_position = self.stream_positions.pop()
        with self.path.open("a", encoding="utf8") as f:
            f.truncate(last_position)


def read_jsonl(
    path: Path | str,
    parse_lines_to: Callable[..., _T],
) -> list[_T]:
    with _to_path(path).open("r", encoding="utf8") as f:
        return [parse_lines_to(**json.loads(line)) for line in f.readlines()]


StateDict: TypeAlias = dict[str, Any]


def save_model_state(
    dir: str | Path,
    checkpoint_name: str,
    model: TorchModule | StateDict,
    loss: float,
) -> tuple[bool, Path]:
    """
    Save a model's state_dict to a checkpoint. Will overwrite a file with the same name.

    Args:
        dir (str | Path): If it does not exist, it will be created recursively
        checkpoint_name (str): The suffix will be set to `.pth` automatically.
        model (torch.nn.Module | StateDict): Any class that inherits from
            `torch.nn.Module` or the actual state dict obtained from `model.state_dict()`

    Returns:
        did_overwrite (bool): Whether or not the file was overwritten
    """
    path = _to_path(dir)
    path.mkdir(parents=True, exist_ok=True)
    file = path.joinpath(checkpoint_name).with_suffix(".pth")

    if isinstance(model, TorchModule):
        model = model.state_dict()
    did_overwrite = file.exists()
    snapshot = {"MODEL": model, "LOSS": loss}
    torch_save(snapshot, file)
    return did_overwrite, file


_TModule = TypeVar("_TModule", bound=TorchModule)


def _embedding_map_extend(
    src_state: dict[str, torch.Tensor],
    tgt_state: dict[str, torch.Tensor],
    map_extend_embeddings: set[str],
    notify: Callable[[str], Any],
) -> dict[str, torch.Tensor]:
    """
    Special loader for when we have module A with an embedding or linear layer
    of size (N, D) and module B with embedding or linear layer of size (N+i, D).
    This function transfers the source weights and biases into the corresponding
    positions in the target, leaving additional entries uninitialized or
    randomly initialized. Hence, it is required that the source size in A is
    smaller than the target size in B. At the moment, we only support this
    extending strategy, not shrinking.
    """
    # TODO: Allow init of newly added embeddings with mean, similar to
    # https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/modeling_utils.py#L2465
    for (src_key, src_tensor), (tgt_key, tgt_tensor) in zip(src_state.items(), tgt_state.items()):
        if src_key != tgt_key:
            raise ValueError(
                f"Expected source and target state dict to match ({src_key} != {tgt_key})"
            )
        if src_key not in map_extend_embeddings:
            # ignore this key
            continue

        if src_key.endswith("weight") and src_tensor.size(1) != tgt_tensor.size(1):
            raise ValueError(
                "Mapping to a smaller embedding dimension is not supported",
            )

        if (src_n_ids := src_tensor.size(0)) > tgt_tensor.size(0):
            raise ValueError(
                "Mapping to a smaller number of embeddings is not supported",
            )
        tgt_state[tgt_key][:src_n_ids] = src_tensor.detach()
        notify(f"Successfully moved embeddings from {src_key}")
    return tgt_state


def _modules_map_rename(
    src_state: dict[str, torch.Tensor],
    map_rename_modules: Iterable[tuple[str, str]],
    notify: Callable[[str], Any],
) -> dict[str, torch.Tensor]:
    """
    Rename entries in a state dict in place by providing a map of
    (source_prefix, target_prefix). As soon as a compatible entry is found,
    i.e., a key in the state dict matches `source_prefix`, `source_prefix` is
    replaced with `target_prefix`
    """
    src_state_updated = src_state.copy()
    for rename_from_refix, rename_to_prefix in map_rename_modules:
        for src_module_name in src_state.keys():
            if src_module_name.startswith(rename_from_refix):
                data = src_state_updated.pop(src_module_name)
                postfix = src_module_name[len(rename_from_refix) :]
                tgt_module_name = rename_to_prefix + postfix
                src_state_updated[tgt_module_name] = data
                notify(f"Successfully renamed {src_module_name +postfix } to {tgt_module_name}")
    return src_state_updated


@torch.no_grad()
def load_model_state(
    checkpoint_file: str | Path,
    model: _TModule,
    map_location: MAP_LOCATION | None = None,
    map_extend_embeddings: set[str] | None = None,
    map_rename_modules: Iterable[tuple[str, str]] | None = None,
    on_success: Callable[[str], Any] | None = None,
) -> tuple[_TModule, float]:
    """
    Restore a model's `state_dict` from a checkpoint and return it.

    Args:
        checkpoint_file: Must point to a `.pth` file
        model: Any class T that inherits from `torch.nn.Module`
        map_location: See `torch.load`
        map_extend_embeddings: Map a smaller source embedding to a
            larger target embedding
        map_rename_modules: Rename modules in the checkpoint state
            before populating the model with it. If provided in the form
            (source_prefix, target_prefix), the state dict will be
            renamed BEFORE `map_extend_embeddings` is applied!

    Returns:
        tuple: A tuple with the original model `T` with
            `updated state_dict` and the associated loss
    """
    snapshot = torch_load(
        checkpoint_file,
        map_location,
        weights_only=True,
    )
    state_dict = snapshot["MODEL"]
    loss = snapshot["LOSS"]

    def _notify(msg: str):
        if on_success:
            on_success(msg)

    if map_rename_modules:
        state_dict = _modules_map_rename(state_dict, map_rename_modules, notify=_notify)
    if map_extend_embeddings:
        state_dict = _embedding_map_extend(
            state_dict,
            model.state_dict(),
            map_extend_embeddings,
            notify=_notify,
        )
    model.load_state_dict(state_dict, strict=True)
    return model, loss
