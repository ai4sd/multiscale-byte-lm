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
from pathlib import Path

from dotenv import dotenv_values

dotenv_file_path_proj_root = Path(__file__) / "../../../.env"

env_config = dotenv_values(dotenv_file_path_proj_root.resolve())


@dataclass
class Config:
    DATASET_PG19_DIR: Path
    DATASET_CLEVR_DIR: Path
    EXPERIMENTS_DIR: Path

    def __post_init__(self):
        for path in (self.DATASET_PG19_DIR, self.DATASET_CLEVR_DIR):
            assert path.is_absolute()


config = Config(
    DATASET_PG19_DIR=Path(env_config.get("DATASET_PG19_DIR") or "/"),
    DATASET_CLEVR_DIR=Path(env_config.get("DATASET_CLEVR_DIR") or "/"),
    EXPERIMENTS_DIR=Path(env_config.get("EXPERIMENTS_DIR") or "/"),
)
