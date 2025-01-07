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

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, TypedDict, overload

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Unpack

from mblm.data.datasets import DistributedDataset, DistributedDatasetConfig
from mblm.data.types import BatchWithLossMask, ModelMode
from mblm.data.utils import (
    Bytes,
    ColorSpace,
    ImagePipeline,
    Tokenizer,
    TokenizerOptions,
    shift_remap_tensor,
    target_loss_mask,
)

if TYPE_CHECKING:
    from mblm.train.mblm import TrainEntryConfig


class ClevrFunction(TypedDict):
    function: str
    inputs: list[int]
    value_inputs: list[str]


class ClevrQuestion(TypedDict):
    question: str
    answer: str
    image_filename: str
    question_family_index: int
    program: list[ClevrFunction]


class ClevrModelGeneration(BaseModel):
    id_model: str
    sample_idx: int
    question: str
    question_type: str
    answer_gen: list[int]
    answer_truth: list[int]
    ce: float
    timestamp: str


class ClevrOptionalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_mode: Literal["qiqa", "a"]
    qiqa_loss_mask: tuple[float, float, float, float]
    answer_categorical: bool = False
    resize_to_h: int | None = None
    resize_to_w: int | None = None
    crop_h_perc: float | None = None
    crop_w_perc: float | None = None
    downsample_channels: int | None = None
    shift_channels_start: int | None = None
    eom_token_id: int | None = None
    som_text_token_id: int | None = None
    som_image_token_id: int | None = None
    enable_jpeg_stream_with_quality: int | None = Field(
        default=None,
        description="When `None`, the image is represented as an array of pixel data. If an integer, the ordinary image processing is applied, however, the final image will consist of the raw, compressed JPEG byte stream with the specified quality.",
    )
    jpeg_in_rgb_color_space: bool = Field(
        default=True,
        description="If `True`, save the JPEG stream in RGB color space instead of YCbCr. This setting only has an effect if `enable_jpeg_stream_with_quality` is not `None`.",
    )

    @field_validator("enable_jpeg_stream_with_quality")
    @classmethod
    def assert_is_jpeg_quality(cls, v: int | None) -> int | None:
        """
        We compress the Clevr input PNG images via Pillow. This ensures the
        quality parameter for JPEG ranges from 0 to 95, where 0 is maximum
        compression and 95 least compression. See
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving
        """
        if v is not None:
            assert 0 <= v <= 95
        return v


class Clevr(DistributedDataset[BatchWithLossMask]):
    """
    https://cs.stanford.edu/people/jcjohns/clevr/

    The `data_dir` is expected to be the of the exact structure as the
    original dataset, although the .txt files are not strictly needed:
        ├── COPYRIGHT.txt
        ├── LICENSE.txt
        ├── README.txt
        ├── images
        ├── questions
        └── scenes
    Note: Clevr does not include answers to test questions
    """

    ANSWER_CLASS_MAP: dict[str, str] = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "A",
        "blue": "B",
        "brown": "C",
        "cube": "D",
        "cyan": "E",
        "cylinder": "F",
        "gray": "G",
        "green": "H",
        "large": "I",
        "metal": "J",
        "no": "K",
        "purple": "L",
        "red": "M",
        "rubber": "N",
        "small": "O",
        "sphere": "P",
        "yellow": "Q",
        "yes": "R",
    }

    QUESTION_TYPES: dict[str, str] = {
        "exist": "exists",
        "count": "count",
        # compare integer
        "equal_integer": "compare_integer",
        "less_than": "compare_integer",
        "greater_than": "compare_integer",
        # query attribute
        "query_color": "query_attribute",
        "query_material": "query_attribute",
        "query_size": "query_attribute",
        "query_shape": "query_attribute",
        # compare attribute
        "equal_size": "compare_attribute",
        "equal_material": "compare_attribute",
        "equal_shape": "compare_attribute",
        "equal_color": "compare_attribute",
    }

    # common (transposed) image shape for all modes
    IMAGE_SHAPE_C_W_H = 3, 480, 320
    MAX_QUESTION_LEN_BYTES = 205  # 203 for validation
    MAX_ANSWER_LEN_BYTESN = 8

    # number of question-answer pairs in the test/validation set
    NUM_ITEMS_TRAIN = 699_989
    NUM_ITEMS_VALID = 149_991

    # technically, Clevr images are RGBA - we drop the alpha channel
    image_color_space = ColorSpace.RGB

    def __init__(
        self,
        data_dir: str | Path,
        mode: ModelMode,
        pad_token_id: int,
        optional_args: ClevrOptionalArgs,
        **config: Unpack[DistributedDatasetConfig],
    ):
        root = Path(data_dir)
        self.mode = mode
        self.args = optional_args

        self.tokenize = Tokenizer(
            TokenizerOptions(
                pad_token_id=pad_token_id,
                eom_token_id=optional_args.eom_token_id,
                som_text_token_id=optional_args.som_text_token_id,
                som_image_token_id=optional_args.som_image_token_id,
            )
        ).pipeline

        if mode == ModelMode.TRAIN:
            questions_path = "questions/CLEVR_train_questions.json"
            img_root = "images/train"
        elif mode == ModelMode.VALID:
            questions_path = "questions/CLEVR_val_questions.json"
            img_root = "images/val"
        else:
            questions_path = "questions/CLEVR_test_questions.json"
            img_root = "images/test"
        with (root / questions_path).open() as file:
            self.entries: list[ClevrQuestion] = json.load(file)["questions"]
            self.images_root = root / img_root
        super().__init__(
            data_size=len(self.entries),
            is_sequential=False,
            **config,
        )

    @staticmethod
    def from_train_entry_config(
        config: TrainEntryConfig,
        mode: ModelMode,
        worker_id: int,
        num_workers: int,
    ) -> DistributedDataset[BatchWithLossMask]:
        # cannot pass None to model_validate
        optional_args = ClevrOptionalArgs.model_validate(config.io.dataset_args or dict())
        return Clevr(
            data_dir=config.io.dataset_dir,
            mode=mode,
            pad_token_id=config.params.pad_token_id,
            seq_len=config.params.input_seq_len,
            worker_id=worker_id,
            num_workers=num_workers,
            optional_args=optional_args,
        )

    @staticmethod
    def supports_test_mode():
        return False

    def process_and_flatten_img(self, image: ImagePipeline) -> torch.Tensor:
        """
        Process an image sample:

        1. Crop
        2. Resize
        3. Discretize/downsample channels
        4. If JPEG compression is enabled, return a tensor with data from the
           raw JPEG stream
        """
        if (h := self.args.crop_h_perc) and (w := self.args.crop_w_perc):
            image.crop((h, w))
        # resize
        if (w := self.args.resize_to_w) and (h := self.args.resize_to_h):
            image.resize((w, h))
        # discretize/downsample
        if disc_channels := self.args.downsample_channels:
            image.downsample_channels(disc_channels)
            # if we discretize, we optionally shift
            if shift_start := self.args.shift_channels_start:
                image_tensor, _, _ = shift_remap_tensor(image.to_tensor(), range_start=shift_start)
                image = ImagePipeline(image_tensor, self.image_color_space)
        if q := self.args.enable_jpeg_stream_with_quality:
            # already flattened
            keep_rgb = self.args.jpeg_in_rgb_color_space
            image_tensor = image.to_jpeg_buffer(q, keep_rgb).to_tensor()
        else:
            image_tensor = image.to_tensor().flatten()
        return image_tensor

    def get_sample_raw(self, from_idx: int) -> ClevrQuestion:
        """
        Get a raw sample as a (question, answer, image) tuple with no
        tokenization or preprocessing applied.
        """
        if self.mode == ModelMode.TEST:
            raise ValueError("Clevr dataset does not support testing!")
        return self.entries[from_idx]

    def iter_images(
        self, shuffle: bool = False, max_items: int | None = None
    ) -> Generator[torch.Tensor, None, None]:
        """
        For every image in Clevr, there are multiple questions and answers. Use
        this iterator if you want to iterate over images and yield a unique
        image each time. No preprocessing is applied.
        """
        image_lst = os.listdir(self.images_root)
        if shuffle:
            random.shuffle(image_lst)
        if max_items is not None:
            image_lst = image_lst[:max_items]
        for img_path in image_lst:
            yield ImagePipeline(self.images_root / img_path, self.image_color_space).to_tensor()

    @overload
    def iter(
        self, *, shuffle: bool = ..., max_items: int | None = ..., raw: Literal[True]
    ) -> Generator[tuple[int, ClevrQuestion], None, None]: ...
    @overload
    def iter(
        self, shuffle: bool = ..., max_items: int | None = ..., raw: Literal[False] = ...
    ) -> Generator[tuple[int, tuple[str, str, ImagePipeline]], None, None]: ...
    def iter(
        self, shuffle: bool = False, max_items: int | None = None, raw: bool = False
    ) -> Generator[tuple[int, tuple[str, str, ImagePipeline] | ClevrQuestion], None, None]:
        """
        Iterate over all the question, answer image tuples in Clevr (or raw
        entries if specified). While question/answer pairs unique, images may
        appear more than once. No preprocessing is applied.
        """
        entries_range = list(range(len(self.entries)))
        if shuffle:
            random.shuffle(entries_range)
        if max_items is not None:
            entries_range = entries_range[:max_items]
        for i in entries_range:
            s = self.get_sample_raw(i)
            if raw:
                yield i, s
            else:
                img = ImagePipeline(self.images_root / s["image_filename"], self.image_color_space)
                yield i, (s["question"], s["answer"], img)

    def get_sample_with_parts(
        self, from_idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]:
        """
        Get a sample at an index with padding and tokenization applied. The
        returned tuple contains:

        1. The concatenated question, image, question, answer: (`q_i_q_a`)
        2. The loss mask with the same shape as `q_i_q_a`
        3. A tuple containing the original `q`, `i`, `a` parts that are
           concatenated in `q_i_q_a` and the size of the right padding applied
        """
        s = self.get_sample_raw(from_idx)
        question_str, answer_str = (s["question"], s["answer"])
        image_pipeline = ImagePipeline(
            self.images_root / s["image_filename"], self.image_color_space
        )
        # question and image as tensors, not yet tokenized
        question = Bytes.str_to_tensor(question_str)
        image = self.process_and_flatten_img(image_pipeline)

        # add special tokens - they will be converted from uint8 to long
        question = self.tokenize(question).with_som_text().with_eom().to_long_tensor()
        image = self.tokenize(image).with_som_image().with_eom().to_long_tensor()

        if self.args.answer_categorical:
            # override the n-length answer with a string of length 1
            answer_str = self.ANSWER_CLASS_MAP[answer_str]

        answer = Bytes.str_to_tensor(answer_str)
        answer = self.tokenize(answer).with_som_text().with_eom().to_long_tensor()

        # concat everything -> dtype long is preserved
        q_i_q_a = torch.concat([question, image, question, answer])
        len_original_q_i_q_a = len(q_i_q_a)

        # create a loss mask for q_i_q_a with the weights
        loss_mask = target_loss_mask(
            zip(
                [
                    question,
                    image,
                    question,
                    answer,
                ],
                self.args.qiqa_loss_mask,
            )
        )
        assert q_i_q_a.shape == loss_mask.shape
        # apply padding
        q_i_q_a = self.tokenize(q_i_q_a).pad_right_to(self.seq_len).to_long_tensor()
        loss_mask = self.tokenize(loss_mask).pad_right_to(self.seq_len).to_long_tensor()

        len_padding = len(q_i_q_a) - len_original_q_i_q_a

        return q_i_q_a, loss_mask, (question, image, answer, len_padding)

    def get_sample(self, from_idx: int) -> BatchWithLossMask:
        """
        Get a processed sample with a loss mask. This method is required by the
        DistributedDataset superclass.
        """
        q_i_q_a, loss_mask, _ = self.get_sample_with_parts(from_idx)
        return q_i_q_a, loss_mask
