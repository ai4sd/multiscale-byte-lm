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


import logging
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

from mblm.data.utils import FileStream

# when the logging module is imported during runtime, PIL picks this up and
# pollutes sys.stdout because it uses a low log level. disable this. see
# https://github.com/camptocamp/pytest-odoo/issues/15
logging.getLogger("PIL").setLevel(logging.WARNING)


# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
class ColorSpace(str, Enum):
    RGB = "RGB"
    GRAY = "L"


BinMode = Literal["lower", "upper", "mean"]
_BinModeValues: dict[str, tuple[float, float]] = {
    "lower": (0.0, 1.0),
    "upper": (1.0, 0.0),
    "mean": (0.5, 0.5),
}

ImagePipelineInput: TypeAlias = str | Path | PILImage | NDArray[np.uint8] | torch.Tensor


class ImagePipeline:
    """
    Create an image processing pipeline.

    - If the image is given as a Numpy array, the shape must be (H, W, C)
    - If given as a Torch tensor, the shape must be (C, H, W)
    - For grayscale images, the shape must be (H, W)

    As of now, this pipeline supports RGB and grayscale color spaces.
    """

    # we assume all images have a depth of 8 bit, which gives 2**8 color values
    MAX_NUM_CHANNELS = 256

    def __init__(self, image: ImagePipelineInput, color_space: ColorSpace):
        self._cs = color_space
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, (np.ndarray, torch.Tensor)):
            image = self._init_from_arraylike(image)

        self._image = image.convert(color_space.value)

    def _init_from_arraylike(self, image: NDArray[np.uint8] | torch.Tensor) -> PILImage:
        if isinstance(image, torch.Tensor):
            if self._cs == ColorSpace.RGB:
                image = image.permute(1, 2, 0)
            image = image.numpy()
        if (dtype := image.dtype) != np.uint8:
            raise ValueError(f"Can only process uint8 images, received {dtype} tensor")
        return Image.fromarray(image)

    def grayscale(self) -> ImagePipeline:
        if self._cs == ColorSpace.GRAY:
            return self
        self._image = self._image.convert(ColorSpace.GRAY.value)
        return self

    def resize(self, new_w_h: tuple[int, int]) -> ImagePipeline:
        """
        Resize an image to a new width and height (in pixels).
        """
        self._image = self._image.resize(new_w_h, resample=Image.Resampling.LANCZOS)
        return self

    def crop(self, factor_perc_h_w: tuple[float, float]) -> ImagePipeline:
        """
        Crop an image by (width, height) crop factors in percent.
        """
        width, height = self._image.size
        factor_h, factor_w = factor_perc_h_w

        margin_width = int(min(width * factor_w, width / 2))
        margin_height = int(min(height * factor_h, height / 2))

        left = margin_width
        top = margin_height
        right = width - margin_width
        bottom = height - margin_height

        self._image = self._image.crop((left, top, right, bottom))
        return self

    def downsample_channels(
        self, max_num_channel_values: int, bin_mode: BinMode = "mean"
    ) -> ImagePipeline:
        """
        Reduces the number of channel values in an image to a specified number.
        Channel values are effectively binned to `max_num_channel_values` bins.
        `min_bins_per_channel` must be greater than 0. If greater than 255,
        returns `self` unchanged.

        The strategy determines what value is assigned to the value in a bin, it
        can be `"upper"`, `"lower"` (where the values at the edges of bins are
        taken) or `"mean"` (which takes the average).

        By default, for `n` bins, values are mapped to the average bin width
        width. E.g., for `n=2` bins, all values between 0 and 127 are mapped to
        64. The values between 127 and 255 are mapped to 191.

        Note that due to floating point precision, channel values at the bin
        edges might fall into either the left or right bin.
        """
        # 255 bins is the maximum: 0 to 255
        if max_num_channel_values + 1 >= self.MAX_NUM_CHANNELS:
            return self

        if max_num_channel_values < 1:
            raise ValueError(
                f"Min number of bins must be positive, received {max_num_channel_values}"
            )

        # create bin decision edges from 0 to 255 at which to discretize values
        bin_borders = np.linspace(
            0, self.MAX_NUM_CHANNELS - 1, max_num_channel_values + 1, endpoint=True
        )
        assert len(bin_borders) == max_num_channel_values + 1
        # take any combination of two consecutive borders (n, n+1) to get the new values
        strategy = _BinModeValues[bin_mode]
        bin_values = np.convolve(bin_borders, strategy, mode="valid")
        # drop 255 - the last index is not needed for binning with np.digitize
        bin_borders = bin_borders[:-1]
        # discretize to get the indices to which midpoint values to map
        bin_indices = np.digitize(self.to_numpy(), bin_borders) - 1
        # extract discretized values at right positions
        binned_image = bin_values[bin_indices].astype(np.uint8)
        self._image = Image.fromarray(binned_image)
        return self

    def to_numpy(self) -> NDArray[np.uint8]:
        """
        Return the image as a numpy array with shape H, W, C.
        """
        return np.array(self._image, dtype=np.uint8)

    def to_tensor(self) -> torch.Tensor:
        """
        Return the image as a numpy array with shape C, H, W.
        """
        img_tensor = torch.from_numpy(self.to_numpy())
        if self._cs == ColorSpace.RGB:
            return img_tensor.permute(2, 0, 1)
        return img_tensor

    def to_image(self) -> PILImage:
        """
        Return the underlying Pillow image.
        """
        return self._image

    def to_jpeg_buffer(self, quality: int = 95, keep_rgb: bool = False) -> FileStream:
        """
        Compress the underlying image using JPEG compression and return a
        wrapped raw file buffer.

        Quality must be an integer between 0 (max compression) and 95 (minimal
        compression). See
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving

        Note that you can transform the FileStream back into a Pillow image,
        however, you will need to manually convert to YCbCr color space.
        Although the choice of `keep_rgb` will be reflected in the buffer,
        reading it back into a Pillow image will automatically assume RGB color
        space.
        """
        if not 0 <= quality <= 95:
            raise ValueError(f"Invalid argument for quality: {quality}")
        buffer = BytesIO()
        self._image.save(buffer, "JPEG", quality=quality, keep_rgb=keep_rgb)
        return FileStream(buffer)
