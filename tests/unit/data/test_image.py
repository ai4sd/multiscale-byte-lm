import numpy as np
import PIL.Image
import pytest
import torch

from mblm.data.utils.image import BinMode, ColorSpace, ImagePipeline

IMAGE_PATH = "tests/fixtures/clevr/images/val/CLEVR_val_000000.png"

IMAGE_SHAPE_NUMPY_RGB = (12, 15, 3)  # H, W, C
IMAGE_SHAPE_TORCH_RGB = (3, 12, 15)  # C, H, W
IMAGE_SHAPE_GRAY = (12, 15)  # H, W

TORCH_IMAGE_RGB = torch.randint(0, 256, IMAGE_SHAPE_TORCH_RGB, dtype=torch.uint8)
TORCH_IMAGE_GRAY = torch.randint(0, 256, IMAGE_SHAPE_GRAY, dtype=torch.uint8)

NUMPY_IMAGE_RGB = np.random.randint(0, 256, IMAGE_SHAPE_NUMPY_RGB, dtype=np.uint8)
NUMPY_IMAGE_GRAY = np.random.randint(0, 256, IMAGE_SHAPE_GRAY, dtype=np.uint8)


class TestImagePipeline:
    def test_from_path(self):
        try:
            ImagePipeline(IMAGE_PATH, ColorSpace.RGB)
        except Exception:
            pytest.fail("Could not create image pipeline from path")

    def test_from_pil(self):
        try:
            img = PIL.Image.new(ColorSpace.RGB.value, (5, 5))
            ImagePipeline(img, ColorSpace.GRAY)
        except Exception:
            pytest.fail("Could not create image pipeline from PIL")

    @pytest.mark.parametrize(
        "image,cs",
        ((TORCH_IMAGE_RGB, ColorSpace.RGB), (TORCH_IMAGE_GRAY, ColorSpace.GRAY)),
    )
    def test_from_to_tensor(self, image: torch.Tensor, cs: ColorSpace):
        assert image.equal(ImagePipeline(image, cs).to_tensor())

    @pytest.mark.parametrize(
        "image,cs",
        ((NUMPY_IMAGE_RGB, ColorSpace.RGB), (NUMPY_IMAGE_GRAY, ColorSpace.GRAY)),
    )
    def test_from_to_numpy(self, image: np.ndarray, cs: ColorSpace):
        assert np.array_equal(image, ImagePipeline(image, cs).to_numpy())

    @pytest.mark.parametrize("image", (NUMPY_IMAGE_RGB.astype(np.long), TORCH_IMAGE_RGB.long()))
    def test_from_wrong_dtype(self, image: torch.Tensor):
        with pytest.raises(ValueError):
            ImagePipeline(image, ColorSpace.RGB)

    @pytest.mark.parametrize(
        "image,cs",
        ((NUMPY_IMAGE_RGB, ColorSpace.RGB), (NUMPY_IMAGE_GRAY, ColorSpace.GRAY)),
    )
    def test_to_grayscale(self, image: np.ndarray, cs: ColorSpace):
        # for already gray images, nothing happens
        assert ImagePipeline(image, cs).grayscale().to_numpy().shape == IMAGE_SHAPE_GRAY

    def test_resize(self):
        new_w, new_h = 10, 15
        resized = ImagePipeline(NUMPY_IMAGE_RGB, ColorSpace.RGB).resize((new_w, new_h)).to_image()
        assert (resized.width, resized.height) == (new_w, new_h)

    def test_crop(self):
        crop_h_w_perc = 0.25, 0.20
        cropped = ImagePipeline(NUMPY_IMAGE_RGB, ColorSpace.RGB).crop((crop_h_w_perc)).to_image()
        # adjust if image constants change
        assert (cropped.height, cropped.width) == (6, 9)

    @pytest.mark.parametrize(
        "image,cs",
        ((NUMPY_IMAGE_RGB, ColorSpace.RGB), (NUMPY_IMAGE_GRAY, ColorSpace.GRAY)),
    )
    def test_compress_jpg(self, image: np.ndarray, cs: ColorSpace):
        buffer = ImagePipeline(image, cs).to_jpeg_buffer(5)
        jpeg_magic_number = b"\xff\xd8\xff"
        # make sure is a jpeg image now
        assert buffer.to_buffer()[:3] == jpeg_magic_number

    @pytest.mark.parametrize(
        "bin_mode,expected_bin_vals",
        (("lower", (0, 127)), ("upper", (127, 255)), ("mean", (63, 191))),
    )
    def test_downsample_two_bins(self, bin_mode: BinMode, expected_bin_vals: tuple[int, int]):
        num_bins = 2
        output_image = (
            ImagePipeline(NUMPY_IMAGE_RGB, ColorSpace.RGB)
            .downsample_channels(
                num_bins,
                bin_mode=bin_mode,
            )
            .to_numpy()
        )
        expected_values = np.array(expected_bin_vals, dtype=np.uint8)
        received_values, _ = np.unique_counts(output_image)
        assert np.array_equal(expected_values, received_values)

    @pytest.mark.parametrize("num_bins,abs_tol", ((254, 1), (255, 0), (256, 0)))
    def test_downsample_boundaries(self, num_bins: int, abs_tol: int):
        pipeline = ImagePipeline(NUMPY_IMAGE_RGB, ColorSpace.RGB)
        original_image = pipeline.to_numpy()
        output_image = pipeline.downsample_channels(num_bins).to_numpy()
        assert np.isclose(original_image, output_image, atol=abs_tol, rtol=0).all()

    def test_downsample_error(self):
        with pytest.raises(TypeError):
            ImagePipeline(NUMPY_IMAGE_RGB, ColorSpace.RGB).downsample_channels(0)
