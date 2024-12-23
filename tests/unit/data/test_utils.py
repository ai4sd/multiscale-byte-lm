import torch

from mblm.data.utils import target_loss_mask
from mblm.data.utils.misc import shift_remap_tensor


def test_loss_mask():
    inp = torch.ones((3), dtype=torch.long)  # special dtype
    loss_mask = target_loss_mask([(inp, 2), (inp, 1), (inp, 0.1)])

    assert len(loss_mask) == 3 * len(inp)
    assert loss_mask.dtype == torch.float  # not long

    mask_2, mask_1, mask_0_1 = torch.chunk(loss_mask, 3)

    assert (mask_2 == 2).all().item()
    assert (mask_1 == 1).all().item()
    assert (mask_0_1 == 0.1).all().item()


def test_shift_unshift():
    input_tensor = torch.tensor(
        [
            [2, 3, 3],
            [1, 2, 3],
            [3, 5, 1],
        ],
        dtype=torch.uint8,
    )
    shifted, unshift, indices = shift_remap_tensor(input_tensor, range_start=11)
    expected_shifted = 11 + torch.tensor(
        [
            [1, 2, 2],
            [0, 1, 2],
            [2, 3, 0],
        ]
    )

    assert shifted.equal(expected_shifted)
    assert shifted.shape == input_tensor.shape
    assert shifted.dtype == input_tensor.dtype
    assert input_tensor.equal(unshift[indices])
