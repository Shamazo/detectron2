import torch
from typing import Tuple


class PatchUtil:
    """Class functioning as a namespace for utils related
    to patching.
    """

    @staticmethod
    def batched_patches(images: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
        # print("Begin batch of {}".format(images.shape))
        patches = images.unfold(-2, patch_size, patch_size)
        patches = patches.unfold(-2, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, -2, -1)
        _, patch_x, patch_y, _, _, _ = patches.shape
        patches = patches.flatten(0, 2)
        # print("Outputting batched size of {}".format(patches.shape))
        return patches, patch_x, patch_y

    @staticmethod
    def reconstruct_from_batched_patches(batched_out, patch_x=4, patch_y=4):
        batches, channels, x, y = batched_out.shape
        assert x == y
        bsz = batches // (patch_x * patch_y)
        out = batched_out.view(bsz, patch_x, patch_y, channels, x, y).permute(0, 3, 1, 2, -2, -1)
        return out.transpose(-3, -2).flatten(-2, -1).flatten(-3, -2)
