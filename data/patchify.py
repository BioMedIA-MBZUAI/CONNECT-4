from typing import Tuple, List
import torch
from torch import Tensor


class Patchify3D:
    """
    Patchify 3D volumes.
    """
    
    def __init__(self, patch_size: Tuple[int, int, int]):
        self.patch_size = patch_size
    
    def patchify(self, x: Tensor) -> List[Tensor]:
        """
        Patchify a 3D volume.
        
        Args:
            x: Tensor of shape [B, C, D, H, W] or [1, C, D, H, W]
        
        Returns:
            List of patches, each [C, pd, ph, pw]
        """
        if x.dim() == 5:
            b, c, d, h, w = x.shape
        elif x.dim() == 4:
            c, d, h, w = x.shape
            x = x.unsqueeze(0)
            b = 1
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D")
        
        pd, ph, pw = self.patch_size
        assert d % pd == 0 and h % ph == 0 and w % pw == 0, \
            f"Volume dimensions ({d}, {h}, {w}) must be divisible by patch_size ({pd}, {ph}, {pw})"
        
        patches = []
        for b_idx in range(b):
            for d_idx in range(0, d, pd):
                for h_idx in range(0, h, ph):
                    for w_idx in range(0, w, pw):
                        patch = x[b_idx, :, d_idx:d_idx+pd, h_idx:h_idx+ph, w_idx:w_idx+pw]
                        patches.append(patch)
        
        return patches


def patchify_3d(x: Tensor, patch_size: Tuple[int, int, int]) -> Tensor:
    """
    Patchify a 3D volume.

    Args:
        x: Tensor of shape [B, C, D, H, W].
        patch_size: (pd, ph, pw).

    Returns:
        patches: [B, N, C, pd, ph, pw] where N = (D/pd)*(H/ph)*(W/pw).
    """
    b, c, d, h, w = x.shape
    pd, ph, pw = patch_size
    assert d % pd == 0 and h % ph == 0 and w % pw == 0, "Volume dimensions must be divisible by patch_size."

    x = x.view(b, c, d // pd, pd, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    patches = x.view(b, -1, c, pd, ph, pw)
    return patches


def unpatchify_3d(patches: Tensor, volume_shape: Tuple[int, int, int]) -> Tensor:
    """
    Reconstruct a 3D volume from patches.

    Args:
        patches: [B, N, C, pd, ph, pw].
        volume_shape: (D, H, W).

    Returns:
        x: [B, C, D, H, W].
    """
    b, n, c, pd, ph, pw = patches.shape
    d, h, w = volume_shape
    assert (d * h * w) // (pd * ph * pw) == n, "Number of patches does not match volume_shape."

    nd = d // pd
    nh = h // ph
    nw = w // pw
    x = patches.view(b, nd, nh, nw, c, pd, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    x = x.view(b, c, d, h, w)
    return x


