from typing import Optional
import os
import torch
from torch import Tensor, nn
import sys
from pathlib import Path
from data.provenance import canonical_sha256, sha256_file

# Locate the BrainIAC source tree robustly. Search, in order:
#   1. $CONNECT4_BRAINIAC                       (explicit override)
#   2. <repo>/BrainIAC/src                      (local copy or symlink)
#   3. known absolute install paths
# whichever first contains load_brainiac.py is used (no fragile single path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BRAINIAC_CANDIDATES = [
    os.environ.get("CONNECT4_BRAINIAC"),
    str(PROJECT_ROOT / "BrainIAC" / "src"),
    "/path/to/BrainIAC/src",
]
BRAINIAC_ROOT = next(
    (p for p in _BRAINIAC_CANDIDATES if p and (Path(p) / "load_brainiac.py").exists()),
    None,
)

BRAINIAC_AVAILABLE = False
BRAINIAC_ERROR = None
if BRAINIAC_ROOT is not None:
    if BRAINIAC_ROOT not in sys.path:
        sys.path.insert(0, BRAINIAC_ROOT)
    try:
        from load_brainiac import load_brainiac
        BRAINIAC_AVAILABLE = True
    except ImportError as e:                      # missing dep (e.g. monai)
        BRAINIAC_ERROR = str(e)
        print(f"Warning: BrainIAC found at {BRAINIAC_ROOT} but import failed: {BRAINIAC_ERROR}")
else:
    BRAINIAC_ERROR = "load_brainiac.py not found in any candidate path"
    print(f"Warning: BrainIAC not found (searched: {[c for c in _BRAINIAC_CANDIDATES if c]}). "
          "Set $CONNECT4_BRAINIAC or place BrainIAC/ at the repo root.")


def _required_sha256(value: Optional[str], label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a complete lowercase SHA-256 digest")
    return digest


def brainiac_source_fingerprint(model_path: str, *, embed_dim: int = 768) -> dict:
    """Bind BrainIAC weights to its loader/architecture code and adapter."""
    if BRAINIAC_ROOT is None:
        raise RuntimeError("BrainIAC source root is unavailable for provenance")
    source_root = Path(BRAINIAC_ROOT).resolve()
    source_files = {
        "connect4/models/brainiac_wrapper.py": sha256_file(Path(__file__).resolve())
    }
    source_files.update(
        {
            f"brainiac/{path.relative_to(source_root).as_posix()}": sha256_file(path)
            for path in sorted(source_root.rglob("*.py"))
            if path.is_file()
        }
    )
    if "brainiac/load_brainiac.py" not in source_files:
        raise RuntimeError("BrainIAC loader source is missing from provenance")
    return {
        "implementation": "BrainIAC",
        "checkpoint_sha256": sha256_file(model_path),
        "source_files_sha256": source_files,
        "source_files_fingerprint_sha256": canonical_sha256(source_files),
        "adapter": {
            "input_channels": 1,
            "resize_shape": [96, 96, 96],
            "resize_mode": "trilinear",
            "align_corners": False,
            "output": "CLS token embedding",
            "embedding_dim": int(embed_dim),
        },
    }


class BrainIACWrapper(nn.Module):
    """
    Wrapper around pre-trained BrainIAC model.
    
    Loads BrainIAC from checkpoint at /path/to/BrainIAC/src/checkpoints/BrainIAC.ckpt
    Uses ViTBackboneNet encoder for patch encoding.
    
    Exposes:
        encode(patch: [C, pd, ph, pw]) -> [embed_dim]
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: int = 768,
        device: str = "cuda",
        expected_checkpoint_sha256: Optional[str] = None,
        expected_source_fingerprint_sha256: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        
        # Default checkpoint path: alongside the resolved BrainIAC source, else
        # the repo-local path. ($CONNECT4_BRAINIAC_CKPT overrides everything.)
        if model_path is None:
            model_path = os.environ.get("CONNECT4_BRAINIAC_CKPT")
        if model_path is None:
            cands = []
            if BRAINIAC_ROOT is not None:
                cands.append(str(Path(BRAINIAC_ROOT) / "checkpoints" / "BrainIAC.ckpt"))
            cands.append(str(PROJECT_ROOT / "BrainIAC" / "src" / "checkpoints" / "BrainIAC.ckpt"))
            model_path = next((c for c in cands if Path(c).exists()), cands[-1])
        
        if not BRAINIAC_AVAILABLE:
            raise RuntimeError(
                "The paper-faithful image stream requires the frozen BrainIAC "
                f"foundation model, but it could not be loaded: {BRAINIAC_ERROR}. "
                "Set CONNECT4_BRAINIAC and CONNECT4_BRAINIAC_CKPT (or place "
                "BrainIAC/ at the repository root)."
            )
        
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"BrainIAC checkpoint not found: {model_path}. "
                "A pretrained checkpoint is required for paper-faithful preprocessing."
            )
        self.model_path = str(Path(model_path).resolve())
        self.source_fingerprint = brainiac_source_fingerprint(
            self.model_path, embed_dim=self.embed_dim
        )
        expected_checkpoint = _required_sha256(
            expected_checkpoint_sha256
            or os.environ.get("CONNECT4_BRAINIAC_SHA256"),
            "BrainIAC checkpoint SHA-256",
        )
        expected_source = _required_sha256(
            expected_source_fingerprint_sha256
            or os.environ.get("CONNECT4_BRAINIAC_SOURCE_SHA256"),
            "BrainIAC source fingerprint SHA-256",
        )
        if self.source_fingerprint["checkpoint_sha256"] != expected_checkpoint:
            raise RuntimeError("BrainIAC checkpoint does not match the pinned SHA-256")
        if (
            self.source_fingerprint["source_files_fingerprint_sha256"]
            != expected_source
        ):
            raise RuntimeError(
                "BrainIAC loader/architecture source does not match the pinned SHA-256"
            )

        print(f"Loading BrainIAC model from {model_path}...")
        
        # Load BrainIAC model
        # The model expects input size (96, 96, 96) by default
        # We'll resize patches to this size
        self.model = load_brainiac(self.model_path, device=device)
        self.model.eval()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print("✓ BrainIAC model loaded")

    def encode(self, patches: Tensor) -> Tensor:
        """
        Encode patches using BrainIAC (supports both single and batch).
        
        Args:
            patches: [C, pd, ph, pw] or [1, C, pd, ph, pw] or [B, C, pd, ph, pw]
                    Patches can be any size, will be resized to (96, 96, 96)
        
        Returns:
            embedding: [embed_dim] or [B, embed_dim] - CLS token embedding(s)
        """
        if not torch.is_tensor(patches):
            raise TypeError("BrainIAC patches must be a torch.Tensor")
        batch_mode = patches.dim() == 5
        if patches.dim() == 4:
            patches = patches.unsqueeze(0)  # [1, C, D, H, W]
        elif patches.dim() != 5:
            raise ValueError(
                "BrainIAC patches must have shape [C,D,H,W] or [B,C,D,H,W]"
            )
        batch_size = int(patches.shape[0])
        if batch_size < 1 or patches.shape[1] != 1:
            raise ValueError(
                f"BrainIAC requires a non-empty single-channel patch batch, got {patches.shape}"
            )
        if not torch.isfinite(patches).all():
            raise ValueError("BrainIAC input patches contain NaN or infinity")
        
        # Move to device
        patches = patches.to(self.device)
        
        # Resize patches to (96, 96, 96) to match BrainIAC input size
        # BrainIAC expects (96, 96, 96) by default
        target_size = (96, 96, 96)
        if patches.shape[2:] != target_size:
            patches = torch.nn.functional.interpolate(
                patches,
                size=target_size,
                mode='trilinear',
                align_corners=False,
            )
        
        # Encode through BrainIAC
        with torch.no_grad():
            # BrainIAC forward returns CLS token embedding: [B, 768]
            cls_embedding = self.model(patches)
        if not torch.is_tensor(cls_embedding) or tuple(cls_embedding.shape) != (
            batch_size,
            self.embed_dim,
        ):
            shape = getattr(cls_embedding, "shape", None)
            raise RuntimeError(
                "BrainIAC must return exactly one configured embedding per patch: "
                f"expected {(batch_size, self.embed_dim)}, got {shape}"
            )
        if not torch.isfinite(cls_embedding).all():
            raise RuntimeError("BrainIAC produced NaN or infinity")
        return cls_embedding if batch_mode else cls_embedding[0]
    
    def encode_batch(self, patches: Tensor) -> Tensor:
        """
        Encode a batch of patches using BrainIAC (optimized for batch processing).
        
        Args:
            patches: [B, C, pd, ph, pw] - batch of patches
        
        Returns:
            embeddings: [B, embed_dim] - CLS token embeddings
        """
        return self.encode(patches)  # encode() handles batches efficiently


__all__ = ["BrainIACWrapper", "brainiac_source_fingerprint"]
