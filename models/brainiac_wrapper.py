from typing import Optional
import os
import torch
from torch import Tensor, nn
import sys
from pathlib import Path

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


class BrainIACWrapper(nn.Module):
    """
    Wrapper around pre-trained BrainIAC model.
    
    Loads BrainIAC from checkpoint at /path/to/BrainIAC/src/checkpoints/BrainIAC.ckpt
    Uses ViTBackboneNet encoder for patch encoding.
    
    Exposes:
        encode(patch: [C, pd, ph, pw]) -> [embed_dim]
    """

    def __init__(self, model_path: Optional[str] = None, embed_dim: int = 768, device: str = "cuda"):
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
            # Fallback to placeholder if BrainIAC not available
            print("⚠ BrainIAC not available. Using placeholder encoder.")
            self.model = None
            self.backbone = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(1, embed_dim),
            )
            return
        
        print(f"Loading BrainIAC model from {model_path}...")
        
        # Load BrainIAC model
        # The model expects input size (96, 96, 96) by default
        # We'll resize patches to this size
        self.model = load_brainiac(model_path, device=device)
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
        # Fallback if BrainIAC not available
        if self.model is None:
            if patches.dim() == 4:
                patches = patches.unsqueeze(0)
            feats = self.backbone(patches)
            return feats.squeeze(0) if feats.dim() > 1 and feats.shape[0] == 1 else feats
        
        # Handle batch dimension
        batch_mode = patches.dim() == 5
        original_batch_size = patches.shape[0] if batch_mode else 1
        
        if patches.dim() == 4:
            patches = patches.unsqueeze(0)  # [1, C, D, H, W]
        
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
        
        # Ensure single channel (BrainIAC expects 1 channel)
        if patches.shape[1] > 1:
            patches = patches[:, 0:1, :, :, :]
        elif patches.shape[1] == 0:
            # Add channel dimension if missing
            patches = patches.unsqueeze(1)
        
        # Encode through BrainIAC
        with torch.no_grad():
            # BrainIAC forward returns CLS token embedding: [B, 768]
            cls_embedding = self.model(patches)
            
            # Ensure correct shape
            if batch_mode:
                # Batch mode: ensure [B, embed_dim]
                if cls_embedding.dim() == 1:
                    cls_embedding = cls_embedding.unsqueeze(0)
                elif cls_embedding.dim() == 2 and cls_embedding.shape[0] != original_batch_size:
                    if cls_embedding.shape[0] == 1 and original_batch_size > 1:
                        cls_embedding = cls_embedding.expand(original_batch_size, -1)
            else:
                # Single patch: return [embed_dim]
                if cls_embedding.dim() == 2:
                    cls_embedding = cls_embedding.squeeze(0)
        
        return cls_embedding
    
    def encode_batch(self, patches: Tensor) -> Tensor:
        """
        Encode a batch of patches using BrainIAC (optimized for batch processing).
        
        Args:
            patches: [B, C, pd, ph, pw] - batch of patches
        
        Returns:
            embeddings: [B, embed_dim] - CLS token embeddings
        """
        return self.encode(patches)  # encode() handles batches efficiently

