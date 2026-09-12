from typing import Optional, List
from pathlib import Path
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
from data.provenance import canonical_sha256, directory_file_sha256


CLINICAL_MODERNBERT_MODEL_ID = "Simonlee711/Clinical_ModernBERT"


def _immutable_revision(value: Optional[str]) -> str:
    revision = str(value or "").strip().lower()
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            "Clinical-ModernBERT requires a full immutable 40- or 64-character "
            "source revision"
        )
    return revision


class ModernBERTWrapper(nn.Module):
    """
    Wrapper around Clinical ModernBERT model.
    
    Uses the actual Clinical_ModernBERT model from HuggingFace:
    'Simonlee711/Clinical_ModernBERT'
    
    Exposes:
        encode(text: str) -> [embed_dim]
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: int = 768,
        device: Optional[torch.device] = None,
        revision: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Load the configured pretrained clinical encoder (paper ref. [23]).
        model_name = model_path or CLINICAL_MODERNBERT_MODEL_ID
        self.model_name = str(model_name)
        local_path = Path(self.model_name).expanduser()
        immutable_revision = _immutable_revision(revision)
        if not local_path.is_dir() and self.model_name != CLINICAL_MODERNBERT_MODEL_ID:
            raise ValueError(
                "remote clinical text encoding must use "
                f"{CLINICAL_MODERNBERT_MODEL_ID!r}"
            )
        print("Loading Clinical ModernBERT model and tokenizer...")
        self.model = AutoModel.from_pretrained(
            model_name, revision=immutable_revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=immutable_revision
        )
        
        # Move model to device if specified (important for DDP)
        if device is not None:
            self.model = self.model.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get actual embedding dimension from model
        # Clinical ModernBERT typically has 768-dim embeddings
        if hasattr(self.model.config, 'hidden_size'):
            self.embed_dim = self.model.config.hidden_size

        local_hashes = {}
        if local_path.is_dir():
            local_hashes = directory_file_sha256(local_path)
        resolved_revision = str(
            getattr(self.model.config, "_commit_hash", "") or ""
        ).strip().lower()
        if not local_hashes and resolved_revision != immutable_revision:
            raise RuntimeError(
                "resolved Clinical-ModernBERT Hub revision differs from the "
                "configured immutable revision"
            )
        self.source_fingerprint = {
            "implementation": "Clinical-ModernBERT",
            "upstream_model_id": CLINICAL_MODERNBERT_MODEL_ID,
            "model_name": self.model_name,
            "revision": immutable_revision,
            "local_files_sha256": local_hashes,
        }
        self.source_fingerprint["fingerprint_sha256"] = canonical_sha256(
            self.source_fingerprint
        )
        
        print(f"Clinical ModernBERT loaded (embed_dim={self.embed_dim})")
        
        # Freeze model parameters (optional - set to False if you want to fine-tune)
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, text: str) -> Tensor:
        """
        Encode a text description using Clinical ModernBERT.
        
        Args:
            text: Text description string
        
        Returns:
            embedding: [embed_dim] - [CLS] token embedding
        """
        device = next(self.model.parameters()).device
        
        # Tokenize input text
        inputs = self.tokenizer(
            str(text),
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token) as sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, embed_dim]
        if cls_embedding.shape != (1, self.embed_dim):
            raise RuntimeError(
                "Clinical-ModernBERT must return one configured CLS embedding"
            )
        cls_embedding = cls_embedding[0]
        if not torch.isfinite(cls_embedding).all():
            raise RuntimeError("Clinical-ModernBERT produced NaN or infinity")
        
        return cls_embedding
    
    def forward(self, texts: List[str]) -> Tensor:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of text description strings
        
        Returns:
            embeddings: [batch_size, embed_dim]
        """
        device = next(self.model.parameters()).device
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, embed_dim]
        if cls_embeddings.shape != (len(texts), self.embed_dim):
            raise RuntimeError(
                "Clinical-ModernBERT returned an invalid batch embedding shape"
            )
        if not torch.isfinite(cls_embeddings).all():
            raise RuntimeError("Clinical-ModernBERT produced NaN or infinity")
        return cls_embeddings
