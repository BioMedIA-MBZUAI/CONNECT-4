from typing import Optional, List
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class ModernBERTWrapper(nn.Module):
    """
    Wrapper around Clinical ModernBERT model.
    
    Uses the actual Clinical_ModernBERT model from HuggingFace:
    'Simonlee711/Clinical_ModernBERT'
    
    Exposes:
        encode(text: str) -> [embed_dim]
    """

    def __init__(self, model_path: Optional[str] = None, embed_dim: int = 768, device: Optional[torch.device] = None):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Load Clinical ModernBERT model and tokenizer
        print("Loading Clinical ModernBERT model and tokenizer...")
        self.model = AutoModel.from_pretrained('Simonlee711/Clinical_ModernBERT')
        self.tokenizer = AutoTokenizer.from_pretrained('Simonlee711/Clinical_ModernBERT')
        
        # Move model to device if specified (important for DDP)
        if device is not None:
            self.model = self.model.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get actual embedding dimension from model
        # Clinical ModernBERT typically has 768-dim embeddings
        if hasattr(self.model.config, 'hidden_size'):
            self.embed_dim = self.model.config.hidden_size
        
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
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # [embed_dim]
        
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
        
        return cls_embeddings


