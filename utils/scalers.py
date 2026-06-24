"""
Feature scaling utilities for CONNECT_4_V2.
Handles saving and loading scalers for all feature types.
"""
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
import json


class FeatureScalerManager:
    """
    Manages scalers for different feature types:
    - radiomics: Radiomics features (per ROI)
    - anatcl: AnatCL embeddings (per ROI)
    - brainiac: BrainIAC patch embeddings
    - modernbert: ModernBERT patch embeddings
    """
    
    def __init__(self, scaler_dir: Optional[Path] = None):
        self.scaler_dir = Path(scaler_dir) if scaler_dir else None
        self.scalers: Dict[str, StandardScaler] = {}
    
    def fit_radiomics_scaler(self, all_radiomics: Dict[int, torch.Tensor]) -> StandardScaler:
        """
        Fit scaler on all radiomics features across all ROIs.
        
        Args:
            all_radiomics: Dict mapping structure_id -> feature tensor
        
        Returns:
            Fitted StandardScaler
        """
        # Collect all radiomics features
        features_list = []
        for structure_id, feats in all_radiomics.items():
            if feats is not None and feats.numel() > 0:
                features_list.append(feats.cpu().numpy().flatten())
        
        if not features_list:
            raise ValueError("No radiomics features found to fit scaler")
        
        # Stack into [num_samples, num_features]
        features_array = np.stack(features_list)
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(features_array)
        
        return scaler
    
    def fit_anatcl_scaler(self, all_anatcl_embeddings: List[torch.Tensor]) -> StandardScaler:
        """
        Fit scaler on all AnatCL embeddings.
        
        Args:
            all_anatcl_embeddings: List of embedding tensors [embed_dim]
        
        Returns:
            Fitted StandardScaler
        """
        if not all_anatcl_embeddings:
            raise ValueError("No AnatCL embeddings found to fit scaler")
        
        # Stack into [num_samples, embed_dim]
        embeddings_array = np.stack([emb.cpu().numpy() for emb in all_anatcl_embeddings])
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(embeddings_array)
        
        return scaler
    
    def fit_brainiac_scaler(self, all_brainiac_embeddings: List[torch.Tensor]) -> StandardScaler:
        """
        Fit scaler on all BrainIAC patch embeddings.
        
        Args:
            all_brainiac_embeddings: List of embedding tensors [num_patches, embed_dim]
        
        Returns:
            Fitted StandardScaler
        """
        if not all_brainiac_embeddings:
            raise ValueError("No BrainIAC embeddings found to fit scaler")
        
        # Concatenate all patches: [total_patches, embed_dim]
        embeddings_list = [emb.cpu().numpy() for emb in all_brainiac_embeddings]
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(embeddings_array)
        
        return scaler
    
    def fit_modernbert_scaler(self, all_modernbert_embeddings: List[torch.Tensor]) -> StandardScaler:
        """
        Fit scaler on all ModernBERT patch embeddings.
        
        Args:
            all_modernbert_embeddings: List of embedding tensors [num_patches, embed_dim]
        
        Returns:
            Fitted StandardScaler
        """
        if not all_modernbert_embeddings:
            raise ValueError("No ModernBERT embeddings found to fit scaler")
        
        # Concatenate all patches: [total_patches, embed_dim]
        embeddings_list = [emb.cpu().numpy() for emb in all_modernbert_embeddings]
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(embeddings_array)
        
        return scaler
    
    def save_scalers(self, output_dir: Path):
        """
        Save all fitted scalers to disk.
        
        Args:
            output_dir: Directory to save scalers
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_dir / f"{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved {scaler_name} scaler to {scaler_path}", flush=True)
        
        # Save metadata
        metadata = {
            'scaler_names': list(self.scalers.keys()),
            'num_scalers': len(self.scalers),
        }
        metadata_path = output_dir / "scaler_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved scaler metadata to {metadata_path}", flush=True)
    
    def load_scalers(self, scaler_dir: Path):
        """
        Load all scalers from disk.
        
        Args:
            scaler_dir: Directory containing saved scalers
        """
        scaler_dir = Path(scaler_dir)
        
        # Load metadata
        metadata_path = scaler_dir / "scaler_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            scaler_names = metadata.get('scaler_names', [])
        else:
            # Fallback: try to find all scaler files
            scaler_files = list(scaler_dir.glob("*_scaler.pkl"))
            scaler_names = [f.stem.replace("_scaler", "") for f in scaler_files]
        
        # Load each scaler
        for scaler_name in scaler_names:
            scaler_path = scaler_dir / f"{scaler_name}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers[scaler_name] = pickle.load(f)
                print(f"Loaded {scaler_name} scaler from {scaler_path}", flush=True)
            else:
                print(f"Warning: Scaler file not found: {scaler_path}", flush=True)
    
    def scale_radiomics(self, radiomics: torch.Tensor) -> torch.Tensor:
        """
        Scale radiomics features.
        
        Args:
            radiomics: Feature tensor [num_features]
        
        Returns:
            Scaled feature tensor
        """
        if 'radiomics' not in self.scalers:
            return radiomics  # No scaler, return as-is
        
        scaler = self.scalers['radiomics']
        # Reshape to [1, num_features] for transform
        radiomics_np = radiomics.cpu().numpy().reshape(1, -1)
        scaled_np = scaler.transform(radiomics_np)
        return torch.from_numpy(scaled_np.flatten()).to(radiomics.device).to(radiomics.dtype)
    
    def scale_anatcl(self, anatcl_emb: torch.Tensor) -> torch.Tensor:
        """
        Scale AnatCL embedding.
        
        Args:
            anatcl_emb: Embedding tensor [embed_dim]
        
        Returns:
            Scaled embedding tensor
        """
        if 'anatcl' not in self.scalers:
            return anatcl_emb  # No scaler, return as-is
        
        scaler = self.scalers['anatcl']
        # Reshape to [1, embed_dim] for transform
        emb_np = anatcl_emb.cpu().numpy().reshape(1, -1)
        scaled_np = scaler.transform(emb_np)
        return torch.from_numpy(scaled_np.flatten()).to(anatcl_emb.device).to(anatcl_emb.dtype)
    
    def scale_brainiac(self, brainiac_emb: torch.Tensor) -> torch.Tensor:
        """
        Scale BrainIAC embeddings.
        
        Args:
            brainiac_emb: Embedding tensor [num_patches, embed_dim] or [embed_dim]
        
        Returns:
            Scaled embedding tensor
        """
        if 'brainiac' not in self.scalers:
            return brainiac_emb  # No scaler, return as-is
        
        scaler = self.scalers['brainiac']
        original_shape = brainiac_emb.shape
        original_device = brainiac_emb.device
        original_dtype = brainiac_emb.dtype
        
        # Flatten to [num_samples, embed_dim]
        if brainiac_emb.ndim == 1:
            emb_np = brainiac_emb.cpu().numpy().reshape(1, -1)
        else:
            emb_np = brainiac_emb.cpu().numpy()
        
        scaled_np = scaler.transform(emb_np)
        
        # Reshape back to original shape
        if len(original_shape) == 1:
            scaled_tensor = torch.from_numpy(scaled_np.flatten())
        else:
            scaled_tensor = torch.from_numpy(scaled_np)
        
        return scaled_tensor.to(original_device).to(original_dtype)
    
    def scale_modernbert(self, modernbert_emb: torch.Tensor) -> torch.Tensor:
        """
        Scale ModernBERT embeddings.
        
        Args:
            modernbert_emb: Embedding tensor [num_patches, embed_dim] or [embed_dim]
        
        Returns:
            Scaled embedding tensor
        """
        if 'modernbert' not in self.scalers:
            return modernbert_emb  # No scaler, return as-is
        
        scaler = self.scalers['modernbert']
        original_shape = modernbert_emb.shape
        original_device = modernbert_emb.device
        original_dtype = modernbert_emb.dtype
        
        # Flatten to [num_samples, embed_dim]
        if modernbert_emb.ndim == 1:
            emb_np = modernbert_emb.cpu().numpy().reshape(1, -1)
        else:
            emb_np = modernbert_emb.cpu().numpy()
        
        scaled_np = scaler.transform(emb_np)
        
        # Reshape back to original shape
        if len(original_shape) == 1:
            scaled_tensor = torch.from_numpy(scaled_np.flatten())
        else:
            scaled_tensor = torch.from_numpy(scaled_np)
        
        return scaled_tensor.to(original_device).to(original_dtype)



