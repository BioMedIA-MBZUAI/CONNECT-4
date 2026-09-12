"""Strict train-partition scaling for the ROI feature stream."""
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.preprocessing import StandardScaler
import json


SCALER_METADATA_SCHEMA = "connect4-feature-scalers-v1"


class FeatureScalerManager:
    """Manage the radiomics and AnatCL scalers fitted on training subjects."""
    
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
    
    def save_scalers(self, output_dir: Path):
        """
        Save all fitted scalers to disk.
        
        Args:
            output_dir: Directory to save scalers
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for scaler_name, scaler in sorted(self.scalers.items()):
            scaler_path = output_dir / f"{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved {scaler_name} scaler to {scaler_path}", flush=True)
        
        # Save metadata
        metadata = {
            'schema': SCALER_METADATA_SCHEMA,
            'scaler_names': sorted(self.scalers),
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
        if not scaler_dir.is_dir():
            raise FileNotFoundError(f"scaler directory not found: {scaler_dir}")

        metadata_path = scaler_dir / "scaler_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(
                f"scaler metadata is required; missing {metadata_path}"
            )
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"invalid scaler metadata: {metadata_path}") from exc
        scaler_names = metadata.get('scaler_names')
        if (
            metadata.get('schema') != SCALER_METADATA_SCHEMA
            or not isinstance(scaler_names, list)
            or not scaler_names
            or any(not isinstance(name, str) or not name for name in scaler_names)
            or len(set(scaler_names)) != len(scaler_names)
            or metadata.get('num_scalers') != len(scaler_names)
        ):
            raise RuntimeError(f"invalid scaler metadata contract: {metadata_path}")
        expected_files = {f"{name}_scaler.pkl" for name in scaler_names}
        actual_files = {path.name for path in scaler_dir.glob("*_scaler.pkl")}
        if actual_files != expected_files:
            raise RuntimeError(
                "scaler files differ from metadata: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )

        loaded: Dict[str, StandardScaler] = {}
        for scaler_name in scaler_names:
            scaler_path = scaler_dir / f"{scaler_name}_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as exc:
                raise RuntimeError(f"cannot load scaler {scaler_path}") from exc
            if not isinstance(scaler, StandardScaler) or not hasattr(scaler, "n_features_in_"):
                raise RuntimeError(f"{scaler_path} is not a fitted StandardScaler")
            loaded[scaler_name] = scaler
            print(f"Loaded {scaler_name} scaler from {scaler_path}", flush=True)
        self.scalers = loaded
        self.scaler_dir = scaler_dir
    
    def scale_radiomics(self, radiomics: torch.Tensor) -> torch.Tensor:
        """
        Scale radiomics features.
        
        Args:
            radiomics: Feature tensor [num_features]
        
        Returns:
            Scaled feature tensor
        """
        if 'radiomics' not in self.scalers:
            raise RuntimeError("radiomics scaler has not been loaded or fitted")
        
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
            raise RuntimeError("AnatCL scaler has not been loaded or fitted")
        
        scaler = self.scalers['anatcl']
        # Reshape to [1, embed_dim] for transform
        emb_np = anatcl_emb.cpu().numpy().reshape(1, -1)
        scaled_np = scaler.transform(emb_np)
        return torch.from_numpy(scaled_np.flatten()).to(anatcl_emb.device).to(anatcl_emb.dtype)
    
