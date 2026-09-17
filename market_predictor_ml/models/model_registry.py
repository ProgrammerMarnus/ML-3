"""
Model registry for version tracking and management.

Implements:
- Model registration with metadata
- Model persistence and retrieval
- Performance tracking
- Best model selection
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from ..core import IPredictionModel, IModelRegistry


class ModelMetadata:
    """Container for model metadata."""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        created_at: str,
        hyperparameters: Dict[str, Any],
        training_metrics: Dict[str, float],
        feature_names: List[str],
        description: str = "",
        tags: Optional[List[str]] = None
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.created_at = created_at
        self.hyperparameters = hyperparameters
        self.training_metrics = training_metrics
        self.feature_names = feature_names
        self.description = description
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'created_at': self.created_at,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'description': self.description,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(**data)


class InMemoryModelRegistry(IModelRegistry):
    """
    In-memory model registry implementation.
    
    Suitable for development and testing.
    For production, use a database-backed registry.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self._models: Dict[str, IPredictionModel] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._storage_path = Path(storage_path) if storage_path else None
        
        # Load from storage if path provided
        if self._storage_path and self._storage_path.exists():
            self._load_from_storage()
    
    def _generate_model_id(self, model: IPredictionModel, metadata: Dict[str, Any]) -> str:
        """Generate unique model ID based on model hash and timestamp."""
        content = f"{datetime.now().isoformat()}{str(metadata)}"
        return f"model_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def register_model(
        self,
        model: IPredictionModel,
        metadata: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a trained model and return its ID."""
        # Generate model ID
        model_id = self._generate_model_id(model, metadata)
        
        # Store model
        self._models[model_id] = model
        
        # Create and store metadata
        model_metadata = ModelMetadata(
            model_id=model_id,
            model_type=metadata.get('model_type', 'unknown'),
            created_at=datetime.now().isoformat(),
            hyperparameters=metadata.get('hyperparameters', {}),
            training_metrics=metadata.get('training_metrics', {}),
            feature_names=metadata.get('feature_names', []),
            description=metadata.get('description', ''),
            tags=metadata.get('tags', [])
        )
        self._metadata[model_id] = model_metadata
        
        # Save to storage if path provided
        if self._storage_path:
            self._save_to_storage(model_id, artifacts)
        
        return model_id
    
    def get_model(self, model_id: str) -> IPredictionModel:
        """Retrieve a model by ID."""
        if model_id not in self._models:
            # Try to load from storage
            if self._storage_path:
                self._load_model_from_storage(model_id)
                if model_id in self._models:
                    return self._models[model_id]
            raise KeyError(f"Model with ID {model_id} not found")
        
        return self._models[model_id]
    
    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List registered models with optional filtering."""
        results = []
        
        for model_id, metadata in self._metadata.items():
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if key == 'model_type' and metadata.model_type != value:
                        match = False
                    elif key == 'tag' and value not in metadata.tags:
                        match = False
                    elif key == 'min_sharpe' and metadata.training_metrics.get('sharpe_ratio', 0) < value:
                        match = False
                
                if not match:
                    continue
            
            results.append(metadata.to_dict())
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x['created_at'], reverse=True)
        
        return results
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Optional[Dict[str, Any]]:
        """Get the best performing model based on a metric."""
        best_model = None
        best_value = float('-inf')
        
        for model_id, metadata in self._metadata.items():
            metric_value = metadata.training_metrics.get(metric, float('-inf'))
            if metric_value > best_value:
                best_value = metric_value
                best_model = metadata.to_dict()
        
        return best_model
    
    def _save_to_storage(self, model_id: str, artifacts: Optional[Dict[str, Any]] = None):
        """Save model and metadata to disk."""
        if not self._storage_path:
            return
        
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save model object
        model_path = self._storage_path / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self._models[model_id], f)
        
        # Save metadata
        metadata_path = self._storage_path / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._metadata[model_id].to_dict(), f, indent=2)
        
        # Save artifacts if provided
        if artifacts:
            artifacts_path = self._storage_path / f"{model_id}_artifacts"
            artifacts_path.mkdir(exist_ok=True)
            for name, data in artifacts.items():
                artifact_file = artifacts_path / f"{name}.pkl"
                with open(artifact_file, 'wb') as f:
                    pickle.dump(data, f)
    
    def _load_from_storage(self):
        """Load all models from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        
        for metadata_file in self._storage_path.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                model_id = metadata_dict['model_id']
                metadata = ModelMetadata.from_dict(metadata_dict)
                self._metadata[model_id] = metadata
                
                # Load model
                model_path = self._storage_path / f"{model_id}.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self._models[model_id] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model {metadata_file.name}: {e}")
    
    def _load_model_from_storage(self, model_id: str):
        """Load a specific model from storage."""
        if not self._storage_path:
            return
        
        model_path = self._storage_path / f"{model_id}.pkl"
        metadata_path = self._storage_path / f"{model_id}_metadata.json"
        
        if model_path.exists() and metadata_path.exists():
            with open(model_path, 'rb') as f:
                self._models[model_id] = pickle.load(f)
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                self._metadata[model_id] = ModelMetadata.from_dict(metadata_dict)


# Global default registry instance
_default_registry: Optional[InMemoryModelRegistry] = None


def get_registry(storage_path: Optional[str] = None) -> IModelRegistry:
    """Get or create the default model registry."""
    global _default_registry
    
    if _default_registry is None:
        _default_registry = InMemoryModelRegistry(storage_path)
    
    return _default_registry


def register_model(
    model: IPredictionModel,
    metadata: Dict[str, Any],
    storage_path: Optional[str] = None
) -> str:
    """Convenience function to register a model."""
    registry = get_registry(storage_path)
    return registry.register_model(model, metadata)


def get_model(model_id: str, storage_path: Optional[str] = None) -> IPredictionModel:
    """Convenience function to retrieve a model."""
    registry = get_registry(storage_path)
    return registry.get_model(model_id)
