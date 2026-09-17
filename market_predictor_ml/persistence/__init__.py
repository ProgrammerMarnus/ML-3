"""
Model Persistence Module.
Handles saving and loading of trained models, configurations, and backtest results.
"""
import os
import json
import joblib
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from ..config.settings import Config, BacktestConfig


class ModelPersistence:
    """
    Manages serialization of models, configs, and results.
    """
    
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.configs_dir = self.base_dir / "configs"
        
        # Create directories
        for d in [self.models_dir, self.results_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_model(self, model: Any, name: str, metadata: Optional[Dict] = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: The trained model object (e.g., LightGBM, Ridge)
            name: Unique name for the model
            metadata: Optional dict with training info (date, params, score)
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        filepath = self.models_dir / filename
        
        save_data = {
            "model": model,
            "metadata": metadata or {},
            "saved_at": str(datetime.now())
        }
        
        joblib.dump(save_data, filepath)
        print(f"Model saved to: {filepath}")
        return str(filepath)
        
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the .pkl file
            
        Returns:
            Dict containing 'model' and 'metadata'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        data = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return data
        
    def save_results(self, results: Dict[str, Any], name: str) -> str:
        """
        Save backtest results to JSON.
        
        Args:
            results: Dict of metrics and equity curve data
            name: Name for the result file
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert non-serializable types (e.g., numpy, pandas)
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to: {filepath}")
        return str(filepath)
        
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)
            
    def save_config(self, config: Config, name: str) -> str:
        """Save configuration to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.configs_dir / filename
        
        # Convert dataclass to dict
        config_dict = self._make_serializable(config.__dict__)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Config saved to: {filepath}")
        return str(filepath)
        
    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert non-serializable objects to serializable types."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return {"type": "DataFrame", "data": obj.to_dict(), "columns": list(obj.columns)}
        elif isinstance(obj, pd.Series):
            return {"type": "Series", "data": obj.tolist()}
        elif isinstance(obj, datetime):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
