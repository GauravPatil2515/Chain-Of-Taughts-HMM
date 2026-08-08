"""
Colab/Kaggle Configuration for CoT-HMM Experiments
Dynamic configuration that adapts to environment (Colab vs Kaggle vs Local)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentConfig:
    """Configuration manager for CoT-HMM experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.is_colab = self._detect_colab()
        self.is_kaggle = self._detect_kaggle()
        self.is_local = not (self.is_colab or self.is_kaggle)
        self.workspace = Path(self._get_workspace())
        self.data_dir = self.workspace / "data"
        self.models_dir = self.workspace / "models"
        self.results_dir = self.workspace / "results"
        self.cache_dir = Path("/tmp/cot_hmm_cache")
        
        self.device = self._get_device()
        self.max_seq_length = 512
        self.batch_size = 8 if self.is_colab or self.is_kaggle else 4
        self.n_ensemble = 12
        self.n_bootstrap = 10000
        self.random_seed = 42
        
        self._setup_directories()
        
        if config_path:
            self.load(config_path)
    
    def _detect_colab(self) -> bool:
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _detect_kaggle(self) -> bool:
        return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    
    def _get_workspace(self) -> str:
        if self.is_colab:
            return "/content/drive/MyDrive/cot_hmm"
        ws = Path.cwd() / "Chain-Of-Taughts-HMM"
        return str(ws) if ws.exists() else "."
    
    def _get_device(self) -> str:
        if self.is_colab:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except:
                pass
        if self.is_kaggle:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except:
                pass
        return "cpu"
    
    def _setup_directories(self) -> None:
        for d in [self.cache_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def load(self, path: str) -> None:
        with open(path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, path: str) -> None:
        data = {}
        for key in ['device', 'batch_size', 'n_ensemble', 'n_bootstrap', 'random_seed']:
            data[key] = getattr(self, key)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'environment': 'colab' if self.is_colab else 'kaggle' if self.is_kaggle else 'local',
            'device': self.device,
            'workspace': str(self.workspace),
            'data_dir': str(self.data_dir),
            'models_dir': str(self.models_dir),
            'results_dir': str(self.results_dir),
            'batch_size': self.batch_size,
            'n_ensemble': self.n_ensemble,
            'n_bootstrap': self.n_bootstrap,
            'random_seed': self.random_seed,
        }

config = ExperimentConfig()

def print_config(cfg: ExperimentConfig = config) -> None:
    """Pretty print configuration."""
    print("\n" + "="*60)
    print("CoT-HMM Experiment Configuration")
    print("="*60)
    print(f"Environment: {'Colab' if cfg.is_colab else 'Kaggle' if cfg.is_kaggle else 'Local'}")
    print(f"Device: {cfg.device}")
    print(f"Workspace: {cfg.workspace}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Ensemble Size: {cfg.n_ensemble}")
    print(f"Bootstrap Samples: {cfg.n_bootstrap}")
    print(f"Random Seed: {cfg.random_seed}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_config()