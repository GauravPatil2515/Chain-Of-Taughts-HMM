"""
HMM Ensemble Module for Colab/Kaggle
"""

import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
ID_TO_TAG = {i: tag for i, tag in enumerate(TAGS)}


class EnsembleHMM:
    """HMM Ensemble for scoring CoT sequences."""
    
    def __init__(self, n_components: int = 5, n_models: int = 12, n_iter: int = 100, seed: int = 42):
        self.n_components = n_components
        self.n_models = n_models
        self.n_iter = n_iter
        self.seed = seed
        self.models: List = []
        self._trained = False
    
    def fit(self, sequences: List[List[str]], verbose: bool = True) -> 'EnsembleHMM':
        """Train ensemble on tag sequences."""
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn required")
        
        self.models = []
        
        encoded_sequences = []
        for seq in sequences:
            if len(seq) >= 2:
                encoded = [TAG_TO_ID.get(t, 0) for t in seq]
                encoded_sequences.append((np.array(encoded).reshape(-1, 1), len(encoded)))
        
        if not encoded_sequences:
            raise ValueError("No valid sequences for training")
        
        for i in range(self.n_models):
            try:
                model = hmm.CategoricalHMM(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    random_state=self.seed + i,
                    verbose=False
                )
                
                X = np.concatenate([s[0] for s in encoded_sequences])
                lengths = [s[1] for s in encoded_sequences]
                model.fit(X, lengths)
                self.models.append(model)
                
                if verbose and (i + 1) % 5 == 0:
                    print(f"  Trained {i+1}/{self.n_models} models")
            except Exception as e:
                if verbose:
                    print(f"  Model {i} failed: {e}")
        
        self._trained = True
        if verbose:
            print(f"Trained {len(self.models)} models successfully")
        
        return self
    
    def score(self, tag_sequence: List[str]) -> Dict[str, Any]:
        """Score a tag sequence with the ensemble."""
        if not tag_sequence:
            return {"ensemble_ll": float('-inf'), "std_ll": 0.0, "median_ll": float('-inf')}
        
        encoded = np.array([TAG_TO_ID.get(t, 0) for t in tag_sequence]).reshape(-1, 1)
        scores = []
        
        for model in self.models:
            try:
                ll = model.score(encoded)
                norm_ll = ll / len(tag_sequence)
                scores.append(norm_ll)
            except:
                pass
        
        if not scores:
            return {"ensemble_ll": float('-inf'), "std_ll": 0.0}
        
        return {
            "ensemble_ll": float(np.mean(scores)),
            "median_ll": float(np.median(scores)),
            "std_ll": float(np.std(scores)),
            "min_ll": float(np.min(scores)),
            "max_ll": float(np.max(scores)),
            "n_models": len(scores)
        }
    
    def save(self, path: str) -> None:
        """Save ensemble to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'n_components': self.n_components,
            'n_models': self.n_models,
            'models': self.models,
            'tag_to_id': TAG_TO_ID
        }
        joblib.dump(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleHMM':
        """Load ensemble from file."""
        data = joblib.load(path)
        ensemble = cls(n_components=data['n_components'], n_models=data['n_models'])
        ensemble.models = data['models']
        ensemble._trained = True
        ensemble.__class__.TAG_TO_ID = data['tag_to_id']
        return ensemble


def train_ensemble(sequences: List[List[str]], n_components: int = 5, n_models: int = 12, seed: int = 42) -> EnsembleHMM:
    """Convenience function to train an ensemble."""
    ensemble = EnsembleHMM(n_components=n_components, n_models=n_models, seed=seed)
    ensemble.fit(sequences, verbose=True)
    return ensemble