"""
HMM Ensemble module adapter for Colab scripts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hmm.ensemble import EnsembleHMM as _EnsembleHMM
from src.hmm.trainer import TAGS, TAG_TO_ID, ID_TO_TAG

__all__ = ['EnsembleHMM', 'TAGS', 'TAG_TO_ID', 'ID_TO_TAG', 'train_ensemble']


class EnsembleHMM:
    """Adapter for existing EnsembleHMM class."""
    
    def __init__(self, n_components: int = 5, n_models: int = 12, n_iter: int = 100, seed: int = 42):
        self.n_components = n_components
        self.n_models = n_models
        self.n_iter = n_iter
        self.seed = seed
        
        original = _EnsembleHMM(
            component_configs=[n_components],
            n_seeds_per_config=n_models,
            n_iter=n_iter
        )
        self._ensemble = original
        self.models = []
    
    def fit(self, sequences, verbose: bool = True):
        self.models = []
        
        for n_comp in [self.n_components]:
            for i in range(self.n_models):
                model_seed = self.seed + i
                from hmmlearn import hmm
                import numpy as np
                
                encoded = []
                lengths = []
                for seq in sequences:
                    if len(seq) >= 2:
                        encoded_seq = [TAG_TO_ID.get(t, TAG_TO_ID["IN"]) for t in seq]
                        encoded.extend(encoded_seq)
                        lengths.append(len(encoded_seq))
                
                if not encoded:
                    continue
                
                model = hmm.CategoricalHMM(
                    n_components=n_comp,
                    n_iter=self.n_iter,
                    random_state=model_seed,
                    verbose=False
                )
                
                X = np.array(encoded).reshape(-1, 1)
                model.fit(X, lengths)
                self.models.append({'model': model, 'n_components': n_comp, 'seed': model_seed})
        
        self._trained = True
        return self
    
    def score(self, tag_sequence):
        if not tag_sequence or not self.models:
            return {"ensemble_ll": float('-inf')}
        
        encoded = np.array([TAG_TO_ID.get(t, 0) for t in tag_sequence]).reshape(-1, 1)
        scores = []
        
        for item in self.models:
            try:
                ll = item['model'].score(encoded)
                norm_ll = ll / len(tag_sequence)
                scores.append(norm_ll)
            except:
                pass
        
        if not scores:
            return {"ensemble_ll": float('-inf')}
        
        return {
            "ensemble_ll": float(np.mean(scores)),
            "median_ll": float(np.median(scores)),
            "std_ll": float(np.std(scores))
        }
    
    def save(self, path):
        import joblib
        joblib.dump({'models': self.models, 'n_components': self.n_components}, path)
    
    @classmethod
    def load(cls, path):
        import joblib
        data = joblib.load(path)
        obj = cls(n_components=data.get('n_components', 5))
        obj.models = data.get('models', [])
        return obj
    
    def to_list(self):
        """Return list of models for compatibility."""
        return [m['model'] for m in self.models]


def train_ensemble(sequences, n_components: int = 5, n_models: int = 12, seed: int = 42):
    """Convenience function."""
    ensemble = EnsembleHMM(n_components=n_components, n_models=n_models, seed=seed)
    ensemble.fit(sequences)
    return ensemble