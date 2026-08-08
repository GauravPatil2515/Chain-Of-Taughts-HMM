"""
HMM-specific modules adapter for Colab scripts
"""

import numpy as np
from typing import List, Dict

TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
ID_TO_TAG = {i: tag for i, tag in enumerate(TAGS)}

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class SimpleHMM:
    """Simple HMM wrapper for Colab."""
    
    def __init__(self, n_components: int = 5, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self._transmat = None
        self._startprob = None
        self._emissionprob = None
    
    def fit(self, sequences: List[List[str]], lengths: List[int] = None):
        """Fit model to sequences."""
        if not HMM_AVAILABLE:
            return
        
        encoded = []
        seq_lengths = []
        for seq in sequences:
            if len(seq) >= 2:
                enc = [TAG_TO_ID.get(t, 0) for t in seq]
                encoded.extend(enc)
                seq_lengths.append(len(enc))
        
        if not encoded:
            return
        
        X = np.array(encoded, dtype=np.int32).reshape(-1, 1)
        lengths = seq_lengths
        
        self.model = hmm.CategoricalHMM(
            n_components=self.n_components,
            random_state=self.random_state,
            verbose=False
        )
        self.model.fit(X, lengths)
        
        self._transmat = self.model.transmat_
        self._startprob = self.model.startprob_
        self._emissionprob = self.model.emissionprob_
        
        return self
    
    def score(self, tag_sequence: List[str]) -> float:
        """Score a tag sequence."""
        if self.model is None or not tag_sequence:
            return float('-inf')
        
        encoded = np.array([TAG_TO_ID.get(t, 0) for t in tag_sequence]).reshape(-1, 1)
        return self.model.score(encoded)


class SimpleEnsemble:
    """Simple ensemble of HMMs."""
    
    def __init__(self, n_models: int = 12, n_components: int = 5):
        self.n_models = n_models
        self.n_components = n_components
        self.models = []
    
    def fit(self, sequences: List[List[str]], seed: int = 42):
        """Train ensemble."""
        self.models = []
        
        for i in range(self.n_models):
            hmm_model = SimpleHMM(
                n_components=self.n_components,
                random_state=seed + i
            )
            hmm_model.fit(sequences)
            self.models.append(hmm_model)
        
        return self
    
    def score(self, tag_sequence: List[str]) -> Dict[str, float]:
        """Score with ensemble."""
        if not tag_sequence or not self.models:
            return {'ensemble_ll': float('-inf'), 'std_ll': 0.0}
        
        scores = []
        for m in self.models:
            try:
                ll = m.score(tag_sequence)
                norm_ll = ll / len(tag_sequence)
                scores.append(norm_ll)
            except:
                pass
        
        if not scores:
            return {'ensemble_ll': float('-inf'), 'std_ll': 0.0}
        
        return {
            'ensemble_ll': float(np.mean(scores)),
            'median_ll': float(np.median(scores)),
            'std_ll': float(np.std(scores))
        }


if __name__ == "__main__":
    test_seqs = [['PR', 'CA', 'IN', 'CO']] * 10
    
    ensemble = SimpleEnsemble(n_models=4, n_components=3)
    ensemble.fit(test_seqs)
    
    test_seq = ['PR', 'CA', 'IN', 'CO']
    score = ensemble.score(test_seq)
    print(f"Test score: {score}")
    print("SimpleHMM adapter working!")