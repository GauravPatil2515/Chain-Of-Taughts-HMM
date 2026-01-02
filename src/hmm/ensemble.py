"""
Ensemble HMM Trainer & Scorer.
Implements a bagged ensemble of HMMs with varying hidden states to improve robustness.
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from hmmlearn import hmm

from src.hmm.trainer import train_hmm, encode_sequences, TAG_TO_ID
from src.hmm.scorer import CoTScore

class EnsembleHMM:
    """
    Ensemble of HMMs with different numbers of hidden states and random seeds.
    Averages the scores (normalized log-likelihoods) to reduce variance.
    """
    
    def __init__(
        self,
        component_configs: List[int] = [3, 5, 8, 12],
        n_seeds_per_config: int = 3,
        n_iter: int = 100
    ):
        """
        Args:
            component_configs: List of n_components to train (e.g., [3, 5, 8])
            n_seeds_per_config: How many models to train for each component size (Bagging)
            n_iter: Max EM iterations per model
        """
        self.component_configs = component_configs
        self.n_seeds_per_config = n_seeds_per_config
        self.n_iter = n_iter
        self.models: List[Dict] = [] # List of {"model": hmm, "weight": float}
        
    def fit(self, sequences: List[List[str]], verbose: bool = True):
        """
        Train the ensemble on the provided sequences.
        """
        self.models = []
        total_models = len(self.component_configs) * self.n_seeds_per_config
        
        if verbose:
            print(f"Training Ensemble of {total_models} HMMs...")
            
        for n_comp in self.component_configs:
            for i in range(self.n_seeds_per_config):
                seed = 42 + i * 100 + n_comp
                
                if verbose:
                    print(f"  - Model: {n_comp} states, seed={seed}")
                
                # Train individual model using the shared trainer utility
                # Note: We re-use train_hmm which handles encoding internally
                # but train_hmm returns a model. We need to handle potential failures
                try:
                    model = train_hmm(
                        sequences, 
                        n_components=n_comp, 
                        n_iter=self.n_iter, 
                        random_state=seed
                    )
                    
                    # Store model
                    self.models.append({
                        "model": model,
                        "n_components": n_comp,
                        "seed": seed,
                        "weight": 1.0 # Can implement weighted voting later
                    })
                except Exception as e:
                    print(f"    Failed to train model (states={n_comp}, seed={seed}): {e}")

        if verbose:
            print(f"Successfully trained {len(self.models)} models.")

    def score(self, tag_sequence: List[str]) -> Dict:
        """
        Score a sequence using the ensemble.
        Returns aggregated statistics.
        """
        if len(tag_sequence) < 1:
            return {"ensemble_ll": float('-inf'), "std_ll": 0.0}

        # Encode once to check validity? 
        # Actually each model needs to score. hmmlearn expects encoded input per model.
        # But our encoding is static (TAG_TO_ID is constant), so we can encode once.
        encoded_seq = [TAG_TO_ID.get(tag, TAG_TO_ID["IN"]) for tag in tag_sequence]
        X = np.array(encoded_seq).reshape(-1, 1)
        
        scores = []
        
        for item in self.models:
            model = item["model"]
            try:
                # Get Log-Likelihood
                ll = model.score(X)
                
                # Normalize by length to make it comparable across different sequence lengths
                # and somewhat comparable across different model complexities
                norm_ll = ll / len(tag_sequence)
                scores.append(norm_ll)
            except:
                scores.append(float('-inf'))
        
        # Filter out errors
        valid_scores = [s for s in scores if s != float('-inf')]
        
        if not valid_scores:
            return {"ensemble_ll": float('-inf'), "std_ll": 0.0}
            
        return {
            "ensemble_ll": float(np.mean(valid_scores)),
            "median_ll": float(np.median(valid_scores)),
            "min_ll": float(np.min(valid_scores)),
            "max_ll": float(np.max(valid_scores)),
            "std_ll": float(np.std(valid_scores)),
            "n_models": len(valid_scores)
        }

    def save(self, directory: str):
        """Save ensemble models to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            "component_configs": self.component_configs,
            "n_seeds_per_config": self.n_seeds_per_config,
            "n_models": len(self.models)
        }
        with open(path / "ensemble_config.json", "w") as f:
            json.dump(config, f)
            
        # Save each model
        for idx, item in enumerate(self.models):
            joblib.dump(item["model"], path / f"model_{idx}.pkl")
            
    @classmethod
    def load(cls, directory: str) -> 'EnsembleHMM':
        """Load ensemble from directory."""
        path = Path(directory)
        
        with open(path / "ensemble_config.json", "r") as f:
            config = json.load(f)
            
        ensemble = cls(
            component_configs=config["component_configs"],
            n_seeds_per_config=config["n_seeds_per_config"]
        )
        
        ensemble.models = []
        for i in range(config["n_models"]):
            model_path = path / f"model_{i}.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                ensemble.models.append({
                    "model": model,
                    "index": i
                })
                
        return ensemble
