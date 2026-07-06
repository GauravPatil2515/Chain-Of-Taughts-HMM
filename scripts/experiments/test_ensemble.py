import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm.ensemble import EnsembleHMM
from src.hmm.trainer import load_tagged_data

def test_ensemble():
    print("Testing Ensemble HMM...")
    
    # 1. Create dummy data if real data not available
    # But we assume the user has some data or we can make a small dummy set
    sequences = [
        ["PR", "IN", "CA", "CO"],
        ["PR", "IN", "IN", "CA", "CO"],
        ["PR", "VE", "CO"],
        ["PR", "IN", "RF", "IN", "CO"],
        ["PR", "CA", "CA", "CO"]
    ] * 10 # 50 samples
    
    # 2. Initialize Ensemble
    ensemble = EnsembleHMM(
        component_configs=[3, 5],
        n_seeds_per_config=2,
        n_iter=10
    )
    
    # 3. Train
    print("Training...")
    ensemble.fit(sequences)
    
    # 4. Score
    print("Scoring...")
    test_seq = ["PR", "IN", "CA", "CO"]
    score = ensemble.score(test_seq)
    print(f"Score for {test_seq}: {score}")
    
    assert score['ensemble_ll'] != float('-inf')
    assert score['n_models'] == 4
    
    # 5. Save and Load
    print("Saving...")
    output_dir = "models/test_ensemble"
    ensemble.save(output_dir)
    
    print("Loading...")
    loaded_ensemble = EnsembleHMM.load(output_dir)
    score_loaded = loaded_ensemble.score(test_seq)
    
    print(f"Score from loaded model: {score_loaded}")
    assert abs(score['ensemble_ll'] - score_loaded['ensemble_ll']) < 1e-9
    
    print("✅ Ensemble Test Passed!")

if __name__ == "__main__":
    test_ensemble()
