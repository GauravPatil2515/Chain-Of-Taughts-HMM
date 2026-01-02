"""
HMM-based Reward Function for RLHF.
Uses the EnsembleHMM score to provide dense rewards for reasoning steps.
"""

import numpy as np
from typing import List
from src.hmm.ensemble import EnsembleHMM
from src.tagging.rule_based_tagger import tag_step

class StructureReward:
    def __init__(self, ensemble_path: str):
        self.ensemble = EnsembleHMM.load(ensemble_path)
        
    def compute_reward(self, current_cot_text: str) -> float:
        """
        Compute reward for the current Chain-of-Thought based on structural plausibility.
        Args:
            current_cot_text: The CoT generated so far.
        Returns:
            Scalar reward.
        """
        # 1. Parse and Tag
        # Ideally we tag the whole sequence
        from src.tagging.rule_based_tagger import tag_cot
        tagged = tag_cot(current_cot_text)
        tag_sequence = [s['tag'] for s in tagged]
        
        # 2. Score with Ensemble
        score_info = self.ensemble.score(tag_sequence)
        ll = score_info['ensemble_ll']
        
        # 3. Normalize/Clip Reward
        # Log-likelihood is negative, we want higher reward for higher LL (closer to 0)
        # We can normalize against a baseline or just use the raw LL scaled.
        # Simple transformation:
        # If typical good LL is -2.0 and bad is -5.0
        return np.clip(ll, -10.0, 0.0)

    def step_reward(self, previous_text: str, new_step: str) -> float:
        """
        Compute incremental reward for a new step.
        """
        full_text = previous_text + " " + new_step
        return self.compute_reward(full_text)
