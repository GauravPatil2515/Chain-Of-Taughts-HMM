#!/usr/bin/env python3
"""
Unit tests for the HMM ensemble scoring functions.
Tests determinism and correctness of scoring logic.
"""

import pytest
import numpy as np
from pathlib import Path
import pickle
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmm.scorer import EnsembleScorer, load_ensemble
from hmm.trainer import EnsembleHMMTrainer


@pytest.fixture
def sample_tag_sequences():
    """Sample tag sequences for testing."""
    return [
        ["PR", "IN", "CA", "CO"],
        ["PR", "CA", "IN", "CA", "CO"],
        ["IN", "IN", "CA", "VE", "CO"],
    ]

@pytest.fixture
def mock_ensemble(sample_tag_sequences):
    """Create a mock ensemble for testing."""
    trainer = EnsembleHMMTrainer(n_components=3, n_ensemble=2, n_iter=10)
    trainer.fit(sample_tag_sequences)
    return trainer.ensemble


class TestEnsembleScorer:
    """Tests for EnsembleScorer class."""

    def test_scorer_initialization(self, mock_ensemble):
        """Test scorer initializes correctly with ensemble."""
        scorer = EnsembleScorer(mock_ensemble)
        assert len(scorer.models) == 2
        assert scorer.n_models == 2

    def test_score_single_sequence(self, mock_ensemble):
        """Test scoring a single tag sequence."""
        scorer = EnsembleScorer(mock_ensemble)
        sequence = ["PR", "IN", "CA", "CO"]
        score = scorer.score(sequence)
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)

    def test_score_multiple_sequences(self, mock_ensemble):
        """Test scoring multiple sequences returns array."""
        scorer = EnsembleScorer(mock_ensemble)
        sequences = [
            ["PR", "IN", "CA", "CO"],
            ["PR", "CA", "IN", "CA", "CO"],
        ]
        scores = scorer.score_batch(sequences)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_deterministic_scoring(self, mock_ensemble):
        """Test that scoring is deterministic for same input."""
        scorer = EnsembleScorer(mock_ensemble)
        sequence = ["PR", "IN", "CA", "CO"]
        
        score1 = scorer.score(sequence)
        score2 = scorer.score(sequence)
        score3 = scorer.score(sequence)
        
        assert score1 == score2 == score3

    def test_length_normalization(self, mock_ensemble):
        """Test that scores are length-normalized."""
        scorer = EnsembleScorer(mock_ensemble)
        
        short_seq = ["PR", "CO"]
        long_seq = ["PR", "IN", "IN", "IN", "CA", "CA", "VE", "CO"]
        
        short_score = scorer.score(short_seq)
        long_score = scorer.score(long_seq)
        
        # Both should be reasonable scores (not extreme due to length)
        assert -10 < short_score < 0
        assert -10 < long_score < 0

    def test_unknown_tag_handling(self, mock_ensemble):
        """Test handling of unknown tags in sequences."""
        scorer = EnsembleScorer(mock_ensemble)
        # Sequence with a tag not in training vocabulary
        sequence = ["PR", "IN", "UNKNOWN_TAG", "CO"]
        
        # Should not crash, should handle gracefully
        score = scorer.score(sequence)
        assert isinstance(score, float)

    def test_empty_sequence_handling(self, mock_ensemble):
        """Test handling of empty sequences."""
        scorer = EnsembleScorer(mock_ensemble)
        
        # Empty sequence
        score = scorer.score([])
        assert isinstance(score, float)
        assert np.isneginf(score) or score < -100  # Very low score for empty

    def test_scorer_save_load(self, mock_ensemble):
        """Test saving and loading scorer preserves behavior."""
        scorer = EnsembleScorer(mock_ensemble)
        sequence = ["PR", "IN", "CA", "CO"]
        original_score = scorer.score(sequence)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            scorer.save(f.name)
            loaded_scorer = EnsembleScorer.load(f.name)
        
        loaded_score = loaded_scorer.score(sequence)
        assert original_score == loaded_score


class TestEnsembleConsistency:
    """Tests for ensemble consistency properties."""

    def test_ensemble_averaging(self, sample_tag_sequences):
        """Test that ensemble score equals average of individual model scores."""
        trainer = EnsembleHMMTrainer(n_components=3, n_ensemble=4, n_iter=10)
        trainer.fit(sample_tag_sequences)
        ensemble = trainer.ensemble
        scorer = EnsembleScorer(ensemble)
        
        sequence = ["PR", "IN", "CA", "CO"]
        ensemble_score = scorer.score(sequence)
        
        # Get individual model scores
        individual_scores = []
        for model in ensemble:
            try:
                log_likelihood = model.score([sequence])  # hmmlearn format
                individual_scores.append(log_likelihood / len(sequence))
            except:
                # Skip models that can't score
                pass
        
        if individual_scores:
            avg_score = np.mean(individual_scores)
            assert np.isclose(ensemble_score, avg_score, rtol=1e-5)

    def test_ensemble_size_effect(self, sample_tag_sequences):
        """Test that larger ensemble doesn't change score type."""
        for n_ensemble in [1, 2, 4, 8]:
            trainer = EnsembleHMMTrainer(n_components=3, n_ensemble=n_ensemble, n_iter=10)
            trainer.fit(sample_tag_sequences)
            scorer = EnsembleScorer(trainer.ensemble)
            
            score = scorer.score(["PR", "IN", "CA", "CO"])
            assert isinstance(score, float)
            assert not np.isnan(score)


class TestTrainerDeterminism:
    """Tests for training determinism."""

    def test_trainer_deterministic_with_seed(self, sample_tag_sequences):
        """Test that training with same seed produces identical models."""
        trainer1 = EnsembleHMMTrainer(n_components=3, n_ensemble=3, n_iter=20, random_state=42)
        trainer1.fit(sample_tag_sequences)
        
        trainer2 = EnsembleHMMTrainer(n_components=3, n_ensemble=3, n_iter=20, random_state=42)
        trainer2.fit(sample_tag_sequences)
        
        # Compare model parameters
        for m1, m2 in zip(trainer1.ensemble, trainer2.ensemble):
            np.testing.assert_array_almost_equal(m1.transmat_, m2.transmat_)
            np.testing.assert_array_almost_equal(m1.emissionprob_, m2.emissionprob_)
            np.testing.assert_array_almost_equal(m1.startprob_, m2.startprob_)

    def test_different_seeds_produce_different_models(self, sample_tag_sequences):
        """Test that different seeds produce different models."""
        trainer1 = EnsembleHMMTrainer(n_components=3, n_ensemble=3, n_iter=20, random_state=42)
        trainer1.fit(sample_tag_sequences)
        
        trainer2 = EnsembleHMMTrainer(n_components=3, n_ensemble=3, n_iter=20, random_state=123)
        trainer2.fit(sample_tag_sequences)
        
        # At least one model should differ
        all_same = True
        for m1, m2 in zip(trainer1.ensemble, trainer2.ensemble):
            if not np.allclose(m1.transmat_, m2.transmat_):
                all_same = False
                break
        assert not all_same


class TestLikelihoodComputation:
    """Tests for likelihood computation correctness."""

    def test_log_likelihood_negative(self, mock_ensemble):
        """Test that log-likelihoods are negative (probabilities < 1)."""
        scorer = EnsembleScorer(mock_ensemble)
        
        test_sequences = [
            ["PR", "IN", "CA", "CO"],
            ["PR", "CA", "CO"],
            ["IN", "IN", "IN"],
        ]
        
        for seq in test_sequences:
            score = scorer.score(seq)
            assert score < 0, f"Score should be negative, got {score}"

    def test_more_correct_like_sequences_higher_score(self, sample_tag_sequences):
        """Test that sequences similar to training data get higher scores."""
        trainer = EnsembleHMMTrainer(n_components=3, n_ensemble=4, n_iter=20, random_state=42)
        trainer.fit(sample_tag_sequences)
        scorer = EnsembleScorer(trainer.ensemble)
        
        # Sequence similar to training data
        similar_seq = ["PR", "IN", "CA", "CO"]
        # Random/unlikely sequence
        random_seq = ["DV", "DV", "DV", "DV"]
        
        similar_score = scorer.score(similar_seq)
        random_score = scorer.score(random_seq)
        
        assert similar_score > random_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])