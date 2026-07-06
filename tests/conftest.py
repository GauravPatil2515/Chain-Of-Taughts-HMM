#!/usr/bin/env python3
"""
Shared test fixtures and configuration for CoT-HMM tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_tag_sequences():
    """Standard sample tag sequences for testing."""
    return [
        ["PR", "IN", "CA", "CO"],
        ["PR", "CA", "IN", "CA", "CO"],
        ["IN", "IN", "CA", "VE", "CO"],
        ["PR", "EQ", "CA", "IN", "CO"],
        ["IN", "CA", "IN", "CA", "CO"],
    ]


@pytest.fixture
def trained_ensemble(sample_tag_sequences):
    """Pre-trained ensemble for scoring tests."""
    from hmm.trainer import EnsembleHMMTrainer
    trainer = EnsembleHMMTrainer(n_components=3, n_ensemble=4, n_iter=20, random_state=42)
    trainer.fit(sample_tag_sequences)
    return trainer.ensemble


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensure deterministic test runs."""
    np.random.seed(42)
    yield
    np.random.seed(None)