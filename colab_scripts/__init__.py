"""
__init__.py for colab_scripts
"""

from .config import ExperimentConfig, config, print_config
from .data_generation import DataGenerator, scale_dataset, GeneratedSample
from .bootstrap_ci import BootstrapAnalyzer, BootstrapResult, run_full_bootstrap
from .cross_architecture_transfer import test_cross_architecture_transfer, run_transfer_experiment
from .entropy_drift import run_edr_experiment, compute_entropy_drift_rate, analyze_trajectory_by_correctness
from .run_pipeline import ExperimentRunner, generate_paper_summary

__all__ = [
    'ExperimentConfig', 'config', 'print_config',
    'DataGenerator', 'scale_dataset', 'GeneratedSample',
    'BootstrapAnalyzer', 'BootstrapResult', 'run_full_bootstrap',
    'test_cross_architecture_transfer', 'run_transfer_experiment',
    'run_plausibility_margin_experiment', 'compute_plausibility_margin',
    'run_edr_experiment', 'compute_entropy_drift_rate', 'analyze_trajectory_by_correctness',
    'ExperimentRunner', 'generate_paper_summary'
]