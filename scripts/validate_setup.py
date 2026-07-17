#!/usr/bin/env python3
"""
validate_setup.py - Validate CoT-HMM project setup and configuration
Run this to check that all components are properly configured.

Usage:
    python scripts/validate_setup.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.9."""
    if sys.version_info < (3, 9):
        print("✗ Python 3.9+ required")
        return False
    print(f"✓ Python {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check required packages are installed."""
    deps = {
        "hmmlearn": "HMM training",
        "numpy": "Numerical operations",
        "scipy": "Statistical tests",
        "sklearn": "Evaluation metrics",
        "transformers": "Neural models (optional)",
        "torch": "Neural models (optional)",
        "sentence-transformers": "ROSCOE baseline (optional)",
    }
    
    missing = []
    for pkg, purpose in deps.items():
        try:
            if pkg == "sklearn":
                import sklearn
            elif pkg == "sentence-transformers":
                import sentence_transformers
            else:
                __import__(pkg)
            print(f"✓ {pkg} ({purpose})")
        except ImportError:
            missing.append(pkg)
            print(f"✗ {pkg} ({purpose}) - MISSING")
    
    return missing

def check_file_structure():
    """Check project file structure."""
    required = [
        "src/hmm/trainer.py",
        "src/hmm/scorer.py",
        "src/tagging/rule_based_tagger.py",
        "src/evaluation/baselines.py",
        "src/experiments_v2/data_schema.py",
    ]
    
    base = Path(".")
    all_exist = True
    
    for f in required:
        path = base / f
        if path.exists():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} - MISSING")
            all_exist = False
    
    return all_exist

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("○ GPU not available (using CPU)")
            return False
    except ImportError:
        print("○ torch not installed - GPU check skipped")
        return False

def check_model_files():
    """Check for trained models."""
    models_dir = Path("models")
    pkl_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    
    if pkl_files:
        for f in pkl_files[:3]:
            print(f"✓ Found model: {f.name}")
        return True
    else:
        print("○ No trained models found (run training first)")
        return False

def check_data_availability():
    """Check for available datasets."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("✗ data/ directory missing")
        return False
    
    json_files = list(data_dir.rglob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith('.json~')]
    
    if json_files:
        for f in json_files[:5]:
            print(f"✓ Found data: {f.relative_to(data_dir)}")
        return True
    else:
        print("○ No dataset files found (run data collection first)")
        return False


def main():
    print("=" * 60)
    print("CoT-HMM Setup Validation")
    print("=" * 60)
    print()
    
    print("1. Python Version")
    print("-" * 30)
    py_ok = check_python_version()
    print()
    
    print("2. Dependencies")
    print("-" * 30)
    missing = check_dependencies()
    print()
    
    print("3. File Structure")
    print("-" * 30)
    struct_ok = check_file_structure()
    print()
    
    print("4. GPU Availability")
    print("-" * 30)
    gpu_ok = check_gpu()
    print()
    
    print("5. Trained Models")
    print("-" * 30)
    model_ok = check_model_files()
    print()
    
    print("6. Dataset Files")
    print("-" * 30)
    data_ok = check_data_availability()
    print()
    
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    critical_ok = py_ok and struct_ok
    warnings = len(missing) > 0
    
    if critical_ok:
        print("✓ Setup is ready for basic HMM experiments")
        if warnings:
            print(f"⚠ Run: pip install {' '.join(missing)} for neural baselines")
    else:
        print("✗ Setup incomplete - fix issues above")
    
    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())