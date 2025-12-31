"""
Data Quality Checker
Validates generated CoT dataset for research standards.
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List


def load_data(file_path: str) -> List[Dict]:
    """Load CoT data from JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_schema(data: List[Dict]) -> Dict:
    """Check if all required fields are present."""
    required_fields = [
        "id", "dataset", "domain", "question", "ground_truth",
        "model", "cot_raw", "predicted_answer", "is_correct"
    ]
    
    issues = []
    for i, item in enumerate(data):
        missing = [f for f in required_fields if f not in item]
        if missing:
            issues.append(f"Sample {i} ({item.get('id', 'NO_ID')}): missing {missing}")
    
    return {
        "passed": len(issues) == 0,
        "issues": issues
    }


def check_duplicates(data: List[Dict]) -> Dict:
    """Check for duplicate IDs."""
    ids = [item.get("id", "") for item in data]
    duplicates = [id for id, count in Counter(ids).items() if count > 1]
    
    return {
        "passed": len(duplicates) == 0,
        "duplicates": duplicates
    }


def check_correctness_balance(data: List[Dict], target_ratio: float = 0.6, 
                               tolerance: float = 0.15) -> Dict:
    """Check correct/incorrect balance."""
    correct = sum(1 for item in data if item.get("is_correct", False))
    total = len(data)
    ratio = correct / total if total > 0 else 0
    
    in_range = abs(ratio - target_ratio) <= tolerance
    
    return {
        "passed": in_range,
        "correct_count": correct,
        "incorrect_count": total - correct,
        "correct_ratio": ratio,
        "target_range": f"{target_ratio-tolerance:.1%} - {target_ratio+tolerance:.1%}",
        "status": "✅ GOOD" if in_range else "⚠️ IMBALANCED"
    }


def check_cot_quality(data: List[Dict]) -> Dict:
    """Check CoT text quality."""
    empty_cots = []
    short_cots = []
    long_cots = []
    
    for item in data:
        cot = item.get("cot_raw", "")
        length = len(cot)
        
        if length == 0:
            empty_cots.append(item.get("id"))
        elif length < 50:
            short_cots.append(item.get("id"))
        elif length > 2000:
            long_cots.append(item.get("id"))
    
    avg_length = sum(len(item.get("cot_raw", "")) for item in data) / len(data)
    
    return {
        "passed": len(empty_cots) == 0,
        "empty_count": len(empty_cots),
        "short_count": len(short_cots),
        "long_count": len(long_cots),
        "avg_length": avg_length,
        "empty_ids": empty_cots[:5]  # Show first 5
    }


def check_generation_consistency(data: List[Dict]) -> Dict:
    """Check generation parameter consistency."""
    params = [item.get("generation_params", {}) for item in data]
    
    # Check if all have same parameters
    if not params:
        return {"passed": False, "message": "No generation params found"}
    
    first_params = params[0]
    inconsistent = []
    
    for i, p in enumerate(params[1:], 1):
        if p != first_params:
            inconsistent.append(i)
    
    return {
        "passed": len(inconsistent) == 0,
        "inconsistent_count": len(inconsistent),
        "reference_params": first_params,
        "status": "✅ CONSISTENT" if len(inconsistent) == 0 else "⚠️ INCONSISTENT"
    }


def generate_statistics(data: List[Dict]) -> Dict:
    """Generate dataset statistics."""
    domains = Counter(item.get("domain") for item in data)
    models = Counter(item.get("model") for item in data)
    datasets = Counter(item.get("dataset") for item in data)
    
    return {
        "total_samples": len(data),
        "domains": dict(domains),
        "models": dict(models),
        "datasets": dict(datasets)
    }


def main(file_path: str):
    """Run all quality checks."""
    print("="*70)
    print("DATA QUALITY CHECK")
    print("="*70)
    print(f"\nFile: {file_path}\n")
    
    # Load data
    try:
        data = load_data(file_path)
        print(f"✅ Loaded {len(data)} samples\n")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return 1
    
    # Run checks
    checks = {
        "Schema Validation": check_schema(data),
        "Duplicate Check": check_duplicates(data),
        "Correctness Balance": check_correctness_balance(data),
        "CoT Quality": check_cot_quality(data),
        "Generation Consistency": check_generation_consistency(data)
    }
    
    # Print results
    all_passed = True
    for name, result in checks.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} {name}")
        
        # Print details
        for key, value in result.items():
            if key != "passed" and not isinstance(value, list):
                print(f"    {key}: {value}")
        
        # Print issues if any
        if not result["passed"] and "issues" in result:
            print(f"    Issues: {len(result['issues'])}")
            for issue in result["issues"][:3]:
                print(f"      - {issue}")
        
        print()
        all_passed = all_passed and result["passed"]
    
    # Statistics
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    stats = generate_statistics(data)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Dataset is research-ready!")
    else:
        print("⚠️ SOME CHECKS FAILED - Review issues above")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_data_quality.py <path_to_cot_data.json>")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1]))
