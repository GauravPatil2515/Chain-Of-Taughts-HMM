"""
Dataset Download Script
Downloads benchmark datasets: GSM8K, StrategyQA, ARC
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_gsm8k(output_dir: str, samples: int = 200) -> str:
    """Download GSM8K math reasoning dataset."""
    print(f"Downloading GSM8K dataset ({samples} samples)...")
    
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Sample if needed
    if samples < len(dataset):
        dataset = dataset.select(range(samples))
    
    output_path = Path(output_dir) / "gsm8k.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for i, item in enumerate(tqdm(dataset, desc="Processing GSM8K")):
        data.append({
            "id": f"gsm8k_{i:04d}",
            "question": item["question"],
            "answer": item["answer"].split("####")[-1].strip(),
            "full_solution": item["answer"],
            "dataset": "gsm8k",
            "domain": "math"
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} samples to {output_path}")
    return str(output_path)


def download_strategyqa(output_dir: str, samples: int = 100) -> str:
    """Download StrategyQA commonsense reasoning dataset."""
    print(f"Downloading StrategyQA dataset ({samples} samples)...")
    
    dataset = load_dataset("metaeval/strategy-qa", split="train", trust_remote_code=True)
    
    if samples < len(dataset):
        dataset = dataset.select(range(samples))
    
    output_path = Path(output_dir) / "strategyqa.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for i, item in enumerate(tqdm(dataset, desc="Processing StrategyQA")):
        data.append({
            "id": f"strategyqa_{i:04d}",
            "question": item["question"],
            "answer": "yes" if item["answer"] else "no",
            "facts": item.get("facts", []),
            "dataset": "strategyqa",
            "domain": "commonsense"
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} samples to {output_path}")
    return str(output_path)


def download_arc(output_dir: str, samples: int = 100) -> str:
    """Download ARC-Challenge logic/science dataset."""
    print(f"Downloading ARC-Challenge dataset ({samples} samples)...")
    
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    
    if samples < len(dataset):
        dataset = dataset.select(range(samples))
    
    output_path = Path(output_dir) / "arc_challenge.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for i, item in enumerate(tqdm(dataset, desc="Processing ARC")):
        choices = item["choices"]
        answer_key = item["answerKey"]
        answer_idx = choices["label"].index(answer_key)
        
        data.append({
            "id": f"arc_{i:04d}",
            "question": item["question"],
            "choices": dict(zip(choices["label"], choices["text"])),
            "answer": answer_key,
            "answer_text": choices["text"][answer_idx],
            "dataset": "arc_challenge",
            "domain": "science"
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} samples to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["gsm8k", "strategyqa", "arc", "all"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset in ["gsm8k", "all"]:
        download_gsm8k(args.output_dir, args.samples)
    
    if args.dataset in ["strategyqa", "all"]:
        download_strategyqa(args.output_dir, args.samples)
    
    if args.dataset in ["arc", "all"]:
        download_arc(args.output_dir, args.samples)
    
    print("\nDataset download complete!")


if __name__ == "__main__":
    main()
