"""
Builds a strictly organic, balanced dataset from real LLM traces.
Since there are exactly 100 incorrect organic traces across the 4 models,
we will subsample 100 correct organic traces to create a perfectly balanced
dataset of 200 traces (100 correct, 100 incorrect).
"""

import json
import random

def build_organic_balanced():
    random.seed(42)
    
    # Load all traces from the organic trace file
    with open("data/v2/gsm8k_500_traces.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    correct_traces = [d for d in data if d.get("is_correct", False)]
    incorrect_traces = [d for d in data if not d.get("is_correct", False)]
    
    print(f"Loaded {len(data)} total organic traces.")
    print(f"Correct organic traces available: {len(correct_traces)}")
    print(f"Incorrect organic traces available: {len(incorrect_traces)}")
    
    # Balance to the minority class (incorrect traces)
    n_samples = len(incorrect_traces)
    sampled_correct = random.sample(correct_traces, n_samples)
    
    # Combine and shuffle
    balanced_dataset = sampled_correct + incorrect_traces
    random.shuffle(balanced_dataset)
    
    # Ensure sample IDs are somewhat clean/trackable if needed
    for i, item in enumerate(balanced_dataset):
        # We preserve the original ID, but just in case we need it:
        item["organic_id"] = item.get("sample_id", f"org_{i}")
        
    out_path = "data/v2/gsm8k_organic_balanced.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(balanced_dataset, f, indent=2)
        
    print(f"\nSuccessfully saved balanced organic dataset to {out_path}.")
    print(f"Final Dataset Size: {len(balanced_dataset)} ({n_samples} correct, {n_samples} incorrect)")

if __name__ == "__main__":
    build_organic_balanced()
