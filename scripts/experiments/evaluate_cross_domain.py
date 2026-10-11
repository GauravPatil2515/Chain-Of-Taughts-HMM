"""
Evaluate a pre-trained HMM Ensemble on a NEW domain (Cross-Domain Test).
Does NOT retrain the model.
"""

import os
import argparse
import sys
import json
import numpy as np
import random
from pathlib import Path
from scipy.stats import pointbiserialr

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.tagging.rule_based_tagger import tag_cot_file
from src.hmm.ensemble import EnsembleHMM
from scripts.analysis import post_process_correctness as ppc

def load_gsm8k_data():
    """Load GSM8K dataset with tag sequences"""
    try:
        # Try to load the proper balanced dataset first
        gsm8k_proper_path = Path('data/v2/gsm8k_500_proper.json')
        if gsm8k_proper_path.exists():
            with open(gsm8k_proper_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded proper GSM8K data: {len(data)} samples")
            return data
        
        # Fallback to old processed data
        gsm8k_path = Path('data/processed/gsm8k_with_tags.json')
        if gsm8k_path.exists():
            with open(gsm8k_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded GSM8K data: {len(data)} samples")
            return data
        else:
            print("GSM8K data not found, generating synthetic data...")
            return generate_synthetic_gsm8k_data()
    except Exception as e:
        print(f"Error loading GSM8K data: {e}")
        return generate_synthetic_gsm8k_data()

def generate_synthetic_gsm8k_data():
    """Generate synthetic GSM8K-like data for testing"""
    print("Generating synthetic GSM8K data...")
    data = []
    
    # Create balanced dataset with realistic variations
    random.seed(42)
    np.random.seed(42)
    
    # Realistic correct and incorrect patterns with VARIATION
    correct_base_patterns = [
        ['PR', 'CA', 'IN', 'CO'],
        ['PR', 'EQ', 'CA', 'IN', 'CO'], 
        ['PR', 'IN', 'CA', 'CO'],
        ['PR', 'CA', 'RF', 'CA', 'CO'],
    ]

    incorrect_base_patterns = [
        ['PR', 'DV', 'CO'],              # Simple diversion
        ['PR', 'CA', 'CA', 'CA', 'CO'],  # Stuck in calculation
        ['PR', 'IN', 'IN', 'IN', 'CO'],  # Stuck in inference
        ['PR', 'RF', 'RF', 'RF', 'CO'],  # Stuck in reflection
    ]

    # Add noise to create diversity
    def add_noise_to_pattern(base_pattern, noise_prob=0.3):
        '''Add random variations to a base pattern'''
        pattern = list(base_pattern)
        # Sometimes insert diversion or reflection
        for i in range(len(pattern)):
            if random.random() < noise_prob:
                if random.random() < 0.5:
                    pattern.insert(i, 'DV')  # Insert diversion
                else:
                    pattern.insert(i, 'RF')  # Insert reflection
        # Sometimes duplicate a step
        if random.random() < 0.2 and len(pattern) > 2:
            idx = random.randint(0, len(pattern)-1)
            pattern.insert(idx, pattern[idx])
        # Sometimes delete a step (but keep min length)
        if len(pattern) > 3 and random.random() < 0.1:
            idx = random.randint(0, len(pattern)-1)
            pattern.pop(idx)
        return pattern

    # Generate samples
    for i in range(500):
        is_correct = i % 2 == 0  # Alternate for balance
        
        if is_correct:
            base_pattern = random.choice(correct_base_patterns)
            tag_seq = add_noise_to_pattern(base_pattern, noise_prob=0.25)
            # Correct samples get better scores
            score = np.random.normal(-1.5, 0.4)
        else:
            base_pattern = random.choice(incorrect_base_patterns)
            tag_seq = add_noise_to_pattern(base_pattern, noise_prob=0.25)
            # Incorrect samples get worse scores
            score = np.random.normal(-3.5, 0.5)
        
        # Ensure minimum length
        if len(tag_seq) < 2:
            tag_seq = ['PR', 'CO'] if is_correct else ['PR', 'DV', 'CO']
        
# Generate simple CoT text that matches tagger patterns
    # Map tags to realistic text patterns that the tagger can recognize
    tag_to_text = {
        'PR': 'First, I need to understand the problem. We are given some numbers to work with.',
        'EQ': 'The equation is x = a + b. We can set up the formula as total = part1 + part2.',
        'CA': 'Calculate the result: 5 + 3 = 8. Compute the sum of the numbers.',
        'IN': 'Therefore, this means the intermediate result is 8. This implies we need to continue.',
        'RF': 'Wait, let me recheck that calculation. Actually, I need to verify the numbers.',
        'DV': 'By the way, this reminds me of a similar problem. Off topic, but interesting.',
        'CO': 'The final answer is 17. So the result is 17 dollars.'
    }
    
    # Generate CoT text by mapping each tag to its text pattern
    cot_parts = []
    for tag in tag_seq:
        if tag in tag_to_text:
            cot_parts.append(tag_to_text[tag])
        else:
            cot_parts.append(f'Processing {tag}')
    
    cot_text = ' '.join(cot_parts)
        
        data.append({
            'sample_id': f'gsm_{i:04d}',
            'problem': f'Problem {i}: Calculate something',
            'cot_text': cot_text,
            'tag_sequence': tag_seq,
            'is_correct': is_correct,
            'hmm_score': {'normalized_ll': float(score)},
        })

    # Shuffle
    random.shuffle(data)
    
    print(f'Created synthetic dataset: {len(data)} samples')
    print(f'Correct: {sum(1 for d in data if d["is_correct"])}')
    print(f'Incorrect: {sum(1 for d in data if not d["is_correct"])}')
    
    return data

def evaluate_cross_domain(input_file, model_dir, output_dir, dataset_name):
    print(f"🚀 Starting Cross-Domain Evaluation")
    print(f"Target Data: {input_file}")
    print(f"Using Model: {model_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Tagging
    tagged_file = str(Path(output_dir) / f"{Path(input_file).stem}_tagged.json")
    print(f"\n[1/4] Tagging data...")
    if not os.path.exists(tagged_file):
        tag_cot_file(input_file, tagged_file)
    else:
        print("Tagged file exists, using it.")

    # 2. Correctness
    corrected_file = str(Path(output_dir) / f"{Path(input_file).stem}_corrected.json")
    print(f"\n[2/4] Verifying correctness for domain '{dataset_name}'...")
    
    # We can use the script via subprocess to avoid import complexity with main blocks
    # OR better, since I verified ppc imports above (if path is right), let's implement the loop here
    
    with open(tagged_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    correct_count = 0
    for item in data:
        # Determine dataset type for extraction
        ds_type = dataset_name if dataset_name else item.get('dataset', 'gsm8k')
        
        # Use simple fallback if ground truth missing
        gt = item.get('ground_truth', '')
        if not gt:
            gt = item.get('answer', '') # Some datasets use 'answer'
            
        cot = item.get('cot_raw', '')
        
        # Trust existing is_correct if valid
        if 'is_correct' in item:
            is_correct = item['is_correct']
            pred = item.get('predicted_answer', '')
        else:
            # Re-compute if missing
            pred = ppc.extract_answer(cot, dataset=ds_type)
            is_correct = ppc.check_correctness(pred, gt)
        
        item['predicted_answer'] = pred
        item['is_correct'] = is_correct
        if is_correct:
            correct_count += 1
            
    print(f"Accuracy on {dataset_name}: {correct_count}/{len(data)} ({correct_count/len(data)*100:.1f}%)")
    
    with open(corrected_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 3. Load Model
    print(f"\n[3/4] Loading Pre-trained Ensemble...")
    try:
        ensemble = EnsembleHMM.load(model_dir)
        print(f"Loaded ensemble with {len(ensemble.models)} models.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Score
    print(f"\n[4/4] Scoring & Evaluating...")
    scores = []
    labels = []
    
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        res = ensemble.score(tag_seq)
        
        # Store score
        item['hmm_score'] = res
        
        scores.append(res['ensemble_ll'])
        labels.append(1 if item['is_correct'] else 0)
        
    # Save final results
    final_file = str(Path(output_dir) / f"{Path(input_file).stem}_final.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    # Stats
    valid_indices = [i for i, s in enumerate(scores) if s != float('-inf')]
    valid_scores = [scores[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    if len(set(valid_labels)) < 2:
        print("⚠️ Cannot calculate correlation: All samples have same label (all correct or all incorrect).")
        return

    r, p = pointbiserialr(valid_labels, valid_scores)
    
    print("\n" + "="*50)
    print("CROSS-DOMAIN RESULTS")
    print("="*50)
    print(f"Domain: {dataset_name}")
    print(f"Model ID: {Path(model_dir).parent.name}")
    print(f"Correlation (r): {r:.4f}")
    print(f"P-value (p):     {p:.4f}")
    
    if p < 0.05:
        print("✅ Statistically Significant!")
    else:
        print("❌ Not Significant (Expected for cross-domain?)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="experiments/results/cross_domain")
    parser.add_argument("--dataset-name", default="arc_challenge")
    args = parser.parse_args()
    
    evaluate_cross_domain(args.input, args.model_dir, args.output_dir, args.dataset_name)
