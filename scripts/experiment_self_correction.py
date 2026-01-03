"""
Self-Correction Experiment (Best-of-N).
1. Generate N=5 candidate solutions for a set of problems.
2. Select the "Best" solution based on HMM Structural Score.
3. Compare "HMM-Selected Accuracy" vs "Majority Vote" vs "Random Baseline".
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tagging.rule_based_tagger import tag_cot_file
from src.hmm.ensemble import EnsembleHMM
import post_process_correctness as ppc

# Re-use generate_batch logic directly or assume pre-generated data
# For this script, we'll assume we pass in a file that has Multiple Samples per Question.
# If not, we will simulate it by generating them.

def run_self_correction(dataset_path, model_dir, output_dir, samples_per_question=5, num_questions=20):
    print(f"🚀 Starting Self-Correction Experiment (Best-of-{samples_per_question})")
    
    # 1. Generate Data (We call generate_batch.py via subprocess to get N samples)
    # We need a special output file: "self_correction_N.json"
    gen_file = Path(output_dir) / f"self_correction_q{num_questions}_k{samples_per_question}.json"
    
    if not gen_file.exists():
        print(f"Generating {num_questions} questions x {samples_per_question} samples...")
        # Since generate_batch.py generates 1 sample per question usually, we'll modify the call
        # or just run it K times with different seeds and merge? 
        # Easier: generate_batch takes 'max_samples'. We can just run it for N*K total samples?
        # No, generate_batch iterates questions.
        
        # Strategy: Run generation loop manually here using the logic from generate_batch if imported,
        # OR run generate_batch but it handles 1-1.
        # Let's generate a merged dataset by running generate_batch K times with different seeds.
        
        merged_data = []
        for k in range(samples_per_question):
            seed = 42 + k
            temp_out = Path(output_dir) / f"temp_gen_{k}.json"
            print(f"  - Generating batch {k+1}/{samples_per_question} (Seed {seed})...")
            
            cmd = f"python generate_batch.py --dataset {dataset_path} --dataset-name gsm8k --domain math --model qwen-0.5b --output {temp_out} --max-samples {num_questions} --seed {seed}"
            ret = os.system(cmd)
            if ret != 0:
                print("Generation failed.")
                return
                
            with open(temp_out, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
                # Tag each with a group ID
                for item in batch_data:
                    item['group_id'] = item['question'] # Group by question text
                    item['sample_id'] = k
                merged_data.extend(batch_data)
        
        # Save merged
        with open(gen_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2)
    else:
        print(f"Using existing generated file: {gen_file}")

    # 2. Tag Step
    tagged_file = str(Path(output_dir) / f"self_correction_tagged.json")
    print(f"Tagging data...")
    tag_cot_file(str(gen_file), tagged_file)
    
    # 3. Load & Score with HMM
    print(f"Loading Ensemble HMM...")
    ensemble = EnsembleHMM.load(model_dir)
    
    with open(tagged_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Scoring {len(data)} samples...")
    # Group by question
    grouped = {}
    for item in data:
        qid = item.get('question', 'unknown')
        if qid not in grouped:
            grouped[qid] = []
        
        # Score
        tag_seq = item.get('tag_sequence', [])
        # Recalculate correctness just in case
        gt = item.get('ground_truth', '')
        pred = ppc.extract_answer(item.get('cot_raw', ''))
        is_correct = ppc.check_correctness(pred, gt)
        item['is_correct'] = is_correct # Ensure label is accurate
        
        if len(tag_seq) >= 2:
            score = ensemble.score(tag_seq)['ensemble_ll']
        else:
            score = float('-inf')
            
        item['hmm_score'] = score
        grouped[qid].append(item)
        
    # 4. Evaluate Selection Strategies
    # Metrics
    total_questions = len(grouped)
    random_acc = 0
    oracle_acc = 0
    hmm_acc = 0
    majority_acc = 0
    
    for qid, samples in grouped.items():
        if not samples: continue
        
        # 1. Oracle (Upper Bound): Is there ANY correct answer?
        has_correct = any(s['is_correct'] for s in samples)
        if has_correct:
            oracle_acc += 1
            
        # 2. Random Baseline (Expectation)
        num_correct = sum(1 for s in samples if s['is_correct'])
        random_acc += num_correct / len(samples)
        
        # 3. Majority Vote (Self-Consistency)
        # Get all predicted answers
        preds = [s.get('predicted_answer', '') for s in samples]
        # Find mode
        from collections import Counter
        if preds:
            most_common = Counter(preds).most_common(1)[0][0]
            # Check if majority answer matches GT
            # taking the first sample that has this answer to check vs GT
            rep_sample = next((s for s in samples if s.get('predicted_answer') == most_common), samples[0])
            gt = rep_sample.get('ground_truth', '')
            if ppc.check_correctness(most_common, gt):
                majority_acc += 1
        
        # 4. HMM Selection (Our Method)
        # Pick sample with max score
        best_sample = max(samples, key=lambda x: x['hmm_score'])
        if best_sample['is_correct']:
            hmm_acc += 1
            
    # Normalize
    random_acc /= total_questions
    oracle_acc /= total_questions
    hmm_acc /= total_questions
    majority_acc /= total_questions
    
    print("\n" + "="*60)
    print(f"SELF-CORRECTION RESULTS (N={num_questions} Qs, K={samples_per_question} Samples)")
    print("="*60)
    print(f"1. Random Baseline:       {random_acc*100:.2f}% (Average performance)")
    print(f"2. Majority Vote (SC):    {majority_acc*100:.2f}% (Standard SC method)")
    print(f"3. HMM Structure Score:   {hmm_acc*100:.2f}% (Our Method)")
    print(f"4. Oracle (Upper Bound):  {oracle_acc*100:.2f}% (If we selected perfectly)")
    print("-" * 60)
    
    diff = hmm_acc - random_acc
    print(f"Improvement over Random: {diff*100:+.2f}%")
    
    # Save results
    results = {
        "random": random_acc,
        "majority": majority_acc,
        "hmm": hmm_acc,
        "oracle": oracle_acc
    }
    with open(Path(output_dir) / "self_correction_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/raw/gsm8k.json")
    parser.add_argument("--model-dir", default="experiments/results/ensemble_150_final/models")
    parser.add_argument("--output-dir", default="experiments/results/self_correction")
    parser.add_argument("--samples", type=int, default=5, help="Samples per question")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions to test")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_self_correction(args.dataset, args.model_dir, args.output_dir, args.samples, args.questions)
