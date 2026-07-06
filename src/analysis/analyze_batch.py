"""Detailed batch analysis script"""
import json
import sys
import numpy as np
from collections import Counter

def analyze_batch(filepath):
    """Perform detailed analysis on generated batch"""
    with open(filepath) as f:
        data = json.load(f)
    
    print("="*70)
    print("DETAILED BATCH ANALYSIS")
    print("="*70)
    
    # Basic stats
    total = len(data)
    correct = sum(1 for d in data if d['is_correct'])
    incorrect = total - correct
    
    print(f"\nTotal samples: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
    
    # CoT length distribution
    lengths = [len(d['cot_raw']) for d in data]
    print(f"\nCoT Length Statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Std: {np.std(lengths):.1f}")
    
    # Empty/error detection
    empty = sum(1 for l in lengths if l < 20)
    print(f"\nEmpty/Very Short CoTs (<20 chars): {empty} ({empty/total*100:.1f}%)")
    
    # Predicted answer distribution
    answers = [d.get('predicted_answer', '') for d in data]
    unique_answers = len(set(answers))
    print(f"\nUnique predicted answers: {unique_answers}")
    
    # Most common answers
    answer_counts = Counter(answers)
    print(f"\nTop 5 most common answers:")
    for ans, count in answer_counts.most_common(5):
        display_ans = ans[:30] + "..." if len(ans) > 30 else ans
        print(f"  '{display_ans}': {count}")
    
    # Model info
    if data:
        print(f"\nModel: {data[0]['model']}")
        print(f"Dataset: {data[0]['dataset']}")
        print(f"Domain: {data[0]['domain']}")
        print(f"Temperature: {data[0]['generation_params']['temperature']}")
        print(f"Max tokens: {data[0]['generation_params']['max_new_tokens']}")
    
    # Length distribution by correctness
    correct_lengths = [len(d['cot_raw']) for d in data if d['is_correct']]
    incorrect_lengths = [len(d['cot_raw']) for d in data if not d['is_correct']]
    
    if correct_lengths:
        print(f"\nCorrect CoT lengths:")
        print(f"  Mean: {np.mean(correct_lengths):.1f}")
        print(f"  Median: {np.median(correct_lengths):.1f}")
    
    if incorrect_lengths:
        print(f"\nIncorrect CoT lengths:")
        print(f"  Mean: {np.mean(incorrect_lengths):.1f}")
        print(f"  Median: {np.median(incorrect_lengths):.1f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_batch.py <batch_file.json>")
        sys.exit(1)
    
    analyze_batch(sys.argv[1])
