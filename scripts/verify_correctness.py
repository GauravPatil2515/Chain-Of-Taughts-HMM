#!/usr/bin/env python3
"""
Correctness Verification Script for GSM8K CoT Traces
Extracts final numeric answer from model output and compares to GSM8K gold answers using regex.
"""

import json
import re
from pathlib import Path
from typing import Tuple, Optional


def extract_final_answer(cot_text: str) -> Tuple[Optional[str], bool]:
    """
    Extract the final answer from CoT text.
    Returns (answer_string, found_flag)
    """
    # Pattern 1: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', cot_text)
    if boxed:
        return boxed[-1].strip(), True
    
    # Pattern 2: "The answer is X" or "Final answer: X" or "Therefore X"
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+(\S+)',
        r'final\s+answer[:\s]+(\S+)',
        r'therefore[,\s]+(?:the\s+)?answer\s+(?:is\s+)?(\S+)',
        r'so\s+(?:the\s+)?answer\s+(?:is\s+)?(\S+)',
        r'\\boxed\{([^}]+)\}',
        r'answer:\s*(\S+)',
        r'= (\d+(?:\.\d+)?)',  # Simple equation ending
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cot_text, re.IGNORECASE)
        if matches:
            return matches[-1].strip(), True
    
    # Pattern 3: Last number in the text (fallback)
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', cot_text)
    if numbers:
        return numbers[-1].strip(), True
    
    return None, False


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove commas, dollar signs, whitespace
    normalized = re.sub(r'[,\$\s]', '', answer.strip())
    # Handle fractions? For GSM8K, answers are integers
    return normalized


def verify_correctness(predicted: str, ground_truth: str) -> bool:
    """
    Verify if predicted answer matches ground truth.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not gt_norm:
        return False
    
    try:
        # Try numeric comparison
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        return abs(pred_num - gt_num) < 0.01
    except ValueError:
        # String comparison
        return pred_norm.lower() == gt_norm.lower()


def load_gsm8k_gold_answers(path: str = "data/raw/gsm8k.json") -> dict:
    """Load GSM8K gold answers indexed by problem ID and question text."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    gold = {}
    gold_by_question = {}
    for item in data:
        # Extract numeric answer from the "answer" field
        # GSM8K answers are typically like "#### 18"
        match = re.search(r'####\s*(\d+)', item['answer'])
        if match:
            answer = match.group(1)
        else:
            # Fallback: use the whole answer
            answer = item['answer'].strip()
        
        gold[item['id']] = answer
        # Also index by question text (normalized)
        q_normalized = re.sub(r'\s+', ' ', item['question'].strip().lower())
        gold_by_question[q_normalized] = answer
    
    return {'by_id': gold, 'by_question': gold_by_question}


def verify_dataset(input_path: str, output_path: str, gold_path: str = "data/raw/gsm8k.json") -> dict:
    """
    Verify correctness of all samples in a dataset against GSM8K gold answers.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    gold = load_gsm8k_gold_answers(gold_path)
    gold_by_id = gold['by_id']
    gold_by_question = gold['by_question']
    
    verified_data = []
    stats = {
        'total': len(data),
        'verified': 0,
        'correct': 0,
        'incorrect': 0,
        'no_gold': 0,
        'no_extraction': 0,
        'mismatched_ids': 0
    }
    
    for item in data:
        # Get problem ID
        sample_id = item.get('sample_id', item.get('id', ''))
        
        # Extract gold answer - try by ID first, then by question text
        gold_answer = None
        gold_key = None
        
        if sample_id in gold_by_id:
            gold_answer = gold_by_id[sample_id]
            gold_key = sample_id
        else:
            # Try matching by question text
            problem_text = item.get('problem', item.get('question', ''))
            if problem_text:
                q_normalized = re.sub(r'\s+', ' ', problem_text.strip().lower())
                # Find best match
                for gq, gans in gold_by_question.items():
                    # Simple substring match for key numbers
                    if q_normalized[:50] in gq or gq[:50] in q_normalized:
                        gold_answer = gans
                        gold_key = gq
                        break
        
        if gold_answer is None:
            stats['no_gold'] += 1
            item['is_correct'] = False
            item['gold_answer'] = None
            item['predicted_answer'] = None
            verified_data.append(item)
            continue
        
        # Extract predicted answer from CoT
        cot_text = item.get('cot_text', item.get('cot_raw', ''))
        predicted_answer, found = extract_final_answer(cot_text)
        
        if not found:
            stats['no_extraction'] += 1
            item['is_correct'] = False
            item['gold_answer'] = gold_answer
            item['predicted_answer'] = None
            verified_data.append(item)
            continue
        
        # Verify
        is_correct = verify_correctness(predicted_answer, gold_answer)
        
        item['is_correct'] = is_correct
        item['gold_answer'] = gold_answer
        item['predicted_answer'] = predicted_answer
        item['gold_key'] = gold_key
        
        stats['verified'] += 1
        if is_correct:
            stats['correct'] += 1
        else:
            stats['incorrect'] += 1
        
        verified_data.append(item)
    
    # Save verified dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(verified_data, f, indent=2)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify GSM8K CoT correctness against gold answers")
    parser.add_argument("--input", type=str, required=True, help="Input dataset JSON")
    parser.add_argument("--output", type=str, required=True, help="Output verified dataset JSON")
    parser.add_argument("--gold", type=str, default="data/raw/gsm8k.json", help="Gold answers path")
    
    args = parser.parse_args()
    
    print(f"Verifying {args.input} against GSM8K gold...")
    stats = verify_dataset(args.input, args.output, args.gold)
    
    print(f"\n=== VERIFICATION RESULTS ===")
    print(f"Total samples:     {stats['total']}")
    print(f"Verified:          {stats['verified']}")
    print(f"Correct:           {stats['correct']}")
    print(f"Incorrect:         {stats['incorrect']}")
    print(f"No gold match:     {stats['no_gold']}")
    print(f"No extraction:     {stats['no_extraction']}")
    if stats['verified'] > 0:
        print(f"Accuracy:          {stats['correct']/stats['verified']*100:.1f}%")
    
    print(f"\nResults saved to: {args.output}")