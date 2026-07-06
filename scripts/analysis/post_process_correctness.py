import json
import re

def extract_answer(response: str, dataset: str = "gsm8k") -> str:
    """Extract answer based on dataset type."""
    response_lower = response.lower()
    
    if dataset == "arc" or dataset == "arc_challenge":
        # Look for single letter A-E enclosed in various ways
        # Prioritize "Answer is (X)" or "Solution: (X)"
        patterns = [
            r"answer is ([a-e])\b",
            r"solution is ([a-e])\b",
            r"answer: ([a-e])\b",
            r"^\s*([a-e])\s*$"
        ]
        for pat in patterns:
            match = re.search(pat, response_lower)
            if match:
                return match.group(1).upper()
        # Fallback: look for last capital letter A-E
        fallback = re.findall(r'\b([A-E])\b', response)
        return fallback[-1] if fallback else ""

    elif dataset == "strategyqa":
        # Look for yes/no
        patterns = [
            r"answer is (yes|no)",
            r"therefore, (yes|no)",
            r"so (yes|no)"
        ]
        for pat in patterns:
            match = re.search(pat, response_lower)
            if match:
                return match.group(1)
        # Fallback: last yes/no
        fallback = re.findall(r'\b(yes|no)\b', response_lower)
        return fallback[-1] if fallback else ""

    else: # gsm8k (Math)
        patterns = ["the answer is", "final answer:", "therefore,", "answer:"]
        
        # Try pattern matching
        for pattern in patterns:
            if pattern in response_lower:
                idx = response_lower.rfind(pattern)
                answer_part = response[idx + len(pattern):].strip()
                # Look for boxed answer first
                boxed = re.findall(r'\\boxed\{([^}]+)\}', answer_part)
                if boxed:
                    return boxed[0]
                # Else extract first number
                nums = re.findall(r'-?\d+\.?\d*', answer_part)
                if nums:
                    return nums[0]
        
        # Fallback: look for boxed anywhere
        boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed[-1]
            
        # Fallback: last number in text
        nums = re.findall(r'-?\d+\.?\d*', response)
        return nums[-1] if nums else ""

def check_correctness(pred, gt):
    if not pred or not gt:
        return False
        
    # String equality for non-math
    if isinstance(pred, str) and isinstance(gt, str):
        if pred.lower().strip() == gt.lower().strip():
            return True
    
    # Clean strings
    pred = str(pred).lower().replace(",", "").replace("$", "")
    gt = str(gt).lower().replace(",", "").replace("$", "")
    
    # Exact match
    if pred == gt:
        return True
        
    # Float match
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except:
        pass
        
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tagged/real_cot_tagged.json")
    parser.add_argument("--output", default="data/tagged/real_cot_tagged_corrected.json")
    parser.add_argument("--dataset", type=str, help="Force dataset type (arc, strategyqa, gsm8k)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.dataset:
        print(f"Forcing dataset type: {args.dataset}")
    

    print(f"Processing {len(data)} samples from {args.input}...")

    corrected_count = 0
    for item in data:
        ds_type = args.dataset if args.dataset else item.get('dataset', 'gsm8k')
        pred = extract_answer(item['cot_raw'], dataset=ds_type)
        is_correct = check_correctness(pred, item['ground_truth'])
        
        item['predicted_answer'] = pred
        item['is_correct'] = is_correct
        
        if is_correct:
            corrected_count += 1

    print(f"Found {corrected_count} correct samples ({corrected_count/len(data)*100:.1f}%)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {args.output}")
