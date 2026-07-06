"""
Simplified CoT Generation - Direct inference without pipeline
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model_simple():
    """Load model without using pipeline to avoid torchvision dependency."""
    print(f"Loading model: {MODEL_ID}")
    
    # Import only what we need
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    
    return tokenizer, model


def generate_text(tokenizer, model, prompt, max_length=512):
    """Generate text without using pipeline."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def extract_answer_math(response: str) -> str:
    """Extract answer from math CoT."""
    response_lower = response.lower()
    
    import re
    # Look for answer patterns
    patterns = ["the answer is", "final answer:", "therefore", "= "]
    for pattern in patterns:
        if pattern in response_lower:
            idx = response_lower.rfind(pattern)
            answer = response[idx + len(pattern):].strip()
            # Extract first number
            nums = re.findall(r'-?\d+\.?\d*', answer)
            if nums:
                return nums[0]
    
    # Return last number
    nums = re.findall(r'-?\d+\.?\d*', response)
    return nums[-1] if nums else ""


def check_answer_math(predicted: str, ground_truth: str) -> bool:
    """Check math answer."""
    import re
    pred_nums = re.findall(r'-?\d+\.?\d*', predicted.replace(",", ""))
    gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth.replace(",", ""))
    
    if pred_nums and gt_nums:
        try:
            return abs(float(pred_nums[-1]) - float(gt_nums[-1])) < 0.01
        except:
            return pred_nums[-1] == gt_nums[-1]
    
    return False


def generate_gsm8k():
    """Generate GSM8K dataset."""
    print("="*70)
    print("GSM8K CoT Generation (Simplified)")
    print("="*70)
    
    # Load data
    with open("data/raw/gsm8k.json", "r") as f:
        questions = json.load(f)
    
    print(f"\nLoaded {len(questions)} questions")
    
    # Load model
    tokenizer, model = load_model_simple()
    print("Model loaded!\n")
    
    # Generate
    results = []
    correct = 0
    
    for item in tqdm(questions, desc="Generating"):
        prompt = f"""Solve this math problem step by step. Show your work clearly.

Question: {item['question']}

Solution:"""
        
        try:
            output = generate_text(tokenizer, model, prompt, max_length=400)
        except Exception as e:
            print(f"\nError on {item['id']}: {e}")
            output = f"[Error: {str(e)}]"
        
        predicted = extract_answer_math(output)
        is_correct = check_answer_math(predicted, item["answer"])
        
        if is_correct:
            correct += 1
        
        results.append({
            "id": f"gsm8k_qwen_{ item['id']}",
            "dataset": "gsm8k",
            "domain": "math",
            "question": item["question"],
            "ground_truth": item["answer"],
            "model": "qwen2.5-0.5b",
            "cot_raw": output,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save every 25 samples
        if len(results) % 25 == 0:
            Path("data/cot_outputs").mkdir(parents=True, exist_ok=True)
            with open("data/cot_outputs/gsm8k_qwen_100.json", "w") as f:
                json.dump(results, f, indent=2)
    
    # Final save
    with open("data/cot_outputs/gsm8k_qwen_100.json", "w") as f:
        json.dump(results, f, indent=2)
    
    accuracy = correct / len(results) * 100
    print(f"\n{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Total: {len(results)} samples")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Incorrect: {len(results) - correct} ({100-accuracy:.1f}%)")
    print(f"Saved to: data/cot_outputs/gsm8k_qwen_100.json")


def generate_arc():
    """Generate ARC dataset."""
    print("="*70)
    print("ARC-Easy CoT Generation")
    print("="*70)
    
    with open("data/raw/arc_easy.json", "r") as f:
        questions = json.load(f)
    
    print(f"\nLoaded {len(questions)} questions")
    
    tokenizer, model = load_model_simple()
    print("Model loaded!\n")
    
    results = []
    correct = 0
    
    for item in tqdm(questions, desc="Generating"):
        choices_str = ", ".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item.get("choices", []))])
        prompt = f"""Answer this science question step by step. Show your reasoning clearly.

Question: {item['question']}
Choices: {choices_str}

Solution:"""
        
        try:
            output = generate_text(tokenizer, model, prompt, max_length=400)
        except Exception as e:
            print(f"\nError on {item['id']}: {e}")
            output = f"[Error: {str(e)}]"
        
        # Extract answer choice
        output_upper = output.upper()
        predicted = "A"  # default
        for choice in ['A', 'B', 'C', 'D']:
            if f"ANSWER IS {choice}" in output_upper or f"CHOICE {choice}" in output_upper:
                predicted = choice
                break
        
        is_correct = predicted == item["answer"].upper()
        
        if is_correct:
            correct += 1
        
        results.append({
            "id": f"arc_{item['id']}",
            "dataset": "arc_easy",
            "domain": "science",
            "question": item["question"],
            "choices": item.get("choices", []),
            "ground_truth": item["answer"],
            "model": "qwen2.5-0.5b",
            "cot_raw": output,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(results) % 25 == 0:
            Path("data/cot_outputs").mkdir(parents=True, exist_ok=True)
            with open("data/cot_outputs/arc_qwen_100.json", "w") as f:
                json.dump(results, f, indent=2)
    
    with open("data/cot_outputs/arc_qwen_100.json", "w") as f:
        json.dump(results, f, indent=2)
    
    accuracy = correct / len(results) * 100
    print(f"\n{'='*70}")
    print(f"Generation Complete!")
    print(f"{'='*70}")
    print(f"Total: {len(results)} samples")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Saved to: data/cot_outputs/arc_qwen_100.json")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "arc":
        generate_arc()
    else:
        generate_gsm8k()
