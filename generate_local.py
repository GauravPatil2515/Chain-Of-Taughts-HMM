"""
Local CoT Generation Script
Generates Chain-of-Thought using a lightweight local model.
Designed to work on CPU with limited RAM.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use a very lightweight model that works on CPU
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Only 0.5B params, works on CPU

COT_PROMPT = """Solve this math problem step by step. Show your work clearly.

Question: {question}

Solution:"""


def load_questions(path: str, max_samples: int = None):
    """Load questions from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    return data


def extract_answer(response: str) -> str:
    """Extract final answer from response."""
    response_lower = response.lower()
    
    # Look for answer patterns
    patterns = ["the answer is", "final answer:", "= ", "therefore"]
    for pattern in patterns:
        if pattern in response_lower:
            idx = response_lower.rfind(pattern)
            answer = response[idx + len(pattern):].strip()
            # Get first number or word
            answer = answer.split("\n")[0].split(".")[0].strip()
            return answer
    
    # Return last line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if answer is correct."""
    pred = predicted.lower().replace("$", "").replace(",", "").strip()
    gt = ground_truth.lower().replace("$", "").replace(",", "").strip()
    
    # Try to extract numbers
    import re
    pred_nums = re.findall(r'-?\d+\.?\d*', pred)
    gt_nums = re.findall(r'-?\d+\.?\d*', gt)
    
    if pred_nums and gt_nums:
        return pred_nums[-1] == gt_nums[-1]
    
    return gt in pred or pred in gt


def main():
    print("="*60)
    print("Local CoT Generation (Lightweight)")
    print("="*60)
    
    # Paths
    input_path = "data/raw/gsm8k.json"
    output_path = "data/cot_outputs/gsm8k_local.json"
    
    # Load questions (generate all 100 for Phase 1)
    print(f"\nLoading questions from {input_path}...")
    questions = load_questions(input_path, max_samples=100)
    print(f"Loaded {len(questions)} questions")
    
    # Load model
    print(f"\nLoading model: {MODEL_ID}")
    print("This may take a few minutes on first run...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,  # Allow longer CoTs
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("Model loaded!\n")
    
    # Generate CoTs
    results = []
    correct = 0
    
    for item in tqdm(questions, desc="Generating CoT"):
        prompt = COT_PROMPT.format(question=item["question"])
        
        try:
            output = generator(prompt, return_full_text=False)[0]["generated_text"]
        except Exception as e:
            print(f"Error on {item['id']}: {e}")
            output = "Error generating response"
        
        predicted = extract_answer(output)
        is_correct = check_answer(predicted, item["answer"])
        
        if is_correct:
            correct += 1
        
        results.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "model": "qwen2.5-0.5b",
            "cot_raw": output.strip(),
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "dataset": "gsm8k",
            "domain": "math",
            "timestamp": datetime.now().isoformat()
        })
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary
    accuracy = correct / len(results) * 100
    print(f"\n" + "="*60)
    print(f"Generation Complete!")
    print(f"="*60)
    print(f"Total: {len(results)} questions")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Saved to: {output_path}")
    print(f"\nNext step: Run tagging with:")
    print(f"  python -c \"from src.tagging.rule_based_tagger import tag_cot_file; tag_cot_file('{output_path}', 'data/tagged/gsm8k_tagged.json')\"")


if __name__ == "__main__":
    main()
