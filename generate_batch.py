"""
Batch CoT Generation for Research Dataset
Generates CoT samples at scale with quality controls.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random


# Model configurations
MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
}

COT_PROMPT = """Question: {question}

Please solve this step by step, showing your reasoning clearly before giving the final answer.

Solution:"""


def load_questions(dataset_path: str, max_samples: int = None, seed: int = 42):
    """Load and optionally sample questions."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(data)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def extract_answer(response: str, domain: str = "math") -> str:
    """Extract final answer from CoT response."""
    response_lower = response.lower()
    
    # Look for answer patterns
    patterns = [
        ("the answer is", 14),
        ("final answer:", 13),
        ("answer:", 7),
        ("therefore, ", 11),
        ("thus, ", 6),
        ("= ", 2),
    ]
    
    for pattern, offset in patterns:
        if pattern in response_lower:
            idx = response_lower.rfind(pattern)
            answer = response[idx + offset:].strip()
            # Get first line/sentence
            answer = answer.split("\n")[0].split(".")[0].strip()
            # Extract numbers for math
            if domain == "math":
                import re
                nums = re.findall(r'-?\d+\.?\d*', answer)
                if nums:
                    return nums[0]
            return answer[:50]  # Limit length
    
    # Fallback: last line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1][:50] if lines else ""


def check_correctness(predicted: str, ground_truth: str, domain: str = "math") -> bool:
    """Check if predicted answer matches ground truth."""
    pred = predicted.lower().replace("$", "").replace(",", "").strip()
    gt = ground_truth.lower().replace("$", "").replace(",", "").strip()
    
    if domain == "math":
        # Extract numbers
        import re
        pred_nums = re.findall(r'-?\d+\.?\d*', pred)
        gt_nums = re.findall(r'-?\d+\.?\d*', gt)
        
        if pred_nums and gt_nums:
            try:
                return abs(float(pred_nums[-1]) - float(gt_nums[-1])) < 0.01
            except:
                pass
    
    # String matching
    return gt in pred or pred in gt


def generate_batch(
    questions: list,
    model_name: str,
    output_path: str,
    dataset_name: str,
    domain: str,
    generation_params: dict,
    batch_size: int = 10
):
    """Generate CoT outputs for a batch of questions."""
    
    # Load model
    print(f"\nLoading model: {MODELS[model_name]}")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name],
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=generation_params["max_new_tokens"],
        temperature=generation_params["temperature"],
        top_p=generation_params["top_p"],
        do_sample=generation_params["do_sample"],
        pad_token_id=tokenizer.eos_token_id
    )
    
    print(f"Model loaded! Generating {len(questions)} samples...")
    
    results = []
    correct_count = 0
    
    for item in tqdm(questions, desc=f"Generating ({model_name})"):
        prompt = COT_PROMPT.format(question=item["question"])
        
        try:
            output = generator(prompt, return_full_text=False)[0]["generated_text"]
        except Exception as e:
            print(f"\nError on {item['id']}: {e}")
            output = f"[Generation Error: {str(e)}]"
        
        predicted = extract_answer(output, domain)
        is_correct = check_correctness(predicted, item.get("answer", ""), domain)
        
        if is_correct:
            correct_count += 1
        
        result = {
            "id": f"{dataset_name}_{model_name}_{item['id']}",
            "dataset": dataset_name,
            "domain": domain,
            "question": item["question"],
            "ground_truth": item.get("answer", ""),
            "model": model_name,
            "model_full_name": MODELS[model_name],
            "cot_raw": output.strip(),
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "generation_params": generation_params,
            "timestamp": datetime.now().isoformat(),
            "prompt_template": "standard_cot"
        }
        
        results.append(result)
        
        # Save intermediate results every batch_size
        if len(results) % batch_size == 0:
            _save_results(results, output_path)
    
    # Final save
    _save_results(results, output_path)
    
    # Statistics
    accuracy = correct_count / len(results) * 100 if results else 0
    print(f"\n{'='*60}")
    print(f"Generation Complete: {model_name} on {dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Correct: {correct_count} ({accuracy:.1f}%)")
    print(f"Incorrect: {len(results) - correct_count} ({100-accuracy:.1f}%)")
    print(f"Saved to: {output_path}")
    
    return results


def _save_results(results: list, path: str):
    """Save results to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Batch CoT Generation")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to dataset JSON")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name (gsm8k, strategyqa, arc)")
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain (math, commonsense, logic)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()),
                        help="Model to use")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Load questions
    print(f"Loading questions from {args.dataset}")
    questions = load_questions(args.dataset, args.max_samples, args.seed)
    print(f"Loaded {len(questions)} questions")
    
    # Generation parameters
    gen_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 0.9,
        "do_sample": True,
        "random_seed": args.seed
    }
    
    # Generate
    generate_batch(
        questions=questions,
        model_name=args.model,
        output_path=args.output,
        dataset_name=args.dataset_name,
        domain=args.domain,
        generation_params=gen_params
    )


if __name__ == "__main__":
    main()
