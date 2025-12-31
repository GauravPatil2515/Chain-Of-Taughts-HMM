"""
CoT Generation Script
Generate Chain-of-Thought outputs from various LLMs
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Model configurations
MODEL_CONFIGS = {
    "phi-3.5": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "max_new_tokens": 512,
    },
    "qwen2.5": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_new_tokens": 512,
    },
    "llama-3.2": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "max_new_tokens": 512,
    },
}

COT_PROMPT = """Question: {question}

Please solve this step by step, showing your reasoning clearly before giving the final answer.

Solution:"""


def load_model(model_name: str):
    """Load model and tokenizer."""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Loading model: {config['model_id']}")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, config


def generate_cot(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate CoT response for a question."""
    prompt = COT_PROMPT.format(question=question)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    cot = response[len(prompt):].strip()
    return cot


def extract_final_answer(cot: str, domain: str) -> str:
    """Extract final answer from CoT."""
    cot_lower = cot.lower()
    
    # Look for common answer patterns
    patterns = [
        "the answer is",
        "final answer:",
        "therefore,",
        "so the answer is",
        "= ",
    ]
    
    for pattern in patterns:
        if pattern in cot_lower:
            idx = cot_lower.rfind(pattern)
            answer_part = cot[idx + len(pattern):].strip()
            # Get first line or up to period
            answer = answer_part.split("\n")[0].split(".")[0].strip()
            return answer
    
    # Return last line as fallback
    lines = [l.strip() for l in cot.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def check_correctness(predicted: str, ground_truth: str, domain: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    
    # Normalize numbers
    pred = pred.replace("$", "").replace(",", "")
    gt = gt.replace("$", "").replace(",", "")
    
    return pred == gt or gt in pred or pred in gt


def generate_cot_outputs(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    max_samples: int = None
):
    """Generate CoT outputs for a dataset."""
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    # Load model
    model, tokenizer, config = load_model(model_name)
    
    # Prepare output
    output_path = Path(output_dir) / f"{Path(dataset_path).stem}_{model_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    correct_count = 0
    
    for item in tqdm(dataset, desc=f"Generating CoT with {model_name}"):
        cot = generate_cot(model, tokenizer, item["question"], config["max_new_tokens"])
        predicted_answer = extract_final_answer(cot, item.get("domain", ""))
        is_correct = check_correctness(predicted_answer, item["answer"], item.get("domain", ""))
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "model": model_name,
            "cot_raw": cot,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "dataset": item.get("dataset", ""),
            "domain": item.get("domain", ""),
            "timestamp": datetime.now().isoformat(),
        })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    accuracy = correct_count / len(results) * 100
    print(f"\nResults saved to {output_path}")
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(results)})")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate CoT outputs")
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--output-dir", type=str, default="data/cot_outputs")
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    generate_cot_outputs(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
