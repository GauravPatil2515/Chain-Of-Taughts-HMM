"""
Generate CoT samples for ARC-Easy dataset
Lightweight CPU-friendly version
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

COT_PROMPT = """Answer this science question step by step. Show your reasoning clearly.

Question: {question}
Choices: {choices}

Solution:"""


def load_questions(path: str, max_samples: int = None):
    """Load ARC questions."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    return data


def format_choices(choices: list) -> str:
    """Format multiple choice options."""
    return ", ".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)])


def extract_answer(response: str) -> str:
    """Extract answer choice from response."""
    response_upper = response.upper()
    # Look for A, B, C, D patterns
    for choice in ['A', 'B', 'C', 'D']:
        if f"ANSWER IS {choice}" in response_upper or f"CHOICE {choice}" in response_upper:
            return choice
    
    # Check first character
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if lines:
        first_char = lines[-1][0].upper()
        if first_char in ['A', 'B', 'C', 'D']:
            return first_char
    
    return "A"  # Default


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if answer is correct."""
    return predicted.strip().upper() == ground_truth.strip().upper()


def main():
    print("="*60)
    print("ARC-Easy CoT Generation")
    print("="*60)
    
    input_path = "data/raw/arc_easy.json"
    output_path = "data/cot_outputs/arc_local.json"
    
    print(f"\nLoading questions from {input_path}...")
    questions = load_questions(input_path, max_samples=100)
    print(f"Loaded {len(questions)} questions")
    
    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("Model loaded!\n")
    
    results = []
    correct = 0
    
    for item in tqdm(questions, desc="Generating CoT"):
        choices_str = format_choices(item.get("choices", []))
        prompt = COT_PROMPT.format(
            question=item["question"],
            choices=choices_str
        )
        
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
            "choices": item.get("choices", []),
            "ground_truth": item["answer"],
            "model": "qwen2.5-0.5b",
            "cot_raw": output.strip(),
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "dataset": "arc_easy",
            "domain": "science",
            "timestamp": datetime.now().isoformat()
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    accuracy = correct / len(results) * 100
    print(f"\n" + "="*60)
    print(f"Generation Complete!")
    print(f"="*60)
    print(f"Total: {len(results)} questions")
    print(f"Correct: {correct} ({accuracy:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
