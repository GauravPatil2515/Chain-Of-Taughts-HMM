"""
Multi-Dataset CoT Generation
Generates CoT for ARC and StrategyQA datasets.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct", 
}

def load_model(model_name):
    print(f"Loading model: {MODELS[model_name]}")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name],
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def generate_one(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def format_arc_prompt(item):
    """Format ARC multiple choice question."""
    choices_str = "\n".join([f"{k}: {v}" for k, v in item['choices'].items()])
    return f"Question: {item['question']}\n\nChoices:\n{choices_str}\n\nThink step by step and select the best answer.\nSolution:"

def format_strategyqa_prompt(item):
    """Format StrategyQA yes/no question."""
    return f"Question: {item['question']}\n\nThink step by step and answer yes or no.\nSolution:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["arc", "strategyqa"], required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="qwen-0.5b", choices=list(MODELS.keys()))
    args = parser.parse_args()
    
    # Load dataset
    input_file = f"data/raw/{'arc_challenge' if args.dataset == 'arc' else 'strategyqa'}.json"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)[:args.limit]
    
    # Load model
    tokenizer, model = load_model(args.model)
    
    # Output file
    output_file = f"data/cot_outputs/{args.dataset}_{args.model}_{args.limit}.json"
    incremental_file = f"data/cot_outputs/{args.dataset}_incremental.json"
    
    # Check for resume
    results = []
    if Path(incremental_file).exists():
        try:
            with open(incremental_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} samples...")
        except:
            pass
    
    remaining = data[len(results):]
    
    if not remaining:
        print("All samples already generated!")
    else:
        print(f"\nGenerating {len(remaining)} {args.dataset} samples...")
        
        for i, item in enumerate(tqdm(remaining)):
            if args.dataset == "arc":
                prompt = format_arc_prompt(item)
                gt = item['answer']
            else:
                prompt = format_strategyqa_prompt(item)
                gt = item['answer']
            
            output = generate_one(tokenizer, model, prompt)
            
            results.append({
                "id": item['id'],
                "question": item['question'],
                "cot_raw": output,
                "ground_truth": gt,
                "model": args.model,
                "dataset": args.dataset,
                "is_real_llm": True
            })
            
            # Save every 5
            if len(results) % 5 == 0:
                with open(incremental_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
