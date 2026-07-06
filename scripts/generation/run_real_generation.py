"""
Robust Real CoT Generation
Uses manual model loading to avoid pipeline hangs. Low memory usage for CPU.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of samples")
    parser.add_argument("--model", type=str, default="qwen-0.5b", choices=MODELS.keys(), help="Model to use")
    args = parser.parse_args()

    # Load data
    with open("data/raw/gsm8k.json", "r") as f:
        data = json.load(f)[:args.limit]

    # Load model
    tokenizer, model = load_model(args.model)

    # Check for existing progress
    model_suffix = args.model.replace("-", "_")
    incremental_file = f"data/cot_outputs/incremental_{model_suffix}_{args.limit}.json"
    results = []
    
    if Path(incremental_file).exists():
        try:
            with open(incremental_file, "r") as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} samples...")
        except:
            print("Could not load incremental file, starting fresh.")
            
    # filter data to only what is remaining
    remaining_data = data[len(results):]
    
    if not remaining_data:
        print("All samples already generated!")
    else:
        print(f"\nGenerating {len(remaining_data)} samples...")
        
        for i, item in enumerate(tqdm(remaining_data)):
            # Adjust index for ID
            true_idx = len(results) 
            
            prompt = f"Question: {item['question']}\nSolve step by step.\nSolution:"
            output = generate_one(tokenizer, model, prompt)
            
            results.append({
                "id": f"real_{item.get('id', true_idx)}",
                "question": item['question'],
                "cot_raw": output,
                "ground_truth": item['answer'],
                "model": args.model,
                "is_real_llm": True
            })
            
            # Save every 5 samples
            if len(results) % 5 == 0:
                with open(incremental_file, "w") as f:
                    json.dump(results, f, indent=2)
        


    # Save
    outfile = f"data/cot_outputs/test_{model_suffix}_{args.limit}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")

if __name__ == "__main__":
    main()
