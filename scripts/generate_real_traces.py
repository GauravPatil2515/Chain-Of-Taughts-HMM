#!/usr/bin/env python3
"""
Generate REAL CoT traces from ACTUAL GSM8K questions using local GPU.
This replaces all synthetic/fabricated data with genuine LLM outputs.
"""

import json
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


MODEL_CONFIGS = {
    'qwen-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',  # Requires HF token
    'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',
}

# Use Qwen-0.5B as primary (no auth needed, fast)
DEFAULT_MODEL = 'qwen-0.5B'

PROMPT_TEMPLATE = """Solve this math problem step by step. Provide your final answer in a boxed format: \boxed{{answer}}.

Question: {question}

Answer:"""


def load_gsm8k_questions(limit: int = 300):
    """Load real GSM8K test questions."""
    ds_items = []
    try:
        from datasets import load_dataset
        ds = load_dataset('openai/gsm8k', 'main', split='test')
        for item in ds.select(range(min(limit, len(ds)))):
            ds_items.append({'question': item['question'], 'answer': item['answer']})
    except Exception as e:
        print(f"Dataset load warning ({e}), loading directly from OpenAI GitHub repo...")
        import urllib.request
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        req = urllib.request.urlopen(url)
        lines = req.read().decode('utf-8').strip().split('\n')
        for line in lines[:limit]:
            ds_items.append(json.loads(line))
            
    questions = []
    for item in ds_items:
        match = re.search(r'####\s*(\d+)', item['answer'])
        gold_answer = match.group(1) if match else item['answer'].strip()
        questions.append({
            'id': f'gsm8k_{len(questions):04d}',
            'question': item['question'],
            'gold_answer': gold_answer
        })
    return questions


def extract_final_answer(cot_text: str) -> tuple:
    """Extract final answer from CoT text."""
    # Pattern 1: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', cot_text)
    if boxed:
        return boxed[-1].strip(), True
    
    # Pattern 2: Explicit answer statements
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+(\S+)',
        r'final\s+answer[:\s]+(\S+)',
        r'therefore[,\s]+(?:the\s+)?answer\s+(?:is\s+)?(\S+)',
        r'so\s+(?:the\s+)?answer\s+(?:is\s+)?(\S+)',
        r'answer:\s*(\S+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cot_text, re.IGNORECASE)
        if matches:
            return matches[-1].strip(), True
    
    # Pattern 3: Last number (fallback)
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', cot_text)
    if numbers:
        return numbers[-1].strip(), True
    
    return None, False


def verify_answer(predicted: str, gold: str) -> bool:
    """Verify predicted answer matches gold."""
    def normalize(s):
        if s is None:
            return ""
        return re.sub(r'[,\$\s]', '', s.strip())
    
    pred_norm = normalize(predicted)
    gold_norm = normalize(gold)
    
    if not pred_norm or not gold_norm:
        return False
    
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 0.01
    except ValueError:
        return pred_norm.lower() == gold_norm.lower()


def generate_cot_traces(
    model_name: str = DEFAULT_MODEL,
    num_samples: int = 300,
    output_path: str = "data/real/gsm8k_real_traces.json",
    device: str = "cuda"
):
    """Generate real CoT traces from actual GSM8K questions."""
    
    print(f"Loading model: {MODEL_CONFIGS[model_name]}")
    model_id = MODEL_CONFIGS[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    
    print(f"Loading {num_samples} GSM8K questions...")
    questions = load_gsm8k_questions(num_samples)
    
    print(f"Generating CoT traces...")
    traces = []
    
    for i, q in enumerate(tqdm(questions, desc="Generating")):
        prompt = PROMPT_TEMPLATE.format(question=q['question'])
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract CoT (text after "Answer:")
        if 'Answer:' in response:
            cot = response.split('Answer:')[-1].strip()
        else:
            cot = response[len(prompt):].strip()
        
        # Extract and verify answer
        predicted_answer, found = extract_final_answer(cot)
        is_correct = verify_answer(predicted_answer, q['gold_answer']) if found else False
        
        trace = {
            'sample_id': q['id'],
            'problem': q['question'],
            'cot_text': cot,
            'gold_answer': q['gold_answer'],
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'model_name': model_name,
            'model_family': 'qwen' if 'qwen' in model_name.lower() else 'other'
        }
        traces.append(trace)
        
        # Periodic save
        if (i + 1) % 50 == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(traces, f, indent=2)
            print(f"  Saved {i+1} traces...")
    
    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(traces, f, indent=2)
    
    # Statistics
    correct = sum(1 for t in traces if t['is_correct'])
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Total traces: {len(traces)}")
    print(f"Correct: {correct} ({correct/len(traces)*100:.1f}%)")
    print(f"Incorrect: {len(traces)-correct}")
    print(f"Saved to: {output_path}")
    
    return traces


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate real GSM8K CoT traces")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--output", type=str, default="data/real/gsm8k_real_traces.json")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    generate_cot_traces(
        model_name=args.model,
        num_samples=args.samples,
        output_path=args.output,
        device=args.device
    )