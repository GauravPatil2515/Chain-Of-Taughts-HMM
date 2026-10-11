# CoT-HMM Real Data Generation for Colab/Kaggle
# 
# INSTRUCTIONS:
# 1. Open this in Google Colab (colab.research.google.com) or Kaggle Notebook
# 2. Enable GPU: Runtime > Change runtime type > GPU > T4/P100
# 3. Run all cells
# 4. Download the generated JSON file
#
# This generates ~300 REAL CoT traces from ACTUAL GSM8K questions using Qwen-0.5B

# @title 1. Install Dependencies
!pip install -q transformers accelerate datasets hmmlearn scikit-learn scipy tqdm

# @title 2. Import Libraries
import os
import json
import re
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# @title 3. Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_SAMPLES = 300  # Target: 300 real traces
OUTPUT_DIR = Path("/content/cot_hmm_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

PROMPT_TEMPLATE = """Solve this math problem step by step. Provide your final answer in a boxed format: \\boxed{{answer}}.

Question: {question}

Answer:"""

# @title 4. Load Real GSM8K Questions
print(f"Loading {NUM_SAMPLES} GSM8K test questions...")

ds_items = []
try:
    # Try HuggingFace dataset with official repo namespace
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    for item in ds.select(range(min(NUM_SAMPLES, len(ds)))):
        ds_items.append({'question': item['question'], 'answer': item['answer']})
except Exception as e1:
    print(f"HuggingFace dataset load error ({e1}), falling back to OpenAI GitHub raw dataset...")
    import urllib.request
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    req = urllib.request.urlopen(url)
    lines = req.read().decode('utf-8').strip().split('\n')
    for line in lines[:NUM_SAMPLES]:
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

print(f"Loaded {len(questions)} questions successfully.")

# @title 5. Load Model
print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
if DEVICE == "cpu":
    model.to(DEVICE)
model.eval()
print("Model loaded ✓")

# @title 6. Answer Extraction & Verification
def extract_final_answer(cot_text: str):
    """Extract final answer from CoT text."""
    boxed = re.findall(r'\\boxed\{([^}]+)\}', cot_text)
    if boxed:
        return boxed[-1].strip(), True
    
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
    
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', cot_text)
    if numbers:
        return numbers[-1].strip(), True
    
    return None, False


def verify_answer(predicted: str, gold: str) -> bool:
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

# @title 7. Generate CoT Traces
print(f"\nGenerating {len(questions)} CoT traces...")
traces = []

for i, q in enumerate(tqdm(questions, desc="Generating")):
    prompt = PROMPT_TEMPLATE.format(question=q['question'])
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if 'Answer:' in response:
        cot = response.split('Answer:')[-1].strip()
    else:
        cot = response[len(prompt):].strip()
    
    predicted_answer, found = extract_final_answer(cot)
    is_correct = verify_answer(predicted_answer, q['gold_answer']) if found else False
    
    trace = {
        'sample_id': q['id'],
        'problem': q['question'],
        'cot_text': cot,
        'gold_answer': q['gold_answer'],
        'predicted_answer': predicted_answer,
        'is_correct': is_correct,
        'model_name': 'qwen-0.5b',
        'model_family': 'qwen',
        'generation_config': {
            'temperature': TEMPERATURE,
            'max_new_tokens': MAX_NEW_TOKENS,
            'prompt_template': PROMPT_TEMPLATE
        }
    }
    traces.append(trace)
    
    # Periodic save
    if (i + 1) % 50 == 0:
        output_file = OUTPUT_DIR / f"gsm8k_qwen_traces_partial_{i+1}.json"
        with open(output_file, 'w') as f:
            json.dump(traces, f, indent=2)

# @title 8. Save Final Results
output_file = OUTPUT_DIR / "gsm8k_qwen_300_real_traces.json"
with open(output_file, 'w') as f:
    json.dump(traces, f, indent=2)

# Statistics
correct = sum(1 for t in traces if t['is_correct'])
incorrect = len(traces) - correct

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")
print(f"Total traces: {len(traces)}")
print(f"Correct: {correct} ({correct/len(traces)*100:.1f}%)")
print(f"Incorrect: {incorrect} ({incorrect/len(traces)*100:.1f}%)")
print(f"Saved to: {output_file}")
print(f"{'='*60}")

# @title 9. Download Results (Colab only)
try:
    from google.colab import files
    files.download(str(output_file))
    print("Download initiated...")
except ImportError:
    print("Not in Colab environment. File saved locally.")
    print(f"File location: {output_file}")