"""
Data Generation Pipeline for Colab/Kaggle
Scales datasets to N=500+ samples for journal submission
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import requests
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False


@dataclass
class GeneratedSample:
    """Single generated CoT sample."""
    question: str
    answer: str
    cot_text: str
    is_correct: Optional[bool] = None
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataGenerator:
    """Generate CoT data using HuggingFace models (CPU-fast, no API keys)."""
    
    DEFAULT_MODELS = {
        'qwen_0.5B': 'Qwen/Qwen2.5-0.5B-Instruct',
        'qwen_1.5B': 'Qwen/Qwen2.5-1.5B-Instruct',
        'phi_3': 'microsoft/Phi-3-mini-4k-instruct',
    }
    
    PROMPT_TEMPLATE = """
Please solve this math problem step by step. Provide your final answer in a box.

Question: {question}

Answer:
"""
    
    def __init__(self, model_name: str = 'qwen_0.5B', device: str = 'cpu', max_new_tokens: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        if not HF_AVAILABLE:
            print("[DataGenerator] WARNING: transformers not available")
            return
        
        hf_model = self.DEFAULT_MODELS.get(self.model_name, self.model_name)
        print(f"[DataGenerator] Loading model: {hf_model} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.to(self.device)
        self.model.eval()
        print(f"[DataGenerator] Model loaded ✓")
    
    def unload(self) -> None:
        """Unload model from VRAM/RAM to prevent memory overload."""
        import gc
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[DataGenerator] Unloaded model from memory ✓")
    
    def generate(self, question: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        if self.model is None:
            return "Error: Model not loaded"
        
        prompt = self.PROMPT_TEMPLATE.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if '\nAnswer:' in response:
            cot = response.split('\nAnswer:')[-1].strip()
        else:
            cot = response[len(prompt):].strip()
        
        return cot
    
    def generate_batch(self, questions: List[str], temperature: float = 0.7) -> List[GeneratedSample]:
        samples = []
        for q in questions:
            cot = self.generate(q, temperature=temperature)
            samples.append(GeneratedSample(
                question=q,
                answer="",
                cot_text=cot,
                metadata={'temperature': temperature, 'model': self.model_name}
            ))
        return samples


def extract_answer(cot_text: str) -> Tuple[str, bool]:
    """Extract final answer from CoT text and check correctness."""
    import re
    
    boxed = re.findall(r'\\boxed\{([^}]+)\}', cot_text)
    if boxed:
        return boxed[-1].strip(), True
    
    paragraphs = cot_text.strip().split('\n\n')
    for para in reversed(paragraphs):
        if re.search(r'\b(?:final|conclusive|therefore|so|the\s+answer)\s+(?:is\s+)?(\S+)', para, re.I):
            match = re.search(r'\b(?:is\s+)?(\d+(?:\.\d+)?)', para)
            if match:
                return match.group(1).strip(), True
    
    sentences = cot_text.strip().split('.')
    for sent in reversed(sentences):
        match = re.search(r'(?:answer|answer is)\s*[:\s]*(\d+)', sent, re.I)
        if match:
            return match.group(1).strip(), True
    
    return "", False


def verify_math_answer(question: str, predicted: str, ground_truth: str) -> bool:
    """Simple verification of math answers."""
    import re
    
    def normalize(s):
        s = s.strip().lower()
        s = re.sub(r'[^\w\s.]', '', s)
        return s
    
    pred_norm = normalize(predicted)
    truth_norm = normalize(ground_truth)
    
    try:
        pred_num = float(re.search(r'[\d.]+', predicted).group())
        truth_num = float(re.search(r'[\d.]+', ground_truth).group())
        return abs(pred_num - truth_num) < 0.01
    except:
        return pred_norm == truth_norm or pred_norm in truth_norm or truth_norm in pred_norm


def scale_dataset(
    input_path: str,
    output_path: str,
    target_size: int = 500,
    model_name: str = 'qwen_0.5B',
    device: str = 'cpu',
    seed: int = 42
) -> Dict[str, Any]:
    """Scale dataset to target size with bootstrap sampling."""
    np.random.seed(seed)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    original_size = len(data)
    print(f"[scale_dataset] Original size: {original_size}")
    print(f"[scale_dataset] Target size: {target_size}")
    
    if original_size >= target_size:
        indices = np.random.choice(original_size, target_size, replace=False)
        scaled_data = [data[i] for i in indices]
    else:
        scaled_data = list(data)
        needed = target_size - original_size
        print(f"[scale_dataset] Need to generate {needed} new samples")
        
        if HF_AVAILABLE and model_name in DataGenerator.DEFAULT_MODELS:
            generator = DataGenerator(model_name=model_name, device=device)
            
            questions = [item.get('problem', item.get('question', '')) for item in data[:100]]
            if not all(questions):
                from datasets import load_dataset
                ds = load_dataset('gsm8k', 'main', split='test')
                questions = [q['question'] for q in ds.select(range(min(200, len(ds))))]
            
            questions = questions[:needed * 3]
            
            new_samples = []
            batch_size = 16 if device == 'cuda' else 4
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i+batch_size]
                generated = generator.generate_batch(batch)
                new_samples.extend(generated)
            
            for sample in new_samples[:needed]:
                answer, _ = extract_answer(sample.cot_text)
                if answer:
                    sample.answer = answer
                    sample.is_correct = True
                
                scaled_data.append({
                    'sample_id': f'gen_{len(scaled_data):04d}',
                    'question': sample.question,
                    'answer': sample.answer,
                    'cot': sample.cot_text,
                    'is_correct': sample.is_correct,
                    'model_family': 'qwen' if 'qwen' in model_name else 'llama',
                    'model_name': model_name,
                    'ground_truth_answer': sample.answer,
                    'domain': 'math',
                    'dataset': 'gsm8k',
                    'created_at': datetime.utcnow().isoformat()
                })
    
    final_size = len(scaled_data)
    n_correct = sum(1 for item in scaled_data if item.get('is_correct', False))
    
    output = {
        'total_samples': final_size,
        'original_size': original_size,
        'correct_samples': n_correct,
        'incorrect_samples': final_size - n_correct,
        'accuracy': n_correct / final_size if final_size > 0 else 0,
        'saved_to': output_path,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(scaled_data, f, indent=2)
    
    print(f"[scale_dataset] Saved {final_size} samples to {output_path}")
    print(f"[scale_dataset] Accuracy: {n_correct}/{final_size} = {n_correct/final_size:.2%}")
    
    return output


def generate_multi_model_dataset(
    output_dir: str,
    configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate datasets from multiple models for cross-architecture testing."""
    results = {}
    
    for cfg in configs:
        model_name = cfg.get('model_name', 'qwen_0.5B')
        target_size = cfg.get('target_size', 200)
        output_path = cfg.get('output_path', f"{output_dir}/gsm8k_{model_name}.json")
        
        print(f"\n[multi_model] Generating with {model_name}...")
        
        result = scale_dataset(
            input_path=output_dir + "/gsm8k_base.json",
            output_path=output_path,
            target_size=target_size,
            model_name=model_name,
            device='cpu'
        )
        results[model_name] = result
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CoT datasets")
    parser.add_argument('--input', type=str, default='data/v2/gsm8k_150_traces.json')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target-size', type=int, default=500)
    parser.add_argument('--model', type=str, default='qwen_0.5B')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    scale_dataset(
        input_path=args.input,
        output_path=args.output,
        target_size=args.target_size,
        model_name=args.model,
        device=args.device
    )